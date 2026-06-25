#!/usr/bin/env python3
# Copyright 2026
#
# Seed a parquet's INITIAL per-problem hint budget B_q from a budget-state JSON.
#
# Re-initializes each row's baked budget from a saved budget-state file (the
# {problem_id: B_q} table BudgetManager writes, e.g. a previously-ratcheted or
# hint-wise-calibrated state) -- i.e. it bakes that table into the dataset as the
# STARTING curriculum. A fresh training run (fresh data.hprl.budget_state_path)
# then begins from these budgets and the ratchet evolves from there.
#
#   * If a row's problem_id IS in the budget-state file -> its budget is set to the
#     table value (clamped to [--min-budget, --max-budget]).
#   * If a row's problem_id is NOT in the file -> the row's budget is KEPT AS-IS
#     (read from extra_info.tools_kwargs.request_hint.create_kwargs.budget, falling
#     back to extra_info.hprl_init_budget).
#
# Per updated row it writes BOTH budget fields the rest of the pipeline reads, so
# they stay consistent:
#   - extra_info.tools_kwargs.request_hint.create_kwargs.budget  (authoritative:
#       the dataset's baked-budget lookup get_create_budget, the budget sampler, and
#       the ratchet read-back _gen_budget_from_extra all read this first)
#   - extra_info.hprl_init_budget                                 (recorded init +
#       _gen_budget_from_extra fallback)
#
# Prompt handling depends on the row TYPE:
#   - AUTO-HINT rows (extra_info.hprl_auto_hint, or simply no hprl_system_base): the
#     prompt is HINT-AGNOSTIC (the policy is never told a budget), so ONLY the two
#     budget fields are updated -- the prompt is left untouched, exactly as
#     HintBudgetDataset._update_auto_hint_budget does at sample time.
#   - TEMPLATED <hint_call/> rows (carry hprl_system_base/hprl_user_base): the
#     budget appears in the system + last-user message, so the prompt is re-rendered
#     for the new budget via set_hint_budget.set_row_budget (the canonical re-render).
#
# Non-destructive by default: --out defaults to "<in stem>-rebudget.parquet" (a new
# file), so the original parquet is never clobbered unless you pass --out == --in.
#
# Usage:
#   python apply_budget_state.py \
#       --in  dataset/dapo-3139-auto-hint.parquet \
#       --budget-state budget_state_hint_wise.json \
#       --out dataset/dapo-3139-auto-hint-hintwise.parquet

from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

# This module now lives in budget_calibration/; put the hint_rl package dir (its
# parent) on sys.path so budget_manager (which stays in the package) still imports.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budget_manager import get_create_budget, load_budget_table


def _clamp(b: int, lo: int, hi: int) -> int:
    return max(lo, min(int(b), hi))


def _read_current_budget(extra_info: dict) -> int:
    """The row's existing baked budget: create_kwargs.budget, else hprl_init_budget, else 0."""
    fallback = int(extra_info.get("hprl_init_budget") or 0)
    return get_create_budget(extra_info.get("tools_kwargs"), fallback)


def _set_auto_hint_budget(extra_info: dict, budget: int) -> dict:
    """Auto-hint (push-hint) row: set ONLY the two budget fields; the prompt is
    hint-agnostic (no budget sentence) so it is left untouched. Returns a new
    extra_info dict with deep-copied tools_kwargs (so the source row is untouched)."""
    extra_info = dict(extra_info)
    extra_info["hprl_init_budget"] = int(budget)
    tools_kwargs = dict(extra_info.get("tools_kwargs") or {})
    req = dict(tools_kwargs.get("request_hint") or {})
    ck = dict(req.get("create_kwargs") or {})
    ck["budget"] = int(budget)
    req["create_kwargs"] = ck
    tools_kwargs["request_hint"] = req
    extra_info["tools_kwargs"] = tools_kwargs
    return extra_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="input parquet")
    ap.add_argument("--budget-state", dest="state_path", required=True,
                    help="budget-state JSON ({problem_id: B_q}, BudgetManager.save format)")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="output parquet (default: '<in stem>-rebudget.parquet'; pass the input "
                         "path to overwrite in place)")
    ap.add_argument("--min-budget", type=int, default=0, help="floor on the applied budget")
    ap.add_argument("--max-budget", type=int, default=8,
                    help="ceiling on the applied budget (<= max_assistant_turns; the auto-hint "
                         "loop can inject at most this many hints)")
    args = ap.parse_args()

    if args.out_path:
        out_path = args.out_path
    else:
        stem, ext = os.path.splitext(args.in_path)
        out_path = f"{stem}-rebudget{ext}"

    table = load_budget_table(args.state_path)
    if not table:
        raise SystemExit(f"budget-state file {args.state_path} has no 'budgets' table (or is unreadable)")
    print(f"read {len(table)} problem budgets from {args.state_path} "
          f"(values {min(table.values())}..{max(table.values())})")

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")
    rows = df.to_dict(orient="records")

    before, after = [], []
    matched = unmatched = clamped = templated = 0
    new_rows = []
    for r in rows:
        extra_info = r["extra_info"]
        pid = extra_info.get("problem_id")
        cur = _read_current_budget(extra_info)
        before.append(cur)

        if pid is not None and str(pid) in table:
            raw = table[str(pid)]
            new_b = _clamp(raw, args.min_budget, args.max_budget)
            clamped += int(new_b != raw)
            matched += 1
            after.append(new_b)
            is_auto_hint = bool(extra_info.get("hprl_auto_hint")) or ("hprl_system_base" not in extra_info)
            if is_auto_hint:
                r = dict(r)
                r["extra_info"] = _set_auto_hint_budget(extra_info, new_b)
            else:
                # templated <hint_call/> row -> re-render the prompt too (lazy import so
                # the auto-hint path carries no prepare_hint_data/hint_prompt dependency).
                from set_hint_budget import set_row_budget

                r = set_row_budget(r, new_b)
                templated += 1
            new_rows.append(r)
        else:
            unmatched += 1
            after.append(cur)
            new_rows.append(r)  # keep as-is

    out = pd.DataFrame(new_rows)
    out.to_parquet(out_path, index=False)

    bd = lambda xs: dict(sorted(Counter(xs).items()))  # noqa: E731
    print(f"matched (set from budget-state): {matched}   unmatched (kept as-is): {unmatched}"
          + (f"   templated re-rendered: {templated}" if templated else "")
          + (f"   clamped to [{args.min_budget},{args.max_budget}]: {clamped}" if clamped else ""))
    print(f"budget dist BEFORE: {bd(before)}")
    print(f"budget dist AFTER : {bd(after)}")
    n_changed = sum(1 for a, b in zip(after, before) if a != b)
    print(f"rows whose budget changed: {n_changed}/{len(out)}")
    print(f"wrote {len(out)} rows to {out_path}" + ("  (IN-PLACE)" if out_path == args.in_path else ""))


if __name__ == "__main__":
    main()
