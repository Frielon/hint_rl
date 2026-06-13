#!/usr/bin/env python3
# Copyright 2026
#
# Re-assign the hint budget B_q on an ALREADY-TEMPLATED multi-turn HPRL parquet.
#
# prepare_hint_data.py does two things in one pass: (1) wrap raw rows in the
# hint-tool template, and (2) set each row's budget B_q. This script is JUST step
# (2), factored out so a new zero-budget curriculum -- the set of problem_ids the
# policy already solves unaided (extract_unaided_solved.py) -- can be applied to
# an existing *-mt.parquet WITHOUT regenerating the template, re-joining the
# hint_full penalty pool, or re-stripping the CoT triggers.
#
# Input : a multi-turn parquet produced by prepare_hint_data.py, e.g.
#   dataset/dapo-3740-hint-verl-simplified-mt.parquet
#   -- every row must already carry extra_info.hprl_system_base / hprl_user_base
#      (the budget-free bases) and extra_info.tools_kwargs.request_hint.create_kwargs.
#
# Per row it:
#   * recomputes the natural budget  B_q = #major steps in the hint pool, capped
#     at --max-budget (the SAME rule as prepare_hint_data.upgrade_row);
#   * forces B_q = 0 for any problem_id in --zero-budget-ids;
#   * re-renders the budget-dependent fields from the budget-free bases:
#       - prompt[0] (system) = render_system(hprl_system_base, B_q)
#       - last user message  = render_user(hprl_user_base, B_q)
#       - extra_info.hprl_init_budget = B_q
#       - extra_info.tools_kwargs.request_hint.create_kwargs.{budget, problem}
# Everything else (hint pool, hint_full, ground_truth, agent_name, other messages)
# is left untouched.
#
# Idempotent: run with the SAME --max-budget and --zero-budget-ids that
# prepare_hint_data used and the parquet is reproduced unchanged. By default it
# updates the dataset IN PLACE (--out defaults to --in); pass --out to fork.
#
# At budget 0 render_system/render_user still emit the full tool template (budget
# shown as 0, user reminder "no hint calls remaining, finish on your own"), so the
# zero-budget prompt matches the budget>0 style and only the budget digit differs.
#
# Usage:
#   python set_hint_budget.py --in dataset/dapo-3740-hint-verl-simplified-mt.parquet \
#       --zero-budget-ids easy_ids.txt                    # in-place
#   python set_hint_budget.py --in IN --out OUT --zero-budget-ids easy_ids.txt
#   python set_hint_budget.py --in IN --max-budget 6      # re-cap only, no zero set

from __future__ import annotations

import argparse
import os

import pandas as pd

from hint_prompt import render_system, render_user
# num_steps + user_problem are the budget-relevant pure helpers from the prep
# script; reuse them so the #major-steps rule stays a single source of truth.
from prepare_hint_data import num_steps, user_problem


def compute_budget(hint_str: str, max_budget: int) -> int:
    """Natural budget B_q = #major steps in the hint pool, capped at max_budget.

    Mirrors prepare_hint_data.upgrade_row exactly: a pool with K_q>0 steps gets
    min(K_q, max_budget); an empty/unparseable pool falls back to max_budget; the
    result is floored at 1 (the zero-budget curriculum is applied separately).
    """
    k = num_steps(hint_str)
    return max(1, min(k if k > 0 else max_budget, max_budget))


def set_row_budget(row: dict, budget: int) -> dict:
    """Re-render an already-templated row's budget-dependent fields for ``budget``.

    Requires extra_info.hprl_system_base / hprl_user_base (the budget-free bases
    baked by prepare_hint_data) and extra_info.tools_kwargs.request_hint.
    create_kwargs. Returns a shallow-copied row with prompt + extra_info replaced.
    """
    row = dict(row)
    extra_info = dict(row["extra_info"])
    base_system = extra_info.get("hprl_system_base")
    user_base = extra_info.get("hprl_user_base")
    if base_system is None or user_base is None:
        raise ValueError(
            "row lacks hprl_system_base/hprl_user_base -- not a templated *-mt "
            "parquet (run prepare_hint_data.py first)"
        )

    # --- re-render the system prompt for the new budget ------------------
    prompt = list(row["prompt"])
    if prompt and prompt[0].get("role") == "system":
        prompt[0] = {"role": "system", "content": render_system(base_system, budget)}
    else:
        prompt = [{"role": "system", "content": render_system(base_system, budget)}, *prompt]

    # --- re-render the last user message's budget reminder ---------------
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i].get("role") == "user":
            prompt[i] = {"role": "user", "content": render_user(user_base, budget)}
            break

    # --- per-sample tool kwargs + recorded initial budget ----------------
    extra_info["hprl_init_budget"] = int(budget)
    tools_kwargs = dict(extra_info.get("tools_kwargs") or {})
    req = dict(tools_kwargs.get("request_hint") or {})
    ck = dict(req.get("create_kwargs") or {})
    ck["budget"] = int(budget)
    # keep create_kwargs.problem in sync with the re-rendered user turn, exactly
    # as prepare_hint_data sets it (problem = user_problem(prompt) post-render).
    ck["problem"] = user_problem(prompt)
    req["create_kwargs"] = ck
    tools_kwargs["request_hint"] = req
    extra_info["tools_kwargs"] = tools_kwargs

    row["prompt"] = prompt
    row["extra_info"] = extra_info
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="templated multi-turn parquet (output of prepare_hint_data.py)")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="output parquet (default: in-place, == --in)")
    ap.add_argument("--zero-budget-ids", default=None,
                    help="file of problem_ids (one per line) to force to budget 0 "
                         "(the hard-problem curriculum's 'easy' bucket from "
                         "extract_unaided_solved.py)")
    ap.add_argument("--max-budget", type=int, default=8,
                    help="cap on the per-problem budget B_q (<= max_assistant_turns); "
                         "MUST match the value prepare_hint_data used to avoid "
                         "silently re-capping non-zero budgets")
    args = ap.parse_args()
    out_path = args.out_path or args.in_path

    zero_budget_ids: set = set()
    if args.zero_budget_ids:
        with open(args.zero_budget_ids) as f:
            zero_budget_ids = {ln.strip() for ln in f if ln.strip()}
        print(f"read {len(zero_budget_ids)} zero-budget problem_ids from {args.zero_budget_ids}")

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")

    rows = []
    for r in df.to_dict(orient="records"):
        extra_info = r.get("extra_info") or {}
        pid = extra_info.get("problem_id")
        budget = compute_budget(extra_info.get("hint", "") or "", args.max_budget)
        if zero_budget_ids and pid in zero_budget_ids:
            budget = 0
        rows.append(set_row_budget(r, budget))
    out = pd.DataFrame(rows)

    budgets = [e["tools_kwargs"]["request_hint"]["create_kwargs"]["budget"] for e in out["extra_info"]]
    n_zero = sum(1 for b in budgets if b == 0)
    if zero_budget_ids:
        matched = sum(1 for e in out["extra_info"] if e.get("problem_id") in zero_budget_ids)
        print(f"zero-budget curriculum: {matched}/{len(zero_budget_ids)} listed ids matched a row "
              f"-> {n_zero} rows at budget 0 ({100*n_zero/len(out):.1f}%)")
    nonzero = [b for b in budgets if b > 0]
    bmean = sum(nonzero) / len(nonzero) if nonzero else 0
    from collections import Counter
    print(f"budget B_q: min={min(budgets)} max={max(budgets)} mean_over_nonzero={bmean:.2f} "
          f"(n_zero={n_zero})  dist={dict(sorted(Counter(budgets).items()))}")

    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}" + ("  (in-place)" if out_path == args.in_path else ""))


if __name__ == "__main__":
    main()
