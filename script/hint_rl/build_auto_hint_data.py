#!/usr/bin/env python3
# Copyright 2026
#
# Build the AUTO-HINT training parquet by joining, on problem_id:
#   * the PLAIN single-turn prompts (dataset/dapo-3139-single-turn.parquet) -- the
#     exact "reason step by step ... \boxed{}" format the auto-hint policy sees
#     (the policy is NEVER told about hints); with
#   * the hint payload (hint pool + difficulty-annotated pool + per-problem budget
#     + ground_truth) from the matching <hint_call/> training parquet
#     (dataset/dapo-3139-hint-verl-mt-clean.parquet).
#
# The result routes through auto_hint_agent_loop.AutoHintAgentLoop via
# agent_name="auto_hint"; the loop reads the pool/budget from
# extra_info.tools_kwargs.request_hint.create_kwargs and injects a hint whenever the
# boxed answer is still wrong, up to the budget. extra_info.hint_full feeds the
# reward's per-hint penalty; extra_info.hprl_auto_hint tells the dynamic-budget
# dataset to update the budget WITHOUT touching the (hint-agnostic) prompt.
#
# This join is used because the repo currently ships the matched 3139 set only as
# the single-turn prompts + the upgraded <hint_call/> parquet (no clean pre-upgrade
# 3139 source). To build from a fresh pre-upgrade source instead, use
# ``prepare_hint_data.py --mode auto_hint``.
#
# Usage:
#   python build_auto_hint_data.py        # defaults under dataset/
#   python build_auto_hint_data.py --plain <p> --hint <h> --out <o>
from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(HINT_RL_HOME, "dataset")


def user_problem(prompt) -> str:
    for m in prompt:
        if m.get("role") == "user":
            return m.get("content", "")
    return prompt[-1].get("content", "") if len(prompt) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain", default=os.path.join(DS, "dapo-3139-single-turn.parquet"))
    ap.add_argument("--hint", default=os.path.join(DS, "dapo-3139-hint-verl-mt-clean.parquet"))
    ap.add_argument("--out", default=os.path.join(DS, "dapo-3139-auto-hint.parquet"))
    args = ap.parse_args()

    plain = pd.read_parquet(args.plain)
    hint = pd.read_parquet(args.hint)
    print(f"plain prompts: {len(plain)} rows ({args.plain})")
    print(f"hint payload : {len(hint)} rows ({args.hint})")

    # index the hint payload by problem_id.
    payload: dict = {}
    for _, r in hint.iterrows():
        e = dict(r["extra_info"])
        pid = e.get("problem_id")
        ck = ((e.get("tools_kwargs") or {}).get("request_hint", {}) or {}).get("create_kwargs", {}) or {}
        payload[pid] = {
            "hints": ck.get("hints", ""),
            "ground_truth": ck.get("ground_truth", ""),
            "budget": int(ck.get("budget", e.get("hprl_init_budget", 0) or 0)),
            "hint_full": e.get("hint_full"),
            "hint": e.get("hint"),
            "reference_solution": e.get("reference_solution"),
        }

    rows, missing, n_full = [], 0, 0
    for _, r in plain.iterrows():
        row = {k: r[k] for k in plain.columns}
        ei = dict(row["extra_info"])
        pid = ei.get("problem_id")
        pay = payload.get(pid)
        if pay is None:
            missing += 1
            continue
        prompt = list(row["prompt"])              # plain single-turn prompt (kept verbatim)
        problem = user_problem(prompt)            # clean problem (no hint/budget text)
        gt = (row.get("reward_model") or {}).get("ground_truth") or pay["ground_truth"]

        ei["need_tools_kwargs"] = True
        ei["tools_kwargs"] = {
            "request_hint": {
                "create_kwargs": {
                    "problem": problem,
                    "hints": pay["hints"],        # the hint pool (JSON string)
                    "ground_truth": str(gt),
                    "budget": int(pay["budget"]),
                }
            }
        }
        ei["hint"] = pay["hint"]
        if pay["reference_solution"] is not None:
            ei["reference_solution"] = pay["reference_solution"]
        if pay["hint_full"] is not None:
            ei["hint_full"] = pay["hint_full"]
            n_full += 1
        ei["hprl_init_budget"] = int(pay["budget"])
        ei["hprl_auto_hint"] = True               # budget-only dataset update; no prompt re-render
        # NB: deliberately NOT setting hprl_system_base / hprl_user_base.

        row["prompt"] = prompt
        row["extra_info"] = ei
        row["agent_name"] = "auto_hint"
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    budgets = [r["extra_info"]["hprl_init_budget"] for r in rows]
    print(f"wrote {len(out)} rows (missing payload: {missing}) -> {args.out}")
    print(f"budget B_q: min={min(budgets)} max={max(budgets)} mean={sum(budgets) / len(budgets):.2f}")
    print(f"rows with hint_full: {n_full}/{len(rows)}")
    print(f"columns: {list(out.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
