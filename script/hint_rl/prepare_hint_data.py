#!/usr/bin/env python3
# Copyright 2026
#
# Upgrade the HPRL training parquet for multi-turn hint-tool rollouts.
#
# Input : dataset/dapo-3740-hint-verl-simplified.parquet
#   columns: data_source, prompt[list[{role,content}]], ability,
#            reward_model{ground_truth}, extra_info{split,index,problem_id,
#            hint(JSON str),reference_solution}
#
# Output: dataset/dapo-3740-hint-verl-simplified-mt.parquet  with, per row:
#   * agent_name = "tool_agent"                 -> routes the row through the
#                                                  tool-calling agent loop.
#   * system prompt augmented with a tool instruction + the per-problem hint
#     budget B_q ("You may call the hint tool at most B_q times ...").
#   * extra_info.need_tools_kwargs = True
#   * extra_info.tools_kwargs.request_hint.create_kwargs = {problem, hints,
#       ground_truth, budget}  -> the per-sample data the stateful HintTool
#       reads at create() time (the hint pool, the problem, the budget B_q).
#
# B_q defaults to the number of major steps K_q in the hint pool, capped at
# --max-budget (so it never exceeds the rollout's max_assistant_turns).
#
# Usage:
#   python prepare_hint_data.py                       # default in/out under dataset/
#   python prepare_hint_data.py --max-budget 8
#   python prepare_hint_data.py --in <path> --out <path>

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_IN = os.path.join(HINT_RL_HOME, "dataset", "dapo-3740-hint-verl-simplified.parquet")
DEFAULT_OUT = os.path.join(HINT_RL_HOME, "dataset", "dapo-3740-hint-verl-simplified-mt.parquet")

TOOL_INSTRUCTION = (
    "\n\nA hint tool named `request_hint` is available. When you are stuck, you "
    "may call it to receive a single step-level hint from a tutor who reads your "
    "reasoning so far. You may call `request_hint` at most {budget} time(s) for "
    "this problem; each hint you use lowers your reward, so solve as much as you "
    "can unaided and only ask for a hint when genuinely stuck. Put your final "
    "answer within \\boxed{{}}."
)


def num_steps(hint_str: str) -> int:
    try:
        pool = json.loads(hint_str)
        steps = pool.get("steps", [])
        return len(steps)
    except Exception:  # noqa: BLE001
        return 0


def user_problem(prompt) -> str:
    """Extract the user problem statement from the chat prompt."""
    for m in prompt:
        if m.get("role") == "user":
            return m.get("content", "")
    # fallback: last message content
    return prompt[-1].get("content", "") if len(prompt) else ""


def upgrade_row(row: dict, max_budget: int) -> dict:
    prompt = list(row["prompt"])  # list of {role, content}
    extra_info = dict(row["extra_info"])
    hint_str = extra_info.get("hint", "") or ""

    k = num_steps(hint_str)
    budget = max(1, min(k if k > 0 else max_budget, max_budget))

    # --- nudge the system prompt to advertise the tool + budget ----------
    instruction = TOOL_INSTRUCTION.format(budget=budget)
    if prompt and prompt[0].get("role") == "system":
        prompt[0] = {"role": "system", "content": prompt[0].get("content", "") + instruction}
    else:
        prompt = [{"role": "system", "content": "You are a math expert." + instruction}, *prompt]

    # --- per-sample stateful-tool data -----------------------------------
    ground_truth = row["reward_model"].get("ground_truth", "")
    extra_info["need_tools_kwargs"] = True
    extra_info["tools_kwargs"] = {
        "request_hint": {
            "create_kwargs": {
                "problem": user_problem(prompt),
                "hints": hint_str,          # the hint pool, JSON string
                "ground_truth": str(ground_truth),
                "budget": int(budget),
            }
        }
    }

    row["prompt"] = prompt
    row["extra_info"] = extra_info
    row["agent_name"] = "tool_agent"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    ap.add_argument("--max-budget", type=int, default=8,
                    help="cap on the per-problem hint budget B_q (<= max_assistant_turns)")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")

    rows = [upgrade_row(dict(r), args.max_budget) for r in df.to_dict(orient="records")]
    out = pd.DataFrame(rows)

    budgets = [r["extra_info"]["tools_kwargs"]["request_hint"]["create_kwargs"]["budget"] for r in rows]
    print(f"budget B_q: min={min(budgets)} max={max(budgets)} mean={sum(budgets)/len(budgets):.2f}")
    print(f"columns: {list(out.columns)}")

    out.to_parquet(args.out_path, index=False)
    print(f"wrote {len(out)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
