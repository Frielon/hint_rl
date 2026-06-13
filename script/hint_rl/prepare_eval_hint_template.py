#!/usr/bin/env python3
# Copyright 2026
#
# Wrap an EVAL/holdout parquet in the HPRL hint-tool template at budget 0.
#
# The held-out eval sets (aime2024.parquet, dapo_sample_hard_100.parquet) ship as
# plain single-turn rows (data_source, prompt, ability, reward_model, extra_info)
# and carry NO hint pool. Training prompts, by contrast, go through the tool
# template (hint_prompt.render_system/render_user) so the policy is asked to solve
# with the <hint_call/> tool advertised. Evaluating on the bare prompt is therefore
# OUT of the training distribution.
#
# This script re-renders each eval row with the SAME tool template the training
# rows use, but at BUDGET 0: the system prompt carries the full tool instruction
# (with "at most 0 time(s)") and the user message ends with "no hint calls
# remaining, finish on your own" (render_user at budget 0). The eval is thus
# unaided -- there is no hint pool to serve anyway -- yet prompt-identical to
# training. Budget 0 also means the agent loop never contacts the selector (the
# len(applied) >= budget branch short-circuits in hint_agent_loop), so the eval
# has no selector dependency.
#
# Per-row this mirrors prepare_hint_data.upgrade_row EXCEPT:
#   * budget is forced to 0 (these problems have no curated hints);
#   * tools_kwargs.request_hint.create_kwargs.hints = "" (empty pool);
#   * extra_info.hint_full is NOT joined (no penalty pool for eval).
#
# The output carries agent_name="hint_agent" + extra_info.hprl_system_base/
# hprl_user_base/hprl_init_budget so the HPRL dataset (HintBudgetDataset) loads it
# exactly like a training row: it re-renders at budget 0 (idempotent -- eval
# problem_ids are absent from the ratchet table, so _budget_for falls back to the
# baked 0) and routes the rollout through HintAgentLoop.
#
# Usage:
#   python prepare_eval_hint_template.py --in dataset/aime2024.parquet \
#                                        --out dataset/aime2024-hint-mt.parquet
#   python prepare_eval_hint_template.py --in dataset/dapo_sample_hard_100.parquet \
#                                        --out dataset/dapo_sample_hard_100-hint-mt.parquet

from __future__ import annotations

import argparse
import os

import pandas as pd

from hint_prompt import (
    DEFAULT_BASE_SYSTEM,
    render_system,
    render_user,
    strip_user_cot_tail,
)
from prepare_hint_data import strip_cot_trigger, user_problem

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))


def upgrade_eval_row(row: dict, agent_name: str = "hint_agent", budget: int = 0) -> dict:
    row = dict(row)
    prompt = list(row["prompt"])  # list of {role, content}
    extra_info = dict(row.get("extra_info") or {})

    # --- system: same tool template as training (budget shown as 0) ------
    if prompt and prompt[0].get("role") == "system":
        base_system = strip_cot_trigger(prompt[0].get("content", "") or DEFAULT_BASE_SYSTEM)
        prompt[0] = {"role": "system", "content": render_system(base_system, budget)}
    else:
        base_system = DEFAULT_BASE_SYSTEM
        prompt = [{"role": "system", "content": render_system(base_system, budget)}, *prompt]

    # --- user: strip the DAPO CoT tail, append the budget-0 reminder -----
    for i in range(len(prompt) - 1, -1, -1):
        if prompt[i].get("role") == "user":
            user_base = strip_user_cot_tail(prompt[i].get("content", "") or "")
            extra_info["hprl_user_base"] = user_base
            prompt[i] = {"role": "user", "content": render_user(user_base, budget)}
            break

    # --- per-sample stateful-tool data (empty hint pool, budget 0) -------
    ground_truth = (row.get("reward_model") or {}).get("ground_truth", "")
    extra_info["need_tools_kwargs"] = True
    extra_info["tools_kwargs"] = {
        "request_hint": {
            "create_kwargs": {
                "problem": user_problem(prompt),
                "hints": "",          # no curated hint pool for eval problems
                "ground_truth": str(ground_truth),
                "budget": int(budget),
            }
        }
    }
    # fields the HPRL dataset reads to re-render under the ratchet (stays 0).
    extra_info["hprl_system_base"] = base_system
    extra_info["hprl_init_budget"] = int(budget)

    row["prompt"] = prompt
    row["extra_info"] = extra_info
    row["agent_name"] = agent_name
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--budget", type=int, default=0,
                    help="hint budget for the eval rows (default 0 -- unaided, "
                         "prompt-matched to training; no hint pool exists to serve)")
    ap.add_argument("--agent-name", default="hint_agent",
                    choices=["hint_agent", "tool_agent"],
                    help="rollout to route rows through (default hint_agent, the "
                         "<hint_call/> sentinel loop)")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")

    rows = [upgrade_eval_row(r, args.agent_name, args.budget) for r in df.to_dict(orient="records")]
    out = pd.DataFrame(rows)

    # sanity: every row carries the tool template + routing fields.
    assert (out["agent_name"] == args.agent_name).all()
    assert all(e.get("hprl_system_base") for e in out["extra_info"])
    assert all(e["tools_kwargs"]["request_hint"]["create_kwargs"]["budget"] == args.budget
               for e in out["extra_info"])
    sys0 = out.iloc[0]["prompt"][0]["content"]
    assert "<hint_call/>" in sys0, "system prompt missing the tool instruction"

    print(f"columns: {list(out.columns)}")
    print(f"budget={args.budget}  agent_name={args.agent_name}")
    print("--- sample system prompt tail ---")
    print(sys0[-220:])
    usr0 = [m for m in out.iloc[0]["prompt"] if m["role"] == "user"][-1]["content"]
    print("--- sample user-message tail ---")
    print(usr0[-160:])

    out.to_parquet(args.out_path, index=False)
    print(f"wrote {len(out)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
