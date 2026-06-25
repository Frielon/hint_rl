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
#   * agent_name = "hint_agent" (default)        -> routes the row through the
#       <hint_call/> rollout (hint_agent_loop.HintAgentLoop). Use
#       --agent-name tool_agent for the legacy hermes request_hint tool path.
#   * system prompt augmented with a tool instruction + the per-problem hint
#     budget B_q (rendered by hint_prompt.render_system).
#   * extra_info.need_tools_kwargs = True
#   * extra_info.tools_kwargs.request_hint.create_kwargs = {problem, hints,
#       ground_truth, budget}  -> the per-sample data the rollout reads: the hint
#       pool, the problem, and the budget B_q (HintAgentLoop reads it directly).
#   * extra_info.hprl_system_base / hprl_init_budget -> the budget-free system
#       preamble and the initial B_q, so the dynamic-budget dataset
#       (hint_dataset.HintBudgetDataset) can RE-RENDER the system prompt for a
#       ratcheted budget at training time. The baked prompt + tools_kwargs still
#       carry the static initial B_q, so the parquet also works unchanged when
#       the budget ratchet is OFF (data.hprl.enable=false).
#   * extra_info.hint_full -> the ORIGINAL difficulty-annotated hint pool (joined
#       from --orig-in by problem_id), read by the reward's hint penalty
#       (hint_penalty.py). Storing it lets the penalty weights be retuned without
#       regenerating the dataset.
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
import re

import pandas as pd

from hint_prompt import (
    DEFAULT_BASE_SYSTEM,
    render_system,
    render_user,
    strip_user_cot_tail,
)

# A "step by step" / "think step by step" phrasing in a sample's base system
# prompt is a chain-of-thought trigger: it pushes the model straight to \boxed{}
# and suppresses <hint_call/> emission (offline bisect 2026-06-09: closer
# "step by step" ~8% emit vs "briefly" ~28%). The DAPO base ("You are a helpful
# assistant. Solve the math problem given by the user, reasoning step by step,
# and put your final answer within \boxed{}.") carries it. Still strip it: the
# ONE deliberate "step by step" lives in render_system()'s closer (Round 13
# trades emission for longer pre-call CoT); an extra unconditional copy in the
# base system stacks on top and pushes emission below the 20% floor.
_COMMA_COT_RE = re.compile(
    r",\s*(?:reasoning|reason|thinking|think)\s+step[\s-]by[\s-]step\s*,",
    re.IGNORECASE,
)
_COT_RE = re.compile(
    r"\s*,?\s*(?:reasoning|reason|thinking|think)?\s*step[\s-]by[\s-]step",
    re.IGNORECASE,
)


def strip_cot_trigger(base_system: str) -> str:
    """Remove "step by step" CoT phrasings from a base system prompt.

    Falls back to DEFAULT_BASE_SYSTEM if stripping leaves nothing meaningful.
    """
    text = _COMMA_COT_RE.sub(",", base_system)        # ", reasoning step by step," -> ","
    text = _COT_RE.sub("", text)                      # any remaining phrase
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip() or DEFAULT_BASE_SYSTEM

# Plain single-turn solver system prompt for AUTO-HINT mode -- byte-identical to
# dataset/dapo-3139-single-turn.parquet, so the policy sees the ORDINARY math prompt
# (no tool instruction, no budget) and the loop, not the model, decides when to hint.
PLAIN_SYSTEM = (
    "You are a helpful assistant. Solve the math problem given by the user, "
    "reasoning step by step, and put your final answer within \\boxed{}."
)

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_IN = os.path.join(HINT_RL_HOME, "dataset", "dapo-3164-hint-verl.parquet")
DEFAULT_OUT = os.path.join(HINT_RL_HOME, "dataset", "dapo-3164-hint-verl-mt.parquet")
# Original (difficulty-annotated) hint pool; joined in for the penalty weights.
DEFAULT_ORIG = os.path.join(HINT_RL_HOME, "dataset", "dapo-3740-hint-verl.parquet")


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


# NOTE: truncated-problem detection/removal lives in remove_truncated.py now.
# Run it on the mt-type output of this script to drop rows whose problem
# statement is cut off mid-sentence.


def upgrade_row(row: dict, max_budget: int, agent_name: str = "hint_agent", hint_full=None,
                zero_budget_ids: set | None = None, mode: str = "hint_call") -> dict:
    prompt = list(row["prompt"])  # list of {role, content}
    extra_info = dict(row["extra_info"])
    hint_str = extra_info.get("hint", "") or ""

    k = num_steps(hint_str)
    budget = max(1, min(k if k > 0 else max_budget, max_budget))
    # Hard-problem curriculum: problems the policy already solves unaided get
    # budget 0 (no hints). On those, a single unaided-correct rollout (reward
    # ~1.1) out-ranks every hinted-correct one (1.1 - penalty) within the GRPO
    # group and trains <hint_call/> away globally; zeroing their budget removes
    # that suppression so hints act only where the model is actually stuck. At
    # budget 0 render_system/render_user STILL render the full tool template
    # (budget shown as 0, user reminder "no hint calls remaining, finish on your
    # own") so the budget-0 prompt matches the budget>0 style/closer and only the
    # budget digit differs. See extract_unaided_solved.py for the id list.
    pid = extra_info.get("problem_id")
    if zero_budget_ids and pid in zero_budget_ids:
        budget = 0

    if mode == "auto_hint":
        # AUTO-HINT: hint-AGNOSTIC single-turn prompt. The policy is never told about
        # hints (the loop, auto_hint_agent_loop, injects them on a wrong answer), so the
        # system prompt is the plain step-by-step solver and the user problem is left
        # untouched -- no tool instruction, no budget reminder. hprl_auto_hint flags the
        # row so the dynamic-budget dataset updates the loop's budget WITHOUT re-rendering
        # the prompt; hprl_system_base / hprl_user_base are deliberately NOT set.
        if prompt and prompt[0].get("role") == "system":
            prompt[0] = {"role": "system", "content": PLAIN_SYSTEM}
        else:
            prompt = [{"role": "system", "content": PLAIN_SYSTEM}, *prompt]
        extra_info["hprl_auto_hint"] = True
    else:
        # --- nudge the system prompt to advertise the tool + budget ----------
        # Keep the budget-free preamble (hprl_system_base) so the dynamic-budget
        # dataset can re-render the sentence for a ratcheted budget; bake the static
        # initial budget into the prompt here so the parquet also works ratchet-off.
        if prompt and prompt[0].get("role") == "system":
            base_system = strip_cot_trigger(prompt[0].get("content", "") or DEFAULT_BASE_SYSTEM)
            prompt[0] = {"role": "system", "content": render_system(base_system, budget)}
        else:
            base_system = DEFAULT_BASE_SYSTEM
            prompt = [{"role": "system", "content": render_system(base_system, budget)}, *prompt]

        # --- budget reminder as the LAST line of the user message ------------
        # Placement matters: as the final user-turn sentence this ~doubles emission
        # and ~triples multi-hint-call vs. the same fact in the system prompt
        # (hint_call_test Round 10). Strip the DAPO "think step by step" CoT tail
        # first (it would sit right before the reminder and blunt it), and store the
        # budget-free, stripped user text as hprl_user_base so the dynamic-budget
        # ratchet (hint_dataset) can re-append the current budget byte-identically.
        for i in range(len(prompt) - 1, -1, -1):
            if prompt[i].get("role") == "user":
                user_base = strip_user_cot_tail(prompt[i].get("content", "") or "")
                extra_info["hprl_user_base"] = user_base
                prompt[i] = {"role": "user", "content": render_user(user_base, budget)}
                break
        # data for the dynamic-budget ratchet (hint_dataset.HintBudgetDataset).
        extra_info["hprl_system_base"] = base_system

    # --- per-sample stateful-tool data (BOTH modes) ----------------------
    # The rollout reads the hint pool / problem / budget from here regardless of
    # whether the prompt advertises the tool. In auto-hint mode the budget is the
    # max number of hint INJECTIONS the loop may make.
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
    extra_info["hprl_init_budget"] = int(budget)
    # the ORIGINAL difficulty-annotated hint pool, for the reward's hint penalty
    # (hint_penalty.py reads extra_info["hint_full"]). Stored verbatim as a JSON
    # string so the penalty weights can be retuned without touching the dataset.
    if hint_full is not None:
        extra_info["hint_full"] = hint_full

    row["prompt"] = prompt
    row["extra_info"] = extra_info
    # "hint_agent" -> <hint_call/> rollout; "auto_hint" -> push-hint rollout
    # (auto_hint_agent_loop); "tool_agent" -> legacy hermes request_hint tool path.
    row["agent_name"] = agent_name
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    ap.add_argument("--orig-in", dest="orig_path", default=DEFAULT_ORIG,
                    help="original difficulty-annotated hint parquet, joined by "
                         "problem_id into extra_info.hint_full for the reward penalty")
    ap.add_argument("--max-budget", type=int, default=8,
                    help="cap on the per-problem hint budget B_q (<= max_assistant_turns)")
    ap.add_argument("--mode", default="hint_call", choices=["hint_call", "auto_hint"],
                    help="'hint_call' (default): the policy emits <hint_call/> and the "
                         "system prompt advertises the tool+budget. 'auto_hint': a plain "
                         "single-turn solver prompt; the loop injects a hint on a wrong "
                         "answer (auto_hint_agent_loop). auto_hint defaults --agent-name "
                         "to 'auto_hint'.")
    ap.add_argument("--agent-name", default=None,
                    choices=["hint_agent", "auto_hint", "tool_agent"],
                    help="rollout to route rows through. Default: 'auto_hint' when "
                         "--mode auto_hint, else 'hint_agent' (<hint_call/> sentinel); "
                         "'tool_agent' is the legacy hermes request_hint tool.")
    ap.add_argument("--zero-budget-ids", default=None,
                    help="file of problem_ids (one per line) to force to budget 0 (no hints) "
                         "-- the hard-problem curriculum's 'easy' bucket from "
                         "extract_unaided_solved.py")
    args = ap.parse_args()

    # auto_hint mode routes through the AutoHintAgentLoop by default.
    if args.agent_name is None:
        args.agent_name = "auto_hint" if args.mode == "auto_hint" else "hint_agent"
    print(f"mode={args.mode}  agent_name={args.agent_name}")

    # Problems to give 0 hint budget (already solvable unaided -> hints there only
    # suppress <hint_call/> under GRPO; see extract_unaided_solved.py).
    zero_budget_ids: set = set()
    if args.zero_budget_ids:
        with open(args.zero_budget_ids) as f:
            zero_budget_ids = {ln.strip() for ln in f if ln.strip()}
        print(f"read {len(zero_budget_ids)} zero-budget problem_ids from {args.zero_budget_ids}")

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")

    # Join the ORIGINAL difficulty-annotated hint pool by problem_id.
    pid_to_full = {}
    if args.orig_path and os.path.exists(args.orig_path):
        odf = pd.read_parquet(args.orig_path)
        for oe in odf["extra_info"]:
            pid_to_full[oe.get("problem_id")] = oe.get("hint")
        print(f"read {len(pid_to_full)} original hint pools from {args.orig_path}")
    else:
        print(f"WARNING: original hint parquet not found ({args.orig_path}); "
              f"extra_info.hint_full will be absent -> hint penalty defaults to 0")

    rows = []
    n_missing = 0
    for r in df.to_dict(orient="records"):
        r = dict(r)
        pid = (r.get("extra_info") or {}).get("problem_id")
        hint_full = pid_to_full.get(pid)
        if hint_full is None and pid_to_full:
            n_missing += 1
        rows.append(upgrade_row(r, args.max_budget, args.agent_name, hint_full=hint_full,
                                zero_budget_ids=zero_budget_ids, mode=args.mode))
    out = pd.DataFrame(rows)
    if n_missing:
        print(f"WARNING: {n_missing} rows had no matching original hint pool (hint_full absent)")

    budgets = [r["extra_info"]["tools_kwargs"]["request_hint"]["create_kwargs"]["budget"] for r in rows]
    n_zero = sum(1 for b in budgets if b == 0)
    if zero_budget_ids:
        matched = sum(1 for r in rows if (r["extra_info"].get("problem_id") in zero_budget_ids))
        print(f"zero-budget curriculum: {matched}/{len(zero_budget_ids)} listed ids matched a row "
              f"-> {n_zero} rows at budget 0 ({100*n_zero/len(rows):.1f}%)")
    nonzero = [b for b in budgets if b > 0]
    bmean = sum(nonzero) / len(nonzero) if nonzero else 0
    print(f"budget B_q: min={min(budgets)} max={max(budgets)} mean_over_nonzero={bmean:.2f} (n_zero={n_zero})")
    n_full = sum(1 for r in rows if r["extra_info"].get("hint_full"))
    print(f"rows with hint_full: {n_full}/{len(rows)}")
    print(f"columns: {list(out.columns)}")

    out.to_parquet(args.out_path, index=False)
    print(f"wrote {len(out)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
