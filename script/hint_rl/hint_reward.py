# Copyright 2026
#
# Reward function for Hint Penalized RL (HPRL).
#
# verl loads this via `custom_reward_function.path` / `.name`. It is called once
# per (prompt, response) pair as:
#
#     compute_score(data_source=..., solution_str=..., ground_truth=...,
#                   extra_info=..., **reward_kwargs)
#
# HPRL reward (paper, Section 3):
#
#     R = R_acc * max(0, 1 - sum_k w_{j_k})
#
# where R_acc in {0,1} is outcome correctness and the {w_{j_k}} are the
# importance weights of the hints applied during the rollout. Correctness is
# verified exactly as in the plain GRPO run (mathruler boxed-answer grading).
#
# The list of hints applied in the rollout is maintained as per-rollout state by
# hint_tool.HintTool (on agent_data.extra_fields["applied_hints"]) and is merged
# into ``extra_info["applied_hints"]`` by hint_reward_manager.HintRewardManager
# before this function runs.
#
# >>> The hint-penalty itself is intentionally LEFT BLANK <<< -- see
# ``hint_penalty()`` below. It currently returns 1.0 (no penalty), so the run is
# equivalent to plain GRPO until the penalty is filled in.

from __future__ import annotations

from typing import Optional

from mathruler.grader import extract_boxed_content, grade_answer

# Sentinel returned by extract_boxed_content when nothing is boxed.
_NO_BOX = "None"


def hint_penalty(applied_hints: list, extra_info: Optional[dict] = None) -> float:
    """Multiplicative penalty factor in [0, 1] for the hints used in a rollout.

    Intended to implement  ``max(0, 1 - sum_k w_{j_k})``  from the paper, where
    each applied hint carries an importance weight ``w``. ``applied_hints`` is
    the per-rollout list recorded by HintTool, each item a dict like::

        {"call_index": 0, "hint_id": "2.1", "major_step_id": 2,
         "hint": "...", "confidence_of_hint": 4}

    ``extra_info`` carries the rest of the sample's metadata (the hint pool, the
    budget, etc.) if the weighting needs it.

    TODO(HPRL): implement the penalty. e.g.::

        w = {...}  # per-hint importance weights, from extra_info / the hint pool
        return max(0.0, 1.0 - sum(w[h["hint_id"]] for h in applied_hints))

    Left BLANK for now: returns 1.0 (no penalty) so training runs end-to-end.
    """
    return 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    format_reward: float = 0.0,
    **kwargs,
) -> dict:
    """HPRL reward: outcome correctness scaled by the (blank) hint penalty.

    Returns a dict whose ``score`` is the optimized reward and whose extra keys
    are logged as ``reward_extra_info`` by the reward manager (``acc`` is also
    used by GRPO group filtering).
    """
    extra_info = extra_info or {}

    # --- outcome correctness (same as the plain GRPO run) ----------------
    pred = extract_boxed_content(solution_str)
    has_format = pred != _NO_BOX
    correct = has_format and grade_answer(pred, ground_truth)

    base = correct_reward if correct else incorrect_reward
    if format_reward and has_format:
        base += format_reward

    # --- hint penalty (multiplicative on the accuracy reward) ------------
    applied_hints = extra_info.get("applied_hints") or []
    # Defensive: applied_hints may arrive as a numpy 0-d object; normalize.
    if hasattr(applied_hints, "tolist"):
        applied_hints = applied_hints.tolist()
    if applied_hints is None:
        applied_hints = []

    penalty_factor = hint_penalty(applied_hints, extra_info)

    # The penalty multiplies the *accuracy* component (R_acc * factor). We apply
    # it to the positive (correct) reward; an incorrect answer stays at its
    # incorrect_reward floor regardless of hints used.
    if correct:
        score = base * penalty_factor
    else:
        score = base

    return {
        "score": float(score),
        "acc": 1.0 if correct else 0.0,
        "pred": pred,
        "has_format": 1.0 if has_format else 0.0,
        "num_hints": float(len(applied_hints)),
        "hint_penalty_factor": float(penalty_factor),
    }
