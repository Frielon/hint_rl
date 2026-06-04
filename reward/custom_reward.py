# Copyright 2025
#
# Custom reward function for the hint_rl DAPO runs.
#
# verl loads this via `custom_reward_function.path` / `custom_reward_function.name`.
# When set, it is passed to the reward manager as `compute_score` and called once
# per (prompt, response) pair as:
#
#     compute_score(data_source=..., solution_str=..., ground_truth=...,
#                   extra_info=..., **reward_kwargs)
#
# The function must return either a float (used directly as the score) or a dict
# containing at least a "score" key. Any other keys in the dict are logged as
# `reward_extra_info` by the reward manager. We deliberately also return an "acc"
# key because the DAPO recipe filters prompt groups on `algorithm.filter_groups.metric=acc`.
#
# Correctness is verified with `mathruler` (boxed-answer extraction + sympy/string
# normalization grading), instead of verl's Minerva "Answer:" verifier.
#
# NOTE: The overlong (length) penalty is applied *on top* of this score by the
# DAPORewardManager itself (via reward_model.overlong_buffer.*), so we do NOT
# handle response length here -- only correctness / formatting.

from __future__ import annotations

from typing import Optional

# mathruler.grader.extract_boxed_content(text) -> content of the last \boxed{...}
#   or the literal string "None" when no boxed answer is present.
# mathruler.grader.grade_answer(given, ground_truth) -> bool, lenient match via
#   string normalization first, then sympy equivalence.
from mathruler.grader import extract_boxed_content, grade_answer

# Sentinel returned by extract_boxed_content when nothing is boxed.
_NO_BOX = "None"


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
    """Compute a customizable math reward using mathruler for correctness.

    Args:
        data_source: dataset tag (e.g. "math_dapo", "aime2024").
        solution_str: the model's decoded response (no prompt, no EOS).
        ground_truth: the gold answer string from reward_model.ground_truth.
        extra_info: per-sample metadata (split, index, problem_id, ...). Also
            carries "rollout_reward_scores" injected by the reward manager.
        correct_reward: reward for a correct final answer.
        incorrect_reward: reward for an incorrect final answer.
        format_reward: bonus added when the response contains a \\boxed{} answer,
            regardless of correctness (set >0 to encourage the boxed format).

    Returns:
        dict with keys:
            score: the scalar reward used for optimization.
            acc:   1.0 if the answer is correct else 0.0 (used by filter_groups).
            pred:  the extracted predicted answer ("None" if no boxed answer).
            has_format: 1.0 if a \\boxed{} answer was present else 0.0.
    """
    # --- extract the model's boxed answer --------------------------------
    pred = extract_boxed_content(solution_str)
    has_format = pred != _NO_BOX

    # --- correctness via mathruler ---------------------------------------
    # grade_answer normalizes and (for non-integers) tries sympy equivalence.
    correct = has_format and grade_answer(pred, ground_truth)

    # --- shaping ----------------------------------------------------------
    score = correct_reward if correct else incorrect_reward
    if format_reward and has_format:
        score += format_reward

    return {
        "score": float(score),
        "acc": 1.0 if correct else 0.0,
        "pred": pred,
        "has_format": 1.0 if has_format else 0.0,
    }
