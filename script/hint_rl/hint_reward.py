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
# HPRL reward:
#
#     R = incorrect_reward                    if the answer is wrong
#     R = correct_reward - sum_k w_{j_k}      if the answer is correct
#
# i.e. a wrong answer gets the base (incorrect) reward; a correct answer gets the
# base (correct) reward minus the summed importance weights {w_{j_k}} of the hints
# applied during the rollout. Correctness is verified exactly as in the plain GRPO
# run (mathruler boxed-answer grading).
#
# The list of hints applied in the rollout is maintained as per-rollout state by
# the rollout (on agent_data.extra_fields["applied_hints"]) and is merged into
# ``extra_info["applied_hints"]`` by hint_reward_manager.HintRewardManager before
# this function runs. The per-hint weights w come from the difficulty-annotated
# hint pool at ``extra_info["hint_full"]`` (see hint_penalty.py).

from __future__ import annotations

from typing import Optional

from mathruler.grader import extract_boxed_content, grade_answer

from hint_penalty import (
    DEFAULT_GUIDANCE_DIFFICULTY,
    DEFAULT_HARD_FACTOR,
    DEFAULT_TOTAL_PENALTY,
    STRATEGY_HINT,
    STRATEGY_MAJOR_STEP,
    applied_penalty,
    applied_step_penalty,
    normalize_strategy,
)

# Sentinel returned by extract_boxed_content when nothing is boxed.
_NO_BOX = "None"


def hint_penalty(
    applied_hints: list,
    extra_info: Optional[dict] = None,
    *,
    total_penalty: float = DEFAULT_TOTAL_PENALTY,
    hard_factor: float = DEFAULT_HARD_FACTOR,
    guidance_difficulty: str = DEFAULT_GUIDANCE_DIFFICULTY,
    strategy: str = STRATEGY_HINT,
) -> float:
    """Hint penalty for a rollout, per the selected ``strategy``.

    The weights are computed from the difficulty-annotated ORIGINAL hint pool,
    stored by prepare_hint_data.py at ``extra_info["hint_full"]``. ``applied_hints``
    is the per-rollout list recorded by the rollout:

      * ``STRATEGY_HINT`` -- each item is one applied hint, e.g.
        ``{"call_index": 0, "hint_id": "2.1", "major_step_id": 2, "hint": "..."}``;
        the penalty sums the per-hint weights ``sum_k w_{j_k}``
        (``compute_hint_penalties`` -> ``applied_penalty``).
      * ``STRATEGY_MAJOR_STEP`` -- each item is one revealed major step, e.g.
        ``{"call_index": 0, "major_step_id": 2, "hint_ids": ["2.0", "2.1", ...],
        "hint": "..."}``; the penalty sums the per-STEP weights directly
        (``compute_step_penalties`` -> ``applied_step_penalty``).

    Returns 0.0 (no penalty) when no hints were used or no ``hint_full`` is
    present. The weighting knobs are passed through from ``compute_score`` reward
    kwargs, so they (and the strategy) can be retuned WITHOUT regenerating the
    dataset.
    """
    extra_info = extra_info or {}
    hint_full = extra_info.get("hint_full")
    if normalize_strategy(strategy) == STRATEGY_MAJOR_STEP:
        return applied_step_penalty(
            applied_hints,
            hint_full,
            total_penalty=total_penalty,
            hard_factor=hard_factor,
        )
    return applied_penalty(
        applied_hints,
        hint_full,
        total_penalty=total_penalty,
        hard_factor=hard_factor,
        guidance_difficulty=guidance_difficulty,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = -1.0,
    format_reward: float = 0.0,
    hint_penalty_total: float = DEFAULT_TOTAL_PENALTY,
    hint_penalty_hard_factor: float = DEFAULT_HARD_FACTOR,
    hint_guidance_difficulty: str = DEFAULT_GUIDANCE_DIFFICULTY,
    hint_strategy: str = STRATEGY_HINT,
    **kwargs,
) -> dict:
    """HPRL reward: outcome correctness minus the summed hint penalty.

    The hint-penalty knobs (``hint_penalty_total`` / ``hint_penalty_hard_factor``
    / ``hint_guidance_difficulty`` / ``hint_strategy``) arrive as
    ``custom_reward_function.reward_kwargs`` so they are tunable at launch without
    regenerating the dataset. ``hint_strategy`` must match the strategy the agent
    loop ran (``data.hprl.strategy``): ``"hint"`` charges per-hint penalties,
    ``"major_step"`` charges per-step penalties directly.

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

    # --- hint penalty (subtracted from the correct reward) ---------------
    applied_hints = extra_info.get("applied_hints") or []
    # Defensive: applied_hints may arrive as a numpy 0-d object; normalize.
    if hasattr(applied_hints, "tolist"):
        applied_hints = applied_hints.tolist()
    if applied_hints is None:
        applied_hints = []

    penalty = hint_penalty(
        applied_hints,
        extra_info,
        total_penalty=hint_penalty_total,
        hard_factor=hint_penalty_hard_factor,
        guidance_difficulty=hint_guidance_difficulty,
        strategy=hint_strategy,
    )

    # Wrong answer -> base (incorrect) reward, ignoring hints. Correct answer ->
    # base (correct) reward minus the summed hint penalty (each hint used lowers
    # the reward).
    if correct:
        score = base - penalty
    else:
        score = base

    # Count of hint calls this rollout made whose selector lookup failed (the
    # policy got the no-op fallback instead of a hint). Recorded by the agent loop
    # on extra_fields; surfaced here so it lands in non_tensor_batch for
    # hint_budget_callback to aggregate -> a selector outage shows up in hprl/*.
    hint_call_failed = extra_info.get("hint_call_failed", 0)
    if hasattr(hint_call_failed, "item"):
        hint_call_failed = hint_call_failed.item()

    return {
        "score": float(score),
        "acc": 1.0 if correct else 0.0,
        "pred": pred,
        "has_format": 1.0 if has_format else 0.0,
        "num_hints": float(len(applied_hints)),
        "hint_penalty": float(penalty),
        "hint_call_failed": float(hint_call_failed or 0),
    }
