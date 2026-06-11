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


def effort_shortfall_sum(turn_lens) -> float:
    """Coefficient-FREE, ORDER-AWARE effort-shaping signal.

    Motivation: the base hint penalty is *timing-blind* -- a hint costs the same
    whether the policy called it after a genuine struggle or after one shallow
    sentence. Empirically that makes the policy front-load hint calls (reason for
    a few tokens, call a hint, repeat) and only reason hard once the budget is
    spent. We want EARLIER turns to reason as hard as LATER turns, so the signal
    must encode turn ORDER -- a turn is bad only if a *later* turn out-reasoned it.

    For the rollout's ordered assistant-turn token lengths ``L_1..L_T``
    (``turn_lens``), the reference for turn ``i`` is the SUFFIX MAXIMUM
    ``M_i = max(L_i, L_{i+1}, ..., L_T)`` -- the hardest reasoning at or after
    turn i. The per-turn shortfall is::

        shortfall_i = relu(M_i - L_i) / M_i = (M_i - L_i) / M_i     in [0, 1)

    and the signal is the SUM over turns::

        shortfall_sum = sum_i shortfall_i

    Equivalent to the recursive rule "from the front, find the longest turn k,
    score turns 1..k against L_k, then recurse on k+1..T": when k = argmax(L_i..L_T),
    every j in [i..k] has max(L_j..L_T) = L_k, so this is exactly the suffix max.
    Computed in one right-to-left pass.

    Order semantics (the point of this design):
      * reasoning NON-INCREASING over turns (hard early, ease off) -> every M_i is
        the turn's own length -> shortfall_sum = 0 ("reason hard first" is free);
      * reasoning RISING (shallow early, long late = front-loading) -> penalized;
      * the longest turn(s) and the final turn contribute 0 (M_i == L_i there), so
        a hint-free single-turn rollout is 0 and the final solve turn is never
        penalized -- it only raises the bar for the turns before it.

    This is the raw quantity logged to wandb (``hint_shape_sum``); the reward
    subtracts ``coeff * shortfall_sum`` (see ``effort_shortfall_penalty``).
    Returns 0.0 for missing/empty/single-turn ``turn_lens`` (incl. legacy rollouts
    without the field).
    """
    if turn_lens is None:
        return 0.0
    if hasattr(turn_lens, "tolist"):
        turn_lens = turn_lens.tolist()
    lens = [float(x) for x in turn_lens if x is not None]
    if len(lens) < 2:
        return 0.0  # 0 or 1 turn: nothing later to fall short of.
    total = 0.0
    suffix_max = 0.0
    # right -> left: after updating, suffix_max == max(L_i .. L_T).
    for length in reversed(lens):
        if length > suffix_max:
            suffix_max = length
        if suffix_max > 0:
            total += (suffix_max - length) / suffix_max
    return total


def effort_shortfall_penalty(turn_lens, coeff: float) -> float:
    """Coeff-scaled effort-shaping penalty: ``coeff * effort_shortfall_sum(...)``.

    Returns 0.0 when disabled (``coeff <= 0``). The caller subtracts this from the
    CORRECT reward only.
    """
    if coeff <= 0:
        return 0.0
    return float(coeff) * effort_shortfall_sum(turn_lens)


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
    hint_shape_coeff: float = 0.0,
    **kwargs,
) -> dict:
    """HPRL reward: outcome correctness minus the summed hint penalty.

    The hint-penalty knobs (``hint_penalty_total`` / ``hint_penalty_hard_factor``
    / ``hint_guidance_difficulty`` / ``hint_strategy``) arrive as
    ``custom_reward_function.reward_kwargs`` so they are tunable at launch without
    regenerating the dataset. ``hint_strategy`` must match the strategy the agent
    loop ran (``data.hprl.strategy``): ``"hint"`` charges per-hint penalties,
    ``"major_step"`` charges per-step penalties directly.

    ``hint_shape_coeff`` (>0) additionally subtracts the effort-shaping penalty
    (``effort_shortfall_penalty``) from the CORRECT reward, discouraging hints
    called after too-little reasoning. 0 disables it.

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

    # --- effort-shaping penalty (order-aware: earlier turns must reason as hard
    # as later ones) -----------------------------------------------------
    # The raw, coefficient-FREE shortfall sum (sum_i relu(suffixmax_i - L_i)/
    # suffixmax_i over the ordered turn lengths) is computed ALWAYS -- even when
    # shaping is disabled (coeff <= 0) -- so the front-loading signal is logged to
    # wandb (hint_shape_sum) in the control arm too. The applied penalty is
    # coeff * that sum, subtracted ONLY from the correct reward (a wrong answer
    # stays at exactly the base incorrect_reward, -1).
    shape_sum = effort_shortfall_sum(extra_info.get("turn_lens"))
    shape_penalty = float(hint_shape_coeff) * shape_sum if hint_shape_coeff > 0 else 0.0

    # Wrong answer -> base (incorrect) reward, ignoring hints. Correct answer ->
    # base (correct) reward minus the summed hint penalty (each hint used lowers
    # the reward) and the effort-shaping penalty (shallow calls lower it further).
    # The correct score is FLOORED at incorrect_reward (-1): even a fully hinted /
    # front-loaded correct rollout never scores below a wrong answer, so the hint
    # + shaping penalties can't make "solve with hints" worse than "fail" (which
    # would push GRPO to suppress hint use entirely).
    if correct:
        score = max(base - penalty - shape_penalty, incorrect_reward)
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
        "hint_shape_sum": float(shape_sum),
        "hint_shape_penalty": float(shape_penalty),
        "hint_call_failed": float(hint_call_failed or 0),
    }
