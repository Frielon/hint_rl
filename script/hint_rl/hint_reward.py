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
#     R = incorrect_reward + b                if the answer is wrong
#     R = correct_reward - sum_k w_{j_k}      if the answer is correct
#
# i.e. a wrong answer gets the base (incorrect) reward plus a one-off, EFFORT-GATED
# hint-call bonus b (= max(0, hint_call_reward - shape_penalty) if the rollout
# RECEIVED >=1 applied hint, else 0 -- the effort-shaping penalty claws the bonus back
# when the hint followed too little reasoning, so a shallow hinted failure cannot
# out-earn an unhinted one); a correct answer gets the base (correct) reward minus the
# summed importance
# weights {w_{j_k}} of the hints applied during the rollout (the bonus does NOT
# apply when correct), FLOORED at the best achievable wrong score
# (incorrect_reward + format_reward + hint_call_reward) so solving always ranks at
# least as high as any failure. Correctness is verified exactly as in the plain
# GRPO run (mathruler boxed-answer grading).
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
    hint_call_reward: float = 0.1,
    hint_penalty_total: float = DEFAULT_TOTAL_PENALTY,
    hint_penalty_hard_factor: float = DEFAULT_HARD_FACTOR,
    hint_guidance_difficulty: str = DEFAULT_GUIDANCE_DIFFICULTY,
    hint_strategy: str = STRATEGY_HINT,
    hint_shape_coeff: float = 0.0,
    budget_exceeded_reward: Optional[float] = None,
    **kwargs,
) -> dict:
    """HPRL reward: outcome correctness minus the summed hint penalty.

    The hint-penalty knobs (``hint_penalty_total`` / ``hint_penalty_hard_factor``
    / ``hint_guidance_difficulty`` / ``hint_strategy``) arrive as
    ``custom_reward_function.reward_kwargs`` so they are tunable at launch without
    regenerating the dataset. ``hint_strategy`` must match the strategy the agent
    loop ran (``data.hprl.strategy``): ``"hint"`` charges per-hint penalties,
    ``"major_step"`` charges per-step penalties directly.

    ``hint_shape_coeff`` (>0) additionally applies the effort-shaping penalty
    (``coeff * effort_shortfall_sum``), discouraging hints called after too-little
    reasoning. It now hits BOTH outcome branches: subtracted from the CORRECT reward,
    AND clawed back from the INCORRECT branch's hint-call bonus (see below), so a
    front-loaded hinted FAILURE cannot out-earn an unhinted failure. (Previously it was
    correct-branch-only; on a low-accuracy regime the ~85% of rollouts that fail never
    saw it, and the policy learned the cheapest hinted failure -- minimal CoT then a
    hint -- for the free hint-call bonus.) 0 disables it.

    ``hint_call_reward`` (default 0.1) is a one-off bonus added to the INCORRECT
    reward when the rollout RECEIVED at least one hint (correct rollouts are
    unaffected). It is binary in hint COUNT -- one applied hint is enough; extra hints
    add nothing, so there is no incentive to spam. The intent is to keep a positive
    gradient on hint use among the failing rollouts (GRPO otherwise ranks
    unhinted-correct above hinted-correct and the policy learns to suppress hint
    use). It gates on an APPLIED hint (``len(applied_hints) >= 1``), NOT a bare
    ``<hint_call/>`` emission: a call the selector failed to serve conveyed nothing
    to the policy, and rewarding the emission alone would pay out during a selector
    outage (when ``hint_call_failed`` spikes). It is EFFORT-GATED by
    ``hint_shape_coeff``: the bonus is ``max(0, hint_call_reward - shape_penalty)``, so
    a hint that followed real reasoning keeps the full bonus while a front-loaded call
    (short CoT then ``<hint_call/>``) has it cancelled back to 0. Without this gate the
    bonus lives ENTIRELY on the incorrect branch -- which the correct-only shape penalty
    never reached -- so the policy learns to emit a minimal CoT then a hint for a free
    reward. 0 disables.

    ``budget_exceeded_reward`` (default ``None`` -> ``incorrect_reward``) is the FLOOR
    score assigned when the agent loop flags an OVER-BUDGET hint call
    (``extra_info["hint_budget_exceeded"]``): the policy emitted ``<hint_call/>`` after
    its budget ``B_q`` was spent. This is treated as a hard protocol violation -- the
    rollout short-circuits to this floor with ``acc=0`` and the boxed answer is NOT
    graded (a correct box earns nothing if the rollout ended on an illegal call). Set
    it BELOW ``incorrect_reward`` to make an over-budget call strictly worse than an
    ordinary failure.

    Returns a dict whose ``score`` is the optimized reward and whose extra keys
    are logged as ``reward_extra_info`` by the reward manager (``acc`` is also
    used by GRPO group filtering).
    """
    extra_info = extra_info or {}

    # --- per-rollout state recorded by the agent loop --------------------
    applied_hints = extra_info.get("applied_hints") or []
    # Defensive: applied_hints may arrive as a numpy 0-d object; normalize.
    if hasattr(applied_hints, "tolist"):
        applied_hints = applied_hints.tolist()
    if applied_hints is None:
        applied_hints = []

    # Count of hint calls whose selector lookup FAILED (the policy got the no-op
    # fallback). Normalized once here (numpy 0-d safe) and reused below.
    hint_call_failed = extra_info.get("hint_call_failed", 0)
    if hasattr(hint_call_failed, "item"):
        hint_call_failed = hint_call_failed.item()

    # Selector latency recorded by the agent loop: total seconds spent in the frozen
    # selector this rollout and the number of selector calls. Passed straight through
    # to the result dict so hint_budget_callback can aggregate hprl/hint_select_time_*
    # (the verl-native hint_select timer is dropped before aggregation; see agent loop).
    hint_select_time = extra_info.get("hint_select_time", 0.0)
    if hasattr(hint_select_time, "item"):
        hint_select_time = hint_select_time.item()
    hint_select_calls = extra_info.get("hint_select_calls", 0)
    if hasattr(hint_select_calls, "item"):
        hint_select_calls = hint_select_calls.item()

    # Box-then-call counters recorded by the agent loop: total hint calls this
    # rollout made, and how many were emitted by a turn that already had a \boxed{}.
    # Passed straight through so hint_budget_callback can log the per-call and
    # per-rollout box-then-call rates (hprl/hint_call_with_box_*).
    hint_calls_total = extra_info.get("hint_calls_total", 0)
    if hasattr(hint_calls_total, "item"):
        hint_calls_total = hint_calls_total.item()
    hint_calls_with_box = extra_info.get("hint_calls_with_box", 0)
    if hasattr(hint_calls_with_box, "item"):
        hint_calls_with_box = hint_calls_with_box.item()

    # --- outcome correctness (same as the plain GRPO run) ----------------
    pred = extract_boxed_content(solution_str)
    has_format = pred != _NO_BOX

    # --- protocol violation: over-budget hint call -> FLOOR score --------
    # The agent loop sets extra_info["hint_budget_exceeded"]=1 when the policy emits
    # <hint_call/> AFTER its budget is spent, then terminates the rollout. Asking for
    # help it cannot have is a hard protocol violation: we SHORT-CIRCUIT to the floor
    # reward (budget_exceeded_reward, default incorrect_reward) and do NOT grade the
    # boxed answer -- even a correct box earns nothing if the rollout ended on an
    # illegal call. acc=0 so it counts as a failure for GRPO group filtering and the
    # budget ratchet; the hint penalty / shaping / call-bonus are all bypassed.
    hint_budget_exceeded = extra_info.get("hint_budget_exceeded", 0)
    if hasattr(hint_budget_exceeded, "item"):
        hint_budget_exceeded = hint_budget_exceeded.item()
    if hint_budget_exceeded:
        floor = incorrect_reward if budget_exceeded_reward is None else float(budget_exceeded_reward)
        return {
            "score": float(floor),
            "acc": 0.0,
            "pred": pred,
            "has_format": 1.0 if has_format else 0.0,
            "num_hints": float(len(applied_hints)),
            "called_hint": 1.0 if len(applied_hints) >= 1 else 0.0,
            "hint_call_bonus": 0.0,
            "hint_penalty": 0.0,
            "hint_shape_sum": 0.0,
            "hint_shape_penalty": 0.0,
            "hint_call_failed": float(hint_call_failed or 0),
            "hint_budget_exceeded": 1.0,
            "hint_select_time": float(hint_select_time or 0.0),
            "hint_select_calls": float(hint_select_calls or 0),
            "hint_calls_total": float(hint_calls_total or 0),
            "hint_calls_with_box": float(hint_calls_with_box or 0),
        }

    correct = has_format and grade_answer(pred, ground_truth)

    base = correct_reward if correct else incorrect_reward
    if format_reward and has_format:
        base += format_reward

    # --- hint penalty (subtracted from the correct reward) ---------------
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
    # coeff * that sum; it is subtracted from the correct reward AND clawed back from
    # the incorrect branch's hint-call bonus (see below), so front-loading is
    # discouraged on BOTH outcomes (a wrong answer with no applied hint still stays at
    # exactly the base incorrect_reward).
    shape_sum = effort_shortfall_sum(extra_info.get("turn_lens"))
    shape_penalty = float(hint_shape_coeff) * shape_sum if hint_shape_coeff > 0 else 0.0

    # --- hint-call bonus (added to the INCORRECT reward only) ------------
    # Reward a FAILING rollout for getting unstuck via a hint: a wrong answer that
    # received a hint scores above one that plowed straight ahead to a wrong answer.
    # We gate on a hint the selector actually SERVED (len(applied_hints) >= 1), NOT
    # on a bare <hint_call/> emission. A call the selector FAILED to serve gave the
    # policy no information to course-correct with (it got the "no hint available,
    # continue on your own" no-op), so it behaves like a rollout that never called;
    # worse, hint_call_failed spikes during a selector OUTAGE (selectors have gone
    # down silently here), and rewarding the bare emission would pay out a bonus for
    # a no-op sentinel exactly then -- training the policy to emit it as a reward-
    # grab decoupled from hint use. BINARY in hint COUNT -- one applied hint is enough,
    # extra hints add nothing -- so there is NO incentive to spam hint calls.
    #
    # EFFORT-GATED: the bonus is reduced by the SAME effort-shaping penalty the correct
    # branch subtracts (shape_penalty = coeff * front-loading shortfall), then clamped at
    # 0. Rationale: this bonus lives ENTIRELY on the incorrect branch, but the shape
    # penalty used to be applied ONLY on the correct branch -- and on a low-accuracy
    # regime most rollouts FAIL, so the policy discovered that the cheapest hinted failure
    # (a near-empty CoT then <hint_call/>) collects the full +hint_call_reward while the
    # shape penalty never reaches it. Subtracting shape_penalty HERE puts the deterrent on
    # the same branch as the incentive: a hint that followed REAL reasoning (shape_penalty
    # ~ 0) keeps the full bonus; a front-loaded call (shape_penalty >= hint_call_reward)
    # has the bonus cancelled to 0 -> same score as not calling. Clamped at 0 so a shallow
    # hinted failure is never pushed BELOW an unhinted one (that would re-suppress hint
    # use, the very thing this bonus exists to prevent) -- only made no-better. Disabling
    # shaping (hint_shape_coeff <= 0 -> shape_penalty == 0) restores the flat bonus exactly.
    called_hint = len(applied_hints) >= 1
    hint_call_bonus = (
        max(0.0, float(hint_call_reward) - shape_penalty)
        if (not correct and called_hint)
        else 0.0
    )

    # Floor for a CORRECT rollout's score: the BEST achievable WRONG score -- a wrong
    # answer that is well-formed (format_reward) AND called a hint (hint_call_reward).
    # Flooring here guarantees a correct rollout is never worth less than ANY wrong
    # rollout in the GRPO group, even after the full hint + shaping penalties, so
    # "solve (with hints)" can never rank below "fail (with a hint)" -- which would
    # otherwise push GRPO to suppress solving-with-hints. (Previously floored at
    # incorrect_reward alone, which left a heavily-penalized correct rollout able to
    # dip just under a hinted failure.)
    correct_floor = incorrect_reward + format_reward + hint_call_reward + 0.05

    # Wrong answer -> base (incorrect) reward + the one-off hint-call bonus (keeps a
    # positive gradient on hint use that GRPO otherwise suppresses). Correct answer
    # -> base (correct) reward minus the summed hint penalty (each hint used lowers
    # the reward) and the effort-shaping penalty (shallow calls lower it further),
    # FLOORED at correct_floor; the bonus does NOT apply when correct.
    if correct:
        score = max(base - penalty - shape_penalty, correct_floor)
    else:
        score = base + hint_call_bonus

    # Count of hint calls this rollout made whose selector lookup FAILED (the
    # policy got the no-op fallback instead of a hint). Recorded by the agent loop
    # on extra_fields; surfaced here so it lands in non_tensor_batch for
    # hint_budget_callback to aggregate -> a selector outage shows up in hprl/*.
    # NOT rewarded (see the hint-call bonus above) -- only logged, as the signal we
    # watch to DETECT such an outage instead of paying it out. (Normalized above.)
    return {
        "score": float(score),
        "acc": 1.0 if correct else 0.0,
        "pred": pred,
        "has_format": 1.0 if has_format else 0.0,
        "num_hints": float(len(applied_hints)),
        "called_hint": 1.0 if called_hint else 0.0,
        "hint_call_bonus": float(hint_call_bonus),
        "hint_penalty": float(penalty),
        "hint_shape_sum": float(shape_sum),
        "hint_shape_penalty": float(shape_penalty),
        "hint_call_failed": float(hint_call_failed or 0),
        "hint_budget_exceeded": 0.0,
        "hint_select_time": float(hint_select_time or 0.0),
        "hint_select_calls": float(hint_select_calls or 0),
        "hint_calls_total": float(hint_calls_total or 0),
        "hint_calls_with_box": float(hint_calls_with_box or 0),
    }
