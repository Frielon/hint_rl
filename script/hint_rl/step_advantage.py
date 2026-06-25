# Copyright 2026
#
# STEP-LEVEL advantage calculation for the AUTO-HINT (push-hint) rollout -- an
# ALTERNATIVE to GRPO's single scalar-per-rollout advantage, gated on
# ``data.hprl.auto_hint.step_adv.enable``. The pure tensor/group transform lives
# here (dependency-light, duck-typed over torch/numpy) so it can be unit-tested
# (test_auto_hint.py) without a live trainer.
#
# THE MODEL (devlog "Step level advantage calculation").
# A problem's hint pool is K ordered substep hints; we define K+1 STATES S_0..S_K,
# where S_k = "the first k hints (in pool order) are completed". A rollout walks
# this chain turn by turn. For each turn the selector (selector_multi) tells us:
#   * how far the turn got -- the contiguous count of completed hints (the state
#     reached), and
#   * which hint it FAILED (the first still-pending hint) -- if any.
# Each turn therefore splits into two segments over its RESPONSE tokens:
#   * a_C = the selector-VERIFIED prefix [turn_start, boundary): it advanced the
#     state S_{ss} -> S_{se} (ss = state at turn start, se = verified state end);
#     reward 0 (no hint needed to get here).
#   * a_I = the unverified tail [boundary, turn_end): it ATTEMPTED hint ``se`` and
#     failed, so the loop injected that hint; reward = r(h_se) = -penalty(h_se) < 0.
# The ENDING turn of a CORRECT rollout is all-a_C reaching S_K (se = K, no a_I).
#
# VALUES from the N rollouts of the problem (one GRPO group). Let H_i = the set of
# hint indices rollout i FAILED (the hints it was given, plus -- for an incorrect
# rollout -- the one identified on its labeled last turn), and V_i = the verified
# final state it reached (K if it solved). With r_k = -penalty[k]:
#
#     V[K] = terminal_value (=1; the last hint is the answer)
#     V[k] = V[k+1] + ( F_k * r_k ) / D_k        for k = K-1 .. 0
#       F_k = #{ i : k in H_i }            (rollouts that FAILED hint k)
#       D_k = #{ i : V_i >= k }            (rollouts that REACHED state k, i.e. that
#                                           took the S_k -> S_{k+1} transition)
# This D_k is exactly the TODO's denominator ``sum_i 1_{E_i > k}`` under the author's
# convention that an incorrect rollout whose selector-verified state is m has E_i = m+1
# (the post-check "attempted" state, with h_{m+1} added to H_i): 1_{E_i > k} = 1_{m >= k}.
# Here V_i carries the verified state m directly (so V_i >= k  <=>  E_i > k); same count.
# F_k <= D_k and r_k <= 0, so V is non-decreasing in k (closer to the goal = higher
# value). Per-segment advantages are the textbook TD form A = r + V(s') - V(s):
#
#     A(a_C) = V[se] - V[ss]                       >= 0   (progress is good)
#     A(a_I) = r_se + V[se+1] - V[se]
#            = r_se * (1 - F_se / D_se)            <= 0   (a needed hint is bad)
#
# So self-achieving a step that OFTEN trips others (F_k/D_k high) is rewarded, and
# failing a step OTHERS clear is penalized -- the GRPO-relative intuition, per step.
#
# ALL-INCORRECT groups get ZERO advantage everywhere (no gradient): without a single
# solve the V[K]=1 anchor is unreached and the positive a_C pulls would chase a goal
# nobody hit. (data.hprl.auto_hint.step_adv.zero_if_no_correct, default true.)

from __future__ import annotations

from typing import Any, Sequence


def prefix_state(pool_order, completed) -> int:
    """The step-adv STATE = length of the leading run of ``pool_order`` all present in
    ``completed`` (the contiguous-completed-prefix count); states range 0..K.

    The multi-round selector only ever marks the next pending hints completed (in pool
    order), so ``completed`` is a contiguous prefix and this equals its size; taking the
    leading run only makes it robust to a stray out-of-order id. Used by the rollout loop
    to label each turn's state_start (and the solving turn's state_start).
    """
    done = {str(x) for x in (completed or [])}
    s = 0
    for h in pool_order:
        if str(h) in done:
            s += 1
        else:
            break
    return s


def compute_state_values(
    final_states: Sequence[int],
    fail_states_list: Sequence[Sequence[int]],
    penalty_vec: Sequence[float],
    K: int,
    *,
    terminal_value: float = 1.0,
) -> list:
    """Backward value recursion V[0..K] over a problem's N rollouts.

    Args:
        final_states: per-rollout verified final state V_i (K if it solved).
        fail_states_list: per-rollout list of FAILED hint indices H_i (0-indexed).
        penalty_vec: per-hint penalty WEIGHTS p[0..K-1] (>= 0); r_k = -p_k.
        K: number of hints in the pool (states are 0..K).
        terminal_value: V[K] (default 1.0).

    Returns:
        list V of length K+1. V[k] <= V[k+1] (non-decreasing toward the goal).
    """
    V = [0.0] * (K + 1)
    if K < 0:
        return V
    V[K] = float(terminal_value)
    # F_k = #rollouts that failed hint k.
    fail_count = [0] * K
    for fs in fail_states_list:
        for k in fs:
            if 0 <= k < K:
                fail_count[k] += 1
    for k in range(K - 1, -1, -1):
        # D_k = #rollouts that reached state k (V_i >= k) = took transition k->k+1.
        d_k = sum(1 for v in final_states if v >= k)
        if d_k > 0 and fail_count[k] > 0:
            r_k = -float(penalty_vec[k]) if k < len(penalty_vec) else 0.0
            V[k] = V[k + 1] + (fail_count[k] * r_k) / d_k
        else:
            V[k] = V[k + 1]
    return V


def assign_row_advantages(adv_row, returns_row, turns, V, penalty_vec, K, *, adv_scale: float = 1.0) -> dict:
    """Overwrite one rollout's per-token advantage from its turn segments (IN PLACE).

    Zeroes the whole row first (dropping the stale GRPO advantage on every response
    token, incl. the masked injected-hint tokens), then writes each segment:
      * a_C [turn_start, boundary)  -> V[se] - V[ss]
      * a_I [boundary, turn_end)    -> r_se + V[se+1] - V[se]  (only a FAILED turn,
                                       se < K)
    ``turns`` items are ``(turn_start, boundary, turn_end, state_start, state_end,
    is_fail)`` in RESPONSE-relative token coords (0 == first response token), exactly
    the columns of ``adv_row``. ``adv_scale`` multiplies every assigned advantage (a
    uniform gradient-magnitude knob; 1.0 == the raw value-based advantage). Returns
    scalar tallies for logging.
    """
    seq_len = int(adv_row.shape[0])
    adv_row[:] = 0.0
    n_tok = 0
    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_neg = 0
    for t in turns:
        ts, b, te, ss, se, is_fail = (int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]))
        ss = max(0, min(K, ss))
        se = max(0, min(K, se))
        ts = max(0, min(seq_len, ts))
        b = max(0, min(seq_len, b))
        te = max(0, min(seq_len, te))
        # a_C: verified prefix advances ss -> se.
        if b > ts:
            a_c = float(V[se] - V[ss]) * adv_scale
            adv_row[ts:b] = a_c
            n = b - ts
            n_tok += n
            if a_c > 0:
                pos_sum += a_c * n
                n_pos += n
            elif a_c < 0:
                neg_sum += a_c * n
                n_neg += n
        # a_I: failed-step tail (only when the turn FAILED a hint that exists).
        if is_fail and se < K and te > b:
            r_se = -float(penalty_vec[se]) if se < len(penalty_vec) else 0.0
            a_i = float(r_se + V[se + 1] - V[se]) * adv_scale
            adv_row[b:te] = a_i
            n = te - b
            n_tok += n
            if a_i > 0:
                pos_sum += a_i * n
                n_pos += n
            elif a_i < 0:
                neg_sum += a_i * n
                n_neg += n
    # returns mirrors advantages (GRPO carries no critic; returns == advantages).
    # No-op when the trainer passes the same tensor object for both.
    returns_row[:] = adv_row
    return {
        "tokens": n_tok,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def _zero_rows(advantages, returns, idxs) -> None:
    for i in idxs:
        advantages[i][:] = 0.0
        returns[i][:] = 0.0


def apply_step_level_advantages(
    advantages,
    returns,
    response_mask,
    uids: Sequence[Any],
    turns_per_row: Sequence[Sequence[Any]],
    correct_per_row: Sequence[bool],
    penalty_per_row: Sequence[Sequence[float]],
    K_per_row: Sequence[int],
    *,
    terminal_value: float = 1.0,
    zero_if_no_correct: bool = True,
    adv_scale: float = 1.0,
) -> tuple[Any, Any, dict]:
    """Replace GRPO advantages with the step-level value-based advantages.

    Groups rows by ``uids`` (the GRPO group = a problem's N rollouts), computes the
    per-problem state values, and writes per-token advantages per rollout. A group
    with no correct rollout is zeroed (``zero_if_no_correct``). Modifies the
    ``advantages``/``returns`` tensors IN PLACE and returns them with scalar stats.

    Per-row inputs are length B (the batch); ``turns_per_row[i]`` is the rollout's
    list of ``(ts, b, te, ss, se, is_fail)`` segments, ``penalty_per_row[i]`` its
    pool's per-hint penalty weights, ``K_per_row[i]`` its pool size.
    """
    n_rows = int(advantages.shape[0])
    groups: dict[Any, list] = {}
    for i in range(n_rows):
        groups.setdefault(uids[i], []).append(i)

    n_groups = len(groups)
    n_zeroed = 0
    n_scored_groups = 0
    n_scored_rows = 0
    tokens_assigned = 0
    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_neg = 0
    v0_sum = 0.0

    for _uid, idxs in groups.items():
        # pool size / penalty vector are the same across a group; take the row with
        # the largest K (a degenerate row may carry K=0 / empty turns).
        K = 0
        pvec: Sequence[float] = []
        for i in idxs:
            ki = int(K_per_row[i] or 0)
            if ki > K:
                K = ki
                pvec = penalty_per_row[i] or []
        has_correct = any(bool(correct_per_row[i]) for i in idxs)
        if K <= 0 or (zero_if_no_correct and not has_correct):
            _zero_rows(advantages, returns, idxs)
            n_zeroed += 1
            continue

        final_states = []
        fail_states_list = []
        for i in idxs:
            turns = turns_per_row[i] or []
            if bool(correct_per_row[i]):
                v_i = K
            else:
                v_i = 0
                for t in turns:
                    se = int(t[4])
                    if se > v_i:
                        v_i = se
            final_states.append(min(K, max(0, v_i)))
            fails = [int(t[4]) for t in turns if int(t[5]) and int(t[4]) < K]
            fail_states_list.append(fails)

        V = compute_state_values(
            final_states, fail_states_list, pvec, K, terminal_value=terminal_value
        )
        v0_sum += V[0]
        n_scored_groups += 1
        for i in idxs:
            tally = assign_row_advantages(
                advantages[i], returns[i], (turns_per_row[i] or []), V, pvec, K,
                adv_scale=adv_scale,
            )
            tokens_assigned += tally["tokens"]
            pos_sum += tally["pos_sum"]
            neg_sum += tally["neg_sum"]
            n_pos += tally["n_pos"]
            n_neg += tally["n_neg"]
            n_scored_rows += 1

    stats = {
        "step_adv/groups_total": float(n_groups),
        "step_adv/groups_zeroed": float(n_zeroed),
        "step_adv/groups_scored": float(n_scored_groups),
        "step_adv/rows_scored": float(n_scored_rows),
        "step_adv/tokens_assigned": float(tokens_assigned),
        "step_adv/adv_pos_mean": float(pos_sum / n_pos) if n_pos else 0.0,
        "step_adv/adv_neg_mean": float(neg_sum / n_neg) if n_neg else 0.0,
        "step_adv/value_s0_mean": float(v0_sum / n_scored_groups) if n_scored_groups else 0.0,
    }
    return advantages, returns, stats
