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
# WHOLE-TURN variant (data.hprl.auto_hint.step_adv.whole_turn). Instead of the a_C/a_I
# split, score the ENTIRE turn [turn_start, turn_end) with ONE advantage -- the TD form
# for the whole turn as a macro-action that carried the rollout from its start state to
# its post-turn state, earning immediate reward = the injected hint's penalty r_se:
#     A = V[se+1] + r_se - V[ss]     (FAILED turn; == a_C + a_I telescoped)
#     A = V[se]          - V[ss]     (solve / no-fail turn; no hint given)
# i.e. V(s_end) + hint_penalty - V(s_start), where a failed turn ENDS at S_{se+1} (the
# hinted step completes -- exactly where the next turn resumes). This is BOUNDARY-FREE:
# it needs only (ss, se, is_fail), not the fuzzy verified-prefix boundary, so the whole
# turn is reinforced/penalized together. Non-fail turns are IDENTICAL to the split mode;
# a failed turn's verified-prefix tokens now share the combined value instead of getting
# a separate V[se]-V[ss]. The per-group value recursion, zeroing, scale/normalize and
# overlong penalty are unchanged.
#
# OVERLONG PENALTY -- two modes (data.hprl.auto_hint.step_adv.overlong_penalty_type):
#   * "post_hoc" (default): AFTER the advantages are assigned, subtract an absolute P_over
#     from each per-turn-cap TRUNCATION tail. Restores a floor the value-based a_I loses as a
#     group co-truncates (a_I = penalty*(fc/d - 1) -> 0 when fc/d -> 1). Only the truncated
#     rows move (down); every other row is untouched (so the non-truncate rows stay ~0).
#   * "value": fold the surcharge into the VALUE recursion instead -- a truncated rollout's
#     failed-step reward at its truncation state k is r_k - P_over, so
#         V[k] = V[k+1] + (F_k*r_k - T_k*P_over) / D_k      (T_k = #truncated-at-k <= F_k)
#     which LOWERS V[k] (and every state below it). With NO mean-centering that LIFTS every
#     non-truncated rollout anchored at state k to a POSITIVE advantage (+T_k*P_over/D_k),
#     while the non-truncate<->truncate gap stays a stable P_over that does NOT collapse as
#     the group co-truncates. Turns the length signal into a "do the within-length thing"
#     reward, not only a "don't truncate" push. (Trade-off: it also positively reinforces a
#     concise-but-WRONG first turn -- keep P_over small.)
# Both modes are SCORED-groups-only: a no-correct group is already zeroed, so it gets NO
# length penalty either way.
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


def classify_length_cut(n_tokens: int, max_turn_tokens: int, room: int) -> str | None:
    """How was an assistant turn's generation cut, for the auto-hint per-turn length cap?

    The rollout loop generates each turn with ``max_tokens = min(max_turn_tokens, room)``,
    where ``room`` is the GLOBAL response tokens still available BEFORE this turn
    (response_length - tokens already used). vLLM reports both an EOS and a length cut as
    "completed", so the cut is detected purely from the produced token count ``n_tokens``.

    Returns:
        "per_turn" -- the PER-TURN cap bound the turn and it ran to that cap: the per-turn
            cap is the binding limit (``max_turn_tokens <= room``, so it is <= room; the
            tie ``max_turn_tokens == room`` counts here) AND ``n_tokens`` reached it. The
            loop PUNISHES this (the turn failed to finish its next hint step).
        "global" -- the GLOBAL ceiling bound the turn: either the per-turn cap is off, or
            the remaining room was STRICTLY smaller than ``max_turn_tokens`` so the turn ran
            out of TOTAL budget (``n_tokens >= room``). The loop scores a LATER such turn
            with advantage 0 (the ceiling is arbitrary w.r.t. the turn).
        None -- the turn stopped EARLY (on EOS), short of every length limit: an ordinary
            turn the loop grades and continues from.

    So at the boundary the rule is: a turn cut at length ``== max_turn_tokens`` is
    "per_turn" (punished) even when that coincides with the global ceiling; only a strictly
    smaller global room (``room < max_turn_tokens``) yields a neutral "global" cut.
    """
    if max_turn_tokens and max_turn_tokens > 0:
        turn_cap = max(1, min(int(max_turn_tokens), int(room)))
        if int(max_turn_tokens) <= int(room) and n_tokens >= turn_cap:
            return "per_turn"
    return "global" if n_tokens >= int(room) else None


def compute_state_values(
    final_states: Sequence[int],
    fail_states_list: Sequence[Sequence[int]],
    penalty_vec: Sequence[float],
    K: int,
    *,
    terminal_value: float = 1.0,
    trunc_counts: Sequence[int] | None = None,
    overlong_penalty: float = 0.0,
) -> list:
    """Backward value recursion V[0..K] over a problem's N rollouts.

    Args:
        final_states: per-rollout verified final state V_i (K if it solved).
        fail_states_list: per-rollout list of FAILED hint indices H_i (0-indexed).
        penalty_vec: per-hint penalty WEIGHTS p[0..K-1] (>= 0); r_k = -p_k.
        K: number of hints in the pool (states are 0..K).
        terminal_value: V[K] (default 1.0).
        trunc_counts: VALUE-mode overlong (see the module header). ``trunc_counts[k]`` = T_k =
            #rollouts whose per-turn-cap TRUNCATION happened at state k (a subset of F_k). None
            (or ``overlong_penalty == 0``) -> the plain recursion (post-hoc / no overlong).
        overlong_penalty: P_over surcharge added to each truncated rollout's failed-step reward
            (r_k -> r_k - P_over) when ``trunc_counts`` is given, so it enters V BEFORE the
            advantage: ``V[k] = V[k+1] + (F_k*r_k - T_k*P_over) / D_k``.

    Returns:
        list V of length K+1. V[k] <= V[k+1] (non-decreasing toward the goal); a value-mode
        overlong only lowers it further at/below the truncation states.
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
    L = float(overlong_penalty)
    for k in range(K - 1, -1, -1):
        # D_k = #rollouts that reached state k (V_i >= k) = took transition k->k+1.
        d_k = sum(1 for v in final_states if v >= k)
        # T_k = #truncated-at-k (value-mode overlong): each carries an extra -P_over reward.
        t_k = int(trunc_counts[k]) if (trunc_counts is not None and k < len(trunc_counts)) else 0
        if d_k > 0 and (fail_count[k] > 0 or t_k > 0):
            r_k = -float(penalty_vec[k]) if k < len(penalty_vec) else 0.0
            V[k] = V[k + 1] + (fail_count[k] * r_k - t_k * L) / d_k
        else:
            V[k] = V[k + 1]
    return V


def assign_row_advantages(
    adv_row, returns_row, turns, V, penalty_vec, K, *, adv_scale: float = 1.0, whole_turn: bool = False,
    overlong_penalty: float = 0.0, row_truncated: bool = False,
) -> dict:
    """Overwrite one rollout's per-token advantage from its turn segments (IN PLACE).

    Zeroes the whole row first (dropping the stale GRPO advantage on every response
    token, incl. the masked injected-hint tokens), then writes each turn.

    Default (``whole_turn=False``) -- the two-segment SPLIT:
      * a_C [turn_start, boundary)  -> V[se] - V[ss]
      * a_I [boundary, turn_end)    -> r_se + V[se+1] - V[se]  (only a FAILED turn,
                                       se < K)
    ``whole_turn=True`` -- ONE advantage over the entire turn [turn_start, turn_end),
    ignoring ``boundary``:
      * FAILED turn (se < K)  -> r_se + V[se+1] - V[ss]   (== a_C + a_I telescoped)
      * else (solve/no-fail)  -> V[se] - V[ss]
    i.e. V(s_end) + hint_penalty - V(s_start), the whole turn scored together. Non-fail
    turns are identical to the split (they already carry b == turn_end).

    ``turns`` items are ``(turn_start, boundary, turn_end, state_start, state_end,
    is_fail)`` in RESPONSE-relative token coords (0 == first response token), exactly
    the columns of ``adv_row``. ``adv_scale`` multiplies every assigned advantage (a
    uniform gradient-magnitude knob; 1.0 == the raw value-based advantage). Returns
    scalar tallies for logging.

    ``row_truncated`` / ``overlong_penalty`` (VALUE-mode overlong, see the module header):
    when the row ended on a per-turn-cap TRUNCATION, its LAST turn segment failed its step
    while running over length, so that segment's reward is charged an extra ``-overlong_penalty``
    (r_se -> r_se - P_over). Paired with the ``T_k`` term in ``compute_state_values`` this keeps
    the whole-turn/a_I telescoping exact. No-op unless both are set (post-hoc mode passes
    neither -- its P_over is subtracted by the caller AFTER this).
    """
    seq_len = int(adv_row.shape[0])
    adv_row[:] = 0.0
    n_tok = 0
    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_neg = 0
    n_turns = len(turns)
    L = float(overlong_penalty)

    def _tally(a_val, n):
        nonlocal n_tok, pos_sum, neg_sum, n_pos, n_neg
        n_tok += n
        if a_val > 0:
            pos_sum += a_val * n
            n_pos += n
        elif a_val < 0:
            neg_sum += a_val * n
            n_neg += n

    for ti, t in enumerate(turns):
        ts, b, te, ss, se, is_fail = (int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]))
        ss = max(0, min(K, ss))
        se = max(0, min(K, se))
        ts = max(0, min(seq_len, ts))
        b = max(0, min(seq_len, b))
        te = max(0, min(seq_len, te))
        # value-mode overlong: the truncation is the row's LAST segment; charge its failed
        # step an extra -P_over so the surcharge rides V / the telescoping (not a post-hoc
        # subtract). Pairs with the T_k term in compute_state_values.
        trunc_here = bool(row_truncated) and L > 0.0 and ti == n_turns - 1
        if whole_turn:
            # ONE value over the whole turn [ts, te): the failed turn (its generation plus
            # the hint the loop injects) carries the rollout from S_ss to S_{se+1}, earning
            # immediate reward r_se (the hint's penalty); a solve/no-fail turn just advances
            # ss -> se with no hint. Boundary is unused (so no fuzzy-prefix dependence).
            if is_fail and se < K:
                r_se = -float(penalty_vec[se]) if se < len(penalty_vec) else 0.0
                if trunc_here:
                    r_se -= L
                a_val = float(r_se + V[se + 1] - V[ss]) * adv_scale
            else:
                a_val = float(V[se] - V[ss]) * adv_scale
            if te > ts:
                adv_row[ts:te] = a_val
                _tally(a_val, te - ts)
            continue
        # --- SPLIT mode ---
        # a_C: verified prefix advances ss -> se.
        if b > ts:
            a_c = float(V[se] - V[ss]) * adv_scale
            adv_row[ts:b] = a_c
            _tally(a_c, b - ts)
        # a_I: failed-step tail (only when the turn FAILED a hint that exists).
        if is_fail and se < K and te > b:
            r_se = -float(penalty_vec[se]) if se < len(penalty_vec) else 0.0
            if trunc_here:
                r_se -= L
            a_i = float(r_se + V[se + 1] - V[se]) * adv_scale
            adv_row[b:te] = a_i
            _tally(a_i, te - b)
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


# A group whose advantage std is below this is treated as having NO spread (degenerate)
# and left unnormalized -- dividing by ~0 would blow it up (the GRPO std->0 failure mode).
_NORM_EPS = 1e-6


def _group_token_std(advantages, response_mask, idxs) -> float:
    """Population std of the per-token advantages over the group's TRAINED tokens
    (``response_mask == 1``). Duck-typed over torch/numpy via element-wise ``*`` +
    ``.sum()`` (``adv * mask`` zeroes the untrained tokens; a trained token with a
    genuinely-0 advantage is still counted). 0.0 if the group has no trained token.
    """
    total = 0.0
    sq = 0.0
    cnt = 0
    for i in idxs:
        a = advantages[i]
        m = response_mask[i]
        am = a * m  # untrained tokens -> 0; trained keep their (possibly 0) advantage
        total += float(am.sum())
        sq += float((am * am).sum())
        cnt += int(float(m.sum()))
    if cnt <= 0:
        return 0.0
    mean = total / cnt
    var = sq / cnt - mean * mean
    return float(var ** 0.5) if var > 0.0 else 0.0


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
    normalize: bool = False,
    overlong_penalty: float = 0.0,
    overlong_penalty_type: str = "post_hoc",
    turn_truncated_per_row: Sequence[Any] | None = None,
    whole_turn: bool = False,
) -> tuple[Any, Any, dict]:
    """Replace GRPO advantages with the step-level value-based advantages.

    Groups rows by ``uids`` (the GRPO group = a problem's N rollouts), computes the
    per-problem state values, and writes per-token advantages per rollout. A group
    with no correct rollout is zeroed (``zero_if_no_correct``). Modifies the
    ``advantages``/``returns`` tensors IN PLACE and returns them with scalar stats.

    Per-row inputs are length B (the batch); ``turns_per_row[i]`` is the rollout's
    list of ``(ts, b, te, ss, se, is_fail)`` segments, ``penalty_per_row[i]`` its
    pool's per-hint penalty weights, ``K_per_row[i]`` its pool size.

    ``normalize`` (GRPO-style, per group): divide each group's per-token advantages by
    the group's advantage std (over its trained tokens), bringing the raw value-based
    advantages (which are small, ~penalty/K) to ~unit scale -- the fix for "the gradient
    is too small", adaptively per group. ``adv_scale`` then sets the TARGET std (1.0 ->
    unit). NO mean-centering: the value function V already baselines the advantages
    (their token-mean is ~0), and centering again would distort the a_C>=0 / a_I<=0 sign.
    A near-zero-std (degenerate) group is left unnormalized (the divide-by-~0 blow-up
    guard). With ``normalize=False`` it is the plain ``adv_scale`` multiply.

    ``whole_turn`` (default false): score each turn with a SINGLE value over its whole
    span (V[se+1]+r_se-V[ss] for a failed turn, V[se]-V[ss] otherwise) instead of the
    a_C/a_I prefix/tail split -- boundary-free, the turn reinforced/penalized together
    (see the module header). Everything else (values, zeroing, scale/normalize, overlong
    penalty) is identical; only the per-row token assignment changes.

    ``overlong_penalty`` (P_over, raw units, 0 = off) + ``overlong_penalty_type`` select how the
    per-turn-failure surcharge is charged (rows with ``turn_truncated_per_row[i]`` truthy:
    a cap hit or a missing box deliberately treated as one; the failure is the row's LAST
    turn segment). SCORED groups only either way -- no-correct groups are already zeroed +
    skipped above, so all-truncate / no-correct groups get NO penalty.

    * ``"post_hoc"`` (default): subtract P_over from each truncation tail AFTER the advantages
      are assigned, BEFORE the per-group std/normalize. The value-based tail
      ``a_I = penalty[se]*(fc/d - 1)`` collapses toward 0 as a group co-truncates (``fc/d -> 1``),
      so the value machinery cannot punish truncation once it is the group norm; the absolute
      P_over restores a floor that survives that collapse, and applied pre-std it is scaled INTO
      the group's unit range (~penalty/K, so ~0.1 lands near unit). Only the truncated rows move.
    * ``"value"``: fold P_over into the VALUE recursion instead (T_k truncated-at-k carry
      reward r_k - P_over), lowering ``V[k]`` and -- with no mean-centering -- LIFTING every
      non-truncated row anchored at k to a positive advantage (+T_k*P_over/D_k) while the
      non-truncate<->truncate gap stays a stable P_over. No post-hoc subtraction runs. See the
      module header for the full rationale / trade-off.
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
    group_std_sum = 0.0
    norm_factor_sum = 0.0
    ov_rows = 0
    ov_tokens = 0

    # overlong penalty routing: "post_hoc" subtracts P_over from truncation tails after the
    # advantages are assigned; "value" folds it into the value recursion (see docstring).
    _mode = str(overlong_penalty_type or "post_hoc").lower()
    _ov_on = float(overlong_penalty) > 0.0 and turn_truncated_per_row is not None
    value_overlong = _ov_on and _mode == "value"
    posthoc_overlong = _ov_on and _mode != "value"

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

        # VALUE-mode overlong: T_k = #rollouts truncated at state k (their last-turn se).
        # Folded into the recursion so the surcharge lowers V[k] and LIFTS the non-truncated
        # rows anchored there (vs the post-hoc tail subtraction further below).
        trunc_counts = None
        if value_overlong:
            trunc_counts = [0] * K
            for i in idxs:
                if not turn_truncated_per_row[i]:
                    continue
                trow = turns_per_row[i] or []
                if not trow:
                    continue
                s_tr = int(trow[-1][4])
                if 0 <= s_tr < K:
                    trunc_counts[s_tr] += 1
        V = compute_state_values(
            final_states, fail_states_list, pvec, K, terminal_value=terminal_value,
            trunc_counts=trunc_counts,
            overlong_penalty=(float(overlong_penalty) if value_overlong else 0.0),
        )
        v0_sum += V[0]
        n_scored_groups += 1

        # pass 1: assign the RAW value-based advantages (scale handled below so the
        # group std for normalization is read off the raw values).
        g_tokens = 0
        g_pos = 0.0
        g_neg = 0.0
        g_np = 0
        g_nn = 0
        for i in idxs:
            row_tr = bool(value_overlong and turn_truncated_per_row[i])
            tally = assign_row_advantages(
                advantages[i], returns[i], (turns_per_row[i] or []), V, pvec, K,
                adv_scale=1.0, whole_turn=whole_turn,
                overlong_penalty=(float(overlong_penalty) if row_tr else 0.0),
                row_truncated=row_tr,
            )
            g_tokens += tally["tokens"]
            g_pos += tally["pos_sum"]
            g_neg += tally["neg_sum"]
            g_np += tally["n_pos"]
            g_nn += tally["n_neg"]
            n_scored_rows += 1
            # value-mode overlong bookkeeping (parity with the post-hoc counters): the last
            # (truncated) segment carried the surcharge INSIDE the value/advantage above.
            if row_tr:
                trow = turns_per_row[i] or []
                if trow:
                    seq_len_i = int(advantages[i].shape[0])
                    b_i = max(0, min(seq_len_i, int(trow[-1][1])))
                    te_i = max(0, min(seq_len_i, int(trow[-1][2])))
                    if te_i > b_i:
                        ov_rows += 1
                        ov_tokens += (te_i - b_i)

        # POST-HOC over-turn-length penalty: subtract an absolute P_over from each per-turn-cap
        # truncation tail (the row's LAST segment) BEFORE the std/normalize below, so it is
        # read into the group std and scaled with the rest -- and, unlike the value-based
        # a_I, it does NOT ride the fc/d brake, so it survives a co-truncating group. Scored
        # groups only (no-correct groups already `continue`d), honoring "no penalty there".
        # Skipped in "value" mode (the surcharge is already inside V / the advantage above).
        if posthoc_overlong:
            for i in idxs:
                if not turn_truncated_per_row[i]:
                    continue
                trow = turns_per_row[i] or []
                if not trow:
                    continue
                seq_len = int(advantages[i].shape[0])
                b = max(0, min(seq_len, int(trow[-1][1])))
                te = max(0, min(seq_len, int(trow[-1][2])))
                if te > b:
                    advantages[i][b:te] -= float(overlong_penalty)
                    returns[i][b:te] = advantages[i][b:te]  # keep returns mirrored
                    ov_rows += 1
                    ov_tokens += (te - b)

        # group scale factor: normalize to unit std (then x adv_scale = target std), or
        # the plain adv_scale multiply. A degenerate (near-0-std) group keeps adv_scale.
        factor = float(adv_scale)
        if normalize:
            gstd = _group_token_std(advantages, response_mask, idxs)
            group_std_sum += gstd
            if gstd > _NORM_EPS:
                factor = float(adv_scale) / gstd
        norm_factor_sum += factor
        if factor != 1.0:
            for i in idxs:
                advantages[i] *= factor
                returns[i][:] = advantages[i]  # re-mirror (no-op when same tensor)
            g_pos *= factor  # factor > 0 -> signs preserved
            g_neg *= factor

        tokens_assigned += g_tokens
        pos_sum += g_pos
        neg_sum += g_neg
        n_pos += g_np
        n_neg += g_nn

    stats = {
        "step_adv/groups_total": float(n_groups),
        "step_adv/groups_zeroed": float(n_zeroed),
        "step_adv/groups_scored": float(n_scored_groups),
        "step_adv/rows_scored": float(n_scored_rows),
        "step_adv/tokens_assigned": float(tokens_assigned),
        "step_adv/adv_pos_mean": float(pos_sum / n_pos) if n_pos else 0.0,
        "step_adv/adv_neg_mean": float(neg_sum / n_neg) if n_neg else 0.0,
        "step_adv/value_s0_mean": float(v0_sum / n_scored_groups) if n_scored_groups else 0.0,
        "step_adv/group_std_mean": float(group_std_sum / n_scored_groups) if n_scored_groups else 0.0,
        "step_adv/norm_factor_mean": float(norm_factor_sum / n_scored_groups) if n_scored_groups else 0.0,
        "step_adv/overlong_rows": float(ov_rows),
        "step_adv/overlong_tokens": float(ov_tokens),
        # 1 = "value" mode folded P_over into V; 0 = post-hoc tail subtraction (or overlong off).
        "step_adv/overlong_value_mode": float(1.0 if value_overlong else 0.0),
    }
    return advantages, returns, stats
