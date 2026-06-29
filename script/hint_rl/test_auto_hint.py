#!/usr/bin/env python3
# Copyright 2026
#
# Unit tests for the auto-hint mechanism's two dependency-light pieces:
#   * auto_hint_mask.apply_positive_adv_masking -- the verified-prefix loss-mask
#     transform (sign-gated span zeroing), exercised with plain numpy arrays.
#   * selector_multi.locate_quote_end / render_hints_with_status / pending_hint_ids
#     -- the fuzzy quote locator and status-rendering used by the rollout.
#
# Run:  python test_auto_hint.py   (or pytest test_auto_hint.py)
from __future__ import annotations

import json

import numpy as np

from auto_hint_mask import apply_positive_adv_masking, sequence_advantage
from budget_manager import BudgetManager, compute_adaptive_budget
from hint_penalty import compute_hint_penalties
from selector_multi import (
    locate_quote_end,
    pending_hint_ids,
    prune_hint_pool,
    render_hints_with_status,
)
from step_advantage import (
    apply_step_level_advantages,
    compute_state_values,
    prefix_state,
)


def _broadcast_adv(scalars, length):
    """(B,) per-sequence advantages -> (B, L) GRPO-style broadcast."""
    return np.array([[s] * length for s in scalars], dtype=np.float64)


def test_positive_adv_masks_spans():
    # row0: positive adv -> its span [2,4) is zeroed; row1: negative -> untouched.
    mask = np.ones((2, 6), dtype=np.int64)
    adv = _broadcast_adv([+1.0, -1.0], 6)
    spans = [[[2, 4]], [[2, 4]]]
    out, stats = apply_positive_adv_masking(mask, adv, spans)
    assert out[0].tolist() == [1, 1, 0, 0, 1, 1], out[0].tolist()
    assert out[1].tolist() == [1, 1, 1, 1, 1, 1], out[1].tolist()
    assert stats["auto_hint/rows_pos_masked"] == 1.0
    assert stats["auto_hint/mask_tokens_dropped"] == 2.0
    assert stats["auto_hint/rows_with_spans"] == 2.0


def test_zero_and_no_spans_are_full():
    # advantage exactly 0 is NOT positive -> full mask kept even with a span.
    mask = np.ones((2, 5), dtype=np.int64)
    adv = _broadcast_adv([0.0, +2.0], 5)
    spans = [[[1, 3]], []]  # row1 has no spans (ending-turn-only rollout)
    out, stats = apply_positive_adv_masking(mask, adv, spans)
    assert out[0].tolist() == [1, 1, 1, 1, 1]
    assert out[1].tolist() == [1, 1, 1, 1, 1]
    assert stats["auto_hint/rows_pos_masked"] == 0.0
    assert stats["auto_hint/mask_tokens_dropped"] == 0.0


def test_multiple_spans_and_clamping():
    mask = np.ones((1, 8), dtype=np.int64)
    adv = _broadcast_adv([+0.5], 8)
    # two hinted turns + an out-of-range end that must clamp to L.
    spans = [[[1, 2], [4, 6], [7, 999]]]
    out, stats = apply_positive_adv_masking(mask, adv, spans)
    assert out[0].tolist() == [1, 0, 1, 1, 0, 0, 1, 0]
    assert stats["auto_hint/mask_tokens_dropped"] == 4.0


def test_advantage_uses_only_unmasked_tokens():
    # masked-out (observation) tokens carry adv 0; the sign must come from the
    # trained tokens. Here trained tokens are +3 -> positive.
    mask = np.array([[1, 1, 0, 0, 1]], dtype=np.int64)
    adv = np.array([[3.0, 3.0, 0.0, 0.0, 3.0]], dtype=np.float64)
    assert sequence_advantage(adv[0], mask[0]) == 3.0
    out, _ = apply_positive_adv_masking(mask, adv, [[[4, 5]]])
    assert out[0].tolist() == [1, 1, 0, 0, 0]


def test_all_masked_row_is_safe():
    mask = np.zeros((1, 4), dtype=np.int64)
    adv = np.zeros((1, 4), dtype=np.float64)
    out, stats = apply_positive_adv_masking(mask, adv, [[[0, 4]]])
    assert out[0].tolist() == [0, 0, 0, 0]
    assert stats["auto_hint/rows_pos_masked"] == 0.0


def test_locate_exact():
    text = "We have k <= sqrt(n) < k+1 here. Then the next part."
    q = "k <= sqrt(n) < k+1"
    end = locate_quote_end(q, text)
    assert end is not None and text[:end].endswith("k+1"), end


def test_locate_whitespace_diff():
    text = "First line of work.\n   Then  we   conclude  the bound."
    q = "Then we conclude the bound."
    end = locate_quote_end(q, text)
    assert end is not None and end >= text.index("conclude")


def test_locate_unicode_latex_rewrite():
    # quote uses unicode; trace uses LaTeX -- loose() should still match.
    text = r"so \(\lceil \sqrt{n} \rceil = k+1\) follows immediately"
    q = "⌈√n⌉ = k+1"
    end = locate_quote_end(q, text, fuzzy_threshold=0.6)
    assert end is not None, "unicode<->latex quote should fuzzy-locate"


def test_locate_no_match_returns_none():
    text = "completely unrelated reasoning about triangles and circles"
    q = "the determinant of the Jacobian vanishes on the boundary"
    assert locate_quote_end(q, text) is None


def test_citation_found_rate_counting():
    # mirrors the loop's per-quote counting: real quotes count as found, a fabricated
    # one does not, and empty/missing quotes are skipped (not counted in the total).
    trace = "We set k = floor(sqrt(n)). Then k^2 <= n < (k+1)^2 holds for nonsquares."
    completed_hints = [
        {"hint_id": "1.1", "quote": "k = floor(sqrt(n))"},        # found (exact)
        {"hint_id": "1.2", "quote": "k^2 <= n < (k+1)^2"},        # found (exact)
        {"hint_id": "1.3", "quote": "the determinant vanishes"},  # fabricated -> not found
        {"hint_id": "1.4", "quote": "   "},                        # blank -> skipped
        {"hint_id": "1.5"},                                        # no quote -> skipped
        "not-a-dict",                                              # malformed -> skipped
    ]
    total = found = 0
    for c in completed_hints:
        q = c.get("quote") if isinstance(c, dict) else None
        if isinstance(q, str) and q.strip():
            total += 1
            if locate_quote_end(q, trace, 0.8) is not None:
                found += 1
    assert total == 3 and found == 2, (total, found)  # 2/3 cited sentences located


_POOL = {
    "steps": [
        {"step_id": 1, "purpose": "p1", "hints": [
            {"hint_id": "1.1", "hint": "a"}, {"hint_id": "1.2", "hint": "b"}]},
        {"step_id": 2, "purpose": "p2", "hints": [
            {"hint_id": "2.1", "hint": "c"}]},
    ]
}


def test_render_status_marks_completed():
    rendered = json.loads(render_hints_with_status(_POOL, ["1.1"]))
    flat = {h["hint_id"]: h["status"] for st in rendered["steps"] for h in st["hints"]}
    assert flat == {"1.1": "completed", "1.2": "pending", "2.1": "pending"}


def test_pending_hint_ids():
    assert pending_hint_ids(_POOL, []) == ["1.1", "1.2", "2.1"]
    assert pending_hint_ids(_POOL, ["1.1", "1.2"]) == ["2.1"]
    assert pending_hint_ids(_POOL, ["1.1", "1.2", "2.1"]) == []  # exhausted
    assert pending_hint_ids(json.dumps(_POOL), ["1.1"]) == ["1.2", "2.1"]  # JSON str ok


def test_prune_hint_pool_drops_x0_and_type():
    # X.0 step-guidance hints (by id OR by type) are dropped; substeps kept; the
    # per-hint `type` field is stripped. Accepts a dict or a JSON string.
    pool = {"steps": [
        {"step_id": 1, "purpose": "p1", "hints": [
            {"hint_id": "1.0", "hint": "guide", "type": "step_guidence_hint"},
            {"hint_id": "1.1", "hint": "a", "type": "substep_hint"}]},
        {"step_id": 2, "purpose": "p2", "hints": [
            {"hint_id": "2.0", "hint": "guide2"},          # dropped by the .0 id alone
            {"hint_id": "2.1", "hint": "c", "type": "substep_hint"}]},
    ]}
    pruned = prune_hint_pool(pool)
    assert pending_hint_ids(pruned, []) == ["1.1", "2.1"]
    assert all("type" not in h for st in pruned["steps"] for h in st["hints"])
    # JSON-string input prunes identically; an unparseable value passes through.
    assert pending_hint_ids(prune_hint_pool(json.dumps(pool)), []) == ["1.1", "2.1"]
    assert prune_hint_pool("not json") == "not json"


# --------------------------------------------------------------------------- #
# adaptive ratchet rule (budget_manager.compute_adaptive_budget)
# --------------------------------------------------------------------------- #
def test_adaptive_raises_when_no_correct():
    # no correct rollout -> raise by 1 (clamped to max_budget).
    upd = compute_adaptive_budget(3, [(False, 3)] * 8, min_budget=0, max_budget=8)
    assert upd.new_budget == 4 and upd.rule == "raise" and upd.changed


def test_adaptive_raise_clamped_at_max():
    upd = compute_adaptive_budget(8, [(False, 8)] * 8, min_budget=0, max_budget=8)
    assert upd.new_budget == 8 and upd.rule == "raise" and not upd.changed


def test_adaptive_over_half_sets_n2_smallest():
    # N=8, 5 correct (over half), correct hint counts [0,1,2,3,4]; N/2-th (4th) smallest = 3.
    results = [(True, 0), (True, 1), (True, 2), (True, 3), (True, 4)] + [(False, 6)] * 3
    upd = compute_adaptive_budget(6, results, min_budget=0, max_budget=8)
    assert upd.new_budget == 3 and upd.rule == "pivot_set" and upd.pivot_hint_count == 3


def test_adaptive_exactly_half_holds():
    # N=8, exactly 4 correct -> NOT over half -> unchanged.
    results = [(True, 0)] * 4 + [(False, 5)] * 4
    upd = compute_adaptive_budget(5, results, min_budget=0, max_budget=8)
    assert upd.new_budget == 5 and upd.rule == "unchanged" and not upd.changed


def test_adaptive_some_but_not_half_holds():
    results = [(True, 1), (True, 2)] + [(False, 4)] * 6  # 2/8 correct
    upd = compute_adaptive_budget(4, results, min_budget=0, max_budget=8)
    assert upd.new_budget == 4 and upd.rule == "unchanged"


def test_adaptive_pivot_clamped_to_min():
    results = [(True, 0)] * 5 + [(False, 3)] * 3  # over half, pivot would be 0
    upd = compute_adaptive_budget(3, results, min_budget=2, max_budget=8)
    assert upd.new_budget == 2 and upd.rule == "pivot_set"  # clamped up to min_budget


def test_budget_manager_dispatch():
    # adaptive mode raises on zero-correct and persists.
    bm = BudgetManager(path=None, default_budget=8, min_budget=0, max_budget=8, ratchet_mode="adaptive")
    upd = bm.update_group("p1", [(False, 3)] * 8, current_budget=3)
    assert upd.new_budget == 4 and upd.rule == "raise" and bm.get("p1") == 4
    # downward mode (default) still uses the frugal-success rule.
    bm2 = BudgetManager(path=None, default_budget=8, ratchet_mode="downward")
    upd2 = bm2.update_group("p2", [(True, 1)] + [(False, 4)] * 7, current_budget=4)
    assert upd2.new_budget == 1 and upd2.rule == "min_frugal"


# --------------------------------------------------------------------------- #
# step-level advantage (step_advantage.py)
# --------------------------------------------------------------------------- #
def test_prefix_state_contiguous_and_robust():
    order = ["1.0", "1.1", "1.2", "2.0", "2.1"]
    assert prefix_state(order, []) == 0
    assert prefix_state(order, ["1.0", "1.1"]) == 2
    assert prefix_state(order, ["1.0", "1.1", "1.2", "2.0", "2.1"]) == 5
    # a stray out-of-order id beyond the gap is ignored (leading run only).
    assert prefix_state(order, ["1.0", "1.1", "2.1"]) == 2


# Worked 3-rollout example (devlog "Step level advantage calculation"). K=5, penalties
# p=[0.3,0.1,0.2,0.1,0.1]. Rollouts: A solves first try; B solves after a hint at step 0;
# C fails (reaches verified state 4), having failed steps 0, 2, 4.
_P5 = [0.3, 0.1, 0.2, 0.1, 0.1]
_FINAL = [5, 5, 4]
_FAILS = [[], [0], [0, 2, 4]]
# V solved backward:  V5=1; V4=1-0.1/3=29/30; V3=V4; V2=29/30-0.2/3=0.9; V1=V2; V0=0.9-0.6/3=0.7
_V_EXPECT = [0.7, 0.9, 0.9, 29 / 30, 29 / 30, 1.0]


def test_compute_state_values_worked_example():
    V = compute_state_values(_FINAL, _FAILS, _P5, 5, terminal_value=1.0)
    assert np.allclose(V, _V_EXPECT), V
    # non-decreasing toward the goal.
    assert all(V[k] <= V[k + 1] + 1e-12 for k in range(5))


def test_compute_state_values_all_full_budget():
    # every rollout fails every step -> V[k] = 1 - sum(p[k:]) (each transition's avg = r_k).
    fails = [[0, 1, 2, 3, 4]] * 4
    final = [5, 5, 5, 5]  # all eventually given every hint then "solve"
    V = compute_state_values(final, fails, _P5, 5, terminal_value=1.0)
    tail = [1 - sum(_P5[k:]) for k in range(5)] + [1.0]
    assert np.allclose(V, tail), (V, tail)


def test_apply_step_level_advantages_worked_example():
    # Token layout per rollout (response-relative). Non-segment tokens stay 0.
    L = 24
    adv = np.full((3, L), 9.9, dtype=np.float64)  # stale GRPO values to be overwritten
    ret = adv.copy()
    mask = np.ones((3, L), dtype=np.int64)
    uids = ["p", "p", "p"]
    turns = [
        [[0, 10, 10, 0, 5, 0]],                                  # A: solve, a_C[0,10)
        [[0, 0, 6, 0, 0, 1], [8, 18, 18, 1, 5, 0]],              # B: fail step0, then solve
        [[0, 0, 5, 0, 0, 1], [7, 9, 12, 1, 2, 1], [14, 16, 20, 3, 4, 1]],  # C
    ]
    pens = [_P5, _P5, _P5]
    Ks = [5, 5, 5]
    a, r, stats = apply_step_level_advantages(
        adv, ret, mask, uids, turns, [True, True, False], pens, Ks
    )
    V = _V_EXPECT
    # Row A: whole solve turn = V5 - V0 = 0.3
    assert np.allclose(a[0, 0:10], 0.3) and np.allclose(a[0, 10:], 0.0)
    # Row B: a_I[0,6) = r0 + V1 - V0 = -0.1 ; solve a_C[8,18) = V5 - V1 = 0.1 ; gap 0
    assert np.allclose(a[1, 0:6], -0.1)
    assert np.allclose(a[1, 6:8], 0.0)
    assert np.allclose(a[1, 8:18], 0.1)
    # Row C: a_I[0,5) = -0.1 ; a_C[7,9)=V2-V1=0 ; a_I[9,12)=r2+V3-V2=-0.2+1/15 ; a_C[14,16)=0 ; a_I[16,20)=r4+V5-V4=-0.1+1/30
    assert np.allclose(a[2, 0:5], -0.1)
    assert np.allclose(a[2, 7:9], 0.0)
    assert np.allclose(a[2, 9:12], -0.2 + (V[3] - V[2]))
    assert np.allclose(a[2, 14:16], 0.0)
    assert np.allclose(a[2, 16:20], -0.1 + (V[5] - V[4]))
    # returns mirror advantages.
    assert np.allclose(r, a)
    assert stats["step_adv/groups_scored"] == 1.0 and stats["step_adv/groups_zeroed"] == 0.0


def test_apply_step_level_advantages_adv_scale():
    # the worked example (V[0]=0.7 driven by others' failures) at scale 1 vs 5 -- row A's
    # solve advantage V5-V0=0.3 must scale exactly 5x.
    L = 12
    turns = [[[0, 6, 6, 0, 5, 0]], [[0, 0, 4, 0, 0, 1]], [[0, 0, 4, 0, 0, 1], [5, 7, 10, 1, 2, 1]]]
    correct = [True, False, False]
    a1 = np.zeros((3, L)); apply_step_level_advantages(a1, a1.copy(), np.ones((3, L)), ["p"]*3, turns, correct, [_P5]*3, [5]*3)
    a5 = np.zeros((3, L)); apply_step_level_advantages(a5, a5.copy(), np.ones((3, L)), ["p"]*3, turns, correct, [_P5]*3, [5]*3, adv_scale=5.0)
    assert np.allclose(a1[0, 0:6], 0.3), a1[0, 0:6]
    assert np.allclose(a5[0, 0:6], 1.5), a5[0, 0:6]  # exactly 5x


def test_normalize_brings_group_to_unit_std():
    # a group with spread; tokens tile each row (mask all-1) so np.std over the tensor ==
    # the group's trained-token std. normalize -> that std becomes adv_scale (1.0 default).
    L = 12
    turns = [
        [[0, 12, 12, 0, 5, 0]],                          # A correct: a_C 0.3
        [[0, 0, 6, 0, 0, 1], [6, 12, 12, 1, 5, 0]],      # B: a_I -0.1 ; a_C 0.1
        [[0, 0, 6, 0, 0, 1], [6, 6, 12, 1, 2, 1]],       # C: a_I -0.1 ; a_I -0.1333
    ]
    correct = [True, True, False]
    mask = np.ones((3, L))
    a_raw = np.zeros((3, L))
    apply_step_level_advantages(a_raw, a_raw.copy(), mask, ["p"] * 3, turns, correct, [_P5] * 3, [5] * 3)
    a_norm = np.zeros((3, L))
    _, _, st = apply_step_level_advantages(
        a_norm, a_norm.copy(), mask, ["p"] * 3, turns, correct, [_P5] * 3, [5] * 3, normalize=True
    )
    assert abs(float(a_norm.std()) - 1.0) < 1e-6, a_norm.std()   # unit std after norm
    assert float(a_raw.std()) < 0.3                              # raw was much smaller
    # signs preserved (no mean-centering): A's progress > 0, C's failed tail < 0.
    assert a_norm[0, 0] > 0.0 and a_norm[2, 0] < 0.0
    assert st["step_adv/group_std_mean"] > 0.0 and st["step_adv/norm_factor_mean"] > 1.0
    # adv_scale is the TARGET std under normalize.
    a3 = np.zeros((3, L))
    apply_step_level_advantages(
        a3, a3.copy(), mask, ["p"] * 3, turns, correct, [_P5] * 3, [5] * 3, normalize=True, adv_scale=3.0
    )
    assert abs(float(a3.std()) - 3.0) < 1e-6, a3.std()


def test_apply_step_level_advantages_all_incorrect_zeroed():
    L = 12
    adv = np.full((2, L), 0.5, dtype=np.float64)
    ret = adv.copy()
    mask = np.ones((2, L), dtype=np.int64)
    turns = [[[0, 0, 6, 0, 0, 1]], [[0, 0, 6, 0, 0, 1]]]
    a, r, stats = apply_step_level_advantages(
        adv, ret, mask, ["p", "p"], turns, [False, False], [_P5, _P5], [5, 5]
    )
    assert np.allclose(a, 0.0) and np.allclose(r, 0.0)
    assert stats["step_adv/groups_zeroed"] == 1.0 and stats["step_adv/groups_scored"] == 0.0


def test_apply_step_level_advantages_zero_if_no_correct_false():
    # with the flag off, an all-incorrect group is still scored (V[K] anchor unreached).
    L = 8
    adv = np.zeros((2, L), dtype=np.float64)
    ret = adv.copy()
    mask = np.ones((2, L), dtype=np.int64)
    turns = [[[0, 0, 4, 0, 0, 1]], [[0, 2, 6, 0, 1, 1]]]
    a, _, stats = apply_step_level_advantages(
        adv, ret, mask, ["p", "p"], turns, [False, False], [_P5, _P5], [5, 5],
        zero_if_no_correct=False,
    )
    assert stats["step_adv/groups_scored"] == 1.0
    # row0 failed step0 (a_I negative); not all-zero.
    assert a[0, 0] < 0.0


# --------------------------------------------------------------------------- #
# guidance-free (X.0 hint penalty -> 0) (hint_penalty.compute_hint_penalties)
# --------------------------------------------------------------------------- #
_PENALTY_POOL = [
    {"step_id": 1, "difficulty": "easy", "substeps": [
        {"substep_id": "1.1", "difficulty": "easy"},
        {"substep_id": "1.2", "difficulty": "hard"}]},
    {"step_id": 2, "difficulty": "hard", "substeps": [
        {"substep_id": "2.1", "difficulty": "moderate"}]},
]


def test_guidance_free_zeros_x0_and_preserves_total():
    base = compute_hint_penalties(_PENALTY_POOL, total_penalty=0.8, hard_factor=1.5)
    free = compute_hint_penalties(_PENALTY_POOL, total_penalty=0.8, hard_factor=1.5, guidance_free=True)
    # X.0 guidance hints are zeroed under the flag, priced normally without it.
    assert free["1.0"] == 0.0 and free["2.0"] == 0.0, free
    assert base["1.0"] > 0.0 and base["2.0"] > 0.0
    # the pool total is preserved in BOTH (cost redistributed, not dropped).
    assert abs(sum(base.values()) - 0.8) < 1e-9
    assert abs(sum(free.values()) - 0.8) < 1e-9
    # each step's hints still sum to that step's (level-1) penalty -> substeps absorbed
    # the guidance share exactly.
    assert abs((free["1.1"] + free["1.2"]) - (base["1.0"] + base["1.1"] + base["1.2"])) < 1e-9
    assert abs(free["2.1"] - (base["2.0"] + base["2.1"])) < 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
