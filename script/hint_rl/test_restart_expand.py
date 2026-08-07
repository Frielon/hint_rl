#!/usr/bin/env python3
# Copyright 2026
#
# Tests for the RESTART segment-row expansion ("every turn gets gradient"):
#   * fit_exclude_per_row -- excluded rows are painted from V but never enter the
#     value fit (numpy-only);
#   * expand_restart_segment_rows -- real-DataProto row materialization, padding,
#     old_log_probs anchoring (verl env; skips cleanly without it);
#   * RestartHintAgentLoop archives restart_segment_rows under
#     HPRL_RESTART_TRAIN_SEGMENTS=all (verl env + tokenizer).
#
# Run:  /share5/users/xutao.ma/miniconda3/envs/verl/bin/python test_restart_expand.py
from __future__ import annotations

import os


def test_fit_exclude_paints_but_does_not_fit():
    """A chain (final row w/ phantoms) + its expanded segment rows: V must equal
    the final-rows-only fit, and segment rows must be painted from that V."""
    import numpy as np

    from step_advantage import apply_step_level_advantages

    K, pvec = 2, [0.4, 0.3]
    # chains: A unhinted solve; B fail h0 -> fail h1, incorrect (2 archived... 1
    # archived segment [fail h0] + terminal fail h1 on the final row).
    final_rows = [
        [[0, 10, 10, 0, 2, 0]],                                # A: solve 0->K
        [[0, 0, 0, 0, 0, 1], [0, 0, 7, 1, 1, 1]],              # B: phantom(h0) + terminal fail h1
    ]
    seg_rows = [
        [[0, 0, 6, 0, 0, 1]],                                  # B's archived segment: failed h0 at s0
    ]
    turns = final_rows + seg_rows
    n = len(turns)
    adv = np.zeros((n, 12)); mask = np.zeros((n, 12))
    for i, L in enumerate((10, 7, 6)):
        mask[i, :L] = 1
    kwargs = dict(
        correct_per_row=[True, False, False],
        penalty_per_row=[pvec] * n, K_per_row=[K] * n,
        terminal_value=1.0, zero_if_no_correct=True, adv_scale=1.0, normalize=False,
        overlong_penalty=0.0, whole_turn=True,
    )
    gv = {}
    apply_step_level_advantages(adv, adv, mask, ["p"] * n, turns,
                                group_values_out=gv,
                                fit_exclude_per_row=[0, 0, 1], **kwargs)
    # V from FINAL rows only: D_0=2, D_1=2 (A reached K; B verified state 1);
    # F_0=1 (B's phantom), F_1=1 (B's terminal fail).
    V1 = 1.0 + 1 * (-0.3) / 2          # 0.85
    V0 = V1 + 1 * (-0.4) / 2           # 0.65
    assert all(abs(a - b) < 1e-9 for a, b in zip(gv["p"], [V0, V1, 1.0])), gv["p"]
    # the segment row IS painted: A = r0 + V[1] - V[0] over its 6 tokens.
    a_seg = -0.4 + V1 - V0
    assert abs(adv[2, 0] - a_seg) < 1e-9 and abs(adv[2, 5] - a_seg) < 1e-9
    assert np.all(adv[2, 6:] == 0.0)
    # sanity: WITHOUT exclusion the fit would double-count B's h0 failure
    adv2 = np.zeros((n, 12)); gv2 = {}
    apply_step_level_advantages(adv2, adv2, mask, ["p"] * n, turns,
                                group_values_out=gv2, fit_exclude_per_row=None, **kwargs)
    assert gv2["p"][0] < gv["p"][0] - 1e-9, "unexcluded segment row must lower V[0] (double count)"


def test_overlong_zeroed_group():
    """All-incorrect group: zero_if_no_correct zeroes it, but overlong_zeroed
    still paints the raw -P_over on each truncated row's tail (2026-08-04
    flipped-price finding: an 8/8 co-truncating all-wrong group previously got
    ZERO anti-truncation gradient)."""
    import numpy as np

    from step_advantage import apply_step_level_advantages

    K, pvec = 2, [0.4, 0.3]
    turns = [
        [[0, 0, 8, 0, 0, 1]],   # truncated at the cap (whole turn = tail)
        [[0, 0, 6, 0, 0, 1]],   # truncated
        [[0, 0, 5, 0, 0, 1]],   # clean wrong (not truncated)
    ]

    def run(ovz):
        adv = np.zeros((3, 8)); mask = np.zeros((3, 8))
        for i, L in enumerate((8, 6, 5)):
            mask[i, :L] = 1
        gv = {}
        _, _, st = apply_step_level_advantages(
            adv, adv, mask, ["p"] * 3, turns,
            correct_per_row=[False] * 3, penalty_per_row=[pvec] * 3, K_per_row=[K] * 3,
            terminal_value=1.0, zero_if_no_correct=True, adv_scale=1.0, normalize=False,
            overlong_penalty=0.1, overlong_penalty_type="value",
            turn_truncated_per_row=[1, 1, 0], whole_turn=True,
            group_values_out=gv, overlong_zeroed=ovz,
        )
        return adv, gv, st

    adv, gv, st = run(True)
    assert gv["p"] is None, "group must STAY zeroed (no value fit)"
    assert np.allclose(adv[0, :8], -0.1) and np.allclose(adv[1, :6], -0.1)
    assert np.all(adv[1, 6:] == 0.0) and np.all(adv[2] == 0.0), "clean-wrong row untouched"
    assert st["step_adv/groups_zeroed"] == 1 and st["step_adv/overlong_rows"] == 2
    adv0, _, st0 = run(False)  # legacy scored-groups-only behavior
    assert np.all(adv0 == 0.0) and st0["step_adv/overlong_rows"] == 0


def _verl_env():
    try:
        import torch  # noqa: F401
        from verl.protocol import DataProto  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] verl env unavailable: {e}")
        return False


def test_expand_rows_and_padding():
    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.protocol import DataProto

    from restart_expand import expand_restart_segment_rows

    P, R = 8, 6
    n = 2
    bb = TensorDict({
        "prompts": torch.arange(n * P).reshape(n, P),
        "responses": torch.arange(n * R).reshape(n, R),
        "input_ids": torch.zeros(n, P + R, dtype=torch.long),
        "attention_mask": torch.ones(n, P + R, dtype=torch.long),
        "position_ids": torch.zeros(n, P + R, dtype=torch.long),
        "response_mask": torch.ones(n, R, dtype=torch.long),
        "rollout_log_probs": torch.zeros(n, R),
        "old_log_probs": torch.zeros(n, R),
        "advantages": torch.zeros(n, R),
        "returns": torch.zeros(n, R),
        "token_level_scores": torch.zeros(n, R),
        "token_level_rewards": torch.zeros(n, R),
    }, batch_size=[n])
    seg = {"prompt_ids": [7, 8, 9], "response_ids": [11, 12, 13, 14],
           "logprobs": [-0.5, -0.25, -0.125, -0.0625],
           "turns": [[0, 0, 4, 0, 0, 1]]}
    ntb = {
        "uid": np.array(["pA", "pB"], dtype=object),
        "acc": np.array([1.0, 0.0], dtype=object),
        "step_adv_turns": np.array([[[0, 5, 5, 0, 2, 0]], [[0, 0, 6, 0, 0, 1]]], dtype=object),
        "turn_truncated": np.array([0, 0], dtype=object),
        "restart_segment_rows": np.empty(n, dtype=object),
    }
    ntb["restart_segment_rows"][0] = [seg]
    ntb["restart_segment_rows"][1] = []
    batch = DataProto(batch=bb, non_tensor_batch=ntb, meta_info={"m": 1})

    out, n_orig, stats = expand_restart_segment_rows(batch, pad_multiple=2, pad_token_id=0)
    assert n_orig == 2 and len(out) == 4, (n_orig, len(out), stats)  # 2 + 1 seg + 1 pad
    assert stats["restart_expand/rows_segments"] == 1 and stats["restart_expand/rows_pad"] == 1

    # original batch untouched
    assert len(batch) == 2 and "restart_expanded" not in batch.non_tensor_batch

    eb, entb = out.batch, out.non_tensor_batch
    # segment row (index 2): prompt LEFT-padded, response RIGHT-padded
    assert eb["prompts"][2].tolist() == [0] * (P - 3) + [7, 8, 9]
    assert eb["responses"][2].tolist() == [11, 12, 13, 14, 0, 0]
    assert eb["response_mask"][2].tolist() == [1, 1, 1, 1, 0, 0]
    assert eb["attention_mask"][2].tolist() == [0] * (P - 3) + [1] * 3 + [1] * 4 + [0] * 2
    # position ids increase over the real span
    real = eb["position_ids"][2][P - 3: P + 4].tolist()
    assert real == sorted(real) and len(set(real)) == len(real)
    # old_log_probs anchored to the archived sampling logprobs (bypass semantics)
    assert torch.allclose(eb["old_log_probs"][2][:4], torch.tensor([-0.5, -0.25, -0.125, -0.0625]))
    assert torch.allclose(eb["rollout_log_probs"][2][:4], eb["old_log_probs"][2][:4])
    # non-tensors: uid inherited from parent; record overridden; marker set
    assert entb["uid"][2] == "pA" and entb["restart_expanded"][2] == 1
    assert entb["step_adv_turns"][2] == [[0, 0, 4, 0, 0, 1]]
    assert entb["turn_truncated"][2] == 0
    # pad row (index 3): fully untrained clone
    assert entb["restart_expanded"][3] == 2
    assert int(eb["response_mask"][3].sum()) == 0
    assert float(eb["advantages"][3].abs().sum()) == 0.0
    # originals keep marker 0
    assert entb["restart_expanded"][0] == 0 and entb["restart_expanded"][1] == 0

    # trainer splice: paint a fake advantage on the segment row and read it back
    # onto the ORIGINAL batch as restart_segment_advs (dump/viewer audit path).
    from hprl_ray_trainer import HPRLRayPPOTrainer

    eb["advantages"][2, :4] = -0.15
    HPRLRayPPOTrainer._hprl_attach_segment_advs(None, batch, out, n_orig)
    col = batch.non_tensor_batch["restart_segment_advs"]
    (pair,) = col[0]                                   # chain A's one archived segment
    assert abs(pair[0] + 0.15) < 1e-6 and abs(pair[1] + 0.15) < 1e-6, col[0]
    assert list(col[1]) == [], col[1]                  # chain B had none


def test_loop_archives_segment_rows():
    """HPRL_RESTART_TRAIN_SEGMENTS=all -> the loop archives per-segment tensors."""
    os.environ["HPRL_RESTART_TRAIN_SEGMENTS"] = "all"
    try:
        import test_restart_chain as tc
        fix = tc._verl_fixtures()
        if fix is None:
            print("  [skip] no verl fixtures")
            return
        tokenizer, make_loop = fix
        texts = [
            "Count multiples of 2: 50. I think the answer is \\boxed{50}.",
            "Inclusion-exclusion: 50 + 20 - 10 = 60. \\boxed{60}.",
        ]
        server = tc._ScriptedServer(tokenizer, texts)
        loop_obj = make_loop(server, step_adv=True)
        loop_obj._selector = tc._ScriptedSelector([
            {"hint_id": "1.1", "hint": "Use inclusion-exclusion.", "completed_hints": []},
        ])
        out = tc._run(loop_obj, raw_prompt=tc._RAW_PROMPT, tools_kwargs=tc._tools_kwargs(4), extra_info={})
        rows = out.extra_fields["restart_segment_rows"]
        assert len(rows) == 1, rows
        seg0_ids = tokenizer.encode(texts[0], add_special_tokens=False)
        assert rows[0]["response_ids"] == seg0_ids
        assert rows[0]["logprobs"] == [-0.1] * len(seg0_ids)  # scripted server logprobs
        assert rows[0]["turns"] == [[0, 0, len(seg0_ids), 0, 0, 1]]
        assert len(rows[0]["prompt_ids"]) > 0
        # the final row's own metadata is unchanged by the archiving
        assert out.extra_fields["step_adv_turns"][0] == [0, 0, 0, 0, 0, 1]  # phantom
    finally:
        os.environ.pop("HPRL_RESTART_TRAIN_SEGMENTS", None)


def main():
    test_fit_exclude_paints_but_does_not_fit()
    print("PASS test_fit_exclude_paints_but_does_not_fit")
    test_overlong_zeroed_group()
    print("PASS test_overlong_zeroed_group")
    if not _verl_env():
        print("1 test passed; verl-env tests SKIPPED")
        return
    test_expand_rows_and_padding()
    print("PASS test_expand_rows_and_padding")
    test_loop_archives_segment_rows()
    print("PASS test_loop_archives_segment_rows")
    print("all 3 tests passed")


if __name__ == "__main__":
    main()
