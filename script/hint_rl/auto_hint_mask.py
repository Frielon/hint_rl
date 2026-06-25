# Copyright 2026
#
# Auto-hint verified-prefix gradient masking -- the pure tensor transform applied
# by HPRLRayPPOTrainer._update_actor when ``data.hprl.auto_hint.enable`` is set.
#
# The auto-hint rollout (auto_hint_agent_loop) records, per rollout, a list of
# ``disable_spans`` -- ``[start, end)`` ranges over the RESPONSE tokens that should
# NOT be trained WHEN the rollout's advantage is positive. The rule (per the
# design):
#
#   * advantage <= 0  -> train on ALL model-generated tokens (the spans are
#     ignored): a worse-than-average rollout is pushed away from in full.
#   * advantage  > 0  -> train ONLY up to each hinted turn's last selector-VERIFIED
#     sentence: every ``[start, end)`` span (the unverified tail of a turn that was
#     followed by a hint injection) is zeroed out of the loss mask, so a
#     better-than-average rollout reinforces only the reasoning the selector
#     confirmed, not the unverified continuation that the hint then corrected. The
#     ENDING turn is never in a span, so it is always fully promoted.
#
# This is split out as a dependency-light, side-effect-localized helper so it can
# be unit-tested (test_auto_hint.py) with plain numpy arrays -- it touches the
# tensors only through ``__getitem__`` / slice-assignment / ``.sum()``, which
# torch and numpy share.

from __future__ import annotations

from typing import Any, Sequence


def sequence_advantage(adv_row, mask_row) -> float:
    """Per-sequence (scalar) advantage = masked mean of the per-token advantage.

    GRPO broadcasts one scalar advantage across a sequence's response tokens and
    zeroes it on masked-out (observation / padding) tokens, so the masked mean
    recovers that scalar. Returns 0.0 for an all-masked row (no trained token).
    """
    denom = float(mask_row.sum())
    if denom <= 0.0:
        return 0.0
    return float((adv_row * mask_row).sum()) / denom


def _row_sequence_advantages(advantages, response_mask) -> list:
    """Per-row sequence advantage (masked mean) as a plain Python list.

    Vectorized so the whole batch costs TWO reductions + one device->host copy
    (``tolist``) -- NOT one sync per row. Duck-typed over torch (``dim=``) and numpy
    (``axis=``): the only API difference is the reduction kwarg.
    """
    try:  # torch
        denom = response_mask.sum(dim=1)
        num = (advantages * response_mask).sum(dim=1)
    except TypeError:  # numpy
        denom = response_mask.sum(axis=1)
        num = (advantages * response_mask).sum(axis=1)
    denom_l, num_l = denom.tolist(), num.tolist()
    return [(num_l[i] / denom_l[i]) if denom_l[i] > 0 else 0.0 for i in range(len(denom_l))]


def apply_positive_adv_masking(
    response_mask,
    advantages,
    disable_spans: Sequence[Any],
    *,
    positive_eps: float = 0.0,
) -> tuple[Any, dict]:
    """Zero ``response_mask`` over each row's disable-spans IFF its advantage > eps.

    Args:
        response_mask: (B, L) 0/1 loss mask (1 == trained token). Modified IN PLACE
            and returned -- this is verl's assistant-only response_mask, the tensor
            the policy loss aggregates over.
        advantages: (B, L) per-token advantage (GRPO scalar broadcast + masked).
        disable_spans: length-B sequence; ``disable_spans[i]`` is a list of
            ``[start, end)`` response-token ranges (or empty / None) to zero when
            row i's advantage is positive. Indices are response-relative (0 == the
            first response token), matching ``response_mask`` columns. Each span lies
            inside a single assistant turn (all-1 tokens), so its token count is
            ``end - start`` -- no per-span device sync needed to tally drops.
        positive_eps: a row counts as positive iff its sequence advantage strictly
            exceeds this (default 0.0 -> strictly positive). adv <= eps keeps the
            full mask, matching "if the advantage is negative, update on all tokens".

    Returns:
        (response_mask, stats) where ``stats`` are scalar diagnostics for logging:
        how many rows carried spans, how many were positive-and-masked, and how many
        mask tokens were dropped (so the effect is VISIBLE in metrics, not silent).
    """
    n_rows = int(response_mask.shape[0])
    seq_len = int(response_mask.shape[1])
    n_pos_masked = 0          # rows that were positive AND had >=1 span applied
    n_tokens_dropped = 0      # total 1->0 flips
    n_rows_with_spans = 0     # rows carrying >=1 span (positive or not)

    # per-row sign, computed once for the whole batch (no per-row device sync).
    seq_adv = _row_sequence_advantages(advantages, response_mask)

    for i in range(n_rows):
        spans = disable_spans[i] if disable_spans is not None else None
        if spans is None or len(spans) == 0:
            continue
        n_rows_with_spans += 1
        if seq_adv[i] <= positive_eps:
            continue  # not positive -> train on all tokens (spans ignored)
        applied = False
        for span in spans:
            s = max(0, int(span[0]))
            e = min(seq_len, int(span[1]))
            if e <= s:
                continue
            response_mask[i, s:e] = 0
            n_tokens_dropped += e - s  # span is all-1 assistant tokens -> exact
            applied = True
        if applied:
            n_pos_masked += 1

    stats = {
        "auto_hint/rows_with_spans": float(n_rows_with_spans),
        "auto_hint/rows_pos_masked": float(n_pos_masked),
        "auto_hint/mask_tokens_dropped": float(n_tokens_dropped),
    }
    return response_mask, stats
