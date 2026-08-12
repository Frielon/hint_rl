# Copyright 2026
#
# RESTART segment-row expansion -- "every turn gets gradient" (devlog 2026-08-04).
#
# Under HPRL_RESTART_TRAIN_SEGMENTS=all, RestartHintAgentLoop archives each
# terminated segment's exact tensors (prompt ids, response ids, sampling
# logprobs, its step-adv record) in ``non_tensor_batch["restart_segment_rows"]``.
# This module materializes them as REAL training rows so a chain with hints
# h_a, h_b, h_c injected trains all four turns:
#
#     turn_i (failed):  A_i = r(h_fail_i) + V[se_i + 1] - V[ss_i]
#     final turn:       A   = V[se] - V[ss]  (solve)  /  fail form (terminal)
#
# each painted on its own row's response tokens, all sharing the group's V.
# The FINAL rows keep their zero-span phantom records, so the value fit stays
# chain-exact when computed from final rows ONLY -- expanded segment rows are
# marked ``restart_expanded`` and must be passed to
# ``apply_step_level_advantages(fit_exclude_per_row=...)`` (double-counting F_k
# otherwise). Chain-level reward/ratchet/dump semantics are untouched: the
# expansion produces a COPY for the actor update; the caller keeps the original
# batch for everything else (hprl_ray_trainer._update_actor).
#
# Row plumbing (mirrors the agent-loop postprocess):
#   * prompt left-padded to prompt_len, response right-padded to resp_len;
#   * attention_mask from real lengths; position_ids = clamp(cumsum-1, 0);
#   * response_mask = 1 on real response tokens (fully model-generated), EXCEPT
#     the first ``prefix_len`` tokens of a reference-prefix segment (the glued
#     reference-solution text the model merely continued: attended, untrained --
#     restart_agent_loop archives the full region + its prefix_len);
#   * rollout_log_probs from the archived sampling logprobs, and old_log_probs
#     set EQUAL to them -- the behavior-policy anchor. In bypass-mode async that
#     matches every other row's anchor exactly; in sync / decoupled runs the
#     final rows keep their trainer-recomputed anchors while segment rows use
#     the sampling ones (a recompute has already happened by _update_actor
#     time), i.e. mixed numerics with the known vllm-vs-fsdp gap -- the same
#     anchor quality bypass mode runs with on ALL rows. The caller guards on
#     rollout_log_probs being present (rollout.calculate_log_probs=true).
#   * every other tensor key is zero-filled (token_level_scores/rewards,
#     advantages, returns -- step-adv overwrites the latter two for all rows);
#     unknown extra keys are zero-filled too, with a warning.
#   * non-tensors are copied from the parent chain row (uid included -- the
#     segment row joins its problem's group for painting/normalize), with
#     ``step_adv_turns`` := the segment's own record, ``turn_truncated`` := 0,
#     ``restart_expanded`` := 1 (0 on originals).
#
# DIVISIBILITY: verl's train path asserts rows % mini-batch == 0 (tensordict
# make_iterator; no autopad). Variable segment counts break that, so the
# expanded batch is padded to ``pad_multiple`` with INERT rows: a duplicate of
# the shortest real row with response_mask zeroed everywhere and advantages 0 --
# zero trained tokens, so token-mean losses are undiluted; cost is a forward
# pass over the duplicate. Pads carry ``restart_expanded`` = 2.

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# tensor keys we know how to fill for a synthesized row; anything else gets
# zeros_like (warned once).
_EXPECTED_ZERO_KEYS = {
    "token_level_scores", "token_level_rewards", "advantages", "returns",
}


def _to_py(v):
    if v is None:
        return None
    if hasattr(v, "item") and getattr(v, "shape", None) == ():
        return v.item()
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:  # noqa: BLE001
            return v
    return v


def _position_ids_like(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)


def expand_restart_segment_rows(batch, pad_multiple: int, pad_token_id: int = 0):
    """Return ``(expanded_batch, n_original, stats)`` -- a COPY of ``batch`` with
    one appended row per archived restart segment, padded with inert rows to a
    multiple of ``pad_multiple``. Returns ``(batch, len(batch), {})`` unchanged
    when no row carries segments.

    The caller owns the guards (bypass mode, step_adv on) and must route
    ``non_tensor_batch["restart_expanded"]`` into
    ``apply_step_level_advantages(fit_exclude_per_row=...)``.
    """
    ntb = batch.non_tensor_batch
    n = len(batch)
    seg_col = ntb.get("restart_segment_rows")
    if seg_col is None:
        return batch, n, {}
    per_row = [(_to_py(seg_col[i]) or []) for i in range(n)]
    total_segs = sum(len(s) for s in per_row)
    if total_segs == 0:
        return batch, n, {}

    bb = batch.batch
    prompts_t = bb["prompts"]
    resp_t = bb["responses"]
    prompt_len = int(prompts_t.shape[1])
    resp_len = int(resp_t.shape[1])
    dev = prompts_t.device

    warned_keys = set()

    def _make_row(parent_i: int, prompt_ids, response_ids, logprobs, turn, prefix_len=0):
        """One synthesized row's tensor dict + non-tensor overrides."""
        p = list(prompt_ids or [])[-prompt_len:]
        r = list(response_ids or [])[:resp_len]
        lp = list(logprobs or [])[:resp_len]
        if len(lp) < len(r):  # missing/short logprobs: pad with 0.0 (neutral ratio)
            lp = lp + [0.0] * (len(r) - len(lp))
        prow = torch.full((prompt_len,), pad_token_id, dtype=prompts_t.dtype, device=dev)
        if p:
            prow[-len(p):] = torch.tensor(p, dtype=prompts_t.dtype, device=dev)
        rrow = torch.full((resp_len,), pad_token_id, dtype=resp_t.dtype, device=dev)
        if r:
            rrow[: len(r)] = torch.tensor(r, dtype=resp_t.dtype, device=dev)
        attn = torch.zeros(prompt_len + resp_len, dtype=bb["attention_mask"].dtype, device=dev)
        attn[prompt_len - len(p): prompt_len + len(r)] = 1
        rmask = torch.zeros(resp_len, dtype=bb["response_mask"].dtype, device=dev)
        rmask[: len(r)] = 1
        # reference-prefix segment: the leading prefix_len response tokens are the
        # glued reference text -- real context (attention stays 1) but untrained
        # (loss mask 0), exactly the live final-row treatment; the segment's turn
        # record already starts at prefix_len, so nothing is painted there either.
        plen = max(0, int(prefix_len or 0))
        if plen:
            rmask[: min(plen, len(r))] = 0
        lprow = torch.zeros(resp_len, dtype=torch.float32, device=dev)
        if r:
            lprow[: len(r)] = torch.tensor(lp[: len(r)], dtype=torch.float32, device=dev)
        row = {
            "prompts": prow,
            "responses": rrow,
            "input_ids": torch.cat([prow, rrow]),
            "attention_mask": attn,
            "position_ids": _position_ids_like(attn),
            "response_mask": rmask,
        }
        if "rollout_log_probs" in bb.keys():
            row["rollout_log_probs"] = lprow
        if "old_log_probs" in bb.keys():
            # bypass mode: the proximal anchor IS the behavior policy's logprobs.
            row["old_log_probs"] = lprow.clone()
        for key in bb.keys():
            if key in row:
                continue
            if key not in _EXPECTED_ZERO_KEYS and key not in warned_keys:
                warned_keys.add(key)
                logger.warning("restart_expand: zero-filling unexpected batch key %r for segment rows", key)
            row[key] = torch.zeros_like(bb[key][parent_i])
        overrides = {
            "step_adv_turns": [list(t) for t in (turn or [])],
            "turn_truncated": 0,
            "restart_expanded": 1,
        }
        return row, overrides

    new_rows = []
    new_overrides = []
    parent_of = []
    for i in range(n):
        for seg in per_row[i]:
            if not isinstance(seg, dict):
                continue
            row, ov = _make_row(
                i, seg.get("prompt_ids"), seg.get("response_ids"),
                seg.get("logprobs"), seg.get("turns"), seg.get("prefix_len"),
            )
            new_rows.append(row)
            new_overrides.append(ov)
            parent_of.append(i)

    # inert pads to the divisibility multiple: shortest real row, fully untrained.
    n_total = n + len(new_rows)
    n_pad = (-n_total) % max(int(pad_multiple), 1)
    if n_pad:
        lens = bb["attention_mask"].sum(dim=-1)
        short_i = int(torch.argmin(lens).item())
        for _ in range(n_pad):
            row = {k: bb[k][short_i].clone() for k in bb.keys()}
            row["response_mask"] = torch.zeros_like(row["response_mask"])
            for k in ("advantages", "returns", "token_level_scores", "token_level_rewards"):
                if k in row:
                    row[k] = torch.zeros_like(row[k])
            new_rows.append(row)
            new_overrides.append({"step_adv_turns": [], "turn_truncated": 0, "restart_expanded": 2})
            parent_of.append(short_i)

    # assemble the expanded COPY: tensors stacked, non-tensors replicated + overridden.
    expanded = batch.select_idxs(np.concatenate([np.arange(n), np.array(parent_of, dtype=np.int64)])) \
        if hasattr(batch, "select_idxs") else None
    assert expanded is not None, "DataProto.select_idxs unavailable"
    ebb = expanded.batch
    for k in ebb.keys():
        stacked = torch.stack([row[k] for row in new_rows], dim=0) if new_rows else None
        if stacked is not None:
            ebb[k][n:] = stacked.to(ebb[k].dtype)
    entb = expanded.non_tensor_batch

    def _as_1d_object(col):
        """Per-element assignment needs a 1-D object column. numpy silently builds
        a MULTI-dim object array when the rows' nested lists happen to share a
        shape (e.g. every row carrying exactly one 6-element step_adv record --
        common in restart batches), and element assignment then broadcasts.
        Rebuild such columns as 1-D object arrays of python lists."""
        if getattr(col, "dtype", None) == object and getattr(col, "ndim", None) == 1:
            return col
        m = len(col)
        out = np.empty(m, dtype=object)
        out[:] = [col[i].tolist() if hasattr(col[i], "tolist") else col[i] for i in range(m)]
        return out

    marker = np.zeros(len(expanded), dtype=object)
    for j, ov in enumerate(new_overrides):
        marker[n + j] = ov["restart_expanded"]
    entb["restart_expanded"] = marker
    override_keys = {k for ov in new_overrides for k in ov} - {"restart_expanded"}
    for key in override_keys:
        if key in entb:
            entb[key] = _as_1d_object(entb[key])
    for j, ov in enumerate(new_overrides):
        for key, val in ov.items():
            if key == "restart_expanded":
                continue
            if key in entb:
                entb[key][n + j] = val
    # expanded rows must not re-expand or re-archive
    if "restart_segment_rows" in entb:
        entb["restart_segment_rows"] = _as_1d_object(entb["restart_segment_rows"])
        for j in range(len(new_rows)):
            entb["restart_segment_rows"][n + j] = []

    stats = {
        "restart_expand/rows_original": float(n),
        "restart_expand/rows_segments": float(total_segs),
        "restart_expand/rows_pad": float(n_pad),
        "restart_expand/rows_total": float(len(expanded)),
    }
    return expanded, n, stats
