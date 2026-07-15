# Copyright 2026
#
# HPRLRayPPOTrainer -- RayPPOTrainer + the HPRL dynamic-budget ratchet.
#
# This is the ONLY trainer-side change for HPRL, and it is a strict override:
# every HPRL action is gated on ``data.hprl.enable``, so with the flag off this
# class behaves byte-for-byte like the stock RayPPOTrainer (it just calls super).
#
# verl exposes no post-reward / end-of-step callback, so we hook ``_update_actor``
# -- the per-step driver method that, for GRPO, always runs and receives the
# fully populated post-reward batch (problem_id in ``extra_info``; ``acc`` /
# ``num_hints`` in the merged reward keys). After the real actor update we fold
# the step's rollouts through the per-problem downward ratchet
# (hint_budget_callback.hprl_update_budgets) and surface its scalar metrics via
# ``actor_output.meta_info["metrics"]`` so they ride the normal logging path.
#
# Wired in by main_hprl.HPRLTaskRunner (which swaps this in for RayPPOTrainer).

from __future__ import annotations

import logging
import os

import numpy as np
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from auto_hint_mask import apply_positive_adv_masking
from budget_manager import BudgetManager, get_create_budget
from hint_budget_callback import _gen_budget_from_extra, _to_py, hprl_update_budgets
from hint_penalty import (
    DEFAULT_GUIDANCE_DIFFICULTY,
    DEFAULT_HARD_FACTOR,
    DEFAULT_TOTAL_PENALTY,
    compute_hint_penalties,
)
from kpack_expand import render_variant_rows
from selector_multi import pending_hint_ids
from step_advantage import apply_step_level_advantages


def _truthy(v) -> bool:
    """Coerce a yaml bool / hydra-string flag (``"true"``/``"false"``) to a bool."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HPRLRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that ratchets per-problem hint budgets after each step."""

    def _hprl_cfg(self):
        # the flag lives at data.hprl.* (the dataset only sees config.data, so we
        # keep a single source of truth both sides can read).
        return self.config.data.get("hprl", {}) or {}

    def _auto_hint_cfg(self):
        # data.hprl.auto_hint.* -- the push-hint mode's knobs (enable +
        # verified-prefix mask). Independent of the ratchet flag (data.hprl.enable).
        return self._hprl_cfg().get("auto_hint", {}) or {}

    def _hprl_budget_mgr(self) -> BudgetManager:
        """Lazily build the driver-side BudgetManager (loads existing state on
        construction, so it resumes across restarts)."""
        if getattr(self, "_budget_mgr", None) is None:
            cfg = self._hprl_cfg()
            path = cfg.get("budget_state_path", None)
            kcfg = cfg.get("kpack", {}) or {}
            default_budget = int(cfg.get("default_budget", 8))
            self._budget_mgr = BudgetManager(
                path=path,
                default_budget=default_budget,
                min_budget=int(cfg.get("min_budget", 0)),
                decrement=int(cfg.get("decrement", 1)),
                kpack_require_successes=int(kcfg.get("require_successes", 2)),
                # adaptive ratchet (raise on zero-correct, N/2-set on majority) + its ceiling.
                ratchet_mode=str(cfg.get("ratchet_mode", "downward")),
                max_budget=int(cfg.get("max_budget", default_budget)),
                # False -> never LOWER B_q: a would-be decrease is held at the current
                # budget (rule "*_held"); the adaptive raise branch still applies.
                allow_decrease=bool(cfg.get("allow_decrease", True)),
            )
            logger.warning(
                "HPRL ratchet enabled: budget_state=%s mode=%s allow_decrease=%s default=%d min=%d max=%d dec=%d "
                "(loaded %d problems)",
                path,
                self._budget_mgr.ratchet_mode,
                self._budget_mgr.allow_decrease,
                self._budget_mgr.default_budget,
                self._budget_mgr.min_budget,
                self._budget_mgr.max_budget,
                self._budget_mgr.decrement,
                len(self._budget_mgr),
            )
        return self._budget_mgr

    # ------------------------------------------------------------------ #
    # k-pack split: config side (divide the repeat factor so total is unchanged)
    # ------------------------------------------------------------------ #
    def _hprl_apply_kpack_split_config(self) -> None:
        """Divide the rollout repeat factor by k so each problem's n rollouts split into
        k packs of n/k -- WITHOUT changing the per-step total.

        verl repeats every prompt row by ``rollout.n`` and groups by uid. To turn one
        problem's ``rollout.n`` rollouts into k GRPO groups of ``rollout.n/k`` (one per
        budget level), we expand the batch x k rows (``_hprl_expand_kpacks``) and set the
        repeat factor to ``rollout.n/k``. Net rollouts == ``train_batch_size * rollout.n``
        (the ORIGINAL n) -- unchanged; only the grouping/budget split changes.

        Done ONCE, before super().fit(). Also scales ``ppo_mini_batch_size`` by k (default)
        so the PPO update's mini-batch SAMPLE count (``ppo_mini_batch_size * rollout.n``)
        is identical to a non-k-pack run -- the update is byte-for-byte unchanged; only
        the advantage grouping differs. Raises if ``rollout.n`` is not divisible by k.

        Safe to call when k-pack is off: it no-ops. The rollout engine does NOT read
        ``rollout.n`` (agent-loop generates one request per repeated row) and GRPO ignores
        ``num_repeat`` (it groups by uid), so dividing it only affects the repeat + the
        mini-batch arithmetic. Validation uses ``val_kwargs.n`` and is unaffected.
        """
        if getattr(self, "_kpack_split_applied", False):
            return
        self._kpack_split_applied = True
        cfg = self._hprl_cfg()
        kcfg = cfg.get("kpack", {}) or {}
        if not (cfg.get("enable", False) and kcfg.get("enable", False)):
            return
        k = int(kcfg.get("k", 2))
        self._kpack_k = k
        if k < 2:
            return

        n = int(self.config.actor_rollout_ref.rollout.n)
        if n % k != 0:
            raise ValueError(
                f"k-pack split requires rollout.n ({n}) to be divisible by kpack.k ({k}); "
                f"set actor_rollout_ref.rollout.n to a multiple of {k}."
            )
        pack = n // k
        scale_mb = bool(kcfg.get("scale_mini_batch", True))
        old_mb = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)

        readonly = OmegaConf.is_readonly(self.config)
        if readonly:
            OmegaConf.set_readonly(self.config, False)
        try:
            self.config.actor_rollout_ref.rollout.n = pack
            if scale_mb:
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size = old_mb * k
        finally:
            if readonly:
                OmegaConf.set_readonly(self.config, True)

        if pack < 2:
            logger.warning(
                "HPRL k-pack: pack size n/k = %d < 2 -> GRPO groups are singletons (no within-group "
                "std/baseline). Increase rollout.n or lower k.",
                pack,
            )
        logger.warning(
            "HPRL k-pack split: rollout.n %d -> %d (k=%d packs of %d each, total per-step rollouts "
            "unchanged); ppo_mini_batch_size %d -> %d%s",
            n, pack, k, pack, old_mb, (old_mb * k if scale_mb else old_mb),
            "" if scale_mb else " (scale_mini_batch=false)",
        )

    def fit(self):
        # divide the repeat factor by k once, before the (unmodified) base training loop.
        # Loud on a bad rollout.n%k (the user asked for this check); a no-op when off.
        if self._hprl_cfg().get("enable", False):
            self._hprl_apply_kpack_split_config()
        return super().fit()

    # ------------------------------------------------------------------ #
    # k-pack split: generation side (expand every problem into k budget packs)
    # ------------------------------------------------------------------ #
    def _get_gen_batch(self, batch):
        """Split every problem into k budget packs before generation (train only).

        For EVERY problem at budget B, emit k-1 extra rows at budgets B-1 .. B-k+1
        (clamped to min_budget); the dataset's original row is pack 0 (budget B). Each
        pack is its own GRPO group (a fresh uid -> verl groups by uid), so the ratchet
        pools the k packs and reads true need from the FORCED low-budget packs instead of
        the corrupted full-budget signal (budget_manager.compute_kpack_budget).

        The batch grows N -> N*k here; the repeat factor was divided by k in
        ``_hprl_apply_kpack_split_config``, so after the repeat the total is N*k*(n/k) ==
        N*n -- the per-step rollout count is UNCHANGED (== train_batch_size * the ORIGINAL
        rollout.n), just split into k packs of n/k per problem.

        Validation calls _get_gen_batch on a batch with NO ``uid`` (uid is assigned only
        on the training path), so the ``uid`` guard makes this a no-op for validation.
        """
        cfg = self._hprl_cfg()
        kcfg = cfg.get("kpack", {}) or {}
        if (
            cfg.get("enable", False)
            and kcfg.get("enable", False)
            and int(kcfg.get("k", 2)) >= 2
            and isinstance(getattr(batch, "non_tensor_batch", None), dict)
            and "uid" in batch.non_tensor_batch
        ):
            try:
                self._hprl_expand_kpacks(batch, kcfg)
            except Exception as e:  # splitting must never crash training
                logger.warning("HPRL k-pack split failed at step %s: %s", self.global_steps, e)
                self._hprl_probe_stats = {}
        return super()._get_gen_batch(batch)

    def _hprl_expand_kpacks(self, batch, kcfg) -> None:
        """In-place: split every problem into k budget packs (grow the batch N -> N*k).

        Every dataset row (one pack at budget B) gains k-1 variant rows at budgets
        B-1..B-k+1 (clamped to min_budget), each a fresh GRPO group (own uid, re-rendered
        prompt + tool budget). Deep-copies per variant so the original rows -- kept as
        pack 0 -- are untouched. The x k growth is matched by the x(1/k) repeat factor set
        in _hprl_apply_kpack_split_config, so the per-step rollout total is unchanged.
        """
        self._hprl_probe_stats = {}
        ntb = batch.non_tensor_batch
        n_rows = len(batch)
        if n_rows == 0 or "extra_info" not in ntb:
            return
        k = int(getattr(self, "_kpack_k", kcfg.get("k", 2)))
        if k < 2:
            return
        min_budget = int(self._hprl_cfg().get("min_budget", 0))
        tool_name = self._hprl_cfg().get("tool_name", "request_hint")
        bm = self._hprl_budget_mgr()
        extra_arr = ntb["extra_info"]
        tk_arr = ntb.get("tools_kwargs")

        def _row_budget(i: int):
            # the budget THIS problem's pack-0 runs under == what the dataset injected.
            if tk_arr is not None:
                return get_create_budget(_to_py(tk_arr[i]), bm.default_budget, tool_name)
            ei = _to_py(extra_arr[i]) or {}
            return get_create_budget(ei.get("tools_kwargs"), bm.default_budget, tool_name)

        # k-1 variant rows per problem at clamp(B-1)..clamp(B-k+1); original row = pack 0.
        # Every problem gets exactly k-1 variants so the batch is uniformly N*k (required:
        # the repeat factor n/k is applied per row, so a non-uniform count would mis-size
        # those problems' rollouts). Train rows are all HPRL; a clamp collapses redundant
        # sub-budgets (e.g. B already at the floor) to repeated min_budget packs, harmless.
        var_src, var_budget = [], []
        for i in range(n_rows):
            B = _row_budget(i)
            for j in range(1, k):
                var_src.append(i)
                var_budget.append(max(min_budget, B - j))

        variants = batch.select_idxs(np.array(var_src, dtype=np.int64))
        render_variant_rows(variants.non_tensor_batch, var_budget, tool_name=tool_name, to_py=_to_py)

        expanded = DataProto.concat([batch, variants])
        batch.batch = expanded.batch
        batch.non_tensor_batch = expanded.non_tensor_batch
        # invariant: every problem now has exactly k packs (uniform x k growth).
        assert len(batch) == n_rows * k, (len(batch), n_rows, k)

        self._hprl_probe_stats = {
            "hprl/kpack_problems": float(n_rows),
            "hprl/kpack_rows_out": float(len(batch)),
            "hprl/kpack_packs_per_problem": float(k),
        }
        logger.warning(
            "[HPRL step=%s] k-pack split: %d problems -> %d rows (k=%d packs of n/k each)",
            self.global_steps,
            n_rows,
            len(batch),
            k,
        )

    def _update_actor(self, batch):
        # AUTO-HINT verified-prefix mask: BEFORE the actor update, zero the loss mask
        # (response_mask) over each rollout's recorded disable_spans -- but only for
        # POSITIVE-advantage rollouts (auto_hint_mask). Done here because by now the
        # batch carries both the GRPO advantages and the assistant-only response_mask
        # the policy loss aggregates over; modifying response_mask propagates into
        # every micro-batch. No-op (returns {}) unless data.hprl.auto_hint.enable.
        auto_hint_stats = {}
        ac = self._auto_hint_cfg()
        if ac.get("enable", False):
            # STEP-LEVEL value-based advantages (step_advantage) SUPERSEDE the
            # verified-prefix loss MASK (auto_hint_mask) when enabled -- they handle the
            # unverified tail by giving it a negative advantage instead of dropping it,
            # so the two are mutually exclusive. Default: mask (step_adv off).
            step_adv_on = _truthy((ac.get("step_adv", {}) or {}).get("enable", False))
            try:
                if step_adv_on:
                    auto_hint_stats = self._hprl_apply_step_advantage(batch)
                else:
                    auto_hint_stats = self._hprl_apply_auto_hint_mask(batch)
            except Exception as e:  # adv/mask edits must never crash training
                logger.warning("HPRL auto-hint adv/mask failed at step %s: %s", self.global_steps, e)

        actor_output = super()._update_actor(batch)

        metrics = {}
        if self._hprl_cfg().get("enable", False):
            try:
                metrics = hprl_update_budgets(
                    batch,
                    self._hprl_budget_mgr(),
                    global_step=self.global_steps,
                    kpack_cfg=(self._hprl_cfg().get("kpack", {}) or {}),
                )
                # fold in the generation-side probe summary (set by _hprl_expand_kpacks).
                metrics = {**metrics, **getattr(self, "_hprl_probe_stats", {})}
            except Exception as e:  # never let budget bookkeeping crash training
                logger.warning("HPRL budget ratchet failed at step %s: %s", self.global_steps, e)
        # surface the auto-hint mask stats alongside the ratchet metrics (independent
        # of the ratchet flag) so the masking effect is visible in wandb.
        metrics = {**metrics, **auto_hint_stats}
        if metrics:
            try:
                md = actor_output.meta_info.setdefault("metrics", {})
                md.update(metrics)  # scalars survive reduce_metrics
            except Exception as e:  # noqa: BLE001
                logger.warning("HPRL metric folding failed at step %s: %s", self.global_steps, e)
        return actor_output

    def _hprl_apply_auto_hint_mask(self, batch) -> dict:
        """Zero response_mask over the auto-hint disable_spans for positive-advantage
        rollouts (the verified-prefix gradient mask). Returns scalar stats for wandb.

        Reads the per-rollout ``disable_spans`` recorded by AutoHintAgentLoop from
        non_tensor_batch (extra_fields flatten there), the GRPO ``advantages`` and the
        assistant-only ``response_mask`` from batch.batch. Each disable span is an
        ``[start, end)`` response-token range; it is dropped from the loss mask ONLY
        when the rollout's (masked-mean) advantage is positive -- a negative-advantage
        rollout trains on all of its tokens. Also folds in the CITATION FOUND RATE
        (selector completed_hints quotes that fuzzy-located in the trace). Returns the
        cite stats alone (no mask) if the tensors/spans are absent (e.g. a val batch).
        """
        ntb = getattr(batch, "non_tensor_batch", None)
        # citation found rate first -- it is independent of the mask tensors, so it
        # is still logged even on a batch that carries no disable_spans.
        stats = self._auto_hint_cite_stats(ntb)

        bb = batch.batch
        if bb is None or "response_mask" not in bb.keys() or "advantages" not in bb.keys():
            return stats
        spans_arr = ntb.get("disable_spans") if isinstance(ntb, dict) else None
        if spans_arr is None:
            return stats
        n = int(bb["response_mask"].shape[0])
        disable_spans = [(_to_py(spans_arr[i]) or []) for i in range(n)]
        eps = float(self._auto_hint_cfg().get("positive_eps", 0.0))
        masked, mask_stats = apply_positive_adv_masking(
            bb["response_mask"], bb["advantages"], disable_spans, positive_eps=eps
        )
        bb["response_mask"] = masked  # same tensor (modified in place); reassign to be safe
        stats.update(mask_stats)
        logger.warning(
            "[HPRL step=%s] auto-hint mask: rows_with_spans=%d pos_masked=%d tokens_dropped=%d "
            "cite_found_rate=%.3f (%d/%d)",
            self.global_steps,
            int(mask_stats["auto_hint/rows_with_spans"]),
            int(mask_stats["auto_hint/rows_pos_masked"]),
            int(mask_stats["auto_hint/mask_tokens_dropped"]),
            stats.get("auto_hint/cite_found_rate", 0.0),
            int(stats.get("auto_hint/cite_quotes_found", 0)),
            int(stats.get("auto_hint/cite_quotes_total", 0)),
        )
        return stats

    def _hprl_apply_step_advantage(self, batch) -> dict:
        """Replace the GRPO advantages with the STEP-LEVEL value-based advantages
        (step_advantage.apply_step_level_advantages). Returns scalar stats for wandb.

        For each GRPO group (a problem's N rollouts) this reconstructs the per-rollout
        turn segments (``step_adv_turns``, recorded by AutoHintAgentLoop), the verified
        final state and failed-hint set, computes the per-problem state values V[0..K]
        and writes per-token advantages: a_C verified-progress tokens get V[se]-V[ss]>=0,
        a_I failed-tail tokens get r_se + V[se+1]-V[se] <=0. An all-incorrect group is
        zeroed. The per-hint penalty weights come from extra_info.hint_full with the
        SAME knobs hint_reward.compute_score uses, so r matches the reward's penalty.

        Also folds in the auto-hint CITATION FOUND RATE. No-op (cite stats only) when the
        tensors / step_adv_turns are absent (e.g. a val batch).
        """
        ntb = getattr(batch, "non_tensor_batch", None)
        stats = self._auto_hint_cite_stats(ntb)

        bb = batch.batch
        if bb is None or "advantages" not in bb.keys() or "response_mask" not in bb.keys():
            return stats
        if not isinstance(ntb, dict):
            return stats
        turns_arr = ntb.get("step_adv_turns")
        uids = ntb.get("uid")
        if turns_arr is None or uids is None:
            return stats

        n = int(bb["advantages"].shape[0])
        acc_arr = ntb.get("acc")
        tt_arr = ntb.get("turn_truncated")  # per-turn-cap cut flag (for the overlong penalty)
        extra_arr = ntb.get("extra_info")
        total_penalty, hard_factor, guidance, guidance_free = self._step_adv_penalty_cfg()

        turns_per_row = [self._coerce_turns(_to_py(turns_arr[i])) for i in range(n)]
        correct_per_row = [
            (float(_to_py(acc_arr[i]) or 0.0) >= 0.5) if acc_arr is not None else False
            for i in range(n)
        ]
        turn_truncated_per_row = (
            [int(_to_py(tt_arr[i]) or 0) for i in range(n)] if tt_arr is not None else None
        )
        penalty_per_row: list = []
        K_per_row: list = []
        cache: dict = {}
        for i in range(n):
            ei = extra_arr[i] if extra_arr is not None else None
            pid = ei.get("problem_id") if isinstance(ei, dict) else None
            key = pid if pid is not None else i
            if key in cache:
                pv, K = cache[key]
            else:
                pv, K = self._step_adv_penalty_vec(ei, total_penalty, hard_factor, guidance, guidance_free)
                cache[key] = (pv, K)
            penalty_per_row.append(pv)
            K_per_row.append(K)

        sa = self._auto_hint_cfg().get("step_adv", {}) or {}
        tv = float(sa.get("terminal_value", 1.0))
        zinc = _truthy(sa.get("zero_if_no_correct", True))
        scale = float(sa.get("adv_scale", 1.0))
        normalize = _truthy(sa.get("normalize", False))
        overlong = float(sa.get("overlong_penalty", 0.0))
        overlong_type = str(sa.get("overlong_penalty_type", "post_hoc"))
        whole_turn = _truthy(sa.get("whole_turn", False))
        # GRPO sets returns == advantages (same tensor); fall back to advantages if absent.
        returns_t = bb["returns"] if "returns" in bb.keys() else bb["advantages"]
        _, _, mstats = apply_step_level_advantages(
            bb["advantages"], returns_t, bb["response_mask"], list(uids),
            turns_per_row, correct_per_row, penalty_per_row, K_per_row,
            terminal_value=tv, zero_if_no_correct=zinc, adv_scale=scale, normalize=normalize,
            overlong_penalty=overlong, overlong_penalty_type=overlong_type,
            turn_truncated_per_row=turn_truncated_per_row,
            whole_turn=whole_turn,
        )
        stats.update(mstats)
        logger.warning(
            "[HPRL step=%s] step-adv: groups=%d scored=%d zeroed=%d rows=%d tokens=%d "
            "pos=%.4f neg=%.4f V0=%.4f gstd=%.4f nfac=%.2f cite=%.3f wt=%d ov=%s(rows=%d)",
            self.global_steps,
            int(mstats["step_adv/groups_total"]), int(mstats["step_adv/groups_scored"]),
            int(mstats["step_adv/groups_zeroed"]), int(mstats["step_adv/rows_scored"]),
            int(mstats["step_adv/tokens_assigned"]),
            mstats["step_adv/adv_pos_mean"], mstats["step_adv/adv_neg_mean"],
            mstats["step_adv/value_s0_mean"], mstats["step_adv/group_std_mean"],
            mstats["step_adv/norm_factor_mean"], stats.get("auto_hint/cite_found_rate", 0.0),
            int(whole_turn),
            (overlong_type if overlong > 0.0 else "off"), int(mstats["step_adv/overlong_rows"]),
        )
        return stats

    def _step_adv_penalty_cfg(self):
        """(total_penalty, hard_factor, guidance_difficulty, guidance_free) from the
        reward kwargs -- the SAME knobs hint_reward.compute_score prices hints with, so
        the step-adv r(h) equals the reward's per-hint penalty (incl. a free X.0 hint)."""
        crf = self.config.get("custom_reward_function", None)
        rkw = crf.get("reward_kwargs", None) if crf is not None else None
        rkw = rkw or {}
        return (
            float(rkw.get("hint_penalty_total", DEFAULT_TOTAL_PENALTY)),
            float(rkw.get("hint_penalty_hard_factor", DEFAULT_HARD_FACTOR)),
            str(rkw.get("hint_guidance_difficulty", DEFAULT_GUIDANCE_DIFFICULTY)),
            _truthy(rkw.get("hint_guidance_free", False)),
        )

    @staticmethod
    def _step_adv_penalty_vec(extra_info, total_penalty, hard_factor, guidance, guidance_free=False):
        """Per-hint penalty weights in POOL ORDER (p[0..K-1]) for a rollout, from its
        extra_info: the selector pool gives the canonical hint order (= the step
        indexing the loop used), hint_full gives the difficulty-weighted penalties.
        Returns ([], 0) when either is absent."""
        if not isinstance(extra_info, dict):
            return [], 0
        tk = extra_info.get("tools_kwargs") or {}
        tool = tk.get("request_hint") or {}
        ck = tool.get("create_kwargs") or {}
        pool = ck.get("hints")
        order = pending_hint_ids(pool, []) if pool is not None else []
        hint_full = extra_info.get("hint_full")
        pens = (
            compute_hint_penalties(
                hint_full, total_penalty=total_penalty, hard_factor=hard_factor,
                guidance_difficulty=guidance, guidance_free=guidance_free,
            )
            if hint_full
            else {}
        )
        pv = [float(pens.get(str(h), 0.0)) for h in order]
        return pv, len(order)

    @staticmethod
    def _coerce_turns(turns):
        """Normalize a rollout's step_adv_turns to a list of 6-int rows; drop malformed."""
        out = []
        if not turns:
            return out
        for t in turns:
            try:
                out.append([int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])])
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def _auto_hint_cite_stats(ntb) -> dict:
        """Citation found rate from the per-rollout completed_hints-quote counters
        recorded by AutoHintAgentLoop (cite_quotes_total / cite_quotes_found): of the
        selector's cited sentences, the fraction that fuzzy-located in the student's
        trace. Returns {} when the counters are absent (a non-auto-hint batch)."""
        if not isinstance(ntb, dict):
            return {}
        ct, cf = ntb.get("cite_quotes_total"), ntb.get("cite_quotes_found")
        if ct is None or cf is None:
            return {}
        n = len(ct)
        tot = sum(int(round(float(_to_py(ct[i]) or 0))) for i in range(n))
        fnd = sum(int(round(float(_to_py(cf[i]) or 0))) for i in range(n))
        return {
            "auto_hint/cite_quotes_total": float(tot),
            "auto_hint/cite_quotes_found": float(fnd),
            "auto_hint/cite_found_rate": float(fnd / tot) if tot else 0.0,
        }

    # ------------------------------------------------------------------ #
    # rollout-log augmentation: add the per-rollout hint state to the dump
    # ------------------------------------------------------------------ #
    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        """Add the per-rollout HINT STATE to the dumped rollout JSONL.

        Stock verl dumps input/output/score/gts + the reward-extra keys (acc,
        num_hints, ...). The structured ``applied_hints`` list and the budget B_q
        the rollout ran under are NOT among them, so we splice them in here (only
        when HPRL is on). Both ride as length-N columns, so _write_generations
        emits one value per rollout.
        """
        if not self._hprl_cfg().get("enable", False):
            return super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

        try:
            reward_extra_infos_dict = self._hprl_rollout_log_columns(batch, reward_extra_infos_dict)
        except Exception as e:  # logging must never crash training
            print(f"[HPRL step={self.global_steps}] rollout-log augmentation failed: {e}", flush=True)

        # Dump the rollout JSONL ourselves (NOT super()._log_rollout_data) so the
        # input/output are decoded WITH special tokens kept (skip_special_tokens=False) --
        # preserving the RAW chat-template structure (turn markers <|im_start|>/<|im_end|>,
        # EOS, and the injected hint user-turns) for inspecting auto-hint trajectories. verl's
        # own _log_rollout_data strips them. Padding is trimmed per row via the attention mask
        # so only real tokens show (no trailing/leading pad noise). Everything else mirrors
        # verl's _log_rollout_data; the write still goes through our robust _dump_generations
        # / _write_generations overrides (non-fatal background write).
        try:
            inputs, outputs = self._decode_raw_with_special(batch)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in batch
            ]
            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id", batch.non_tensor_batch["request_id"].tolist()
                )
            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
        except Exception as e:  # noqa: BLE001 -- logging must never crash training
            print(
                f"[HPRL step={self.global_steps}] rollout dump skipped (failed): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
        return None

    def _decode_raw_with_special(self, batch):
        """Decode the batch's prompts/responses WITH special tokens kept, trimming pad.

        skip_special_tokens=False preserves the raw chat-template structure (turn markers,
        EOS, injected hint user-turns); the attention mask drops the left-pad (prompts) and
        right-pad (responses) so the dump shows only real tokens, not a long pad run. Returns
        ``(inputs, outputs)`` -- length-N lists of decoded strings, one per rollout.
        """
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attn = batch.batch["attention_mask"]
        prompt_len = prompts.size(1)
        resp_len = responses.size(1)
        inputs, outputs = [], []
        for i in range(len(batch)):
            p_keep = attn[i, :prompt_len].bool()
            r_keep = attn[i, prompt_len:prompt_len + resp_len].bool()
            inputs.append(self.tokenizer.decode(prompts[i][p_keep], skip_special_tokens=False))
            outputs.append(self.tokenizer.decode(responses[i][r_keep], skip_special_tokens=False))
        return inputs, outputs

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Robust + diagnostic replacement for verl's background JSONL writer.

        The batch columns were all length-consistent (the `dumplens` check showed uniform),
        yet verl's `_write_generations` still IndexErrors at `entry = {k: v[i] ...}` for
        `i in range(len(inputs))` -- so the short column is one of the DERIVED base_data
        lists (`input`/`output`/`gts`/`score`) or a `reward_*` column, not a batch column.
        This override builds base_data exactly like verl, then (1) PRINTS any column whose
        length != len(inputs) -- naming the culprit -- and (2) writes only `min(lengths)`
        rows so it can NEVER IndexError. Self-resolves via `self._write_generations` from
        verl's `_dump_generations`. print() so it reaches the cluster console log.
        """
        import json
        import os

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lens = {k: len(v) for k, v in base_data.items()}
        n_safe = min(lens.values()) if lens else 0
        if n_safe != n:
            short = {k: L for k, L in lens.items() if L != n}
            print(
                f"[HPRL DUMP step={global_steps}] SHORT base_data columns (len(inputs)={n}, "
                f"writing {n_safe} rows): {short}  (full lens={lens})",
                flush=True,
            )

        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")
        with open(filename, "w") as f:
            for i in range(n_safe):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        print(f"Dumped generations to {filename}", flush=True)

    def _dump_generations(self, *args, **kwargs):
        """Make the background rollout-dump truly non-fatal.

        verl's ``_dump_generations`` re-raises a failed background write via ``f.result()``
        and then SKIPS clearing ``self._dump_futures`` (the line after the raise is never
        reached) -- so a single failed write re-raises on EVERY subsequent step and finally
        propagates uncaught (shutdown). Here we swallow the surfaced error and PURGE all
        done futures, so a bad write can never re-surface and never kills training.
        """
        if not self._hprl_cfg().get("enable", False):
            return super()._dump_generations(*args, **kwargs)
        try:
            super()._dump_generations(*args, **kwargs)
        except Exception as e:  # a previously-submitted background write failed
            print(
                f"[HPRL step={self.global_steps}] rollout dump background write failed "
                f"(ignored): {type(e).__name__}: {e}",
                flush=True,
            )
        finally:
            try:  # drop DONE futures (success or failure) so none can re-raise again
                self._dump_futures = [f for f in getattr(self, "_dump_futures", []) if not f.done()]
            except Exception:  # noqa: BLE001
                pass

    def _shutdown_dump_executor(self):
        """Drain the rollout-dump executor without letting a failed background dump crash.

        Belt-and-suspenders for the non-essential rollout dump: verl writes generations in
        a background thread and re-raises a failed write via ``f.result()`` -- not only in
        ``_log_rollout_data`` (guarded above) but ALSO here at shutdown/checkpoint/exception
        cleanup (ray_trainer.py:1399/1758/1770), which is OUTSIDE that guard. Swallow + log
        any such surfaced dump error so training/checkpointing is never killed by logging.
        """
        if not self._hprl_cfg().get("enable", False):
            return super()._shutdown_dump_executor()
        try:
            return super()._shutdown_dump_executor()
        except Exception as e:  # a background dump failed; the dump is non-essential
            # print(), not logger.warning -- see _log_rollout_data note (logger.warning
            # does not reach the cluster console log from the trainer).
            print(
                f"[HPRL] rollout-dump executor surfaced a failed dump at shutdown (ignored): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            try:  # still drain + close so the executor is not leaked
                self._dump_futures.clear()
                self._dump_executor.shutdown(wait=True)
            except Exception:  # noqa: BLE001
                pass

    def _hprl_rollout_log_columns(self, batch, reward_extra_infos_dict) -> dict:
        ntb = batch.non_tensor_batch
        n = len(batch)
        out = dict(reward_extra_infos_dict)

        # applied_hints: flattened top-level key, else nested in tool_extra_fields.
        applied = None
        if "applied_hints" in ntb:
            applied = [_to_py(ntb["applied_hints"][i]) for i in range(n)]
        elif "tool_extra_fields" in ntb:
            applied = [(_to_py(ntb["tool_extra_fields"][i]) or {}).get("applied_hints") for i in range(n)]
        if applied is not None:
            out.setdefault("applied_hints", applied)

        # budget B_q the rollout actually ran under (from extra_info.tools_kwargs).
        if "extra_info" in ntb:
            default_budget = int(self._hprl_cfg().get("default_budget", 8))
            out.setdefault(
                "hint_budget",
                [_gen_budget_from_extra(_to_py(ntb["extra_info"][i]) or {}, default_budget) for i in range(n)],
            )

        # auto-hint: the per-rollout verified-prefix spans (response-token ranges
        # dropped from the loss under positive advantage), so the mask is inspectable
        # in the rollout dump alongside applied_hints.
        if "disable_spans" in ntb:
            out.setdefault("disable_spans", [_to_py(ntb["disable_spans"][i]) for i in range(n)])
        # step-adv: the per-rollout [ts, boundary, te, state_start, state_end, is_fail]
        # turn segments, so the value-based advantage assignment is inspectable.
        if "step_adv_turns" in ntb:
            out.setdefault("step_adv_turns", [_to_py(ntb["step_adv_turns"][i]) for i in range(n)])
        return out
