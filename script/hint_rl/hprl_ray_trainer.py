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

from budget_manager import BudgetManager, get_create_budget
from hint_budget_callback import _gen_budget_from_extra, _to_py, hprl_update_budgets
from kpack_expand import render_variant_rows

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HPRLRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that ratchets per-problem hint budgets after each step."""

    def _hprl_cfg(self):
        # the flag lives at data.hprl.* (the dataset only sees config.data, so we
        # keep a single source of truth both sides can read).
        return self.config.data.get("hprl", {}) or {}

    def _hprl_budget_mgr(self) -> BudgetManager:
        """Lazily build the driver-side BudgetManager (loads existing state on
        construction, so it resumes across restarts)."""
        if getattr(self, "_budget_mgr", None) is None:
            cfg = self._hprl_cfg()
            path = cfg.get("budget_state_path", None)
            kcfg = cfg.get("kpack", {}) or {}
            self._budget_mgr = BudgetManager(
                path=path,
                default_budget=int(cfg.get("default_budget", 8)),
                min_budget=int(cfg.get("min_budget", 0)),
                decrement=int(cfg.get("decrement", 1)),
                kpack_require_successes=int(kcfg.get("require_successes", 2)),
            )
            logger.warning(
                "HPRL ratchet enabled: budget_state=%s default=%d min=%d dec=%d (loaded %d problems)",
                path,
                self._budget_mgr.default_budget,
                self._budget_mgr.min_budget,
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
        actor_output = super()._update_actor(batch)
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
                if metrics:
                    md = actor_output.meta_info.setdefault("metrics", {})
                    md.update(metrics)  # scalars survive reduce_metrics
            except Exception as e:  # never let budget bookkeeping crash training
                logger.warning("HPRL budget ratchet failed at step %s: %s", self.global_steps, e)
        return actor_output

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
            logger.warning("HPRL rollout-log augmentation failed at step %s: %s", self.global_steps, e)

        # The rollout-generation DUMP is non-essential and must NEVER crash training.
        # verl's _write_generations builds `gts` via `for item in batch`, which iterates
        # by integer indexing until a non_tensor column runs out -> if ANY non_tensor key
        # is shorter than the tensor batch, gts comes up short and _write_generations dies
        # with IndexError. Defend: log any length-mismatched non_tensor key (so the
        # culprit is named) and drop it for the dump, then guard the call and restore.
        orig_ntb = batch.non_tensor_batch
        try:
            n = len(batch)
            mism = {k: len(v) for k, v in orig_ntb.items() if len(v) != n}
            if mism:
                logger.warning(
                    "[HPRL step=%s] rollout dump: non_tensor keys with len != batch len %d: %s "
                    "-- dropping them for the dump (training unaffected; please report this key set)",
                    self.global_steps, n, mism,
                )
                batch.non_tensor_batch = {k: v for k, v in orig_ntb.items() if len(v) == n}
            super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
        except Exception as e:  # any other dump failure: skip the dump, keep training
            logger.warning(
                "HPRL rollout dump failed at step %s (skipped; training continues): %s",
                self.global_steps, e,
            )
        finally:
            batch.non_tensor_batch = orig_ntb
        return None

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
        return out
