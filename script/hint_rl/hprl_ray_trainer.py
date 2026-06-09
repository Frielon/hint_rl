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

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from budget_manager import BudgetManager
from hint_budget_callback import _gen_budget_from_extra, _to_py, hprl_update_budgets

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
            self._budget_mgr = BudgetManager(
                path=path,
                default_budget=int(cfg.get("default_budget", 8)),
                min_budget=int(cfg.get("min_budget", 0)),
                decrement=int(cfg.get("decrement", 1)),
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

    def _update_actor(self, batch):
        actor_output = super()._update_actor(batch)
        if self._hprl_cfg().get("enable", False):
            try:
                metrics = hprl_update_budgets(
                    batch, self._hprl_budget_mgr(), global_step=self.global_steps
                )
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
        if self._hprl_cfg().get("enable", False):
            try:
                reward_extra_infos_dict = self._hprl_rollout_log_columns(batch, reward_extra_infos_dict)
            except Exception as e:  # logging must never crash training
                logger.warning("HPRL rollout-log augmentation failed at step %s: %s", self.global_steps, e)
        return super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

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
