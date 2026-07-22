# Copyright 2026
#
# HPRL x verl fully-async policy: the trainer/task-runner subclasses.
#
# verl's fully_async_policy splits the job into two Ray actors -- a Rollouter
# (owns the train/val dataloaders + the vLLM replicas + the agent loops, streams
# one prompt-group at a time into a MessageQueue) and a FullyAsyncTrainer (pulls
# require_batches*ppo_mini_batch_size groups per iteration, updates the actor,
# NCCL-pushes weights back every trigger_parameter_sync_step iterations). All
# HPRL trainer-side logic lives in HPRLRayPPOTrainer's method overrides of
# RayPPOTrainer (_update_actor: step-adv / verified-prefix mask + the budget
# ratchet; _log_rollout_data + dump robustness), and FullyAsyncTrainer inherits
# RayPPOTrainer through SeparateRayPPOTrainer -- so the port is a cooperative
# subclass that puts HPRLRayPPOTrainer FIRST on the MRO:
#
#   _HPRLFullyAsyncTrainerImpl(HPRLRayPPOTrainer, <FullyAsyncTrainer body>)
#   MRO: Impl -> HPRLRayPPOTrainer -> FullyAsyncTrainer -> SeparateRayPPOTrainer
#        -> RayPPOTrainer
#
#   * FullyAsyncTrainer.fit_step() -> SeparateRayPPOTrainer._fit_update_actor()
#     -> self._update_actor() resolves to the HPRL override, whose
#     super()._update_actor() falls through to RayPPOTrainer's real update.
#     The batch there already carries GRPO advantages + the reward extra keys
#     (acc / num_hints, merged into non_tensor_batch by _fit_compute_advantage),
#     which is exactly what the mask/step-adv/ratchet read -- same contract as
#     the sync job.
#   * _fit_dump_data() -> self._log_rollout_data() resolves to the HPRL dump
#     (special-tokens-kept decode + applied_hints/budget columns).
#   * HPRLRayPPOTrainer.fit (sync def; applies the k-pack config split) would
#     shadow FullyAsyncTrainer's async fit -- overridden below (k-pack is
#     unsupported in async mode, so the override just guards and delegates).
#   * HPRLRayPPOTrainer._get_gen_batch (k-pack expansion) is dead code here:
#     the async trainer pulls finished rollouts from the queue and never builds
#     a gen batch (the Rollouter feeds via prepare_single_generation_data).
#
# The Rollouter runs STOCK: HintBudgetDataset injection rides data.custom_cls
# (create_rl_dataset), the auto_hint agent loop rides the per-row agent_name +
# rollout.agent.agent_loop_config_path, and the streaming reward rides the
# RewardLoopManager -- none of which need a subclass. The budget state crosses
# the actor boundary through the shared-FS JSON (atomic os.replace writer on the
# trainer, mtime-cached reader in the dataset), exactly as it crossed the
# process boundary in the sync job.
#
# Ray mechanics: FullyAsyncTrainer / FullyAsyncTaskRunner are @ray.remote-
# decorated AT DEFINITION, and Ray forbids subclassing an ActorClass. The
# undecorated body is reachable as <ActorClass>.__ray_metadata__.modified_class
# (verified on ray 2.55.1: subclass + re-decorate works); that private-API
# reach-in is the one concession this port makes, kept in this single module.
#
# No verl core edits (per the project rule); wired by main_hprl_async.py.

from __future__ import annotations

import logging
import os

import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from hprl_ray_trainer import HPRLRayPPOTrainer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The undecorated class bodies behind the @ray.remote wrappers (see header).
_FULLY_ASYNC_TRAINER_BODY = FullyAsyncTrainer.__ray_metadata__.modified_class
_FULLY_ASYNC_RUNNER_BODY = FullyAsyncTaskRunner.__ray_metadata__.modified_class


class _HPRLFullyAsyncTrainerImpl(HPRLRayPPOTrainer, _FULLY_ASYNC_TRAINER_BODY):
    """FullyAsyncTrainer with the HPRL trainer-side overrides on the MRO."""

    async def fit(self):
        # HPRLRayPPOTrainer.fit is a SYNC def whose only job is the k-pack
        # config split before the sync training loop; on the async MRO it would
        # shadow FullyAsyncTrainer's async fit. k-pack is unsupported here
        # (main_hprl_async also guards at launch -- this is the backstop), so
        # guard and delegate to the fully-async loop.
        cfg = self._hprl_cfg()
        kcfg = cfg.get("kpack", {}) or {}
        if cfg.get("enable", False) and kcfg.get("enable", False):
            raise NotImplementedError(
                "HPRL k-pack (data.hprl.kpack.enable=true) is not supported in "
                "fully-async mode: the probe rewrites rollout.n and expands the "
                "gen batch in the sync trainer's _get_gen_batch, which the async "
                "queue path never calls. Run the sync job for k-pack."
            )
        # Unbound call through the async body (super() from here would resolve
        # to HPRLRayPPOTrainer.fit, the sync one we are shadowing).
        return await _FULLY_ASYNC_TRAINER_BODY.fit(self)


HPRLFullyAsyncTrainer = ray.remote(num_cpus=10)(_HPRLFullyAsyncTrainerImpl)


class HPRLFullyAsyncTaskRunnerImpl(_FULLY_ASYNC_RUNNER_BODY):
    """FullyAsyncTaskRunner that builds HPRLFullyAsyncTrainer instead.

    Only _create_trainer is overridden (a verbatim copy of the base body with
    the trainer class swapped -- the base hardcodes the module-global
    FullyAsyncTrainer, so there is no narrower hook). The Rollouter, the
    MessageQueue and the whole run() orchestration are inherited stock.
    """

    def _create_trainer(self, config) -> None:
        print("[HPRL ASYNC MAIN] Starting create trainer (HPRLFullyAsyncTrainer)...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = HPRLFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[HPRL ASYNC MAIN] HPRLFullyAsyncTrainer created and initialized successfully")
