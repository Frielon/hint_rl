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
#   * _compute_old_log_prob (decoupled rollout correction, bypass_mode=False):
#     stock brackets the recompute with fsdp2/megatron-only CPU snapshots that
#     assert on our fsdp1 trainer; the snapshot is dead weight at
#     trigger_parameter_sync_step=1, so the override below skips it.
#
# The Rollouter needs ONE fix on top of stock: verl's
# _process_single_sample_streaming REPLACES RolloutSample.full_batch with the
# agent-loop output (`rollout_sample.full_batch = ret`), where the sync trainer
# UNIONS the gen output into the dataloader batch. Loop-attached non-tensors
# (step_adv_turns / acc / applied_hints) survive, but the dataset-side columns
# -- extra_info (hint pool: hint_full, tools_kwargs), data_source, reward_model
# -- never reach the trainer. Trainer-side consequences (found 2026-07-23 on
# the staleness grid): the budget-ratchet block guards on "extra_info" in
# non_tensor_batch and silently skips, and step-adv prices every row at K=0 so
# EVERY group is zeroed -- advantages==0, pg_loss==0, the policy never moves.
# HPRLFullyAsyncRollouter below restores the missing dataset non-tensors onto
# the agent-loop output before the sample enters the MessageQueue. Everything
# else rides stock: HintBudgetDataset injection via data.custom_cls
# (create_rl_dataset), the auto_hint agent loop via the per-row agent_name +
# rollout.agent.agent_loop_config_path, the streaming reward via the
# RewardLoopManager. The budget state crosses the actor boundary through the
# shared-FS JSON (atomic os.replace writer on the trainer, mtime-cached reader
# in the dataset), exactly as it crossed the process boundary in the sync job.
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
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role

from hprl_ray_trainer import HPRLRayPPOTrainer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The undecorated class bodies behind the @ray.remote wrappers (see header).
_FULLY_ASYNC_TRAINER_BODY = FullyAsyncTrainer.__ray_metadata__.modified_class
_FULLY_ASYNC_RUNNER_BODY = FullyAsyncTaskRunner.__ray_metadata__.modified_class
_FULLY_ASYNC_ROLLOUTER_BODY = FullyAsyncRollouter.__ray_metadata__.modified_class


def restore_dataset_non_tensors(src, out):
    """Copy every non-tensor column of ``src`` (the pre-generation sample batch,
    which still carries the dataset fields) that is MISSING from ``out`` (the
    agent-loop output) into ``out``. Returns (out, n_restored).

    Row alignment is trivially safe in the fully-async streaming path: ``src``
    is ONE dataset sample repeated rollout.n times, so all rows share the same
    non-tensor values. Columns whose length disagrees with ``out`` are skipped
    with a warning rather than corrupting the batch. Keys already present in
    ``out`` (loop-attached fields, and uid which the rollouter re-stamps) are
    never touched.
    """
    if src is None or out is None:
        return out, 0
    n = len(out)
    restored = 0
    for key, val in src.non_tensor_batch.items():
        if key in out.non_tensor_batch:
            continue
        if len(val) != n:
            logger.warning(
                "[HPRL ASYNC] not restoring non-tensor %r: %d rows vs batch %d", key, len(val), n
            )
            continue
        out.non_tensor_batch[key] = val
        restored += 1
    return out, restored


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

    def _compute_old_log_prob(self, batch):
        # The module-header skip of stock FullyAsyncTrainer's CPU param
        # snapshots. At trigger_parameter_sync_step=1, local_trigger_step stays
        # 1 forever (_fit_update_local_step resets, never increments), so the
        # stock save_model_to_cpu(1) is written every step and NEVER read back
        # -- and its handler is fsdp2-only (fsdp2_sharded_save_to_cpu asserts
        # "No DTensor-type parameters found" on this job's fsdp1 trainer). Go
        # straight to RayPPOTrainer's snapshot-free recompute, unbound like the
        # fit() delegation above: super(_FULLY_ASYNC_TRAINER_BODY, self) would
        # NOT work -- ray's modified_class is a SUBCLASS of the stock
        # FullyAsyncTrainer, which therefore sits once more on the MRO right
        # after the body, snapshot wrapper included.
        if self.trigger_parameter_sync_step == 1:
            return RayPPOTrainer._compute_old_log_prob(self, batch)
        # trigger > 1 genuinely needs the version-1 snapshot (old_log_prob is
        # anchored to the last-synced params across local steps); upstream only
        # implements it for fsdp2/megatron, so fail fast with the fix instead
        # of the DTensor assert three actors deep.
        if self.config.actor_rollout_ref.actor.strategy == "fsdp":
            raise NotImplementedError(
                "async_training.trigger_parameter_sync_step > 1 requires the CPU "
                "param snapshots in stock _compute_old_log_prob, whose handlers "
                "are fsdp2/megatron-only (fsdp1 has no DTensor params). Set "
                "actor_rollout_ref.actor.strategy=fsdp2 (mind ckpt-format "
                "compatibility on resume) or keep trigger_parameter_sync_step=1."
            )
        return super()._compute_old_log_prob(batch)


HPRLFullyAsyncTrainer = ray.remote(num_cpus=10)(_HPRLFullyAsyncTrainerImpl)


class _HPRLFullyAsyncRollouterImpl(_FULLY_ASYNC_ROLLOUTER_BODY):
    """FullyAsyncRollouter that restores the dataset non-tensors (extra_info /
    data_source / reward_model / ...) onto the agent-loop output before each
    sample is queued -- see the module header for why stock loses them and what
    that breaks (all-zero step-adv, dead budget ratchet).

    The wrap lives on the manager's ``generate_sequences_single`` -- the single
    seam between the original batch and ``rollout_sample.full_batch = ret`` --
    instead of copying the whole ``_process_single_sample_streaming`` body, so
    upstream edits to the streaming loop (counters, queue put) keep working.
    """

    async def _init_async_rollout_manager(self):
        await super()._init_async_rollout_manager()
        mgr = self.async_rollout_manager
        inner = mgr.generate_sequences_single
        logged = False

        async def _generate_and_restore(prompts):
            nonlocal logged
            out = await inner(prompts)
            out, restored = restore_dataset_non_tensors(prompts, out)
            if restored and not logged:
                logged = True
                print(
                    f"[HPRL ASYNC] restoring {restored} dataset non-tensor keys onto agent-loop "
                    f"outputs (extra_info et al.; logged once per rollouter)"
                )
            return out

        mgr.generate_sequences_single = _generate_and_restore


HPRLFullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(_HPRLFullyAsyncRollouterImpl)


class HPRLFullyAsyncTaskRunnerImpl(_FULLY_ASYNC_RUNNER_BODY):
    """FullyAsyncTaskRunner that builds HPRLFullyAsyncTrainer and
    HPRLFullyAsyncRollouter instead.

    _create_trainer / _create_rollouter are overridden (verbatim copies of the
    base bodies with the classes swapped -- the base hardcodes the
    module-global classes, so there is no narrower hook). The MessageQueue and
    the whole run() orchestration are inherited stock.
    """

    def _create_rollouter(self, config) -> None:
        print("[HPRL ASYNC MAIN] Starting create rollouter (HPRLFullyAsyncRollouter)...")
        rollouter = HPRLFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[HPRL ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[HPRL ASYNC MAIN] HPRLFullyAsyncRollouter created and initialized successfully")

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
