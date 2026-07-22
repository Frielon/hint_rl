# Copyright 2026
#
# HPRL FULLY-ASYNC training entry point -- the async sibling of main_hprl.py.
#
# Mirrors verl's fully_async_main.main() (the async_training assert, device
# autoselect, the rollout-node-split copy into actor_rollout_ref, the legacy
# reward-config migration) and then hands run_ppo the HPRL task runner
# (hprl_fully_async.HPRLFullyAsyncTaskRunnerImpl), which builds the
# HPRLFullyAsyncTrainer -- FullyAsyncTrainer with the HPRL _update_actor
# (verified-prefix mask / step-level advantage + the budget ratchet) and the
# HPRL rollout dump on the MRO. The Rollouter side needs no swap: the dynamic-
# budget dataset rides data.custom_cls and the auto-hint loop rides the per-row
# agent_name, both honored by the stock FullyAsyncRollouter.
#
# Async-mode restrictions enforced here, loudly, at launch:
#   * data.hprl.kpack.enable=true        -> hard error (sync-trainer-only path).
#   * data.hprl.budget_sampling.enable   -> forced false with a warning: the
#     Rollouter streams samples one at a time (gen_batch_size=1), so there is no
#     generation batch to keep budget-homogeneous -- the queue already absorbs
#     the per-sample duration variance the sampler existed to remove. (The wrap
#     in main_hprl only patches the SYNC TaskRunner's sampler anyway; silently
#     ignoring the flag would be worse than refusing it.)
#
# Launched by run_hprl_async.sh (config = config/hprl_fully_async_trainer.yaml
# = fully_async_ppo_trainer + data.hprl). Launch cwd must be the verl repo root
# (the Ray working-dir) so the config's file:// searchpath resolves.

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def _apply_hprl_async_guards(config) -> None:
    """Enforce the async-mode HPRL restrictions on the composed config."""
    hprl = OmegaConf.select(config, "data.hprl")
    if hprl is None:
        return
    if bool(hprl.get("enable", False)) and bool((hprl.get("kpack") or {}).get("enable", False)):
        raise NotImplementedError(
            "data.hprl.kpack.enable=true is not supported in fully-async mode "
            "(the probe lives in the sync trainer's _get_gen_batch / rollout.n "
            "split, which the async queue path never runs). Use run_hprl_qwen2.5_7b.sh."
        )
    bs = hprl.get("budget_sampling")
    if bs is not None and bool(bs.get("enable", False)):
        print(
            "[HPRL ASYNC MAIN] WARNING: data.hprl.budget_sampling.enable=true has no effect "
            "in fully-async mode (the Rollouter streams samples one at a time; there is no "
            "generation batch to group). Forcing it to false."
        )
        OmegaConf.update(config, "data.hprl.budget_sampling.enable", False)


@hydra.main(config_path="config", config_name="hprl_fully_async_trainer", version_base=None)
def main(config):
    # Ensure async training config exists (same fail-fast as fully_async_main).
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    print(f"[HPRL ASYNC MAIN] driver hostname: {socket.gethostname()}, PID: {os.getpid()}")

    auto_set_device(config)
    # TODO(upstream): unify rollout config with actor_rollout_ref -- verbatim
    # from fully_async_main.main(): the replica manager reads the rollout node
    # split from actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}.
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    _apply_hprl_async_guards(config)

    from hprl_fully_async import HPRLFullyAsyncTaskRunnerImpl

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(HPRLFullyAsyncTaskRunnerImpl))


if __name__ == "__main__":
    main()
