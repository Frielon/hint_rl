# Copyright 2026
#
# StepHint FULLY-ASYNC training entry point.
#
# Mirrors verl's fully_async_main.main() exactly (the async_training assert,
# device autoselect, the rollout-node-split copy into actor_rollout_ref, the
# legacy reward-config migration) and then hands run_ppo the StepHint task
# runner (stephint_fully_async.StepHintFullyAsyncTaskRunnerImpl), whose only
# delta from stock is the Rollouter: it stamps the per-rollout solution-prefix
# columns + agent_name="stephint_agent" onto every training prompt-group and
# restores the dataset non-tensors onto the agent-loop output. Trainer, queue
# and the GRPO update are verl stock.
#
# Config: verl's OWN fully_async_ppo_trainer.yaml, addressed by absolute path
# into the installed package (no yaml copy to keep in sync -- unlike the HPRL
# entry, stephint needs no custom config blocks; its two knobs ride
# "+data.stephint.*" CLI additions from the run script).
#
# Launched by run_stephint_qwen3_4b_instruct_2507_npu_async.sh. Launch cwd must
# be the verl repo root (the Ray working-dir), same as every sibling.

import os
import socket

import hydra
import ray

import verl.experimental.fully_async_policy as _fully_async_pkg
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

_VERL_ASYNC_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(_fully_async_pkg.__file__)), "config"
)


@hydra.main(config_path=_VERL_ASYNC_CONFIG_DIR, config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    # Ensure async training config exists (same fail-fast as fully_async_main).
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    print(f"[STEPHINT ASYNC MAIN] driver hostname: {socket.gethostname()}, PID: {os.getpid()}")

    auto_set_device(config)
    # TODO(upstream): unify rollout config with actor_rollout_ref -- verbatim
    # from fully_async_main.main(): the replica manager reads the rollout node
    # split from actor_rollout_ref.rollout.{nnodes,n_gpus_per_node}.
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)

    from stephint_fully_async import StepHintFullyAsyncTaskRunnerImpl

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(StepHintFullyAsyncTaskRunnerImpl))


if __name__ == "__main__":
    main()
