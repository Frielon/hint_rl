# Copyright 2026
#
# HPRL training entry point.
#
# Reuses verl's standard PPO TaskRunner / run_ppo and only swaps the trainer
# class for HPRLRayPPOTrainer (the dynamic-budget ratchet). Rather than copy the
# evolving TaskRunner.run() body (the DAPO recipe's copy is already stale -- it
# predates add_teacher_model_resource_pool), HPRLTaskRunner patches the trainer
# global INSIDE the actor and delegates to the stock run(): the base run()
# resolves ``RayPPOTrainer`` from verl.trainer.main_ppo's module globals, so
# rebinding that name makes it build our subclass with no other change.
#
# Launched by run_hprl_qwen2.5_7b.sh. The config (config/hprl_trainer.yaml) is
# stock ppo_trainer + the data.hprl knobs; with data.hprl.enable=false this runs
# exactly like vanilla multi-turn GRPO.

import os
import socket

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class HPRLTaskRunner(TaskRunner):
    """TaskRunner that builds HPRLRayPPOTrainer instead of RayPPOTrainer."""

    def run(self, config):
        import verl.trainer.main_ppo as main_ppo

        from budget_sampler import wrap_create_rl_sampler
        from hprl_ray_trainer import HPRLRayPPOTrainer

        print(f"HPRLTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # The base run() instantiates the module-global `RayPPOTrainer`; rebind it
        # so the (unmodified) base flow constructs our flag-gated subclass.
        main_ppo.RayPPOTrainer = HPRLRayPPOTrainer
        # Same flag-gated override for the train sampler: wrap the module-global
        # create_rl_sampler (resolved by the base run() at call time) so that with
        # data.hprl.budget_sampling.enable=true it returns the BudgetGroupedSampler
        # (budget-homogeneous batches -> no high-budget straggler bottlenecking the
        # generation step) and is byte-identical to stock when off.
        main_ppo.create_rl_sampler = wrap_create_rl_sampler(main_ppo.create_rl_sampler)
        return super().run(config)


@hydra.main(config_path="config", config_name="hprl_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(HPRLTaskRunner))


if __name__ == "__main__":
    main()
