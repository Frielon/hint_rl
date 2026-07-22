#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_async_grid.sh -- platform entry point for the fully-async
# GRID SEARCH on a 5-pod job: 1 selector pod + 4 training pods (Ray cluster),
# split 1 trainer : 3 rollout inside the cluster.
#
# Submit THIS path with 5 replicas. The head node then runs
# grid_search_async.sh in the TRAIN_SCRIPT slot, which sweeps the async knobs
# (trainer:rollout split, staleness, partial rollout) as sequential bounded
# ray jobs on the live cluster and writes logs/async_grid_<id>/results.tsv.
#
# The selector/training pod split CANNOT change mid-job: to also test the
# 2-selector + 2-training topology, submit this script again with
# SELECTOR_NNODES=2 baked in (or exported, if the platform ever allows env).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SELECTOR_NNODES=${SELECTOR_NNODES:-1}

# 2026-07-20 (2nd sweep of the day) staleness fine-scan 0.8..2 (supersedes the
# {1,3} sweep in logs/async_grid_20260720-141549; both its trials failed --
# st1: update_weights OOM on a rollout pod after 10 versions, st3: EADDRINUSE
# from st1's orphaned vLLM workers. Fixes now in place: UPDATE_WEIGHTS_BUCKET_MB
# 4096->1024 in run_hprl_async.sh, per-node orphan reap between trials in
# grid_search_async.sh.)
# Why 0.8..2: per version the rollouter may START (1+s)*B samples; the counter
# resets to the in-flight carryover at each sync, and the carryover ~= the
# concurrency that saturates the pool (from the 20260720-065108 sync run:
# ~32 concurrent responses/GPU, i.e. ~64 groups = B on 16 rollout GPUs).
# Steady state needs s*B >= carryover. At the old 1:2 split that put the knife
# edge at s ~= 1, confirmed by grid-141549 st1: every reset_staleness landed on
# exactly 64 and ShouldPause fired each version. At THIS 1:3 split the rollout
# pool is 24 GPUs -> carryover ~= 96 groups -> predicted knife edge s ~= 1.5,
# so the sweep brackets it: 0.8/1.2 below (expect rollouter pauses / capped
# concurrency), 1.6/2 above (expect pause-free; realized staleness stays ~1
# regardless since the allowance is a cap, not a target, while generation is
# the slow side).
#   * trainer stays 1 node: it already starved at 1:2 (idle-waited ~90%/step),
#     so the 5th pod goes to rollout, not the trainer.
#   * partial=False dropped: at trigger=1 every push truncates in-flight turns.
export SPLITS=${SPLITS:-"1:3"}
export STALENESS_LIST=${STALENESS_LIST:-"0.8 1.2 1.6 2"}
export PARTIAL_LIST=${PARTIAL_LIST:-"True"}

# OOM guard for every trial (see run_hprl_async.sh for the full note): the
# 4 GiB default weight-sync bucket no longer fits next to vLLM at
# gpu_memory_utilization=0.80 once the allocator fragments.
export UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-1024}

export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/grid_search_async.sh"}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster.sh" "$@"
