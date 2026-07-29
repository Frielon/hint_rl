#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_openai_staleness_grid.sh -- platform entry point for the
# SYNC-baseline + ASYNC-staleness benchmark (staleness_grid.sh) with the
# OPENAI API as the hint selector (no selector pods).
#
# Submit THIS path with N replicas (N >= 2; the sweep was sized on 5). Every
# pod joins the Ray cluster (the OpenAI launcher has no selector split); the
# head then runs staleness_grid.sh in the TRAIN_SCRIPT slot:
#
#   trials 1..  FULLY-ASYNC    -- 1 trainer : (N-1) rollout nodes, sweeping
#               async_training.staleness_threshold over 2 2.2 2.4 2.6 2.8 3
#               (the 20260722 sweep covered 0.2-2: s=2 hit ~94% of the
#               ~152 s/version trainer-compute ceiling, so this probes the
#               remaining tail up to the expected flatline at s~3)
#   trial 0     SYNC baseline  -- default OFF (RUN_SYNC=0): measured separately
#               on 4 nodes (~9.3 vph, run ...dapo-20260723-100408), and on a
#               5-pod cluster it dies on DP divisibility (512 % 10 != 0).
#
# All trials start from the SAME trained policy: the Qwen3-8B-Base auto-hint
# run's global_step_320, merged once to eval/merged/hprl-qwen3-base-512-step320
# (idempotent; shared with the ckpt-eval harness). Results land in
# logs/staleness_grid_<id>/results.tsv (versions/hour, ranked).
#
# NOTE (2026-07-23): first submission after the fully-async extra_info fix
# (HPRLFullyAsyncRollouter) -- trials now actually train. Verify at step 1 of
# the first trial: "restoring N dataset non-tensor keys" in the rollouter log,
# step-adv scored>0, nonzero actor/pg_loss, hprl/* metrics present.
#
# Selector cost note: ~6 trials x 16 bounded steps of hint calls on the model
# the openai launcher defaults to -- see its header for the per-call estimate.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STALENESS_LIST=${STALENESS_LIST:-"1.6 1.8 2.1"}
export RUN_SYNC=${RUN_SYNC:-0}
export TRAINER_NNODES=${TRAINER_NNODES:-1}    # async trials: 1 trainer, rest rollout

export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/staleness_grid.sh"}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster_openai.sh" "$@"
