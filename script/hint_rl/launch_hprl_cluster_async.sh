#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_async.sh -- FULLY-ASYNC entry point for training platforms
# that can only submit a script PATH (no env overrides).
#
# Identical to launch_hprl_cluster.sh (run on EVERY pod; selector/training
# split, rendezvous, Ray bring-up all unchanged) except TRAIN_SCRIPT points at
# the fully-async run script (run_hprl_async.sh = the Olmo auto-hint async run;
# its defaults were aligned to run_auto_hint_olmo3_7b_instruct_async.sh).
# Equivalent to:  TRAIN_SCRIPT=<...>/run_hprl_async.sh bash launch_hprl_cluster.sh
#
# Async knobs (TRAINER_NNODES/ROLLOUT_NNODES split, STALENESS_THRESHOLD,
# PARTIAL_ROLLOUT, ...) keep their run_hprl_async.sh defaults (2:2, 0.5, on);
# export them before this script runs if the platform ever allows env.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_hprl_async.sh"}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster.sh" "$@"
