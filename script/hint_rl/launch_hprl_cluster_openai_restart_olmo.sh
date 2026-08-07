#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_openai_restart_olmo.sh -- SYNC + OPENAI-API-selector entry
# point for the Olmo TERMINATE-AND-RESTART (segment-chain) auto-hint run, for
# training platforms that can only submit a script PATH.
#
# Identical to launch_hprl_cluster_openai.sh (run on EVERY pod; no selector
# pods, OPENAI_API_KEY sourcing, API reachability probe, Ray bring-up all
# inherited) except TRAIN_SCRIPT points at the SYNC Olmo restart wrapper
# (run_auto_hint_olmo3_7b_instruct_restart.sh -- devlog 2026-08-05): fresh
# single-turn segment per hint, all-turn segment-row training
# (HPRL_RESTART_TRAIN_SEGMENTS=all, anchored on sampling logprobs via
# HPRL_CALC_LOGPROBS=true), pool-worded recap, per-segment cap 4096, prompt
# 4096 / response 8192, budget-grouped sampling ON (sync-only). Equivalent to:
#
#   TRAIN_SCRIPT=<...>/run_auto_hint_olmo3_7b_instruct_restart.sh \
#       bash launch_hprl_cluster_openai.sh
#
# COST NOTE (unchanged vs multi-turn): the loop keeps the full logical
# transcript for the selector, so selector prompts grow with the chain --
# restart shrinks the POLICY context, not the selector bill.
#
# RESUME_EXP_NAME stays empty here (fresh stamped exp); set it only to resume a
# previous RESTART run in place. All wrapper knobs are env-overridable through.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_auto_hint_olmo3_7b_instruct_restart.sh"}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
export RESUME_EXP_NAME=${RESUME_EXP_NAME-}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster_openai.sh" "$@"
