#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_openai_async.sh -- FULLY-ASYNC + OPENAI-API-selector
# entry point for training platforms that can only submit a script PATH.
#
# Identical to launch_hprl_cluster_openai.sh (run on EVERY pod; no selector
# pods, OPENAI_API_KEY sourcing, API reachability probe, Ray bring-up all
# unchanged) except TRAIN_SCRIPT points at the QWEN3-8B-BASE auto-hint
# fully-async wrapper (run_auto_hint_qwen3_8b_base_async.sh: fresh run from
# the eos-derived base dir, max_response_length=30720, 1:N-1 trainer:rollout
# split at staleness 2, overlong_type=value -- see its header). Equivalent to:
# The cumulative progress-message mode is enabled by default here as in the
# synchronous OpenAI launcher; both paths order completed and previously given
# hints by their original position in the hint set. Override with
# HPRL_AUTO_HINT_PROGRESS_MESSAGE=false to restore hint-only messages.
#
#   TRAIN_SCRIPT=<...>/run_auto_hint_qwen3_8b_base_async.sh \
#       bash launch_hprl_cluster_openai.sh
#
# For a different run, export TRAIN_SCRIPT before this script (platform env
# permitting), e.g. the bare run_hprl_async.sh (= the Olmo auto-hint async
# defaults: 2:2 split, staleness 1.9). Async knobs (TRAINER_NNODES/
# ROLLOUT_NNODES, STALENESS_THRESHOLD, PARTIAL_ROLLOUT, RESUME_FROM_PATH, ...)
# are likewise env-overridable through to the wrapper.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_auto_hint_qwen3_8b_base_async.sh"}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster_openai.sh" "$@"
