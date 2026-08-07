#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_openai_async_restart.sh -- FULLY-ASYNC + OPENAI-API-
# selector entry point for the TERMINATE-AND-RESTART (segment-chain) auto-hint
# run, for training platforms that can only submit a script PATH.
#
# Identical to launch_hprl_cluster_openai_async.sh (run on EVERY pod; no
# selector pods, OPENAI_API_KEY sourcing, API reachability probe, Ray bring-up
# all inherited from launch_hprl_cluster_openai.sh) except TRAIN_SCRIPT points
# at the RESTART wrapper (run_auto_hint_qwen3_8b_base_async_restart.sh --
# devlog 2026-08-04): on a wrong answer the rollout does NOT grow one
# multi-turn context; it ends the attempt and starts a fresh single-turn
# segment whose prompt = problem + verified-progress recap + the new hint.
# Registry swap hint_agent_config_restart.yaml, prompt/response budget
# 4096/28672, per-segment cap 8192, step_adv/whole_turn ON at full parent knob
# parity (phantom fail records keep the value fit chain-exact), resume default
# = the 020431 global_step_200 ckpt -- the matched-start restart-vs-multi-turn
# A/B against the parent async wrapper. Equivalent to:
#
#   TRAIN_SCRIPT=<...>/run_auto_hint_qwen3_8b_base_async_restart.sh \
#       bash launch_hprl_cluster_openai.sh
#
# COST NOTE (unchanged vs multi-turn): the loop keeps the full logical
# transcript for the selector (build_trace parity), so selector prompts grow
# with the chain exactly as they do in multi-turn mode -- restart shrinks the
# POLICY context, not the selector bill. Same per-call accounting applies
# (HPRL_SELECTOR_DUMP_DIR, HintSelector token logs).
#
# Async knobs (TRAINER_NNODES/ROLLOUT_NNODES, STALENESS_THRESHOLD,
# PARTIAL_ROLLOUT, RESUME_FROM_PATH, ...) and restart knobs
# (HPRL_STEP_ADV, HPRL_RESTART_ARCHIVE_IDS, max_prompt_length, ...) are
# env-overridable through to the wrapper. RESUME_EXP_NAME stays empty here
# (fresh stamped exp; the wrapper's RESUME_FROM_PATH does the weights-level
# resume) -- set it only to resume a previous RESTART run in place.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_auto_hint_qwen2.5-7b_async_restart_10k.sh"}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
exec bash "${SCRIPT_DIR}/launch_hprl_cluster_openai.sh" "$@"
