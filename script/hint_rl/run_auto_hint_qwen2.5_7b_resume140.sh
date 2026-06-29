#!/usr/bin/env bash
#
# FORK an auto-hint run from a SPECIFIC checkpoint + a seeded budget state.
#
# Non-destructive: starts a NEW experiment (distinct exp_name) whose actor weights,
# optimizer and dataloader state are loaded from RESUME_FROM_PATH (a global_step_N
# dir), and whose initial per-problem budgets are seeded from SRC_BUDGET_STATE. The
# source run's later checkpoints and its budget_state.json are left untouched -- we
# copy the budget file into the new run's log dir so the ratchet writes the COPY.
#
# Defaults fork run HPRL-AutoHint-Qwen2.5-7B-dapo-20260625-230822 from global_step_140
# with that run's (post-step-180) budget_state.json. Override any path via env, e.g.
#   RESUME_FROM_PATH=.../global_step_120 bash run_auto_hint_qwen2.5_7b_resume140.sh
#
# Requires the Ray cluster + selector to already be up (launch_hprl_cluster.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

# --- checkpoint to fork from (must be a .../global_step_N dir; verl reads N from it) ---
export RESUME_MODE=${RESUME_MODE:-resume_path}
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-"${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen2.5-7B-Instruct/HPRL-AutoHint-Qwen2.5-7B-dapo-20260625-230822/global_step_140"}

# --- budget state to seed the new run from (copied into the new log dir; the
#     original is never written) ---
SRC_BUDGET_STATE=${SRC_BUDGET_STATE:-"${HINT_RL_HOME}/logs/HPRL-AutoHint-Qwen2.5-7B-dapo-20260625-230822/budget_state.json"}

# --- new run identity (distinct wandb/log/ckpt namespace) ---
export exp_name=${exp_name:-"HPRL-AutoHint-Qwen2.5-7B-dapo-resume140-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

# --- seed the budget state: copy the source file into the new run's log dir so the
#     ratchet writes the COPY (BudgetManager loads it on construction) ---
NEW_LOG_DIR="${HINT_RL_HOME}/logs/${exp_name}"
mkdir -p "${NEW_LOG_DIR}"
if [ ! -f "${SRC_BUDGET_STATE}" ]; then
    echo "[resume140] ERROR: source budget state not found: ${SRC_BUDGET_STATE}" >&2
    exit 1
fi
cp "${SRC_BUDGET_STATE}" "${NEW_LOG_DIR}/budget_state.json"
export BUDGET_STATE_PATH="${NEW_LOG_DIR}/budget_state.json"

echo "[resume140] forking new auto-hint run"
echo "[resume140]   exp_name=${exp_name}"
echo "[resume140]   resume_from=${RESUME_FROM_PATH}"
echo "[resume140]   budget_state(seed)=${SRC_BUDGET_STATE}"
echo "[resume140]   budget_state(live)=${BUDGET_STATE_PATH}"

exec bash "${SCRIPT_DIR}/run_auto_hint_qwen2.5_7b.sh" "$@"
