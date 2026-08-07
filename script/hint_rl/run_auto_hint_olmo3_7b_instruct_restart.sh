#!/usr/bin/env bash
#
# TERMINATE-AND-RESTART (segment-chain) auto-hint HPRL run for
# Olmo-3-7B-Instruct -- SYNC (colocated) edition. The restart sibling of
# run_auto_hint_olmo3_7b_instruct.sh: every science knob defaults to that
# wrapper's value; the restart deltas are the blocks marked "RESTART DELTA".
# Restart mechanics + smoke checklist: run_auto_hint_qwen3_8b_base_async_restart.sh
# (the first restart wrapper) + devlog 2026-08-04 -- in short: on a wrong answer
# the loop ENDS the attempt and starts a FRESH single-turn rollout whose prompt =
# problem + pool-worded verified-progress recap + the next hint (pool wording);
# with HPRL_RESTART_TRAIN_SEGMENTS=all EVERY turn of a chain becomes a training
# row (A = r + V[se+1] - V[ss] each, phantom fail records keep the value fit
# chain-exact); applied_hints / GRPO groups / ratchet semantics unchanged.
#
# SYNC-specific notes (first sync restart wrapper):
#   1. Segment-row expansion needs the archived SAMPLING logprobs as the
#      synthesized rows' old_log_probs anchor -> HPRL_CALC_LOGPROBS=true below
#      (rollout.calculate_log_probs; the trainer guard skips expansion without
#      it). Final rows keep their recomputed anchors; segment rows use the
#      sampling ones (mixed numerics = the known vllm-vs-fsdp gap, the anchor
#      quality bypass-mode async runs with everywhere).
#   2. Budget-grouped sampling stays ON (sync-only feature): same-budget batches
#      give uniform max chain depth -- a natural fit for restart chains.
#   3. k-pack + restart is UNTESTED -> guarded off below (hard error if forced).
#
# Launch (OpenAI selector mode, on every pod):
#   bash launch_hprl_cluster_openai_restart_olmo.sh
# or  TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Olmo-3-7B-Instruct (as the parent wrapper) --------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct"}

# --- RESTART DELTA 1: the agent-loop registry swap ----------------------------
# auto_hint rows -> RestartHintAgentLoop (same parquet, no dataset rebuild).
export AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config_restart.yaml"}

# --- RESTART DELTA 2: context budget ------------------------------------------
# Prompt 2048->4096 for the recap+hints block (an over-long restart prompt is
# refused by the loop: chain ends, hint not charged, restart_prompt_overflow).
# Response budget 32768->8192: per-segment generation is bounded by the parent's
# HPRL_MAX_TURN_TOKENS=4096 anyway (rows are ONE segment now, no cross-turn
# accumulation), so 8192 = the cap with 2x headroom that only VALIDATION uses
# (bare single-turn val gets the full response room; the parent's val ran at
# 32768 -- mind the comparability of val truncation). Keep response >= cap.
export max_prompt_length=${max_prompt_length:-4096}
export max_response_length=${max_response_length:-8192}

# --- RESTART DELTA 3: sampling logprobs for the segment-row anchor (note 1) ----
export HPRL_CALC_LOGPROBS=${HPRL_CALC_LOGPROBS:-true}

# --- auto-hint mechanism knobs (identical to the parent wrapper) ---------------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}
# POOL WORDING (restart-only; forwarded via the sync script's HPRL_RESTART_ENV):
# recap + delivered hint use the hint set's ORIGINAL sentences. =false ->
# selector wording (multi-turn parity arm).
export HPRL_RESTART_POOL_WORDING=${HPRL_RESTART_POOL_WORDING:-true}
# TRAIN SEGMENTS = all: every turn of a chain trains on its own expanded row.
# =final -> v0 final-segment-only rows.
export HPRL_RESTART_TRAIN_SEGMENTS=${HPRL_RESTART_TRAIN_SEGMENTS:-all}

# --- train parquet + resume: identical to the parent wrapper -------------------
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-512-auto-hint.parquet"}
# Fresh run by default; a .../global_step_N dir resumes weights+optimizer+
# dataloader (TRAIN_FILE must match the ckpt's dataset).
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# --- validation: the BARE eval sets, identical to the parent wrapper -----------
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: identical to the parent wrapper -----------------------------------
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage: identical to the parent wrapper (restart-compatible
#     via phantom fail records; all-turn rows via the expansion) ---------------
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}
# Absolute -P_over on truncation tails inside ZEROED (no-correct) groups
# (2026-08-04 flipped-price finding); =false restores scored-groups-only.
export HPRL_OVERLONG_ZEROED=${HPRL_OVERLONG_ZEROED:-true}

# --- actor LR schedule: CONSTANT (parent parity) -------------------------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-SEGMENT generation-length cap (was per-turn): parent parity -----------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-4096}

# --- k-pack: UNTESTED with restart chains -> hard-guarded off (note 3) ---------
if [ "${HPRL_KPACK_ENABLE:-false}" = "true" ]; then
    echo "[run_auto_hint_olmo_restart] FATAL: HPRL_KPACK_ENABLE=true is untested with" \
         "restart (segment-chain) mode -- probe packs + chain expansion have unverified" \
         "grouping interplay. Run the multi-turn wrapper for k-pack." >&2
    exit 1
fi
export HPRL_KPACK_ENABLE=false

# --- budget-grouped sampling ON (sync-only; parent parity, note 2) -------------
export HPRL_BUDGET_SAMPLING=${HPRL_BUDGET_SAMPLING:-true}
export HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER=${HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER:-true}

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (parent parity) ----------------------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- RESTART DELTA 4: distinct run labels --------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-restart'}
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-restart-dapo-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_olmo_restart] TERMINATE-AND-RESTART auto-hint SYNC mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_olmo_restart]   agent registry: ${AGENT_LOOP_CONFIG_PATH} (auto_hint -> RestartHintAgentLoop)"
echo "[run_auto_hint_olmo_restart]   context: prompt=${max_prompt_length} response=${max_response_length} per-segment cap=${HPRL_MAX_TURN_TOKENS} (live context <= $((max_prompt_length + HPRL_MAX_TURN_TOKENS)) tok)  calc_logprobs=${HPRL_CALC_LOGPROBS}"
echo "[run_auto_hint_olmo_restart]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_olmo_restart]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_olmo_restart]   strategy=${HINT_STRATEGY} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE} budget_sampling=${HPRL_BUDGET_SAMPLING} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} pool_wording=${HPRL_RESTART_POOL_WORDING} train_segments=${HPRL_RESTART_TRAIN_SEGMENTS}"
echo "[run_auto_hint_olmo_restart]   step_adv=${HPRL_STEP_ADV} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} norm=${HPRL_STEP_ADV_NORM} overlong=${HPRL_OVERLONG_PENALTY}(${HPRL_OVERLONG_PENALTY_TYPE}) overlong_zeroed=${HPRL_OVERLONG_ZEROED}"
exec bash "${SCRIPT_DIR}/run_hprl_qwen2.5_7b.sh" "$@"
