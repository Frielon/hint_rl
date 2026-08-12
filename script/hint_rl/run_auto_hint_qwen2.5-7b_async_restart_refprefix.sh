#!/usr/bin/env bash
#
# REFERENCE-PREFIX restart auto-hint HPRL run for Qwen2.5-7B-Instruct --
# the assistant-prefix delivery A/B arm of run_auto_hint_qwen2.5-7b_async_restart_10k.sh.
# Every science knob defaults to that wrapper's value; the deltas are the blocks
# marked "REFPREFIX DELTA".
#
# WHAT CHANGES (HPRL_RESTART_REFERENCE_PREFIX=true; mechanics in
# restart_agent_loop.py's module header): on a wrong answer the loop still ends
# the attempt, asks the selector for the next hint, and restarts -- but instead
# of a fresh prompt carrying "verified progress recap + hint" as USER text, the
# fresh prompt is the BARE original problem and the new segment's ASSISTANT turn
# is pre-filled with the completed steps' reference solutions + the new hint's
# reference (glued in pool order, selector_multi.build_reference_prefix): the
# model CONTINUES a verified partial solution that ends at the step to do next.
# The prefix tokens ride the response region with response_mask=0 -- attended,
# never trained, never painted -- so the turn advantage is computed exactly as
# before and lands only on the model-generated continuation. applied_hints /
# penalties / ratchet / phantom fail records / train_segments=all expansion are
# unchanged (expanded segment rows carry prefix_len and keep the prefix
# untrained). A restart whose hint has no reference, or whose prefix would not
# leave a full per-segment cap of generation room, FALLS BACK to the
# user-message delivery for that round (restart_ref_fallback counts it).
#
# DATA: needs the hint_reference parquet -- the 492-problem dolci set re-emitted
# with a per-substep ``reference`` (the reference solution's prose for exactly
# that substep). The 9517 zero set has no references, hence this separate
# wrapper. HintBudgetDataset copies extra_info.hint_reference into
# create_kwargs at sample time when the env flag is on (env reaches the rollout
# pool via run_hprl_async.sh's HPRL_RESTART_ENV runtime-env block).
#
# CONTEXT BUDGET: the prefix consumes RESPONSE-length budget (measured over the
# 492 set: full-pool glue p50 ~620 / p99 ~1330 / max ~1650 tokens), so
# max_response_length = 6144 = per-segment cap 4096 + 2048 prefix headroom --
# room after the largest real prefix stays >= the cap, keeping the per_turn
# length-cut classification intact (classify_length_cut flips to the neutral
# "global" label once room < max_turn_tokens). Prompt budget is unchanged: the
# fresh prompt is the BARE problem (no recap block), and the fallback path's
# recap prompt fits 4096 exactly as in the parent wrapper.
#
# Cluster launch (OpenAI selector mode):
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Qwen2.5-7B-Instruct, as-is (parent parity) --------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix] ERROR: model dir not found: ${MODEL_PATH}" >&2
    exit 1
fi

# --- the agent-loop registry swap (parent parity) ------------------------------
export AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config_restart.yaml"}

# --- REFPREFIX DELTA 1: context budget (header "CONTEXT BUDGET") ---------------
export max_prompt_length=${max_prompt_length:-4096}
export max_response_length=${max_response_length:-6144}

# --- async pipeline + trainer-throughput knobs (parent parity; microbatch token
#     budget = MULT*(4096+6144)/SP = 10240/GPU at the defaults, well inside the
#     parent's proven 36864 envelope) ------------------------------------------
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-3} - TRAINER_NNODES ))}
export SP_SIZE=${SP_SIZE:-2}
export PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-2}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}

# --- auto-hint mechanism knobs (parent parity) ---------------------------------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}
export HPRL_RESTART_POOL_WORDING=${HPRL_RESTART_POOL_WORDING:-true}
export HPRL_RESTART_TRAIN_SEGMENTS=${HPRL_RESTART_TRAIN_SEGMENTS:-all}

# --- REFPREFIX DELTA 2: the assistant-prefix delivery (header) -----------------
export HPRL_RESTART_REFERENCE_PREFIX=${HPRL_RESTART_REFERENCE_PREFIX:-true}

# --- REFPREFIX DELTA 3: the hint_reference parquet (header "DATA") -------------
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8-hint-reference.parquet"}
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix] ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
    exit 1
fi
# --- trainer resume: empty default = fresh run (export RESUME_FROM_PATH=... to
#     branch from a ckpt; ${VAR-...} on purpose so an exported empty opts out) --
export RESUME_FROM_PATH=${RESUME_FROM_PATH-}
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
    echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix] ERROR: RESUME_FROM_PATH has no actor/ ckpt: ${RESUME_FROM_PATH}" >&2
    exit 1
fi

# --- validation: identical to the parent wrapper -------------------------------
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

# --- STEP-LEVEL advantage: identical to the parent wrapper ---------------------
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.03}
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (parent parity) -------------------------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-SEGMENT generation-length cap: parent parity (the prefix rides the
#     response headroom ABOVE this cap, never inside it) ------------------------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-4096}

# --- k-pack / budget-grouped sampling: async-unsupported, pinned off -----------
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (parent parity) ----------------------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- distinct run labels -------------------------------------------------------
export project_name=${project_name:-'HPRL-Qwen2.5-7B-Instruct'}
export exp_name=${exp_name:-"HPRL-Qwen2.5-7B-Instruct-dolci492-refprefix-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix] REFERENCE-PREFIX restart mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   agent registry: ${AGENT_LOOP_CONFIG_PATH} (auto_hint -> RestartHintAgentLoop)"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   reference_prefix=${HPRL_RESTART_REFERENCE_PREFIX} (untrained assistant prefix; fallback -> user-message restart)"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   context: prompt=${max_prompt_length} response=${max_response_length} per-segment cap=${HPRL_MAX_TURN_TOKENS} (prefix headroom $((max_response_length - HPRL_MAX_TURN_TOKENS)) tok; measured max glue ~1650)"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  sp=${SP_SIZE} ppo_token_mult=${PPO_TOKEN_MULT}  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS}"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   strategy=${HINT_STRATEGY} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE} pool_wording=${HPRL_RESTART_POOL_WORDING} train_segments=${HPRL_RESTART_TRAIN_SEGMENTS}"
echo "[run_auto_hint_qwen2.5-7b_async_restart_refprefix]   step_adv=${HPRL_STEP_ADV} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} norm=${HPRL_STEP_ADV_NORM} overlong=${HPRL_OVERLONG_PENALTY}(${HPRL_OVERLONG_PENALTY_TYPE})"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
