#!/usr/bin/env bash
#
# TERMINATE-AND-RESTART (segment-chain) auto-hint HPRL run for
# Qwen3-4B-Instruct-2507 -- FULLY-ASYNC edition, on the hinted
# Dolci-Instruct-RL set. The restart sibling of
# run_auto_hint_qwen3_4b_instruct_2507_async.sh (multi-turn parent): every
# science knob defaults to that wrapper's value; the restart deltas are the
# blocks marked "RESTART DELTA". Restart mechanics, smoke checklist, and the
# knob docs live in run_auto_hint_qwen3_8b_base_async_restart.sh (the first
# restart wrapper) + devlog 2026-08-04 -- in short: on a wrong answer the loop
# ENDS the attempt and starts a FRESH single-turn rollout whose prompt =
# problem + "You have attempted this problem before and you made the following
# verified progress, which is correct: ..." (pool-worded recap) + the next hint
# (pool wording); the trained row is the chain's FINAL segment; applied_hints /
# GRPO groups / ratchet semantics unchanged; step_adv+whole_turn stay ON
# (phantom fail records keep the value fit chain-exact).
#
# Qwen3-4B-Instruct-2507 model deltas (vs the qwen3-8b-BASE restart wrapper):
#   1. model -- ${BASE_HOME}/model/Qwen3-4B-Instruct-2507 AS-IS: instruct
#      checkpoint with proper eos [151645, 151643] in generation_config.json and
#      the chat template in tokenizer_config.json -- no derived -hprl dir build.
#   2. context -- 262144 NATIVE, no model ceiling to duck: prompt 2048->4096
#      (recap+hints headroom); max_response_length = per-segment cap = 32768
#      (raised from 16384 on 2026-08-05 -- the 16384 run 233932's truncation had
#      fallen to 0-1% with final p99 ~12.7k tok, so the raise is cheap today;
#      but final p50 grew 4.3k->7.6k tok over 125 steps, so expect the headroom
#      to be consumed as length inflation continues). Live per-request context
#      <= 4096 + 32768 = 36864 tokens. Keep the two knobs EQUAL: cap >
#      response_length shrinks the room; cap < response_length wastes width.
#   3. per-segment cap -- HPRL_MAX_TURN_TOKENS=32768 (== max_response_length).
#   4. data -- dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8.parquet, the
#      parent's pinned subset (hard-100 text-duplicates excluded upstream).
#
# START / RESUME: default = the multi-turn parent run 20260802-000856's
# global_step_100 ckpt -- a FIXED mid-run branch point (the parent already has
# steps 100..150+, so the restart arm overlaps existing multi-turn steps
# immediately: the matched-start A/B). World size fits (parent trained on
# TRAINER_NNODES=1 x 8 = world 8; same here) and TRAIN_FILE is the ckpt's own
# parquet -- keep both in sync if you override either. export RESUME_FROM_PATH=
# (empty) for a fresh run; budget_state is fresh either way (ratchet re-fits).
#
# Cluster launch (OpenAI selector mode):
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
# (launch_hprl_cluster_openai_async_restart.sh defaults to the QWEN3-8B-BASE
#  restart wrapper -- point TRAIN_SCRIPT here for the 4B run.)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Qwen3-4B-Instruct-2507, as-is (header pt. 1) ------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_auto_hint_qwen2.5-7b_async_restart] ERROR: model dir not found: ${MODEL_PATH}" >&2
    exit 1
fi

# --- RESTART DELTA 1: the agent-loop registry swap ----------------------------
# auto_hint rows -> RestartHintAgentLoop (same parquet, no dataset rebuild).
export AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config_restart.yaml"}

# --- RESTART DELTA 2: context budget (header pt. 2) ---------------------------
# Prompt budget doubles for the recap+hints block; a restart prompt that would
# exceed it is refused by the loop (chain ends, hint not charged,
# restart_prompt_overflow counter). Response budget = parent parity (no model
# ceiling to duck at 262144 native).
export max_prompt_length=${max_prompt_length:-4096}
export max_response_length=${max_response_length:-4096}

# --- async pipeline + traner-throughput knobs ---------------------------------
# Run 214953 (1:4 split, sp=4, MULT=2) was TRAINER-bound at ~4 versions/hour:
# 57-group queue backlog, ~1s collect wait, 13-17 min/step -- segment-row
# training triples the trained tokens/step (~10M at step 3) and one trainer
# node at dp=2 with a 10k-token/rank microbatch budget couldn't keep up (devlog
# 2026-08-04). Defaults below = the throughput fix at UNCHANGED lengths, 5-node
# cluster: 2 trainer : 3 rollout nodes, Ulysses sp 4->2 (a 4B at <=20480-token
# rows doesn't need 4; trainer dp 2->8), microbatch token budget x2. Expected
# ~3.5-4x -> ~14-16 vph. Divisibility holds: 64 prompts x n8 = 512 rows, dp=8.
# NOTE: ckpts saved under this split have actor world 16 (2x8) -- resuming THEM
# later needs TRAINER_NNODES=2; the multi-turn parent's ckpts (world 8) need
# TRAINER_NNODES=1 instead.
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-3} - TRAINER_NNODES ))}
export SP_SIZE=${SP_SIZE:-2}
# MULT 4->2 at the 32768 cap: microbatch budget = MULT*(4096+32768)/SP = 36864
# tok/GPU -- the same proven memory envelope as the 16384 run's 40960 (which
# ran MULT=4). MULT=4 here would be 73728/GPU: untested activation/logit memory
# on this cluster. A single max-length row (36864 tok = 18432/rank at sp=2)
# still fits the budget.
export PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-2}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}
# Measured on the 16384 run (233932, 2:3/sp2/mult4, ~10.6 vph): the balance
# FLIPPED to rollout-bound -- mq_len 0, trainer collect waits 150-245s. The
# 2-node trainer is KEPT anyway: its idle headroom is what absorbs the growing
# row lengths at the 32768 cap. Watch after relaunch: mq_len pinned high ->
# trainer-bound again (consider 3:2); mq_len ~0 with growing waits -> still
# rollout-bound (expected; a staleness raise only smooths bursts, no capacity).

# --- auto-hint mechanism knobs (shared with the restart loop) ------------------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Recap in the fresh prompt -- the restart analog of the v2 injection. Keep ON.
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}
# POOL WORDING (restart-only; forwarded via run_hprl_async.sh HPRL_RESTART_ENV):
# recap + delivered hint use the hint set's ORIGINAL sentences, not the
# selector's rephrasings. =false -> selector wording.
export HPRL_RESTART_POOL_WORDING=${HPRL_RESTART_POOL_WORDING:-true}
# TRAIN SEGMENTS = all: EVERY turn of a chain becomes a training row (trainer
# expands archived segments; each turn painted A = r + V[se+1] - V[ss] from the
# group fit, final rows keep the chain-exact phantom fit). =final -> v0
# final-segment-only rows. Requires step_adv + bypass_mode (both on here).
export HPRL_RESTART_TRAIN_SEGMENTS=${HPRL_RESTART_TRAIN_SEGMENTS:-all}
# REFERENCE-PREFIX delivery (restart-only; forwarded via run_hprl_async.sh
# HPRL_RESTART_ENV): =true delivers progress+hint by PREFIXING the completed
# steps' reference solutions + the new hint's reference to the ASSISTANT turn
# (untrained response_mask=0 tokens; the fresh prompt is the bare problem) --
# turn adv unchanged, gradient only on the model's continuation. NEEDS a
# hint_reference parquet (dataset/*-hint-reference.parquet); this wrapper's
# dolci-rl-zero-9517 set has none, so =true here would fall back to the
# user-message restart on every chain. Use the dedicated wrapper instead:
# run_auto_hint_qwen2.5-7b_async_restart_refprefix.sh.
export HPRL_RESTART_REFERENCE_PREFIX=${HPRL_RESTART_REFERENCE_PREFIX:-false}

# --- train parquet: identical to the parent wrapper (header pt. 4) -------------
# export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8.parquet"}
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-rl-zero-9517-auto-hint.parquet"}
# --- trainer resume: DEFAULT = the multi-turn parent's global_step_100 (header
#     "START / RESUME"). ${VAR-...} (not :-) on purpose: export RESUME_FROM_PATH=
#     (empty) to opt into a fresh run. ---
export RESUME_FROM_PATH=${RESUME_FROM_PATH-}
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
    echo "[run_auto_hint_qwen2.5-7b_async_restart] ERROR: RESUME_FROM_PATH has no actor/ ckpt: ${RESUME_FROM_PATH}" >&2
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

# --- STEP-LEVEL advantage: identical to the parent wrapper (restart-compatible
#     via phantom fail records; final-segment rows -- see the 8B restart header)
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.03}
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (parent parity) -------------------------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-SEGMENT generation-length cap (was per-turn): parent parity -----------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-4096}

# --- k-pack / budget-grouped sampling: async-unsupported, pinned off -----------
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (parent parity) ----------------------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- RESTART DELTA 3: distinct run labels --------------------------------------
export project_name=${project_name:-'HPRL-Qwen2.5-7B-Instruct'}
export exp_name=${exp_name:-"HPRL-Qwen2.5-7B-Instruct-dolci-10k-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
# export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async'}
# export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async-dapo-20260805-141447"}

echo "[run_auto_hint_qwen2.5-7b_async_restart] TERMINATE-AND-RESTART auto-hint mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   agent registry: ${AGENT_LOOP_CONFIG_PATH} (auto_hint -> RestartHintAgentLoop)"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   context: prompt=${max_prompt_length} response=${max_response_length} (262144 native) per-segment cap=${HPRL_MAX_TURN_TOKENS} (live context <= $((max_prompt_length + HPRL_MAX_TURN_TOKENS)) tok)"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  sp=${SP_SIZE} ppo_token_mult=${PPO_TOKEN_MULT}  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS}"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   strategy=${HINT_STRATEGY} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} pool_wording=${HPRL_RESTART_POOL_WORDING} train_segments=${HPRL_RESTART_TRAIN_SEGMENTS}"
echo "[run_auto_hint_qwen2.5-7b_async_restart]   step_adv=${HPRL_STEP_ADV} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} norm=${HPRL_STEP_ADV_NORM} overlong=${HPRL_OVERLONG_PENALTY}(${HPRL_OVERLONG_PENALTY_TYPE}) -- final-segment rows, phantom fail records keep the value fit chain-exact"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
