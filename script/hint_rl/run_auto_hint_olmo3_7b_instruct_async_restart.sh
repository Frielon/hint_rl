#!/usr/bin/env bash
#
# TERMINATE-AND-RESTART (segment-chain) auto-hint HPRL run for
# Olmo-3-7B-Instruct -- FULLY-ASYNC edition, on the hinted Dolci-Instruct-RL
# set. Cloned from run_auto_hint_qwen3_4b_instruct_2507_async_restart.sh at
# FULL training-param parity (context budgets, async/throughput split, reward
# terms, step-adv config, restart knobs); the deltas are the model and the run
# labels. Restart mechanics, smoke checklist, and the knob docs live in
# run_auto_hint_qwen3_8b_base_async_restart.sh (the first restart wrapper) +
# devlog 2026-08-04 -- in short: on a wrong answer the loop ENDS the attempt
# and starts a FRESH single-turn rollout whose prompt = problem + "You have
# attempted this problem before and you made the following verified progress,
# which is correct: ..." (pool-worded recap) + the next hint (pool wording);
# with HPRL_RESTART_TRAIN_SEGMENTS=all EVERY turn of a chain trains on its own
# expanded row; applied_hints / GRPO groups / ratchet semantics unchanged;
# step_adv+whole_turn stay ON (phantom fail records keep the value fit
# chain-exact).
#
# Olmo-3-7B-Instruct model deltas (vs the qwen3-4b restart wrapper):
#   1. model -- ${BASE_HOME}/model/Olmo-3-7B-Instruct AS-IS (no derived -hprl
#      dir): instruct checkpoint with proper eos [100265 <|im_end|>, 100257
#      <|endoftext|>] in generation_config.json and the chat template in
#      chat_template.jinja, which transformers >= 4.51 (verl env: 4.57.6)
#      auto-loads -- the same layout the Olmo-3-7B-Instruct-SFT runs train with.
#   2. context -- 65536 via YaRN (max_position_embeddings), NOT the 4B's 262144
#      native: prompt 4096 + response 20480 = live per-request context <= 24576
#      tokens, comfortably under the ceiling. The 4B wrapper's 2026-08-05 raise
#      option (response = per-segment cap = 32768) also fits here (36864 <<
#      65536) -- if you take it, flip max_response_length and
#      HPRL_MAX_TURN_TOKENS TOGETHER (cap > response_length shrinks the room;
#      cap < response_length wastes width) AND drop PPO_TOKEN_MULT back to 2:
#      the microbatch budget scales with the lengths, and MULT=4 at 32768
#      would be 73728 tok/GPU, far past any proven envelope.
#   3. data -- dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8.parquet, the
#      SAME parquet as the 4B restart arm (hard-100 text-duplicates excluded
#      upstream), so this run is data-matched to both the 4B restart run and
#      the GRPO-Olmo-3-7B-dolci-492-async baseline (which trains this model on
#      the set's -single-turn sibling). B_q seeds = per-problem major-step
#      counts (problem-intrinsic); the raise-only ratchet re-fits online.
#      Olmo-SELECTED alternatives exist if you want a difficulty-matched pool
#      instead: dolci-instruct-rl-1983/1773-auto-hint-olmo3-7b-le2of8.parquet.
#
# START / RESUME: default FRESH -- no multi-turn auto-hint parent run exists
# for THIS model (the HPRL-AutoHint-Olmo ckpt projects are all the SFT
# checkpoint), so there is no matched-start branch point like the 4B's
# parent-step-100 default. To branch a later A/B, point RESUME_FROM_PATH at a
# .../global_step_N dir (world size must match: ckpts saved under this 2:3
# split have actor world 16 -> resume with TRAINER_NNODES=2; TRAIN_FILE must
# be the ckpt's own parquet).
#
# Cluster launch (OpenAI selector mode):
#   bash launch_hprl_cluster_openai_async_restart_olmo.sh
# or  TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Olmo-3-7B-Instruct, as-is (header pt. 1) ----------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct"}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_auto_hint_olmo_async_restart] ERROR: model dir not found: ${MODEL_PATH}" >&2
    exit 1
fi

# --- RESTART DELTA 1: the agent-loop registry swap ----------------------------
# auto_hint rows -> RestartHintAgentLoop (same parquet, no dataset rebuild).
export AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config_restart.yaml"}

# --- RESTART DELTA 2: context budget (header pt. 2) ---------------------------
# Prompt budget doubles for the recap+hints block; a restart prompt that would
# exceed it is refused by the loop (chain ends, hint not charged,
# restart_prompt_overflow counter). Live context <= 4096 + 20480 = 24576 <<
# the 65536 YaRN ceiling.
export max_prompt_length=${max_prompt_length:-4096}
export max_response_length=${max_response_length:-20480}

# --- async pipeline + trainer-throughput knobs ---------------------------------
# Base layout = 4B-wrapper parity (the 2026-08-04/05 throughput fixes, devlog):
# segment-row training triples the trained tokens/step, so 2 trainer :
# NNODES-2 rollout nodes (2:4 on the 6-node cluster), Ulysses sp=2 (trainer
# dp=8; 64 prompts x n8 = 512 rows, 512 %% 8 = 0). Throughput knobs RETUNED
# 2026-08-05 from run 204213's first 15 steps (~560 s/step, trainer idle
# 43-58%, rollout-bound; diagnosis in devlog):
#   GPU_MEMORY_UTIL=0.90 (was the shared 0.80 default) -- Olmo3 is MHA (no
#     GQA, ~0.3 GB KV per 1k tok): at 0.80 each of the 32 rollout instances
#     fits only ~20 concurrent ~6k-tok chains, so of the 185 admitted groups
#     only ~65 made progress (Little's law on 64 groups/560 s at ~530 s chain
#     latency) and the trainer starved. vLLM-only pods; KV -> throughput ~1:1
#     while KV-bound.
#   PPO_TOKEN_MULT=4 (was 2) -- microbatch budget MULT*(4096+20480)/sp =
#     49152 tok/GPU. Run 204213 peaked at 28 GB alloc / 79 GB HBM at MULT=2
#     with update_actor 275-295 s: the conservative halving bought nothing.
#     49152 is ~20% past the 4B run 233932's proven 40960 -- if the first
#     fit_step OOMs, step back to 3 (36864) before 2.
#   UPDATE_WEIGHTS_BUCKET_MB=512 (was 1024) -- insurance: 0.90 shrinks the
#     rollout pods' non-vLLM headroom 15.8 -> 7.9 GB and the sync-buffer OOM
#     of grid 20260720-141549 happened at 0.80. Costs a few extra flush
#     round-trips on a ~6 s/step sync. If update_weights STILL OOMs, lower
#     GPU_MEMORY_UTIL to 0.85 before touching anything else.
# STALENESS_THRESHOLD stays 1.9: run 204213 sat pinned at the 185-group cap
# with vLLM already ~1.5-2x oversubscribed -- raising it deepens the stale
# queue (partial spans already 6) without adding throughput; after the 0.90
# bump the cap still has ~35% headroom over the ~100 groups that can run.
# NOTE: ckpts saved under this split have actor world 16 (2x8) -- resuming
# THEM later needs TRAINER_NNODES=2.
export TRAINER_NNODES=${TRAINER_NNODES:-2}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-6} - TRAINER_NNODES ))}
export SP_SIZE=${SP_SIZE:-2}
export PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-4}
export GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.90}
export UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-512}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}
# Selector API concurrency, PER agent-loop worker (8 workers). Run 204213 at
# 16/worker: 11.6 s mean per call but 117 s max = queueing spikes; 32 keeps
# the selector off the chain critical path as rollout throughput rises (429
# backoff already guards the API side).
export SELECTOR_MAX_CONCURRENCY=${SELECTOR_MAX_CONCURRENCY:-32}

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

# --- train parquet: data-matched to the 4B restart arm (header pt. 3) ----------
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8.parquet"}
# --- trainer resume: DEFAULT = fresh (header "START / RESUME"). ${VAR-...}
#     (not :-) on purpose so an explicitly exported empty value stays fresh too.
export RESUME_FROM_PATH=${RESUME_FROM_PATH-}
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
    echo "[run_auto_hint_olmo_async_restart] ERROR: RESUME_FROM_PATH has no actor/ ckpt: ${RESUME_FROM_PATH}" >&2
    exit 1
fi

# --- validation: identical to the 4B restart wrapper ---------------------------
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: identical to the 4B restart wrapper -------------------------------
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage: identical to the 4B restart wrapper (restart-
#     compatible via phantom fail records; all-turn rows via the expansion) -----
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}
# Absolute -P_over on truncation tails inside ZEROED (no-correct) groups -- the
# async default is already true; pinned here because a FRESH 7B's early steps
# are exactly the all-wrong co-truncating regime it exists for (devlog
# 2026-08-05 flipped-price finding). =false restores scored-groups-only.
export HPRL_OVERLONG_ZEROED=${HPRL_OVERLONG_ZEROED:-true}

# --- actor LR schedule: CONSTANT (4B-wrapper parity) ---------------------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-SEGMENT generation-length cap: keep == max_response_length ------------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-20480}

# --- k-pack / budget-grouped sampling: async-unsupported, pinned off -----------
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (4B-wrapper parity) ------------------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- RESTART DELTA 3: distinct run labels --------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-async'}
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-async-restart-dolci-instruct-492-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_olmo_async_restart] TERMINATE-AND-RESTART auto-hint mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_olmo_async_restart]   agent registry: ${AGENT_LOOP_CONFIG_PATH} (auto_hint -> RestartHintAgentLoop)"
echo "[run_auto_hint_olmo_async_restart]   context: prompt=${max_prompt_length} response=${max_response_length} (65536 YaRN) per-segment cap=${HPRL_MAX_TURN_TOKENS} (live context <= $((max_prompt_length + HPRL_MAX_TURN_TOKENS)) tok)"
echo "[run_auto_hint_olmo_async_restart]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_olmo_async_restart]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_olmo_async_restart]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  sp=${SP_SIZE} ppo_token_mult=${PPO_TOKEN_MULT} gpu_util=${GPU_MEMORY_UTIL} sync_bucket=${UPDATE_WEIGHTS_BUCKET_MB}MB selector_conc=${SELECTOR_MAX_CONCURRENCY}/worker  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS}"
echo "[run_auto_hint_olmo_async_restart]   strategy=${HINT_STRATEGY} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} pool_wording=${HPRL_RESTART_POOL_WORDING} train_segments=${HPRL_RESTART_TRAIN_SEGMENTS}"
echo "[run_auto_hint_olmo_async_restart]   step_adv=${HPRL_STEP_ADV} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} norm=${HPRL_STEP_ADV_NORM} overlong=${HPRL_OVERLONG_PENALTY}(${HPRL_OVERLONG_PENALTY_TYPE}) overlong_zeroed=${HPRL_OVERLONG_ZEROED} -- segment rows + phantom fail records keep the value fit chain-exact"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
