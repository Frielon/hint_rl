#!/usr/bin/env bash
#
# AUTO-HINT (push-hint) HPRL run for Olmo-3-7B-Instruct-DPO -- FULLY-ASYNC
# edition, on the hinted Dolci-Instruct-RL set.
#
# Cloned from run_auto_hint_qwen3_8b_base_async.sh (same async architecture,
# reward terms, step-level advantage config incl. its value-routed overlong
# penalty, ratchet policy); the deltas are the model, the train parquet and the
# length budgets:
#
#   1. model -- ${BASE_HOME}/model/Olmo-3-7B-Instruct-DPO, used AS-IS (no
#      derived -hprl dir): being an instruct checkpoint it already closes turns
#      properly -- generation_config.json ships eos [100265 <|im_end|>, 100257
#      <|endoftext|>] -- and the chat template lives in chat_template.jinja,
#      which transformers >= 4.51 (verl env: 4.57.6) auto-loads; the agent-loop
#      init that crashed on template-less BASE checkpoints is fine here (the
#      Olmo-3-7B-Instruct-SFT runs use the identical layout).
#   2. context -- 65536 via YaRN (max_position_embeddings), so the requested
#      max_response_length=34768 FITS: prompt(2048)+response = 36816 << 65536.
#      (On the 32768-native qwen3 base the same 34768 was fatal -- vLLM "no
#      room to generate" killed the Rollouter; not a risk on this model.)
#   3. per-turn cap -- HPRL_MAX_TURN_TOKENS=16384 (vs 8192 in the reference):
#      a single turn may spend up to 16384 tokens, the whole rollout up to
#      34768 across turns.
#   4. data -- dataset/dolci-instruct-rl-6762-auto-hint-ex-hard100.parquet:
#      hinted Dolci-Instruct-RL (latest step0/step1 pools, step0 keep-filter,
#      B_q = number of major hint steps, built by pack_dolci_auto_hint.py),
#      with the 25 dapo_sample_hard_100 text-duplicates EXCLUDED -- so the
#      hard-100 val set below is uncontaminated by construction.
#
# Async knob defaults follow the reference script: 1 trainer : N-1 rollout
# nodes, TRIGGER=1/REQUIRE=1 (param version == sync step), partial rollout on,
# and DECOUPLED off-policy correction (ROLLOUT_CORR_BYPASS=False -- bypass
# mode's one-way clip rectifier regrew the bare-'assistant' tic, devlog
# 2026-07-30). Bump STALENESS_THRESHOLD for throughput (s=2 was the 20260722
# sweep's production pick at ~94% of the trainer ceiling).
#
# START / RESUME: default fresh; RESUME_FROM_PATH=.../global_step_N continues
# optimizer+dataloader (world size + TRAIN_FILE must match the ckpt; async
# resume loses queue-in-flight samples). MODEL_PATH=... swaps the checkpoint.
# Budget state: fresh budget_state.json by default (parquet B_q seeds; the
# raise-only ratchet re-fits online); point BUDGET_STATE_PATH at a COPY to
# carry a prior run's budgets.
#
# NOT available in async mode: HPRL_KPACK_ENABLE (hard error in the base
# script) and HPRL_BUDGET_SAMPLING (forced off -- no generation batch to
# budget-group; the queue absorbs the variance).
#
# Cluster launch (OpenAI selector mode -- every pod trains, no selector pods):
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
# or behind the classic selector-pod launcher:
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster.sh
#
# Override any knob via env, e.g.:
#   STALENESS_THRESHOLD=2 TRAINER_NNODES=1 bash run_auto_hint_olmo3_7b_instruct_dpo_async.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Olmo-3-7B-Instruct-DPO, as-is (see header pt. 1) --------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct-SFT"}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_auto_hint_olmo3_dpo_async] ERROR: model dir not found: ${MODEL_PATH}" >&2
    exit 1
fi

# --- context budget: 34768 fits the 65536 YaRN context (header pt. 2) ----------
export max_response_length=${max_response_length:-34768}

# --- async pipeline knobs (identical to the reference wrapper) -----------------
# NNODES is exported by the cluster launcher (total training pods); 5 is the
# base script's default for a bare head-node launch.
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-5} - TRAINER_NNODES ))}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
# DECOUPLED off-policy correction (NOT the stock bypass mode): recompute
# old_log_probs on the trainer as an on-policy proximal anchor + seq-TIS
# weights vs the rollout logprobs. Bypass mode's clip-vs-stale-policy ratio
# is a one-way rectifier for rare suppressed tokens -- the 'assistant'-tic
# resurgence that collapsed the 0723/0724/0729 dolci async runs (devlog
# 2026-07-30). Watch rollout_corr/* + actor/entropy (both absent under
# bypass) and the bare-'assistant' rollout fraction.
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}

# --- turn on the auto-hint mechanism (identical to the reference wrapper) ------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
# Prune X.0 step-guidance hints from the pool before the selector sees it
# (eval/train parity with the offline selector eval).
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}

# --- train parquet: hinted Dolci-Instruct-RL, hard-100 overlap excluded
#     (header pt. 4). B_q seeds = per-problem major-step counts; the raise-only
#     ratchet re-fits online. ---
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-dapo-2180-auto-hint.parquet"}
# --- trainer resume: OFF by default = fresh run (resume_mode=auto still picks
#     up THIS exp's own latest ckpt if the same exp_name is relaunched). Before
#     pointing it at a .../global_step_N dir, see the header's world-size /
#     dataset constraints -- or merge the ckpt and pass MODEL_PATH instead. ---
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# --- validation: the BARE (plain-prompt, no agent_name) eval sets, matched to
#     auto-hint training. Runs on the ROLLOUT pool every trainer.test_freq
#     param versions. hard-100 is train-disjoint by construction (header pt. 4).
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: per-hint penalty; the <hint_call/>-specific terms disabled --------
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
# Free X.0 guidance hint (weight dropped, step penalty borne by the substeps).
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage (identical to the reference wrapper) -----------------
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
# value = fold P_over into the value recursion (the qwen3/sync routing).
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (no decay), as in the reference wrapper -------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-TURN generation-length cap: 16384 (header pt. 3) ----------------------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-16384}

# --- k-pack / budget-grouped sampling: architecture-inherently sync-only;
#     pinned off (the base script errors/forces respectively). ---
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (identical to the reference wrapper) -
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- distinct run labels in wandb ----------------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-sft-async'}
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-sft-async-dolci-instruct-2180-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
# export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-DPO-async-dolci-instruct-6762-20260731-010034"}

echo "[run_auto_hint_olmo3_dpo_async] auto-hint FULLY-ASYNC mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_olmo3_dpo_async]   context: max_response_length=${max_response_length} (65536 YaRN) max_turn_tokens=${HPRL_MAX_TURN_TOKENS}"
echo "[run_auto_hint_olmo3_dpo_async]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_olmo3_dpo_async]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_olmo3_dpo_async]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} require=${REQUIRE_BATCHES} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS}"
echo "[run_auto_hint_olmo3_dpo_async]   strategy=${HINT_STRATEGY} call_reward=${HINT_CALL_REWARD} shape=${HINT_SHAPE_COEFF} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE}"
echo "[run_auto_hint_olmo3_dpo_async]   step_adv=${HPRL_STEP_ADV} step_adv_scale=${HPRL_STEP_ADV_SCALE} step_adv_norm=${HPRL_STEP_ADV_NORM} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} overlong_penalty=${HPRL_OVERLONG_PENALTY} overlong_type=${HPRL_OVERLONG_PENALTY_TYPE} guidance_free=${HINT_GUIDANCE_FREE} prune_guidance=${HPRL_PRUNE_GUIDANCE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} lr_sched=${HPRL_LR_SCHEDULER}"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
