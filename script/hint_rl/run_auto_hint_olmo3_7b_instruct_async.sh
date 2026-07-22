#!/usr/bin/env bash
#
# AUTO-HINT (push-hint) HPRL run for Olmo-3-7B-Instruct -- FULLY-ASYNC edition.
#
# The async sibling of run_auto_hint_olmo3_7b_instruct.sh: every auto-hint
# science knob below is IDENTICAL to that script (same model, data, reward
# terms, step-level advantage config, ratchet policy); what changes is the
# training architecture underneath -- run_hprl_async.sh splits the training
# pods into a streaming ROLLOUT pool and a TRAINER pool that overlap in time
# (verl fully_async_policy), instead of the colocated synchronous trainer whose
# every step waited on its slowest rollout (gen was 71.6% of step time, with a
# 3.6x slowest/mean rollout tail -- run 20260705-095232).
#
# Cluster launch (5-node example: 4 training pods + selector pods unchanged):
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster.sh     # on every pod
# or edit launch_hprl_cluster.sh's TRAIN_SCRIPT default. The Ray cluster spans
# the SAME training pods as before; the trainer/rollout split happens INSIDE
# that cluster via placement groups (default 2 trainer : 2 rollout nodes --
# override with TRAINER_NNODES / ROLLOUT_NNODES).
#
# Async knobs (see run_hprl_async.sh header for the full semantics):
#   STALENESS_THRESHOLD=0.5   bounded off-policyness; 0 = synchronous streaming
#   TRIGGER_PARAMETER_SYNC_STEP=1, REQUIRE_BATCHES=1
#                             64 prompts per weight sync == one sync-job step,
#                             so test/save freq + epochs keep their meanings
#   PARTIAL_ROLLOUT=True      weight sync interrupts + resumes in-flight
#                             rollouts (kills the long-tail wait)
# Watch fully_async/rollouter/idle_ratio vs fully_async/trainer/idle_ratio in
# wandb to rebalance the node split.
#
# NOT available in async mode: HPRL_KPACK_ENABLE (hard error) and
# HPRL_BUDGET_SAMPLING (forced off -- streaming rollout has no generation batch
# to budget-group; the queue absorbs the duration variance that sampler fixed).
#
# Build the train parquet first (same as sync):
#   python prepare_hint_data.py --mode auto_hint \
#       --in   dataset/dapo-3164-hint-verl.parquet \
#       --out  dataset/dapo-3139-auto-hint.parquet
#
# Then:  bash run_auto_hint_olmo3_7b_instruct_async.sh
# Override any knob via env, e.g.  STALENESS_THRESHOLD=0.3 bash run_auto_hint_...sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Olmo-3-7B-Instruct ------------------------------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct-SFT"}

# --- async pipeline knobs (the only genuinely new block vs the sync wrapper) --
export TRAINER_NNODES=${TRAINER_NNODES:-2}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}

# --- turn on the auto-hint mechanism (identical to the sync wrapper) ---------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Prune X.0 step-guidance hints from the pool before the selector sees it (eval/train
# parity with the offline selector eval). On by default, as in the sync wrapper.
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}

# --- train parquet: same default as the sync wrapper (dapo-512 with hint-wise
#     calibrated initial budgets; see run_auto_hint_olmo3_7b_instruct.sh for the
#     dapo-512 <-> dapo-3139 tradeoff and the resume coupling). NOTE: a resumed
#     ASYNC run restores the trainer/dataloader state but loses queue-in-flight
#     samples (upstream limitation logged by the Rollouter at load). ---
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-512-auto-hint.parquet"}

# --- resume from a prior checkpoint (a .../global_step_N dir); empty = fresh.
#     N counts PARAM VERSIONS, which correspond 1:1 to sync global steps at the
#     default TRIGGER/REQUIRE settings. Sync and async checkpoints share the
#     actor layout (actor/ + data.pt), but the dataloader state is NOT
#     interchangeable across modes (sync steps 64-prompt batches, async streams
#     single samples) -- when forking a SYNC ckpt, expect the dataloader restore
#     to warn and restart the epoch stream. ---
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# --- validation: the BARE (plain-prompt, no agent_name) eval sets, matched to
#     auto-hint training. Runs on the ROLLOUT pool every trainer.test_freq param
#     versions. ---
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: per-hint penalty; the <hint_call/>-specific terms disabled -------
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
# Free X.0 guidance hint (weight dropped, step penalty borne by the substeps).
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage (identical to the sync wrapper) ---------------------
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- PER-TURN generation-length cap (identical to the sync wrapper) -----------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-8192}

# --- k-pack: UNSUPPORTED in async mode; keep the flag pinned false (the base
#     script exits loudly if someone exports true). ---
export HPRL_KPACK_ENABLE=false

# --- budget-grouped sampling: NOT NEEDED in async mode -- the sync wrapper
#     enabled it to stop high-budget problems straggling a generation BATCH,
#     but the async rollout pool streams samples continuously (no batch, no
#     straggler barrier). The base script forces it off; pinned here for
#     clarity. ---
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (identical to the sync wrapper) -----
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- distinct run labels in wandb ---------------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async'}
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async-dapo-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_async] auto-hint FULLY-ASYNC mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_async]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_async]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_async]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} require=${REQUIRE_BATCHES} partial_rollout=${PARTIAL_ROLLOUT}"
echo "[run_auto_hint_async]   strategy=${HINT_STRATEGY} call_reward=${HINT_CALL_REWARD} shape=${HINT_SHAPE_COEFF} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE}"
echo "[run_auto_hint_async]   step_adv=${HPRL_STEP_ADV} step_adv_scale=${HPRL_STEP_ADV_SCALE} step_adv_norm=${HPRL_STEP_ADV_NORM} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} overlong_penalty=${HPRL_OVERLONG_PENALTY} guidance_free=${HINT_GUIDANCE_FREE} prune_guidance=${HPRL_PRUNE_GUIDANCE} max_turn_tokens=${HPRL_MAX_TURN_TOKENS}"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
