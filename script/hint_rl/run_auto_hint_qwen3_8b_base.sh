#!/usr/bin/env bash
#
# AUTO-HINT (push-hint) HPRL run for Qwen3-8B-Base.
#
# Copy of run_auto_hint_olmo3_7b_instruct.sh with the base model swapped to
# Qwen3-8B-Base; every auto-hint science knob below is IDENTICAL to that script
# (same data, reward terms, step-level advantage config, ratchet policy, wandb
# label scheme). Qwen3-8B-Base is a BASE model (no instruct tuning) and the
# first base model actually run through this pipeline (the olmo3-1025 wrapper
# was never exercised), which forces two mechanical deltas:
#
#   1. Turn termination (eos). The pristine checkpoint's generation_config.json
#      stops only on <|endoftext|> (151643), but its chat template closes every
#      turn with <|im_end|> (151645). vLLM builds each request's stop set from
#      the generation_config eos ids (v1 input_processor -> SamplingParams.
#      update_from_generation_config; verl's agent loop and the single-turn val
#      loop pass no stop_token_ids of their own), so a base model that emits
#      <|im_end|> would sail through it and keep generating. Every turn would
#      then hit the HPRL_MAX_TURN_TOKENS cap, which the auto-hint loop scores as
#      a FAILED rollout (length_truncated -> acc=0), and every val completion
#      would run to the full response budget: dead on arrival. Fix: this script
#      idempotently builds ${BASE_HOME}/model/Qwen3-8B-Base-hprl -- per-file
#      symlinks into the pristine dir plus ONE real file, a patched
#      generation_config.json with eos_token_id [151643, 151645] (exactly how
#      Qwen ships its instruct/reasoning siblings) -- and points MODEL_PATH at
#      it. The pristine checkpoint dir is never modified. vLLM keeps the
#      finishing <|im_end|> in the returned token ids, so it is trained on, and
#      saved/merged ckpts inherit the fixed generation_config.
#
#   2. Context length. Qwen3-8B-Base is 32768-native (no rope scaling), while
#      the Olmo3 models run 65536 (yarn) -- the main script's old hardcoded
#      max_response_length=32768 (+2048 prompt) came from the olmo era. verl's
#      async server pins vLLM max_model_len to max_position_embeddings and
#      clamps every request to it, so on this model a 32768 response budget
#      carries a phantom 2048 tokens that can never be generated (and skews the
#      dynamic-bsz token-budget math sized from it). This wrapper exports
#      max_response_length=30720 so prompt+response = exactly 32768.
#
# NOTE: the main script (run_hprl_qwen2.5_7b.sh) hardcodes
#   +actor_rollout_ref.model.override_config.{embd_pdrop,resid_pdrop}=0.
# which Qwen3's HF config does not define -- same situation as Olmo3: verl sets
# them as harmless ignored extras; only attention_dropout=0. is consumed (and
# Qwen3 already ships attention_dropout=0.0). No fork of the main script needed.
#
# In this mode the policy runs the ORDINARY single-turn math prompt (it is never
# told about hints); the rollout loop (auto_hint_agent_loop.AutoHintAgentLoop,
# routed by the train parquet's agent_name="auto_hint") grades each answer and, if
# wrong, injects the next selector hint (multi-round Template F) as a user message,
# up to the per-problem budget B_q.
#
# Build the train parquet first (plain prompt + hint pool + difficulty pool):
#   python prepare_hint_data.py --mode auto_hint \
#       --in   dataset/dapo-3164-hint-verl.parquet \
#       --out  dataset/dapo-3139-auto-hint.parquet
#
# Then:  bash run_auto_hint_qwen3_8b_base.sh
# Override any knob via env, e.g.  HPRL_MAX_TURN_TOKENS=8192 bash run_auto_hint_...sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Qwen3-8B-Base via the derived -hprl dir (see header pt. 1) ---
# Skipped entirely when MODEL_PATH is already set (you then own the eos fix).
if [ -z "${MODEL_PATH:-}" ]; then
    QWEN3_SRC_DIR=${QWEN3_SRC_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base"}
    QWEN3_HPRL_DIR=${QWEN3_HPRL_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base-hprl"}
    if [ ! -f "${QWEN3_SRC_DIR}/config.json" ]; then
        echo "[run_auto_hint_qwen3] ERROR: pristine model dir not found: ${QWEN3_SRC_DIR}" >&2
        exit 1
    fi
    mkdir -p "${QWEN3_HPRL_DIR}"
    # Symlink every pristine file except generation_config.json; re-run safe
    # (links already pointing at the right target are left untouched), and a
    # file added upstream later gets linked on the next launch.
    for f in "${QWEN3_SRC_DIR}"/*; do
        b="$(basename "${f}")"
        if [ "${b}" = "generation_config.json" ]; then
            continue
        fi
        if [ "$(readlink "${QWEN3_HPRL_DIR}/${b}" 2>/dev/null || true)" != "${f}" ]; then
            ln -sfn "${f}" "${QWEN3_HPRL_DIR}/${b}"
        fi
    done
    # Pristine file + eos_token_id widened to [<|endoftext|>, <|im_end|>].
    cat > "${QWEN3_HPRL_DIR}/generation_config.json.tmp" <<'EOF'
{
  "bos_token_id": 151643,
  "do_sample": false,
  "eos_token_id": [151643, 151645],
  "max_new_tokens": 2048,
  "transformers_version": "4.37.0"
}
EOF
    mv -f "${QWEN3_HPRL_DIR}/generation_config.json.tmp" "${QWEN3_HPRL_DIR}/generation_config.json"
    export MODEL_PATH="${QWEN3_HPRL_DIR}"
fi

# --- context budget: fit the model's native 32768 (see header pt. 2) ----------
export max_response_length=${max_response_length:-30720}

# --- turn on the auto-hint mechanism (identical to the olmo wrapper) ----------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Optional cumulative "steps done correctly" prefix before each new hint.
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-false}
# Prune X.0 step-guidance hints from the pool before the selector sees it (eval/train
# parity with the offline selector eval). On by default, as in the olmo wrapper.
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}

# --- the plain-prompt + hint-pool train parquet (prepare_hint_data --mode auto_hint),
#     with INITIAL budgets re-seeded from budget_calibration/budget_state_hint_wise.json
#     (budget_calibration/apply_budget_state.py). NOTE: those calibrated budgets were
#     measured on the Qwen2.5/Olmo3-era runs -- for Qwen3-8B-Base they are only initial
#     SEEDS; the adaptive ratchet below re-fits them online (raise-only by default, so
#     an under-seeded hard problem ratchets UP on its first zero-correct step). Use
#     dataset/dapo-3139-auto-hint.parquet for the full pool with (#major-steps) baked
#     budgets. ---
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-rl-zero-10324-auto-hint.parquet"}

# --- resume from a prior checkpoint (a .../global_step_N dir); empty = fresh (the
#     default -- there are no prior Qwen3-8B-Base ckpts to fork). When set, TRAIN_FILE
#     must match the ckpt's dataset (the resumed dataloader data.pt requires it). ---
# export RESUME_FROM_PATH=${RESUME_FROM_PATH:-"${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen3-8B-Base/HPRL-AutoHint-Qwen3-8B-Base-dapo-20260720-235159/global_step_320"}
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# --- validation: the BARE (plain-prompt, no agent_name) eval sets, so eval is
#     prompt-matched to auto-hint training (single-turn, unaided) -- NOT the
#     <hint_call/>-template *-hint-mt eval files. ---
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: per-hint penalty; DISABLE the <hint_call/>-specific terms --------
# This mode emits no <hint_call/>, so the anti-suppression hint-call bonus and the
# effort-shape penalty have nothing to act on -> off. The no-hint (pool-exhausted)
# penalty is also off: the LOOP, not the policy, decides when no hint is available.
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
# Free X.0 guidance hint (weight dropped, step penalty borne by the substeps).
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage (identical to the olmo wrapper) ---------------------
# Replaces GRPO's scalar-per-rollout advantage with the value-based per-segment
# one (step_advantage.py); the verified-prefix gradient mask is skipped when on.
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
# GRPO-style per-group normalization: adv std -> HPRL_STEP_ADV_SCALE (1.0 = unit).
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
# Absolute over-turn-length penalty, applied BEFORE the per-group normalize so it
# survives group co-truncation; ~0.1 lands near unit scale after normalize.
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
# value = fold P_over into the value recursion (non-truncated rollouts get the
# positive lift); post_hoc = subtract from the truncation tail after the advantage.
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
# WHOLE-TURN advantage: one boundary-free A = V(s_end) + hint_penalty - V(s_start)
# per turn instead of the verified-prefix a_C / failed-tail a_I split.
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (no decay), as in the olmo wrapper -----------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-TURN generation-length cap (identical to the olmo wrapper) -----------
# Each assistant turn capped at min(this, remaining global room); a cap-hit turn is
# scored as failing its next hint step and the rollout terminates. For THIS model the
# cap is also the backstop against base-model rambling; the eos fix above is what
# keeps ordinary turns from ever needing it.
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-8192}

# --- k-pack OFF (as in the olmo wrapper); set true to combine with the probe --
export HPRL_KPACK_ENABLE=${HPRL_KPACK_ENABLE:-false}

# --- budget-grouped data sampling ON: pack same-budget problems per step so the
#     high-budget (many-injection) problems don't straggle the generation batch.
export HPRL_BUDGET_SAMPLING=${HPRL_BUDGET_SAMPLING:-true}
export HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER=${HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER:-true}

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (identical to the olmo wrapper) -----
# Raise B_q by 1 on zero-correct steps; decreases are vetoed (held). For a fresh
# base model this doubles as online budget calibration for the stale seeds above.
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- distinct run labels in wandb ---------------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Qwen3-8B-Base'}
export exp_name=${exp_name:-"HPRL-AutoHint-Qwen3-8B-Base-dolci-rl-zero-10324-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint] auto-hint mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint]   base-model deltas: eos=[151643,151645] via derived dir, max_response_length=${max_response_length}"
echo "[run_auto_hint]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint]   strategy=${HINT_STRATEGY} call_reward=${HINT_CALL_REWARD} shape=${HINT_SHAPE_COEFF} kpack=${HPRL_KPACK_ENABLE} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE}"
echo "[run_auto_hint]   step_adv=${HPRL_STEP_ADV} step_adv_scale=${HPRL_STEP_ADV_SCALE} step_adv_norm=${HPRL_STEP_ADV_NORM} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} overlong_penalty=${HPRL_OVERLONG_PENALTY} overlong_type=${HPRL_OVERLONG_PENALTY_TYPE} guidance_free=${HINT_GUIDANCE_FREE} prune_guidance=${HPRL_PRUNE_GUIDANCE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} max_turn_tokens=${HPRL_MAX_TURN_TOKENS} lr_sched=${HPRL_LR_SCHEDULER}"
exec bash "${SCRIPT_DIR}/run_hprl_qwen2.5_7b.sh" "$@"
