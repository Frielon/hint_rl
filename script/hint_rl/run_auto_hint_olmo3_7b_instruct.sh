#!/usr/bin/env bash
#
# AUTO-HINT (push-hint) HPRL run for Olmo-3-7B-Instruct.
#
# Copy of run_auto_hint_qwen2.5_7b.sh with the base model swapped to
# Olmo-3-7B-Instruct (${BASE_HOME}/model/Olmo-3-7B-Instruct, exported as
# MODEL_PATH so the main run_hprl script picks it up) and distinct wandb
# project/exp names. Every auto-hint knob below is identical to the Qwen run.
#
# NOTE: the main script (run_hprl_qwen2.5_7b.sh) hardcodes
#   +actor_rollout_ref.model.override_config.{embd_pdrop,resid_pdrop}=0.
# which Olmo3's HF config does not define. These are harmless (verl just sets
# them as ignored extra attributes on the config); only attention_dropout=0. is
# actually consumed. So no fork of the main script is needed.
#
# In this mode the policy runs the ORDINARY single-turn math prompt (it is never
# told about hints); the rollout loop (auto_hint_agent_loop.AutoHintAgentLoop,
# routed by the train parquet's agent_name="auto_hint") grades each answer and, if
# wrong, injects the next selector hint (multi-round Template F) as a user message,
# up to the per-problem budget B_q. The trainer applies the VERIFIED-PREFIX
# gradient mask (auto_hint_mask): for positive-advantage rollouts only, the loss
# beyond each hinted turn's last selector-verified sentence is dropped.
#
# Build the train parquet first (plain prompt + hint pool + difficulty pool):
#   python prepare_hint_data.py --mode auto_hint \
#       --in   dataset/dapo-3164-hint-verl.parquet \
#       --out  dataset/dapo-3139-auto-hint.parquet
#
# Then:  bash run_auto_hint_olmo3_7b_instruct.sh
# Override any knob via env, e.g.  HPRL_KPACK_ENABLE=true bash run_auto_hint_...sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Olmo-3-7B-Instruct ------------------------------------------
export MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct-SFT"}

# --- turn on the auto-hint mechanism + its verified-prefix gradient mask -----
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Prune X.0 step-guidance hints from the pool before the selector sees it, so the rollout
# presents the SAME substep-only pools the offline selector eval (multi-cite-gpt-eval) was
# scored on (eval/train parity). On here by default; set false for the full pool (X.0 incl.).
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}

# --- the plain-prompt + hint-pool train parquet (prepare_hint_data --mode auto_hint),
#     with INITIAL budgets re-seeded from budget_calibration/budget_state_hint_wise.json
#     (budget_calibration/apply_budget_state.py;
#     problems absent from that file keep their original baked budget). A fresh run reads
#     these as B_q for epoch 1, then the ratchet evolves them. Override TRAIN_FILE back to
#     dataset/dapo-3139-auto-hint.parquet for the ORIGINAL (#major-steps) baked budgets. ---
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-3139-auto-hint.parquet"}

# --- validation: the BARE (plain-prompt, no agent_name) eval sets, so eval is
#     prompt-matched to auto-hint training (single-turn, unaided) -- NOT the
#     <hint_call/>-template *-hint-mt eval files. The main script builds VAL_FILES
#     from TEST_FILE + HARD_TEST_FILE + AIME2025_FILE (aime2024 / dapo-hard / aime2025). ---
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
# add hmmt_nov_2025 (data_source=hmmt_nov_2025, reported as val-core/hmmt_nov_2025/*)
# as a 4th bare val set; override VAL_FILES to change.
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
# Free X.0 guidance hint: true makes every <step_id>.0 guidance hint cost 0 (its weight
# is dropped and the step's penalty is borne by the substeps, so the pool total stays
# HINT_PENALTY_TOTAL). Applies to the per-hint reward AND the step-adv r(h). Default off.
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage (supersedes the verified-prefix mask) ---------------
# HPRL_STEP_ADV=true replaces GRPO's single scalar-per-rollout advantage with the
# value-based per-segment one (step_advantage.py): each turn's selector-verified prefix
# gets a non-negative advantage and its failed tail a non-positive one, with state values
# solved backward over the problem's N rollouts (V[S_K]=1). When ON the verified-prefix
# gradient MASK is skipped (the two are mutually exclusive), and the loop labels the
# terminal turn of an over-budget incorrect rollout (one extra selector call, no hint
# given). The raw advantages are small (~ HINT_PENALTY_TOTAL / #hints, e.g. ~0.06 for a
# 13-hint pool); HPRL_STEP_ADV_SCALE multiplies them to a usable gradient magnitude
# without retuning the LR (try ~5-10). Default OFF -> the mask runs as before.
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
# GRPO-style per-group normalization: divide each group's advantages by their std so the
# (small) raw value-based advantages become ~unit scale ADAPTIVELY -- the fix for a too-small
# gradient. With it ON, HPRL_STEP_ADV_SCALE is the TARGET std (1.0 -> unit). Default ON here
# since step-adv is on; set HPRL_STEP_ADV_NORM=false for the plain adv_scale multiply.
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}

# --- k-pack OFF for the first auto-hint runs (the verified-prefix mask is the new
#     lever); set HPRL_KPACK_ENABLE=true to combine with the counterfactual probe. ---
export HPRL_KPACK_ENABLE=${HPRL_KPACK_ENABLE:-false}

# --- budget-grouped data sampling ON: auto-hint problems span budget 0..~6, so a
#     uniformly-sampled batch lets the high-budget (many-injection) problems straggle
#     and idle the GPUs that finished the budget-0 ones. Grouping same-budget problems
#     per step removes that straggler. Set HPRL_BUDGET_SAMPLING=false for the stock
#     uniform sampler. (Re-affirms the main script's default.) ---
export HPRL_BUDGET_SAMPLING=${HPRL_BUDGET_SAMPLING:-false}
export HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER=${HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER:-true}

# --- budget ratchet: ADAPTIVE (two-sided) for auto-hint -----------------------
# Default the auto-hint runs to the adaptive rule (budget_manager.compute_adaptive_budget):
# RAISE B_q by 1 when NO rollout is correct, set B_q to the N/2-th smallest correct hint
# count when OVER HALF solve, else hold. This avoids the downward rule's aggressive
# auto-hint collapse (a single 0-hint first-try solve snapping B_q to 0). max_budget (the
# raise ceiling) defaults to the turn cap; override HPRL_MAX_BUDGET to change it.
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}

# --- distinct run labels in wandb --------------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-SFT'}
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-dapo-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint] auto-hint mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint]   strategy=${HINT_STRATEGY} call_reward=${HINT_CALL_REWARD} shape=${HINT_SHAPE_COEFF} kpack=${HPRL_KPACK_ENABLE}"
echo "[run_auto_hint]   step_adv=${HPRL_STEP_ADV} step_adv_scale=${HPRL_STEP_ADV_SCALE} step_adv_norm=${HPRL_STEP_ADV_NORM} guidance_free=${HINT_GUIDANCE_FREE} prune_guidance=${HPRL_PRUNE_GUIDANCE}"
exec bash "${SCRIPT_DIR}/run_hprl_qwen2.5_7b.sh" "$@"
