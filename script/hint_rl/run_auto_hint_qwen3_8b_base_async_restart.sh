#!/usr/bin/env bash
#
# TERMINATE-AND-RESTART (segment-chain) auto-hint HPRL run for Qwen3-8B-Base --
# FULLY-ASYNC edition. The restart sibling of run_auto_hint_qwen3_8b_base_async.sh
# (devlog 2026-08-04): every knob below defaults to that wrapper's value; the
# restart deltas are exactly the four blocks marked "RESTART DELTA".
#
# WHAT CHANGES vs multi-turn auto-hint (restart_agent_loop.RestartHintAgentLoop):
#   * On a wrong answer the loop does NOT append the hint as a user turn of one
#     growing conversation. It ENDS the attempt and starts a FRESH single-turn
#     rollout whose prompt = original problem + "You have attempted this problem
#     before and you made the following verified progress, which is correct:
#     ..." (v2 recap list; a progress-free restart carries just the hint
#     block) + the new hint
#     (selector_multi.format_restart_hint_message). Per-segment context is
#     prompt + ONE attempt: max ~4096 + 8192 tokens instead of 20-40k chains.
#   * The trained row is the chain's FINAL segment only (v0): prompt = its fresh
#     recap+hint prompt, response = its attempt. applied_hints accumulate across
#     the chain, so the reward's hint penalties, the GRPO group (n chains per
#     problem), and the budget ratchet are byte-identical in semantics. Earlier
#     segments are archived in extra_fields.restart_segments -> rollout dump.
#   * Selector view is UNCHANGED (the loop keeps the full logical transcript in
#     agent_data.messages for build_trace), making the selector a controlled
#     variable in the restart-vs-multi-turn A/B.
#
# Activation is a registry swap (hint_agent_config_restart.yaml maps the
# dataset's agent_name="auto_hint" rows to RestartHintAgentLoop) -- same
# parquet, no dataset rebuild; flag-off runs are untouched.
#
# First-launch smoke checklist:
#   * rollouter log shows "RestartHintAgentLoop: terminate-and-restart
#     (segment-chain) mode ACTIVE";
#   * rollouts/<step>.jsonl rows with num_hints>0 have `input` CONTAINING the
#     recap+hint block ("You have attempted this problem before") and
#     restart_segments carrying the earlier attempts;
#   * hprl/restart_prompt_overflow stays ~0 (else raise max_prompt_length);
#   * nonzero actor/pg_loss and the hprl/* metric block present (the
#     extra_info-restore path is shared with the parent wrapper).
#
# Cluster launch (OpenAI selector mode) -- dedicated platform entry point:
#   bash launch_hprl_cluster_openai_async_restart.sh        (on every pod)
# or, equivalently, behind the generic async launcher:
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Qwen3-8B-Base via the derived -hprl dir (eos [151643,151645]) --
# Verbatim from run_auto_hint_qwen3_8b_base_async.sh; skipped when MODEL_PATH is
# already set (you then own the eos fix).
if [ -z "${MODEL_PATH:-}" ]; then
    QWEN3_SRC_DIR=${QWEN3_SRC_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base"}
    QWEN3_HPRL_DIR=${QWEN3_HPRL_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base-hprl"}
    if [ ! -f "${QWEN3_SRC_DIR}/config.json" ]; then
        echo "[run_auto_hint_qwen3_async_restart] ERROR: pristine model dir not found: ${QWEN3_SRC_DIR}" >&2
        exit 1
    fi
    mkdir -p "${QWEN3_HPRL_DIR}"
    for f in "${QWEN3_SRC_DIR}"/*; do
        b="$(basename "${f}")"
        if [ "${b}" = "generation_config.json" ]; then
            continue
        fi
        if [ "$(readlink "${QWEN3_HPRL_DIR}/${b}" 2>/dev/null || true)" != "${f}" ]; then
            ln -sfn "${f}" "${QWEN3_HPRL_DIR}/${b}"
        fi
    done
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

# --- RESTART DELTA 1: the agent-loop registry swap ----------------------------
# auto_hint rows -> RestartHintAgentLoop (same parquet, no dataset rebuild).
export AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config_restart.yaml"}

# --- RESTART DELTA 2: context budget ------------------------------------------
# Fresh prompts carry problem + recap + up-to-budget hints, so the prompt budget
# doubles (2048 -> 4096); the response budget shrinks to keep prompt+response at
# the model-native 32768 EXACTLY (the async server pins vLLM max_model_len to
# max_position_embeddings -- exceeding it kills the Rollouter; see the parent
# wrapper's header). Per-segment generation is bounded by HPRL_MAX_TURN_TOKENS
# (8192, below), so live per-request context is <= 4096+8192 = 12288 tokens --
# the throughput mechanism of restart mode. A restart prompt that would exceed
# 4096 is refused by the loop (chain ends, hint not charged,
# restart_prompt_overflow counter) instead of being left-truncated.
export max_prompt_length=${max_prompt_length:-4096}
export max_response_length=${max_response_length:-28672}

# --- async pipeline knobs: identical to the parent wrapper ---------------------
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-5} - TRAINER_NNODES ))}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}
export clip_ratio_low=${clip_ratio_low:-0.2}

# --- PPO update geometry + optimizer: identical to the parent wrapper ----------
export actor_lr=${actor_lr:-1e-6}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-64}
export n_resp_per_prompt=${n_resp_per_prompt:-8}

# --- auto-hint mechanism knobs (shared with the restart loop) ------------------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Recap in the fresh prompt (format_restart_hint_message include_progress) --
# the restart analog of the v2 progress-recap injection. Keep ON.
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}
# POOL WORDING (restart-only knob; forwarded to the agent-loop workers via
# run_hprl_async.sh's HPRL_RESTART_ENV block): recap lines + the delivered hint
# use the hint set's ORIGINAL sentences instead of the selector's
# student-notation rephrasings -- a fresh prompt has no reasoning trace to adapt
# wording to. Selection/verification is unchanged. =false -> selector wording.
export HPRL_RESTART_POOL_WORDING=${HPRL_RESTART_POOL_WORDING:-true}
# TRAIN SEGMENTS = all: EVERY turn of a chain becomes a training row (trainer
# expands archived segments; each turn painted A = r + V[se+1] - V[ss] from the
# group fit, final rows keep the chain-exact phantom fit). =final -> v0
# final-segment-only rows. Requires step_adv + bypass_mode (both on here).
export HPRL_RESTART_TRAIN_SEGMENTS=${HPRL_RESTART_TRAIN_SEGMENTS:-all}

# --- train parquet + resume: identical to the parent wrapper -------------------
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-rl-zero-10324-auto-hint.parquet"}
# Default resume = the 20260731-020431 run's global_step_200 ckpt, SAME as the
# parent wrapper -- so restart-vs-multi-turn continues from a matched start
# point (the controlled A/B). export RESUME_FROM_PATH= (empty) for a fresh run.
QWEN3_ASYNC_S200_CKPT="${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen3-8B-Base-async/HPRL-AutoHint-Qwen3-8B-Base-async-dolci-rl-zero-10324-20260731-020431/global_step_200"
export RESUME_FROM_PATH=${RESUME_FROM_PATH-}
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
    echo "[run_auto_hint_qwen3_async_restart] ERROR: RESUME_FROM_PATH has no actor/ ckpt: ${RESUME_FROM_PATH}" >&2
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

# --- STEP-LEVEL advantage: ON, identical knobs to the parent wrapper -----------
# Compatible with restart mode via PHANTOM fail records: the loop keeps a
# zero-token-span [0,0,0,ss,se,1] record per consumed hint, so the per-group
# value fit sees each CHAIN's full failure set (F_k) and final state (D_k)
# exactly as a multi-turn row would -- V is identical, and the final segment's
# whole_turn advantage A = r_se + V[se+1] - V[ss] (fail) / V[se] - V[ss] (solve)
# equals the multi-turn final turn's (numeric parity test:
# test_restart_chain.test_step_adv_value_parity). v0 trains ONLY the final
# segment, so the intermediate turns' advantage terms are absent (a subset
# gradient over the same V) until the expand-segments-to-rows follow-up.
# whole_turn=true recommended: boundary-free, and the solve/terminal segment is
# priced whole. The verified-prefix mask path stays a structural no-op on
# restart rows (empty disable_spans by design).
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (parent parity) -------------------------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-SEGMENT generation-length cap (was per-turn): parent parity -----------
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-8192}

# --- k-pack / budget-grouped sampling: async-unsupported, pinned off -----------
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (parent parity) ----------------------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- RESTART DELTA 4: distinct run labels --------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Qwen3-8B-Base-async'}
export exp_name=${exp_name:-"HPRL-AutoHint-Qwen3-8B-Base-async-restart-dolci-rl-zero-10324-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_qwen3_async_restart] TERMINATE-AND-RESTART auto-hint mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_qwen3_async_restart]   agent registry: ${AGENT_LOOP_CONFIG_PATH} (auto_hint -> RestartHintAgentLoop)"
echo "[run_auto_hint_qwen3_async_restart]   context: prompt=${max_prompt_length} response=${max_response_length} per-segment cap=${HPRL_MAX_TURN_TOKENS} (live context <= $((max_prompt_length + HPRL_MAX_TURN_TOKENS)) tok)"
echo "[run_auto_hint_qwen3_async_restart]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_qwen3_async_restart]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_qwen3_async_restart]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS} clip_ratio_low=${clip_ratio_low}"
echo "[run_auto_hint_qwen3_async_restart]   lr=${actor_lr} mini_bsz=${train_prompt_mini_bsz} rollout_n=${n_resp_per_prompt}"
echo "[run_auto_hint_qwen3_async_restart]   strategy=${HINT_STRATEGY} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} pool_wording=${HPRL_RESTART_POOL_WORDING} train_segments=${HPRL_RESTART_TRAIN_SEGMENTS}"
echo "[run_auto_hint_qwen3_async_restart]   step_adv=${HPRL_STEP_ADV} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} norm=${HPRL_STEP_ADV_NORM} overlong=${HPRL_OVERLONG_PENALTY}(${HPRL_OVERLONG_PENALTY_TYPE}) -- final-segment rows, phantom fail records keep the value fit chain-exact"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
