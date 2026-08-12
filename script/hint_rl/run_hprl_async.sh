#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Hint Penalized RL (HPRL) -- FULLY-ASYNC (disaggregated) training.
#
# The async sibling of run_hprl_qwen2.5_7b.sh, with every DEFAULT aligned to
# the Olmo auto-hint run (run_auto_hint_olmo3_7b_instruct_async.sh -- model,
# labels, data and science knobs included), so a bare launch of this script IS
# that run. Same algorithm (GRPO clip-higher,
# no KL), same HPRL machinery (auto-hint loop, budget ratchet, step-adv /
# verified-prefix mask, hint reward) -- but instead of the colocated synchronous
# trainer (generate -> update -> generate on the same GPUs, each step gated on
# its slowest rollout; measured 71.6% of step time in generation), the cluster's
# training pods are SPLIT into two pools running concurrently:
#
#   * ROLLOUT pool  (ROLLOUT_NNODES x N_GPUS_PER_NODE): vLLM replicas + the
#     agent loops, STREAMING one prompt-group (rollout.n trajectories, one GRPO
#     group) at a time into a message queue. No step barrier: a long rollout
#     (many hint rounds + selector calls) never idles the GPUs that finished.
#   * TRAINER pool  (TRAINER_NNODES x N_GPUS_PER_NODE): FSDP actor. Pulls
#     REQUIRE_BATCHES x ppo_mini_batch_size groups from the queue per iteration,
#     updates, and every TRIGGER_PARAMETER_SYNC_STEP iterations NCCL-pushes
#     fresh weights to the rollout replicas (~sub-second for a 7B).
#
# Off-policy correctness: trajectories are trained with the logprobs the
# ROLLOUT policy actually sampled them under (rollout.calculate_log_probs=True
# + actor.use_rollout_log_probs=True + algorithm.rollout_correction.bypass_mode
# =True, all defaulted by the base fully_async config) -- PPO's importance
# ratio then correctly accounts for the bounded staleness.
#
# Equivalence bookkeeping vs the sync job (defaults below):
#     REQUIRE_BATCHES(1) x ppo_mini(64) x TRIGGER(1) = 64 prompts per weight
#     sync == one sync-job step (train_prompt_bsz=64). So a "param version"
#     here corresponds 1:1 to a sync global step: trainer.test_freq /
#     save_freq / total_epochs keep their sync meanings.
#
# Staleness knobs:
#   STALENESS_THRESHOLD  max fraction of extra samples generated ahead under
#                        the previous weights (0 = synchronous streaming;
#                        0.5 = benchmark-validated default; keep < 1).
#   PARTIAL_ROLLOUT      True -> weight sync INTERRUPTS in-flight generation
#                        and resumes it on the new weights (invisible to the
#                        agent loop -- the server client re-issues with the
#                        accumulated tokens). The direct fix for the long-tail
#                        (slowest/mean rollout measured 3.6x in the sync job).
#
# NOT supported in async mode (guarded loudly at launch by main_hprl_async.py):
#   * HPRL_KPACK_ENABLE=true       -> hard error (sync-trainer-only code path).
#   * HPRL_BUDGET_SAMPLING=true    -> forced false: the rollout pool streams
#     samples one at a time, so there is no generation batch to keep budget-
#     homogeneous -- the queue already absorbs duration variance.
#
# Prereqs identical to the sync job (parquet built, selector serving, Ray
# cluster up across the TRAINING pods -- both pools live inside that one Ray
# cluster; launch_hprl_cluster.sh needs no change beyond pointing TRAIN_SCRIPT
# at the async wrapper).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# this script lives at <HINT_RL_HOME>/script/hint_rl/<this>
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

# Activate the verl conda environment
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}

# miniconda is not relocatable; symlink the original prefix to the current mount
# so baked-in absolute paths keep resolving when the tree is mounted elsewhere.
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Load secrets (wandb key, ...) from .envrc.
if [ -f "${HINT_RL_HOME}/.envrc" ]; then
    source "${HINT_RL_HOME}/.envrc"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-${wandb_key:-}}"

# Run labels: default to the SAME names the Olmo auto-hint async wrapper
# (run_auto_hint_olmo3_7b_instruct_async.sh) uses -- this script's defaults
# now mirror that run entirely. Override via env to relabel (e.g. a Qwen run).
project_name=${project_name:-'HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async'}
exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async-dapo-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
wandb_project=${wandb_project:-"hint_rl"}

# ---- GRPO algorithm (identical to the sync job) ----------------------------
adv_estimator=grpo
norm_adv_by_std_in_grpo=True
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}
loss_agg_mode="token-mean"

# ---- lengths: multi-turn trajectories accumulate tokens across turns ------
# Env-overridable like the sync job (they were hardcoded before, which silently
# ignored staleness_grid.sh's max_response_length export): the qwen3-8b-base
# async wrapper exports max_response_length=30720 so prompt+response fits that
# model's native 32768 exactly.
max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-32768}

# ---- multi-turn / hint knobs (identical to the sync job) -------------------
max_turns=${max_turns:-10}
AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config.yaml"}
max_tool_response_length=${max_tool_response_length:-2048}

# ---- frozen selector model (OpenAI-compatible endpoint) -------------------
# Identical to the sync job; the agent loops run on the ROLLOUT pool and reach
# the selector over HTTP exactly as before (env forwarded via the runtime env).
export SELECTOR_BASE_URL=${SELECTOR_BASE_URL:-"http://localhost:30000/v1"}
export SELECTOR_BASE_URLS=${SELECTOR_BASE_URLS:-""}
export SELECTOR_MODEL=${SELECTOR_MODEL:-"gpt-oss-20b"}
export SELECTOR_API_KEY=${SELECTOR_API_KEY:-"EMPTY"}
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.7}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-1.0}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}
export SELECTOR_REQUEST_TIMEOUT_S=${SELECTOR_REQUEST_TIMEOUT_S:-600}
export SELECTOR_MAX_RETRIES=${SELECTOR_MAX_RETRIES:-3}
# ---- selector API mode: local vLLM endpoints (default) vs the REAL OpenAI API.
# SELECTOR_API_MODE=openai (set by launch_hprl_cluster_openai_async.sh, which
# also picks SELECTOR_MODEL) sends every hint call to SELECTOR_OPENAI_BASE_URL
# with OPENAI_API_KEY -- no selector pods at all; the SELECTOR_BASE_URL(S) pair
# above is ignored. Reasoning models (gpt-5*/o*) switch to max_completion_tokens
# + SELECTOR_REASONING_EFFORT and drop temperature/top_p. SELECTOR_MAX_CONCURRENCY
# caps in-flight calls PER agent-loop worker process (openai mode only).
export SELECTOR_API_MODE=${SELECTOR_API_MODE:-local}
export SELECTOR_OPENAI_BASE_URL=${SELECTOR_OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export SELECTOR_REASONING_EFFORT=${SELECTOR_REASONING_EFFORT:-low}
export SELECTOR_MAX_CONCURRENCY=${SELECTOR_MAX_CONCURRENCY:-16}
export OPENAI_API_KEY=${OPENAI_API_KEY:-}

# ---- cluster: split the NNODES training pods into the two pools ------------
# NNODES is the TOTAL training-pod count (what ray_cluster_launch.sh waited
# for; the selector pods are outside the Ray cluster and unaffected). Default
# split half/half -- 2 trainer + 2 rollout nodes of the standard 4 -- matching
# verl's benchmarked 16:16 GPU split for a 7B multi-turn tool workload (~1.6x).
# Rebalance with the fully_async/{trainer,rollouter}/idle_ratio wandb metrics:
# high rollouter idle + low trainer idle -> shift a node to the trainer pool,
# and vice versa.
NNODES=${NNODES:-5}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-4}
TRAINER_NNODES=${TRAINER_NNODES:-$((NNODES - ROLLOUT_NNODES))}
if [ "${TRAINER_NNODES}" -lt 1 ] || [ "${ROLLOUT_NNODES}" -lt 1 ]; then
    echo "[run_hprl_async] FATAL: need >=1 trainer node and >=1 rollout node" \
         "(got trainer=${TRAINER_NNODES} rollout=${ROLLOUT_NNODES} of NNODES=${NNODES})." >&2
    exit 1
fi
if [ "$((TRAINER_NNODES + ROLLOUT_NNODES))" -gt "${NNODES}" ]; then
    echo "[run_hprl_async] FATAL: TRAINER_NNODES(${TRAINER_NNODES}) + ROLLOUT_NNODES(${ROLLOUT_NNODES})" \
         "> NNODES(${NNODES}) -- the Ray cluster cannot host both pools." >&2
    exit 1
fi

# ---- batch geometry ---------------------------------------------------------
# train_batch_size MUST be 0 and gen_batch_size MUST be 1 in fully-async mode
# (the Rollouter asserts both): samples are streamed, not stepped. The PPO
# update geometry is unchanged from the sync job -- ppo_mini_batch_size prompts
# x rollout.n responses per optimizer step. n_resp_per_prompt /
# train_prompt_mini_bsz / actor_lr are env-overridable so per-run wrappers can
# pin them without editing this file; queue sizing / staleness math
# (required_samples etc.) derive from them downstream.
n_resp_per_prompt=${n_resp_per_prompt:-8}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-64}
actor_lr=${actor_lr:-1e-6}

# ---- async pipeline knobs ---------------------------------------------------
# See header for semantics. Defaults: sync-equivalent update cadence (64
# prompts per weight sync), 0.5 staleness, partial rollout on.
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
USE_TRAINER_DO_VALIDATE=${USE_TRAINER_DO_VALIDATE:-False}
# Off-policy ratio anchor (algorithm.rollout_correction.bypass_mode):
#   True  (stock fully-async default) -- old_log_probs := the ROLLOUT policy's
#         sampling-time logprobs; PPO clips against the stale behavior policy.
#         Caveat: for a rare token the trainer is suppressing, ratio<1-eps_low
#         from the start -> negative-advantage grad clips to ZERO while the
#         positive side stays live (one-way rectifier; 'assistant'-tic
#         resurgence, devlog 2026-07-30).
#   False (decoupled mode) -- the trainer RECOMPUTES old_log_probs as an
#         on-policy proximal anchor (ratio starts at 1, both grad directions
#         at full strength) and corrects staleness separately via seq-level
#         truncated-IS weights vs rollout_log_probs (rollout_corr/* metrics).
#         Costs one extra actor forward pass per step on the trainer pool.
ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}

# Optional HARD step bound, in PARAM VERSIONS (== sync global steps at the
# defaults above). 0/empty = off (the total_epochs bound alone applies). The
# async stack bounds steps on the ROLLOUTER side (rollout.total_rollout_steps,
# in streamed prompt-groups; the trainer's step count derives from it), so the
# count converts as steps x ppo_mini x require_batches x trigger. Used by
# grid_search_async.sh to run short fixed-length benchmark trials.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-0}
if [ "${TOTAL_TRAINING_STEPS}" -gt 0 ] 2>/dev/null; then
    TOTAL_ROLLOUT_STEPS=$((TOTAL_TRAINING_STEPS * train_prompt_mini_bsz * REQUIRE_BATCHES * TRIGGER_PARAMETER_SYNC_STEP))
else
    TOTAL_ROLLOUT_STEPS=null
fi

# Ray
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${VERL_HOME}"}

# Paths. Base model: Olmo-3-7B-Instruct-SFT (same as the auto-hint wrapper).
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Olmo-3-7B-Instruct-SFT"}
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}

# ---- AUTO-HINT (push-hint) mode -- ON BY DEFAULT ----------------------------
# Defaults ALIGNED with run_auto_hint_olmo3_7b_instruct_async.sh (the Olmo
# wrapper): model, run labels, data, reward, step-adv and ratchet included, so
# a bare launch of this script IS the Olmo auto-hint async run (the wrapper
# remains as an env-override convenience). See run_hprl_qwen2.5_7b.sh for full
# knob docs; each `:=`/`:-` is a default, any exported value wins.
HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Prefix each new hint with cumulative completed/given progress, interleaved in
# the original hint-set order. On by default everywhere (the v2 recap injection
# eliminates the post-hint restart drift); set =false for hint-only messages.
HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
# Prune X.0 step-guidance hints from the pool before the selector sees it
# (eval/train parity with the offline selector eval).
HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}
# STEP-LEVEL advantage ON (replaces GRPO's scalar advantage, skips the
# verified-prefix mask), with GRPO-style per-group std-normalization
# (adv_scale = TARGET std when normalize is on).
HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
# Absolute over-turn-length surcharge on per-turn-cap truncation tails.
HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
# NOTE: the async Olmo wrapper leaves the routing at post_hoc (the SYNC Olmo
# wrapper defaults to value); export HPRL_OVERLONG_PENALTY_TYPE=value to fold
# the penalty into the value recursion instead.
HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-post_hoc}
# Absolute -P_over on truncation tails inside ZEROED (no-correct) groups --
# without it an all-wrong co-truncating group gets zero anti-truncation
# gradient (2026-08-04 flipped-price finding). false restores scored-only.
HPRL_OVERLONG_ZEROED=${HPRL_OVERLONG_ZEROED:-true}
# WHOLE-TURN advantage: one A = V(s_end) + hint_penalty - V(s_start) per turn
# (boundary-free; no verified-prefix a_C / failed-tail a_I split).
HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}
# PER-TURN generation-length cap (tokens); a capped turn is scored as failing
# its next hint step and the rollout terminates. 0 = off.
HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-8192}
case "${HPRL_AUTO_HINT}" in
  true | True | 1 | yes | on)
    # dapo-512 with hint-wise calibrated initial budgets (the wrapper's pin);
    # set dataset/dapo-3139-auto-hint-hintwise.parquet for the full 3139 set.
    : "${TRAIN_FILE:=${HINT_RL_HOME}/dataset/dapo-512-auto-hint.parquet}"
    : "${TEST_FILE:=${HINT_RL_HOME}/dataset/aime2024.parquet}"
    : "${HARD_TEST_FILE:=${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet}"
    : "${AIME2025_FILE:=${HINT_RL_HOME}/dataset/aime2025.parquet}"
    # 4th bare val set (data_source=hmmt_nov_2025 -> val-core/hmmt_nov_2025/*).
    : "${HMMT_FILE:=${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet}"
    : "${VAL_FILES:=['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']}"
    : "${HINT_STRATEGY:=hint}"
    : "${HINT_CALL_REWARD:=0.0}"
    : "${HINT_SHAPE_COEFF:=0.0}"
    : "${NO_HINT_PENALTY_FACTOR:=0.0}"
    : "${HINT_FINALIZE_INCORRECT:=false}"
    # Free X.0 guidance hint (weight dropped; the step's penalty is borne by
    # the substeps, pool total stays HINT_PENALTY_TOTAL).
    : "${HINT_GUIDANCE_FREE:=true}"
    # Budget ratchet: ADAPTIVE + RAISE-ONLY (decreases vetoed and held).
    : "${HPRL_RATCHET_MODE:=adaptive}"
    : "${HPRL_ALLOW_DECREASE:=false}"
    ;;
esac

TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-3139-hint-verl-mt-clean.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024-hint-mt.parquet"}
HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100-hint-mt.parquet"}
AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025-hint-mt.parquet"}
VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}']"}

# HPRL reward function + manager (identical to the sync job; the reward runs
# streaming on the ROLLOUT pool via verl's reward loop, as it already did).
REWARD_FN_PATH=${REWARD_FN_PATH:-"${SCRIPT_DIR}/hint_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}
REWARD_MGR_PATH=${REWARD_MGR_PATH:-"${SCRIPT_DIR}/hint_reward_manager.py"}
REWARD_MGR_CLASS=${REWARD_MGR_CLASS:-"HintRewardManager"}
HINT_PENALTY_TOTAL=${HINT_PENALTY_TOTAL:-1.0}
HINT_PENALTY_HARD_FACTOR=${HINT_PENALTY_HARD_FACTOR:-1.5}
HINT_GUIDANCE_DIFFICULTY=${HINT_GUIDANCE_DIFFICULTY:-easy}
HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-false}
NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.1}
HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
HINT_STRATEGY=${HINT_STRATEGY:-major_step}
HINT_STEP_EXCLUDE_MODE=${HINT_STEP_EXCLUDE_MODE:-cumulative}
HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-true}

# The directory holding hint_tool.py / the HPRL modules must be importable by
# every actor (trainer, rollouter, agent-loop workers, reward workers).
TOOL_PYTHONPATH="${SCRIPT_DIR}"

# Local logs
RUN_ID=${RUN_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${HINT_RL_HOME}/logs"}
EXP_LOG_DIR=${EXP_LOG_DIR:-"${LOG_DIR}/${exp_name}"}
LOG_FILE=${LOG_FILE:-"${EXP_LOG_DIR}/${exp_name}.jsonl"}
CONSOLE_LOG=${CONSOLE_LOG:-"${EXP_LOG_DIR}/${exp_name}.${RUN_ID}.console.log"}
mkdir -p "${EXP_LOG_DIR}"

# ---- HPRL dynamic-budget ratchet -------------------------------------------
# Identical knobs to the sync job. In async mode the ratchet runs on the
# trainer actor after each local update and lands in the shared budget-state
# JSON; the dataset (rollout pool) re-reads it by mtime, so ratcheted budgets
# reach newly-sampled problems with the pipeline's (bounded) staleness lag.
HPRL_ENABLE=${HPRL_ENABLE:-True}
BUDGET_STATE_PATH=${BUDGET_STATE_PATH:-"${EXP_LOG_DIR}/budget_state.json"}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
if [ "${RESUME_FROM_PATH}" != "null" ] && [ -n "${RESUME_FROM_PATH}" ]; then
    RESUME_MODE=${RESUME_MODE:-resume_path}
else
    RESUME_MODE=${RESUME_MODE:-auto}
fi
HPRL_MIN_BUDGET=${HPRL_MIN_BUDGET:-0}
HPRL_DECREMENT=${HPRL_DECREMENT:-1}
HPRL_DEFAULT_BUDGET=${HPRL_DEFAULT_BUDGET:-6}
HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-downward}
HPRL_MAX_BUDGET=${HPRL_MAX_BUDGET:-6}
HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-true}

# ---- async-mode restrictions (see header) -----------------------------------
if [ "${HPRL_KPACK_ENABLE:-false}" = "true" ]; then
    echo "[run_hprl_async] FATAL: HPRL_KPACK_ENABLE=true is not supported in fully-async mode" \
         "(sync-trainer-only code path). Use run_hprl_qwen2.5_7b.sh for k-pack." >&2
    exit 1
fi
if [ "${HPRL_BUDGET_SAMPLING:-false}" = "true" ]; then
    echo "[run_hprl_async] WARN: HPRL_BUDGET_SAMPLING has no effect in fully-async mode" \
         "(streaming rollout has no generation batch to group); forcing false." >&2
fi
HPRL_KPACK_ENABLE=false
HPRL_BUDGET_SAMPLING=false

# Archive a verbatim snapshot of the WHOLE hint_rl script folder alongside the
# run's logs (same rationale as the sync job).
SRC_SNAPSHOT_DIR="${EXP_LOG_DIR}/hint_rl_src.${RUN_ID}"
mkdir -p "${SRC_SNAPSHOT_DIR}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "${SCRIPT_DIR}/" "${SRC_SNAPSHOT_DIR}/"
else
    cp -r "${SCRIPT_DIR}/." "${SRC_SNAPSHOT_DIR}/"
    find "${SRC_SNAPSHOT_DIR}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "${SRC_SNAPSHOT_DIR}" -name '*.pyc' -delete 2>/dev/null || true
fi
echo "[run_hprl_async] archived hint_rl source snapshot -> ${SRC_SNAPSHOT_DIR}"

# Algorithm sampling
temperature=1.0
top_p=1.0
top_k=-1

# Performance (identical to the sync job; the trainer pool is smaller, so the
# per-GPU token budget math is unchanged -- FSDP shards across TRAINER_NNODES*8).
# Ulysses sequence-parallel size. 4 suits 7-8B x 30k+ multi-turn rows; for a
# SMALL model on short restart segments (e.g. 4B at prompt 4096 + cap 8192 =
# 12288 max seq) sp=2 or 1 frees data parallelism (dp = 8/sp) and drops the
# all-to-all overhead -- the 4B restart run 214953 was trainer-bound at dp=2.
sp_size=${SP_SIZE:-4}
use_dynamic_bsz=True
PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-2}
actor_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
offload=${OFFLOAD:-False}
gen_tp=1
# vLLM KV-pool fraction on the (vLLM-only) rollout pods. Historical default
# 0.80. MHA models (Olmo3, no GQA) are KV-bound there, so their wrappers may
# raise it -- but the non-vLLM residue of the 79 GB pods must still hold the
# CANN runtime + the checkpoint-engine sync buffers below: shrink
# UPDATE_WEIGHTS_BUCKET_MB in step when you raise this.
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.80}

# Weight-sync bucket for the disaggregated checkpoint engine. 4096 OOMed on the
# rollout pods (grid 20260720-141549 st1: update_weights tried to alloc exactly
# 4.00 GiB with 3.67 GiB free next to vLLM at gpu_memory_utilization=0.80, ~10
# syncs in once the allocator fragmented). 1024 fits the observed free headroom
# with margin; cost is a few extra flush round-trips per sync. NOTE: that
# headroom was measured at util 0.80 -- wrappers raising GPU_MEMORY_UTIL lose
# ~0.8 GB per 0.01 and should pass a smaller bucket (olmo restart: 0.90 + 512).
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-1024}

# Per-run Ray runtime env (same rationale + air-gap warning as the sync job:
# NEVER add a runtime_env pip block on this fabric).
HPRL_DUMP_SELECTOR=${HPRL_DUMP_SELECTOR:-1}
HPRL_SELECTOR_DUMP_DIR=${HPRL_SELECTOR_DUMP_DIR:-""}
if [ "${HPRL_DUMP_SELECTOR}" != "0" ] && [ -z "${HPRL_SELECTOR_DUMP_DIR}" ]; then
  HPRL_SELECTOR_DUMP_DIR="${EXP_LOG_DIR}/selector_calls"
fi

# Egress-proxy forwarding (openai selector mode on a fabric that needs one):
# any proxy var set at submit time is forwarded verbatim into the workers'
# runtime env. Conditional lines -- injecting an EMPTY proxy var could confuse
# httpx/vLLM networking in the (default) proxy-less local mode.
SELECTOR_PROXY_ENV=""
for _pv in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [ -n "${!_pv:-}" ]; then
        SELECTOR_PROXY_ENV="${SELECTOR_PROXY_ENV}  ${_pv}: \"${!_pv}\""$'\n'
    fi
done

# Restart-mode (segment-chain) agent-loop knobs: forwarded verbatim into the
# workers' runtime env WHEN SET (same conditional pattern as the proxy vars --
# absent in non-restart runs, so their runtime env is unchanged). Env is the
# ONLY channel that reaches the agent-loop workers for these: registry-yaml
# extra keys are lost after the first rollout (restart_agent_loop's registry
# re-pin keeps only _target_), and data.hprl.* would need new hydra keys.
HPRL_RESTART_ENV=""
for _rv in HPRL_RESTART_POOL_WORDING HPRL_RESTART_TRAIN_SEGMENTS HPRL_RESTART_ARCHIVE_IDS \
           HPRL_RESTART_REFERENCE_PREFIX; do
    if [ -n "${!_rv:-}" ]; then
        HPRL_RESTART_ENV="${HPRL_RESTART_ENV}  ${_rv}: \"${!_rv}\""$'\n'
    fi
done

RUNTIME_ENV_RUN=${RUNTIME_ENV_RUN:-"${LOG_DIR}/.runtime_env.${RUN_ID}.yaml"}
( umask 077
  cat > "${RUNTIME_ENV_RUN}" <<EOF
working_dir: "${WORKING_DIR}"
excludes: ["/.git/"]
env_vars:
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  VLLM_USE_V1: "1"
  VERL_FILE_LOGGER_PATH: "${LOG_FILE}"
  WANDB_API_KEY: "${WANDB_API_KEY}"
  HINT_RL_HOME: "${HINT_RL_HOME}"
  SELECTOR_BASE_URL: "${SELECTOR_BASE_URL}"
  SELECTOR_BASE_URLS: "${SELECTOR_BASE_URLS}"
  SELECTOR_MODEL: "${SELECTOR_MODEL}"
  SELECTOR_API_KEY: "${SELECTOR_API_KEY}"
  SELECTOR_TEMPERATURE: "${SELECTOR_TEMPERATURE}"
  SELECTOR_TOP_P: "${SELECTOR_TOP_P}"
  SELECTOR_MAX_TOKENS: "${SELECTOR_MAX_TOKENS}"
  SELECTOR_REQUEST_TIMEOUT_S: "${SELECTOR_REQUEST_TIMEOUT_S}"
  SELECTOR_MAX_RETRIES: "${SELECTOR_MAX_RETRIES}"
  SELECTOR_API_MODE: "${SELECTOR_API_MODE}"
  SELECTOR_OPENAI_BASE_URL: "${SELECTOR_OPENAI_BASE_URL}"
  SELECTOR_REASONING_EFFORT: "${SELECTOR_REASONING_EFFORT}"
  SELECTOR_MAX_CONCURRENCY: "${SELECTOR_MAX_CONCURRENCY}"
  OPENAI_API_KEY: "${OPENAI_API_KEY}"
${SELECTOR_PROXY_ENV}${HPRL_RESTART_ENV}  HPRL_SELECTOR_DUMP_DIR: "${HPRL_SELECTOR_DUMP_DIR}"
  # make the HPRL modules importable in every actor of the job.
  PYTHONPATH: "${TOOL_PYTHONPATH}"
EOF
)

# Fail fast with a clear message if no Ray cluster is listening. The usual
# cause: THIS script (or its auto-hint wrapper) was submitted as the training
# platform's entry point -- but it only SUBMITS the ray job; it must run on the
# Ray head after the cluster is up. The platform entry must be
# launch_hprl_cluster_async.sh (single run) / launch_hprl_cluster_async_grid.sh
# (grid search), which bring up Ray + the selector on every pod first.
if ! python3 -c 'import sys, urllib.request as u; u.urlopen(sys.argv[1].rstrip("/") + "/api/version", timeout=10)' "${RAY_ADDRESS}" 2>/dev/null; then
    echo "[run_hprl_async] FATAL: no Ray cluster answering at ${RAY_ADDRESS}." >&2
    echo "  This script only submits the training job to an EXISTING Ray cluster." >&2
    echo "  On the training platform, submit launch_hprl_cluster_async.sh (single run)" >&2
    echo "  or launch_hprl_cluster_async_grid.sh (grid search) as the entry script." >&2
    exit 1
fi

# Submit to the Ray cluster's job server. The cluster must already be up.
# HPRL ASYNC entry: fully_async_main's flow with the trainer swapped for
# HPRLFullyAsyncTrainer (config = config/hprl_fully_async_trainer.yaml =
# fully_async_ppo_trainer + data.hprl). cwd is the Ray working-dir (= verl repo
# root) so the config's file:// searchpath resolves.
#
# Key async-specific overrides vs the sync submit:
#   data.train_batch_size=0, data.gen_batch_size=1   (streaming; asserted)
#   actor_rollout_ref.hybrid_engine=False            (pools are disaggregated)
#   trainer.nnodes/rollout.nnodes                    (the resource split)
#   async_training.*                                 (pipeline knobs)
#   rollout.calculate_log_probs=True                 (behavior-policy logprobs;
#                                                     also a base-config default)
ray job submit --runtime-env="${RUNTIME_ENV_RUN}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 "${SCRIPT_DIR}/main_hprl_async.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.hprl.enable=${HPRL_ENABLE} \
    data.hprl.budget_state_path="${BUDGET_STATE_PATH}" \
    data.hprl.min_budget=${HPRL_MIN_BUDGET} \
    data.hprl.decrement=${HPRL_DECREMENT} \
    data.hprl.default_budget=${HPRL_DEFAULT_BUDGET} \
    data.hprl.ratchet_mode=${HPRL_RATCHET_MODE} \
    data.hprl.max_budget=${HPRL_MAX_BUDGET} \
    data.hprl.allow_decrease=${HPRL_ALLOW_DECREASE} \
    data.hprl.strategy=${HINT_STRATEGY} \
    data.hprl.step_exclude_mode=${HINT_STEP_EXCLUDE_MODE} \
    data.hprl.finalize_incorrect=${HINT_FINALIZE_INCORRECT} \
    data.hprl.kpack.enable=${HPRL_KPACK_ENABLE} \
    data.hprl.budget_sampling.enable=${HPRL_BUDGET_SAMPLING} \
    data.hprl.auto_hint.enable=${HPRL_AUTO_HINT} \
    data.hprl.auto_hint.fuzzy_threshold=${HPRL_AUTO_HINT_FUZZY} \
    data.hprl.auto_hint.progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} \
    data.hprl.auto_hint.prune_guidance=${HPRL_PRUNE_GUIDANCE} \
    data.hprl.auto_hint.max_turn_tokens=${HPRL_MAX_TURN_TOKENS} \
    data.hprl.auto_hint.step_adv.enable=${HPRL_STEP_ADV} \
    data.hprl.auto_hint.step_adv.adv_scale=${HPRL_STEP_ADV_SCALE} \
    data.hprl.auto_hint.step_adv.normalize=${HPRL_STEP_ADV_NORM} \
    data.hprl.auto_hint.step_adv.overlong_penalty=${HPRL_OVERLONG_PENALTY} \
    data.hprl.auto_hint.step_adv.overlong_penalty_type=${HPRL_OVERLONG_PENALTY_TYPE} \
    data.hprl.auto_hint.step_adv.overlong_zeroed=${HPRL_OVERLONG_ZEROED} \
    data.hprl.auto_hint.step_adv.whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORR_BYPASS} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${HPRL_LR_SCHEDULER:-constant} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager="${REWARD_MGR_CLASS}" \
    reward_model.reward_loop_source=importlib \
    reward_model.reward_loop_module_path="${REWARD_MGR_PATH}" \
    reward_model.reward_loop_class_name="${REWARD_MGR_CLASS}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    +custom_reward_function.reward_kwargs.correct_reward=1.0 \
    +custom_reward_function.reward_kwargs.incorrect_reward=0.0 \
    +custom_reward_function.reward_kwargs.format_reward=0.0 \
    +custom_reward_function.reward_kwargs.hint_call_reward=${HINT_CALL_REWARD} \
    +custom_reward_function.reward_kwargs.hint_penalty_total=${HINT_PENALTY_TOTAL} \
    +custom_reward_function.reward_kwargs.hint_penalty_hard_factor=${HINT_PENALTY_HARD_FACTOR} \
    +custom_reward_function.reward_kwargs.hint_guidance_difficulty=${HINT_GUIDANCE_DIFFICULTY} \
    +custom_reward_function.reward_kwargs.hint_guidance_free=${HINT_GUIDANCE_FREE} \
    +custom_reward_function.reward_kwargs.hint_strategy=${HINT_STRATEGY} \
    +custom_reward_function.reward_kwargs.hint_shape_coeff=${HINT_SHAPE_COEFF} \
    +custom_reward_function.reward_kwargs.no_hint_penalty_factor=${NO_HINT_PENALTY_FACTOR} \
    +custom_reward_function.reward_kwargs.finalize_incorrect=${HINT_FINALIZE_INCORRECT} \
    trainer.logger="['console','file','wandb']" \
    trainer.project_name="${wandb_project}" \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=${TRAINER_NNODES} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    rollout.nnodes=${ROLLOUT_NNODES} \
    rollout.n_gpus_per_node=${N_GPUS_PER_NODE} \
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
    async_training.staleness_threshold=${STALENESS_THRESHOLD} \
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP} \
    async_training.require_batches=${REQUIRE_BATCHES} \
    async_training.partial_rollout=${PARTIAL_ROLLOUT} \
    async_training.use_trainer_do_validate=${USE_TRAINER_DO_VALIDATE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-20} \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=${TOTAL_EPOCHS:-100} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${LOG_DIR}/${exp_name}/rollouts" \
    trainer.validation_data_dir="${LOG_DIR}/${exp_name}/val_rollouts" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path=${RESUME_FROM_PATH} \
    trainer.device=cuda \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    2>&1 | tee "${CONSOLE_LOG}"
