#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Hint Penalized RL (HPRL) -- multi-turn GRPO with the hint tool.
#
# This is the multi-turn tool-use sibling of run_grpo_qwen2.5_7b_npu.sh. The
# algorithm is unchanged (GRPO: group-std-normalized advantage, clip-higher, no
# KL). What is added is the agent loop + the stateful `request_hint` tool:
#
#   * rollout.mode=async + multi_turn.enable=True  -> the agent loop.
#   * multi_turn.tool_config_path=hint_tool_config.yaml  -> declares the
#     `request_hint` tool (hint_tool.HintTool), which routes each hint call to
#     a frozen selector model and records the applied hints as per-rollout state.
#   * per-row agent_name="tool_agent" (added by prepare_hint_data.py) routes
#     training prompts through the tool-calling loop. The val set (aime2024) has
#     NO agent_name, so validation stays single-turn / unaided.
#   * reward: hint_reward.compute_score (outcome reward * blank hint penalty),
#     with HintRewardManager merging the applied-hints state into extra_info.
#
# Prereqs:
#   1. Build the multi-turn parquet:   python prepare_hint_data.py
#   2. Serve the frozen selector model on an OpenAI-compatible endpoint and set
#      SELECTOR_BASE_URL / SELECTOR_MODEL below (default sglang @ :30000).
#
# All paths derive from this script's location; override any VAR via the env.
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

project_name='HPRL-Qwen2.5-7B-Instruct'
exp_name="HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"
wandb_project=${wandb_project:-"hint_rl"}

# ---- GRPO algorithm (identical to the plain GRPO run) ---------------------
adv_estimator=grpo
norm_adv_by_std_in_grpo=True
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

# ---- lengths: multi-turn trajectories accumulate tokens across turns ------
max_prompt_length=2048
# bumped from 8192: prompt + N*(assistant turn + hint tool result).
max_response_length=16384

# ---- multi-turn / hint knobs ----------------------------------------------
# Cap on assistant turns; must be >= the largest per-problem hint budget B_q.
max_turns=${max_turns:-8}
# Hint mechanism: the policy emits the sentinel <hint_call/> and the custom agent
# loop (hint_agent_loop.HintAgentLoop) detects it, calls the selector, and injects
# the hint as the next USER message. Registered via this agent-loop config; the
# hermes request_hint tool is NOT loaded (no tool schema in the prompt).
AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config.yaml"}
# Hints can be a full sentence or two; allow more than the 256-token default.
max_tool_response_length=${max_tool_response_length:-2048}

# ---- frozen selector model (OpenAI-compatible endpoint) -------------------
# All of these are read by hint_selector.HintSelector.from_env() in the agent
# loop and forwarded into the Ray job below so the rollout workers can reach the
# selector. launch_hprl_cluster.sh sets SELECTOR_BASE_URL (the served gpt-oss-20b
# endpoint) and the call params; the defaults here let the script also run solo.
export SELECTOR_BASE_URL=${SELECTOR_BASE_URL:-"http://localhost:30000/v1"}
# Comma-separated list of INDEPENDENT selector endpoints (set by
# launch_hprl_cluster.sh). HintSelector load-balances + fails over across them;
# empty -> falls back to the single SELECTOR_BASE_URL above.
export SELECTOR_BASE_URLS=${SELECTOR_BASE_URLS:-""}
export SELECTOR_MODEL=${SELECTOR_MODEL:-"gpt-oss-20b"}
export SELECTOR_API_KEY=${SELECTOR_API_KEY:-"EMPTY"}
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.7}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-1.0}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}
export SELECTOR_REQUEST_TIMEOUT_S=${SELECTOR_REQUEST_TIMEOUT_S:-600}
export SELECTOR_MAX_RETRIES=${SELECTOR_MAX_RETRIES:-3}

# Cluster
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

train_prompt_bsz=128
n_resp_per_prompt=16
train_prompt_mini_bsz=16

# Ray
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${VERL_HOME}"}

# Paths
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-3740-hint-verl-simplified-mt.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}

# HPRL reward function (outcome reward minus the summed hint penalty).
REWARD_FN_PATH=${REWARD_FN_PATH:-"${SCRIPT_DIR}/hint_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}
# HPRL reward manager: merges per-rollout applied-hints state into extra_info.
REWARD_MGR_PATH=${REWARD_MGR_PATH:-"${SCRIPT_DIR}/hint_reward_manager.py"}
REWARD_MGR_CLASS=${REWARD_MGR_CLASS:-"HintRewardManager"}
# Hint-penalty knobs (hint_penalty.py). Passed as reward_kwargs so they retune
# WITHOUT regenerating the dataset: total penalty across all hints of a problem,
# the per-difficulty-level multiplier (harder = HARD_FACTOR x), and the difficulty
# assigned to the X.0 guidance hint.
HINT_PENALTY_TOTAL=${HINT_PENALTY_TOTAL:-1.8}
HINT_PENALTY_HARD_FACTOR=${HINT_PENALTY_HARD_FACTOR:-1.5}
HINT_GUIDANCE_DIFFICULTY=${HINT_GUIDANCE_DIFFICULTY:-moderate}

# ---- hint-selection + penalty strategy ------------------------------------
# Selects HOW a hint call is answered and penalized. The same value is given to
# BOTH the agent loop (data.hprl.strategy -> HintAgentLoop) and the reward
# (reward_kwargs.hint_strategy -> hint_reward.compute_score); they MUST agree.
#   hint        : the selector reveals ONE hint within the major step it identifies;
#                 that single hint is excluded from the next call; the reward
#                 subtracts the per-hint difficulty weight (the default behavior).
#   major_step  : the selector identifies the major step and ALL of that step's
#                 hints are injected at once; the WHOLE step goes into the rollout
#                 state and is excluded from the next call; the reward subtracts the
#                 per-step penalty directly (hint_penalty.applied_step_penalty).
HINT_STRATEGY=${HINT_STRATEGY:-major_step}

# The directory holding hint_tool.py must be importable by the rollout workers
# (the tool config's class_name "hint_tool.HintTool" is resolved with importlib).
TOOL_PYTHONPATH="${SCRIPT_DIR}"

# Local logs
RUN_ID=${RUN_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${HINT_RL_HOME}/logs"}
EXP_LOG_DIR=${EXP_LOG_DIR:-"${LOG_DIR}/${exp_name}"}
LOG_FILE=${LOG_FILE:-"${EXP_LOG_DIR}/${exp_name}.jsonl"}
CONSOLE_LOG=${CONSOLE_LOG:-"${EXP_LOG_DIR}/${exp_name}.${RUN_ID}.console.log"}
mkdir -p "${EXP_LOG_DIR}"

# ---- HPRL dynamic-budget ratchet (paper Section 7) ------------------------
# Master switch + per-experiment budget-state store (written by the trainer
# ratchet, read by the dynamic-budget dataset). HPRL_ENABLE=false -> ordinary
# multi-turn GRPO (HintBudgetDataset + HPRLRayPPOTrainer become no-ops).
HPRL_ENABLE=${HPRL_ENABLE:-True}
BUDGET_STATE_PATH=${BUDGET_STATE_PATH:-"${EXP_LOG_DIR}/budget_state.json"}
HPRL_MIN_BUDGET=${HPRL_MIN_BUDGET:-0}
HPRL_DECREMENT=${HPRL_DECREMENT:-1}
HPRL_DEFAULT_BUDGET=${HPRL_DEFAULT_BUDGET:-${max_turns}}

# Archive a verbatim snapshot of the WHOLE hint_rl script folder alongside the
# run's logs -- not just this launcher, but every HPRL source it pulls in (agent
# loop, reward fn, selector, penalty, budget manager, configs, ...). This makes
# each run reproducible from its own log dir even after the source tree changes.
# __pycache__ is excluded; the folder includes this script itself.
SRC_SNAPSHOT_DIR="${EXP_LOG_DIR}/hint_rl_src.${RUN_ID}"
mkdir -p "${SRC_SNAPSHOT_DIR}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "${SCRIPT_DIR}/" "${SRC_SNAPSHOT_DIR}/"
else
    cp -r "${SCRIPT_DIR}/." "${SRC_SNAPSHOT_DIR}/"
    find "${SRC_SNAPSHOT_DIR}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "${SRC_SNAPSHOT_DIR}" -name '*.pyc' -delete 2>/dev/null || true
fi
echo "[run_hprl] archived hint_rl source snapshot -> ${SRC_SNAPSHOT_DIR}"

# Algorithm sampling
temperature=1.0
top_p=1.0
top_k=-1

# Performance
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=1

# Per-run Ray runtime env. Env vars from this shell do NOT propagate to a
# submitted job, so wandb key, file-logger path, the selector endpoint, and the
# PYTHONPATH for the hint tool must all be injected here.
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
  # make hint_tool.py / hint_reward_manager.py importable in the job env.
  PYTHONPATH: "${TOOL_PYTHONPATH}"
# custom_reward.py + the selector client import mathruler / openai; ensure both
# exist in the job env (remove if already baked into the image).
pip:
  - mathruler
  - openai
EOF
)

# Submit to the Ray cluster's job server. The cluster must already be up.
# HPRL entry: verl's TaskRunner/run_ppo with the trainer swapped for
# HPRLRayPPOTrainer (config = config/hprl_trainer.yaml = ppo_trainer + data.hprl).
# Launched by file path; cwd is the Ray working-dir (= verl repo root) so the
# config's `searchpath: file://verl/trainer/config` resolves the base config.
ray job submit --runtime-env="${RUNTIME_ENV_RUN}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 "${SCRIPT_DIR}/main_hprl.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.hprl.enable=${HPRL_ENABLE} \
    data.hprl.budget_state_path="${BUDGET_STATE_PATH}" \
    data.hprl.min_budget=${HPRL_MIN_BUDGET} \
    data.hprl.decrement=${HPRL_DECREMENT} \
    data.hprl.default_budget=${HPRL_DEFAULT_BUDGET} \
    data.hprl.strategy=${HINT_STRATEGY} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
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
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
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
    actor_rollout_ref.rollout.val_kwargs.n=1 \
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
    +custom_reward_function.reward_kwargs.incorrect_reward=-1.0 \
    +custom_reward_function.reward_kwargs.format_reward=0.1 \
    +custom_reward_function.reward_kwargs.hint_penalty_total=${HINT_PENALTY_TOTAL} \
    +custom_reward_function.reward_kwargs.hint_penalty_hard_factor=${HINT_PENALTY_HARD_FACTOR} \
    +custom_reward_function.reward_kwargs.hint_guidance_difficulty=${HINT_GUIDANCE_DIFFICULTY} \
    +custom_reward_function.reward_kwargs.hint_strategy=${HINT_STRATEGY} \
    trainer.logger="['console','file','wandb']" \
    trainer.project_name="${wandb_project}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${LOG_DIR}/${exp_name}/rollouts" \
    trainer.validation_data_dir="${LOG_DIR}/${exp_name}/val_rollouts" \
    trainer.resume_mode=auto \
    trainer.device=cuda \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    2>&1 | tee "${CONSOLE_LOG}"
