#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# DAPO variant for Qwen3-8B-Base. This is a copy of run_grpo_qwen3_8b_npu.sh
# with the algorithm switched from plain GRPO to DAPO; the model, datasets,
# lr schedule, batch / mini-batch sizes, n, lengths, sp_size, offload,
# gpu-mem, dynamic bsz, gradient checkpointing, logging and ckpt layout are
# kept identical so the only difference vs. the GRPO run is the algorithm.
#
#   DAPO changes vs. run_grpo_qwen3_8b_npu.sh
#     * Entry point recipe.dapo.main_dapo (the dynamic-sampling DAPO trainer)
#       instead of verl.trainer.main_ppo.
#     * Dynamic sampling: algorithm.filter_groups.enable=True with metric=acc
#       drops prompts whose rollout group is all-correct or all-wrong (their
#       group advantage is identically zero) and keeps generating -- rounds of
#       gen_batch_size = 3x train_batch_size prompts, at most
#       max_num_gen_batches rounds -- until train_batch_size prompts with
#       mixed outcomes are accumulated; training still sees exactly
#       train_prompt_bsz prompts x n responses per step.
#     * Overlong soft penalty: reward manager 'dapo' + overlong_buffer_cfg;
#       the last overlong_buffer_len tokens of the response budget are a
#       soft-punish zone where reward ramps linearly down to -penalty_factor
#       at full length. Sized 2048 = 1/4 of the 8192 budget, the same ratio
#       as the DAPO paper's 4096-of-16384.
#     * Reward scale: correct=0.9 / incorrect=0.0 (the GRPO run's scale)
#       instead of the DAPO paper's +-1 that overlong_penalty_factor=1.0 was
#       calibrated against. format_reward stays 0.1.
#     * clip_ratio_c=10.0 (dual-clip PPO lower bound, from the DAPO recipe).
#   Already DAPO-style in the GRPO script and kept unchanged:
#     * clip-higher: clip_ratio_low=0.2 / clip_ratio_high=0.28.
#     * No KL: use_kl_in_reward=False, use_kl_loss=False, coefs 0.
#     * loss_agg_mode=token-mean (token-level policy-gradient loss).
#     * norm_adv_by_std_in_grpo=True (standard (r - mean)/std group adv).
#
# Config-style NOTE: this verl's migrate_legacy_reward_impl does NOT map the
# legacy reward_model.overlong_buffer.* keys (reward_model is deleted at
# startup without migrating them), so the manager + overlong buffer are passed
# new-style as reward.reward_manager.name / reward.reward_kwargs.*, matching
# upstream recipe/dapo/run_dapo_qwen3_8b_base_npu.sh. Do NOT copy the
# reward_model.* flags from run_dapo_qwen2.5_7b_npu.sh -- on current verl they
# are dropped silently. custom_reward_function.* stays legacy top-level (it is
# migrated to reward.custom_reward_function at startup), same as the working
# GRPO script.
#
# All paths are derived from this script's own location so the tree can be
# mounted anywhere. Expected layout (relative positions must be preserved):
#
#   <base>/
#     miniconda3/                 -> CONDA_HOME
#     model/Qwen3-8B-Base/        -> MODEL_PATH
#     project/
#       verl/                     -> VERL_HOME
#       hint_rl/                  -> HINT_RL_HOME
#         script/<this script>
#         dataset/ reward/ ckpt/
#
# Any path can still be overridden from the environment (VAR=... ./script.sh).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

# Activate the verl conda environment
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}

# miniconda was installed with the prefix below baked into conda.sh and every
# script shebang (conda, torchrun, ray, pip, vllm ...). Conda is not
# relocatable, so when this tree is mounted somewhere else (e.g. ${BASE_HOME})
# those absolute paths must still resolve. Symlink the original prefix to the
# current mount point so the baked-in paths keep working.
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Load secrets (wandb key, ...) from .envrc, which exports `wandb_key`.
# wandb expects WANDB_API_KEY, so map it across.
if [ -f "${HINT_RL_HOME}/.envrc" ]; then
    source "${HINT_RL_HOME}/.envrc"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-${wandb_key:-}}"

project_name='DAPO-Qwen3-8B-Base'
# Fresh timestamped run. To resume an interrupted run in place instead, pin
# exp_name to the existing checkpoint dir name (as run_grpo_qwen3_8b_npu.sh
# does): CKPTS_DIR then points at it and trainer.resume_mode=auto reads its
# latest_checkpointed_iteration.txt, keeping the same wandb curve, log dir and
# checkpoint dir.
# exp_name="DAPO-Qwen3-8B-Base-dolci-zero-rl-8192-16-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"
exp_name="DAPO-Qwen3-8B-Base-dolci-zero-rl-8192-16-20260727-223725"
wandb_project=${wandb_project:-"hint_rl"}

adv_estimator=grpo
# DAPO keeps the standard GRPO group-normalized advantage (r - mean)/std.
norm_adv_by_std_in_grpo=True

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
# DAPO clip-higher (asymmetric clip): lower clip 0.2, upper clip 0.28.
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=2048
max_response_length=8192
# Overlong soft penalty: responses that enter the last overlong_buffer_len
# tokens of the budget get a linearly growing penalty, reaching
# -overlong_penalty_factor at the full response length.
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0
# DAPO token-level PG loss (identical to the GRPO script).
loss_agg_mode="token-mean"
# Dynamic sampling (see header): filter zero-advantage prompt groups on the
# "acc" value returned by the custom reward fn, regenerate until the train
# batch is full, at most max_num_gen_batches generation rounds per step.
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Cluster: NNODES x 8 GPUs (default 2 x 8 H100 = 16). This requires a Ray
# cluster spanning all nodes (see the `ray job submit` at the bottom); the job
# is dispatched across whatever GPUs the cluster actually has.
NNODES=${NNODES:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# Dynamic-sampling oversample: each generation round rolls out 3x the train
# batch so a single round usually fills the 64 kept prompts. The optimizer
# still sees exactly train_prompt_bsz prompts x n responses per step, so the
# per-step training math is unchanged vs. the GRPO run.
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# Ray
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${VERL_HOME}"}
# The Ray runtime env is generated per-run just before submit (see below), so we
# can inject secrets/log paths via env_vars; ${VERL_HOME}/recipe/dapo/runtime_env.yaml
# is the upstream template it mirrors.
# Paths
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen3-8B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}
# Single-turn, dapo_17k-style prompts (no agent_name column), so every row routes
# through verl's built-in single_turn_agent -> plain single-turn DAPO, no hint
# agent loop. Use dapo-3139-hint-verl-mt-clean.parquet (agent_name="hint_agent")
# only with the HPRL launcher (run_hprl), which registers that loop.
TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/Dolci-RL-Zero-Math-7B_dapo_formatted-single-turn.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
TEST_FILE2=${TEST_FILE2:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
TEST_FILE3=${TEST_FILE3:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
TEST_FILE4=${TEST_FILE4:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}

# Custom reward function (loaded by verl via custom_reward_function.path/.name).
# It returns a dict with an "acc" key, which filter_groups.metric=acc reads.
REWARD_FN_PATH=${REWARD_FN_PATH:-"${HINT_RL_HOME}/reward/custom_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}

# Local logs
#   - LOG_FILE   : verl 'file' logger, one JSON object of metrics per step
#   - CONSOLE_LOG: full stdout/stderr of the training driver (text)
RUN_ID=${RUN_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${HINT_RL_HOME}/logs"}
EXP_LOG_DIR=${EXP_LOG_DIR:-"${LOG_DIR}/${exp_name}"}
LOG_FILE=${LOG_FILE:-"${EXP_LOG_DIR}/${exp_name}.jsonl"}
CONSOLE_LOG=${CONSOLE_LOG:-"${EXP_LOG_DIR}/${exp_name}.${RUN_ID}.console.log"}
mkdir -p "${EXP_LOG_DIR}"

# Archive a verbatim copy of this training script into the per-run log dir so the
# exact hyperparameters that produced the run are always recoverable alongside its
# logs/checkpoints. RUN_ID in the name keeps re-launches of the same exp distinct.
cp "${BASH_SOURCE[0]}" "${EXP_LOG_DIR}/$(basename "${BASH_SOURCE[0]}").${RUN_ID}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=1

# Build a per-run Ray runtime env. Env vars from this shell do NOT propagate to
# a submitted job (the driver/workers run elsewhere, often inside a container),
# so the wandb key and file-logger path must be injected via runtime_env.env_vars
# -- this is the mechanism Ray actually guarantees to forward. Written with
# private perms (umask 077) so the wandb key never appears in the set -x trace.
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
# The job runs in the cluster's container (e.g. /opt/venv), not the conda env we
# activated for submitting. custom_reward.py imports mathruler, so ensure it is
# present in the job env. Remove this if mathruler is already baked into the image.
pip:
  - mathruler
EOF
)

# Submit to the Ray cluster's job server (dashboard at ${RAY_ADDRESS}). This
# spans all nodes of the cluster, which is required for the multi-node
# (NNODES>1) run. The cluster must already be up: a head with
# `ray start --head --dashboard-port 8265` and the worker(s) joined via
# `ray start --address <head>:6379`. Run attached so the driver's console
# output streams back here; tee it to a local console log.
ray job submit --runtime-env="${RUNTIME_ENV_RUN}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="['${TEST_FILE}','${TEST_FILE2}','${TEST_FILE3}','${TEST_FILE4}']" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
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
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    +custom_reward_function.reward_kwargs.correct_reward=0.9 \
    +custom_reward_function.reward_kwargs.incorrect_reward=0.0 \
    +custom_reward_function.reward_kwargs.format_reward=0.1 \
    trainer.logger="['console','file','wandb']" \
    trainer.project_name="${wandb_project}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    `# trainer.max_actor_ckpt_to_keep=1` \
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
