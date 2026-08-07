#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Plain GRPO variant, kept as an apples-to-apples sibling of
# run_drgrpo_qwen2.5_7b_npu.sh. It is standard GRPO with the clip-higher
# (asymmetric clip) trick and NO KL penalty, but WITHOUT the two Dr. GRPO
# unbiased-estimator fixes:
#
#   GRPO changes vs. run_drgrpo_qwen2.5_7b_npu.sh
#     * algorithm.norm_adv_by_std_in_grpo=True  -> advantage is (r - mean)/std,
#       the standard GRPO group-normalized advantage (keeps the std divide that
#       Dr. GRPO drops to remove question-difficulty bias).
#     * actor.loss_agg_mode=seq-mean-token-mean -> per-sequence token mean,
#       then mean over sequences (the original GRPO sample-level loss),
#       instead of Dr. GRPO's seq-mean-token-sum-norm constant-length
#       normalization.
#   Kept the same as the Dr. GRPO run (per the request):
#     * No KL: use_kl_in_reward=False, use_kl_loss=False, coefs 0.
#     * clip-higher: asymmetric clip with clip_ratio_low / clip_ratio_high.
#   Entry point is verl.trainer.main_ppo (the standard GRPO trainer).
#
# Everything else (model, data, lr schedule, batch sizes, n, lengths, sp_size,
# offload, gpu-mem, dynamic bsz, gradient checkpointing, logging, ckpt) is kept
# identical so GRPO vs Dr. GRPO is an apples-to-apples comparison.
#
# All paths are derived from this script's own location so the tree can be
# mounted anywhere. Expected layout (relative positions must be preserved):
#
#   <base>/
#     miniconda3/                 -> CONDA_HOME
#     model/Qwen2.5-7B-Instruct/  -> MODEL_PATH
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

project_name='GRPO-Qwen2.5-7B-Instruct'
exp_name="GRPO-Qwen2.5-7B-Instruct-dapo-512-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"
wandb_project=${wandb_project:-"hint_rl"}

adv_estimator=grpo
# GRPO: divide the advantage by the group std (standard group normalization).
norm_adv_by_std_in_grpo=True

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
# DAPO-style asymmetric clip (clip-higher): lower clip 0.2, upper clip 0.28.
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=2048
max_response_length=4096
# GRPO: sample-level loss — mean over tokens within each sequence, then mean
# over sequences (the original GRPO objective).
loss_agg_mode="seq-mean-token-mean"

# Cluster: 6 nodes x 8 H100 = 48 GPUs. This requires a Ray cluster spanning
# all nodes (see the `ray job submit` at the bottom); the job is dispatched
# across whatever GPUs the cluster actually has.
NNODES=${NNODES:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# No dynamic-sampling oversample here, so the generation batch equals the
# training batch (the trainer generates train_prompt_bsz prompts per step).
train_prompt_bsz=64
n_resp_per_prompt=16
train_prompt_mini_bsz=64

# Ray
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${VERL_HOME}"}
# The Ray runtime env is generated per-run just before submit (see below), so we
# can inject secrets/log paths via env_vars; ${VERL_HOME}/recipe/dapo/runtime_env.yaml
# is the upstream template it mirrors.
# Paths
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}
# Single-turn, dapo_17k-style prompts (no agent_name column), so every row routes
# through verl's built-in single_turn_agent -> plain single-turn GRPO, no hint
# agent loop. Use dapo-3139-hint-verl-mt-clean.parquet (agent_name="hint_agent")
# only with the HPRL launcher (run_hprl), which registers that loop.
# TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/StepHint_train.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-rl-zero-9517-auto-hint-single-turn.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
TEST_FILE2=${TEST_FILE2:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
TEST_FILE3=${TEST_FILE3:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
TEST_FILE4=${TEST_FILE4:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}

# Custom reward function (loaded by verl via custom_reward_function.path/.name)
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
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="['${TEST_FILE}','${TEST_FILE2}','${TEST_FILE3}','${TEST_FILE4}']" \
    data.prompt_key=prompt \
    data.truncation='left' \
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
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=naive \
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
    trainer.test_freq=20 \
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
