#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# STEPHINT fully-async GRPO on Qwen3-4B-Instruct-2507.
#
# Clone of run_grpo_qwen3_4b_instruct_2507_npu_async.sh (same GRPO algorithm,
# lengths, node split, async knobs, reward -- the sibling's header documents
# all of that); the deltas are ONLY the stephint rollout scheme:
#
#   1. data -- dataset/dolci-instruct-rl-481-auto-hint-qwen3-4b-le1of8-stephint
#      .parquet: each problem carries a reference solution segmented into
#      M <= 4 parts (extra_info.segments).
#   2. rollout -- of the N = rollout.n responses in a GRPO group, the first
#      r*(M-1) (r = STEPHINT_ROLLOUTS_PER_SEGMENT, default 2) are PREFIXED
#      with solution segments 1..i (r rollouts per depth i, i = 1..M-1; the
#      final segment is never given), the remaining N - r*(M-1) >= r generate
#      from scratch. All N stay in ONE GRPO group for the advantage; the
#      prefix tokens are loss-masked (response_mask=0) so only the model's own
#      tokens receive gradient. Implemented by stephint_agent_loop.py (the
#      "stephint_agent" loop) + stephint_fully_async.py (the rollouter wrap
#      that stamps the per-rollout prefixes); entry main_stephint_async.py.
#   3. sanity check -- N >= r*M against the dataset's max segment count,
#      enforced below BEFORE submit (and per-group at runtime). At r=2 this is
#      the "N >= 2M" rule; the dataset maxes at M=4 so the sibling's
#      n_resp_per_prompt=8 passes exactly.
#
# Validation is untouched: val batches bypass the prefix stamping (they route
# through the batch generate path) and roll out plain single-turn.
#
# Cluster: bring up Ray on all pods first by pointing the platform entry at
#   bash ray_cluster_launch_stephint_qwen3_4b_instruct_2507_async.sh
# START / RESUME semantics are identical to the sibling (pin exp_name to
# resume in place; RESUME_FROM_PATH for an explicit ckpt).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

# Activate the verl conda environment
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}

# conda is not relocatable: the install prefix is baked into conda.sh and every
# script shebang. Symlink it to the current mount point so those paths resolve.
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Load secrets (wandb key, ...) from .envrc, which exports `wandb_key`.
if [ -f "${HINT_RL_HOME}/.envrc" ]; then
    source "${HINT_RL_HOME}/.envrc"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-${wandb_key:-}}"

project_name=${project_name:-'StepHint-GRPO-Qwen3-4B-Instruct-2507-dolci-481-async'}
exp_name=${exp_name:-"StepHint-GRPO-Qwen3-4B-Instruct-2507-dolci-481-async-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
wandb_project=${wandb_project:-"hint_rl"}

# ---- GRPO algorithm (identical to the plain-GRPO sibling) -------------------
adv_estimator=grpo
norm_adv_by_std_in_grpo=True
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}
loss_agg_mode="token-mean"

max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-16384}

# ---- cluster: split the NNODES pods into the two pools ----------------------
# NNODES is the TOTAL pod count (exported by the cluster launcher). Default
# 3 = 1 trainer + 2 rollout.
NNODES=${NNODES:-3}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-$((NNODES - TRAINER_NNODES))}
if [ "${TRAINER_NNODES}" -lt 1 ] || [ "${ROLLOUT_NNODES}" -lt 1 ]; then
    echo "[run_stephint_qwen3_4b_async] FATAL: need >=1 trainer node and >=1 rollout node" \
         "(got trainer=${TRAINER_NNODES} rollout=${ROLLOUT_NNODES} of NNODES=${NNODES})." >&2
    exit 1
fi
if [ "$((TRAINER_NNODES + ROLLOUT_NNODES))" -gt "${NNODES}" ]; then
    echo "[run_stephint_qwen3_4b_async] FATAL: TRAINER_NNODES(${TRAINER_NNODES}) + ROLLOUT_NNODES(${ROLLOUT_NNODES})" \
         "> NNODES(${NNODES}) -- the Ray cluster cannot host both pools." >&2
    exit 1
fi

# ---- batch geometry ---------------------------------------------------------
# train_batch_size MUST be 0 and gen_batch_size MUST be 1 in fully-async mode
# (the Rollouter asserts both): samples are streamed, not stepped. The PPO
# update geometry is unchanged from the sync job -- ppo_mini_batch_size
# prompts x rollout.n responses per optimizer step.
n_resp_per_prompt=${n_resp_per_prompt:-8}
train_prompt_mini_bsz=64

# ---- stephint rollout scheme ------------------------------------------------
# r rollouts per prefix depth (the "2 rollouts for each segment" of the
# scheme); the loop + rollouter wrap live next to this script and reach the
# Ray workers via the runtime env's PYTHONPATH below.
STEPHINT_ROLLOUTS_PER_SEGMENT=${STEPHINT_ROLLOUTS_PER_SEGMENT:-2}
STEPHINT_AGENT_CONFIG=${STEPHINT_AGENT_CONFIG:-"${SCRIPT_DIR}/stephint_agent_config.yaml"}

# ---- async pipeline knobs ---------------------------------------------------
# Identical to the plain-GRPO sibling (see its header for the sweep lore).
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
USE_TRAINER_DO_VALIDATE=${USE_TRAINER_DO_VALIDATE:-False}
ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}
# Actor sharding backend. The STOCK fully-async trainer's decoupled rollout
# correction (bypass_mode=False) recompute path snapshots weights via an
# fsdp2-only helper -- an fsdp1 actor dies on its first fit step. Fail fast
# instead of burning a launch.
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp}
if [ "${ROLLOUT_CORR_BYPASS}" != "True" ] && [ "${ACTOR_STRATEGY}" != "fsdp2" ]; then
    echo "[run_stephint_qwen3_4b_async] FATAL: ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS} needs" \
         "ACTOR_STRATEGY=fsdp2 (stock trainer's old-logprob recompute snapshots via an" \
         "fsdp2-only path; fsdp1 dies on the first fit step -- run 20260731-010034)." >&2
    exit 1
fi

# Optional HARD step bound, in PARAM VERSIONS (== sync global steps at the
# defaults above). 0/empty = off (the total_epochs bound alone applies; the
# stock config's total_rollout_steps default is a tiny 100, so null must be
# passed explicitly). The async stack bounds steps on the ROLLOUTER side in
# streamed prompt-groups: steps x ppo_mini x require_batches x trigger.
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

# Model: Qwen3-4B-Instruct-2507, used AS-IS (see the sibling's header).
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen3-4B-Instruct-2507"}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[run_stephint_qwen3_4b_async] ERROR: model dir not found: ${MODEL_PATH}" >&2
    exit 1
fi
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}

# Data: the SEGMENTED-solution dolci build (extra_info.segments per problem,
# header pt. 1) + the same 4 bare val sets as every sibling.
TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-instruct-rl-481-auto-hint-qwen3-4b-le1of8-stephint.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
TEST_FILE2=${TEST_FILE2:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
TEST_FILE3=${TEST_FILE3:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
TEST_FILE4=${TEST_FILE4:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${TEST_FILE2}','${TEST_FILE3}','${TEST_FILE4}']"}

# ---- STEPHINT SANITY CHECK: N >= r*M ----------------------------------------
# Scan TRAIN_FILE's segment counts and refuse to submit unless rollout.n covers
# r prefixed rollouts per depth 1..M-1 PLUS at least r from-scratch rollouts
# for every problem (N >= r*M; the "N >= 2M" rule at the default r=2). The
# rollouter re-checks per group at runtime, but failing here costs seconds,
# not a cluster launch. Also hard-fails on any row without segments.
MAX_SEGMENTS=$(python3 - "${TRAIN_FILE}" <<'EOF'
import json
import sys

import pandas as pd

df = pd.read_parquet(sys.argv[1], columns=["extra_info"])
max_segments = 0
for row_idx, extra_info in enumerate(df["extra_info"]):
    segments = extra_info.get("segments") if isinstance(extra_info, dict) else None
    if isinstance(segments, bytes):
        segments = segments.decode("utf-8")
    if isinstance(segments, str):
        segments = json.loads(segments)
    if segments is None or len(segments) == 0:
        sys.exit(f"row {row_idx} of {sys.argv[1]} has no extra_info.segments -- not a stephint build?")
    max_segments = max(max_segments, len(segments))
print(max_segments)
EOF
)
STEPHINT_MIN_N=$((STEPHINT_ROLLOUTS_PER_SEGMENT * MAX_SEGMENTS))
if [ "${n_resp_per_prompt}" -lt "${STEPHINT_MIN_N}" ]; then
    echo "[run_stephint_qwen3_4b_async] FATAL: stephint sanity check N >= r*M failed:" \
         "rollout.n=${n_resp_per_prompt} < r(${STEPHINT_ROLLOUTS_PER_SEGMENT}) * max M(${MAX_SEGMENTS})" \
         "= ${STEPHINT_MIN_N} over ${TRAIN_FILE}. Raise n_resp_per_prompt or lower" \
         "STEPHINT_ROLLOUTS_PER_SEGMENT." >&2
    exit 1
fi
echo "[run_stephint_qwen3_4b_async] stephint sanity check OK: rollout.n=${n_resp_per_prompt} >=" \
     "r(${STEPHINT_ROLLOUTS_PER_SEGMENT}) * max M(${MAX_SEGMENTS}) = ${STEPHINT_MIN_N}"

# Custom reward function (streams on the rollout pool via the reward loop; the
# prefix tokens sit in the response region, so the decoded solution_str the
# grader sees is prefix + continuation and the LAST \boxed{} -- the model's own
# final answer -- is what gets graded, same extractor as every sibling).
REWARD_FN_PATH=${REWARD_FN_PATH:-"${HINT_RL_HOME}/reward/custom_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}

# Trainer resume (see the sibling). RESUME_FROM_PATH wins over resume_mode=auto.
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
if [ "${RESUME_FROM_PATH}" != "null" ] && [ -n "${RESUME_FROM_PATH}" ]; then
    RESUME_MODE=${RESUME_MODE:-resume_path}
else
    RESUME_MODE=${RESUME_MODE:-auto}
fi

# Local logs
RUN_ID=${RUN_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${HINT_RL_HOME}/logs"}
EXP_LOG_DIR=${EXP_LOG_DIR:-"${LOG_DIR}/${exp_name}"}
LOG_FILE=${LOG_FILE:-"${EXP_LOG_DIR}/${exp_name}.jsonl"}
CONSOLE_LOG=${CONSOLE_LOG:-"${EXP_LOG_DIR}/${exp_name}.${RUN_ID}.console.log"}
mkdir -p "${EXP_LOG_DIR}"

# Resume-in-place (exp_name pinned to a prior run) re-enters an existing exp
# dir, and verl's FileLogger opens VERL_FILE_LOGGER_PATH with mode "wb" --
# which would TRUNCATE the prior segment's metrics. Rotate it aside first
# (this script runs on the Ray head only, so no cross-pod race); cat the
# segments in stamp order to rebuild the full curve. Fresh runs never hit
# this: their stamped exp_name makes a brand-new dir.
if [ -f "${LOG_FILE}" ]; then
    mv "${LOG_FILE}" "${LOG_FILE%.jsonl}.pre-${RUN_ID}.jsonl"
fi

# Archive a verbatim copy of this training script into the per-run log dir so
# the exact hyperparameters that produced the run are always recoverable.
cp "${BASH_SOURCE[0]}" "${EXP_LOG_DIR}/$(basename "${BASH_SOURCE[0]}").${RUN_ID}"

# Algorithm sampling
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance. The trainer pool is vLLM-free in disaggregated mode, so offload
# defaults OFF and the per-GPU token budget gets a 2x headroom multiplier
# (OFFLOAD=True PPO_TOKEN_MULT=1 restores the exact sync values).
sp_size=4
use_dynamic_bsz=True
PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-2}
actor_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
offload=${OFFLOAD:-False}
gen_tp=1

# Weight-sync bucket for the disaggregated checkpoint engine (4096 OOMed next
# to vLLM at gpu_memory_utilization=0.80 on the rollout pods; see the Olmo
# sibling's header).
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-1024}

# Per-run Ray runtime env. Env vars from this shell do NOT propagate to a
# submitted job, so the wandb key and file-logger path must ride
# runtime_env.env_vars. PYTHONPATH points every Ray worker at this script dir
# so the stephint modules (agent loop / rollouter wrap / entry import) resolve
# on all pods. Written with private perms (umask 077) so the wandb key never
# appears in the set -x trace. NO pip block: it hangs Ray forever on this
# air-gapped fabric (deps are baked into the verl conda env).
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
  PYTHONPATH: "${SCRIPT_DIR}"
EOF
)

# Fail fast with a clear message if no Ray cluster is listening. The usual
# cause: THIS script was submitted as the training platform's entry point --
# but it only SUBMITS the ray job; it must run on the Ray head after the
# cluster is up. The platform entry must be
# ray_cluster_launch_stephint_qwen3_4b_instruct_2507_async.sh, which brings up
# Ray on every pod first and then execs this script on the head.
if ! python3 -c 'import sys, urllib.request as u; u.urlopen(sys.argv[1].rstrip("/") + "/api/version", timeout=10)' "${RAY_ADDRESS}" 2>/dev/null; then
    echo "[run_stephint_qwen3_4b_async] FATAL: no Ray cluster answering at ${RAY_ADDRESS}." >&2
    echo "  This script only submits the training job to an EXISTING Ray cluster." >&2
    echo "  On the training platform, submit ray_cluster_launch_stephint_qwen3_4b_instruct_2507_async.sh" >&2
    echo "  as the entry script instead." >&2
    exit 1
fi

# Submit to the Ray cluster's job server. Entry: main_stephint_async.py --
# verl's stock fully-async main with the StepHint task runner swapped in
# (config = verl's own fully_async_ppo_trainer.yaml, addressed by absolute
# package path, so no cwd dependency and no yaml copy).
#
# Stephint-specific overrides vs the plain-GRPO sibling's submit (everything
# else below is verbatim):
#   rollout.agent.agent_loop_config_path   (registers "stephint_agent")
#   +data.stephint.enable=true             (rollouter prefix stamping gate)
#   +data.stephint.rollouts_per_segment    (r in the N >= r*M scheme)
ray job submit --runtime-env="${RUNTIME_ENV_RUN}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 "${SCRIPT_DIR}/main_stephint_async.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.stephint.enable=true \
    +data.stephint.rollouts_per_segment=${STEPHINT_ROLLOUTS_PER_SEGMENT} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${STEPHINT_AGENT_CONFIG}" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORR_BYPASS} \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
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
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB} \
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
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    +custom_reward_function.reward_kwargs.correct_reward=0.9 \
    +custom_reward_function.reward_kwargs.incorrect_reward=0.0 \
    +custom_reward_function.reward_kwargs.format_reward=0.1 \
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
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.save_freq=${SAVE_FREQ:-50} \
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
