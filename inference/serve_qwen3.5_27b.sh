#!/usr/bin/env bash
# Launch Qwen3.5-27B on 2x A100-40GB with sglang (tp=2).
# strict mode is enabled WITHOUT -u so unbound vars in the cuda-nvcc
# conda activate hook don't trip us up
set -eo pipefail

# NOTE: don't name this var HOST — conda's cuda-nvcc activate hook
# overwrites $HOST with the GCC target triple (x86_64-conda-linux-gnu),
# which then propagates into the sglang --host flag and breaks bind.
MODEL_PATH="${MODEL_PATH:-/share5/users/xutao.ma/model/Qwen3.5-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-27B}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
# 2x A100-40GB. TP=2 is the only viable multi-GPU split for this model:
# num_key_value_heads=4 and intermediate_size=17408 are not divisible by 3,
# so TP=3 is impossible — TP must be a power of two here.
GPU="${GPU:-0,1}"
TP="${TP:-2}"
# weights are ~27.8 GB/GPU at TP=2, leaving ~8 GB/GPU for KV cache
# (head_dim=256 makes KV heavy: ~128 KB/token/GPU). keep context modest.
MEM_FRAC="${MEM_FRAC:-0.9}"
MAX_LEN="${MAX_LEN:-16384}"
DTYPE="${DTYPE:-bfloat16}"
LOG_DIR="${LOG_DIR:-/share5/users/xutao.ma/project/hint_rl/inference/logs}"

CONDA_SH="/shared_home/xutao.ma/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="inference"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_${SERVED_NAME}_${PORT}.log"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

# flashinfer JITs CUDA kernels; point it at the env's bundled nvcc/headers
# and at the system libcuda.so (driver stub lives in /usr/lib/x86_64-linux-gnu)
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
# conda's cuda-nvcc ships the compiler but no CUDA library headers (curand.h,
# cublas.h, ...); those live in the pip-installed nvidia/cu13 wheel. Without
# this, flashinfer's JIT compile of the sampler kernel fails with
# `fatal error: curand.h: No such file or directory` and sglang SIGQUITs.
NV_CU13="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13"
export CPATH="$NV_CU13/include:${CPATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NV_CU13/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NV_CU13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Launching sglang server"
echo "  model:       $MODEL_PATH"
echo "  served name: $SERVED_NAME"
echo "  host:port:   $SERVE_HOST:$PORT"
echo "  GPUs:        $GPU  (tp=$TP)"
echo "  context:     $MAX_LEN"
echo "  log:         $LOG_FILE"

exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host "$SERVE_HOST" \
    --port "$PORT" \
    --tp "$TP" \
    --dtype "$DTYPE" \
    --mem-fraction-static "$MEM_FRAC" \
    --context-length "$MAX_LEN" \
    --trust-remote-code \
    2>&1 | tee "$LOG_FILE"
