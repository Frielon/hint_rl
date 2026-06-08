#!/usr/bin/env bash
# strict mode is enabled WITHOUT -u so unbound vars in the cuda-nvcc
# conda activate hook don't trip us up
set -eo pipefail

# NOTE: don't name this var HOST — conda's cuda-nvcc activate hook
# overwrites $HOST with the GCC target triple (x86_64-conda-linux-gnu),
# which then propagates into the sglang --host flag and breaks bind.
MODEL_PATH="${MODEL_PATH:-/share5/users/xutao.ma/model/Qwen2.5-Math-7B}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-Math-7B}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
GPU="${GPU:-1}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_LEN="${MAX_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
LOG_DIR="${LOG_DIR:-/share5/users/xutao.ma/project/hint_rl/data_pipeline/inference/logs}"

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
echo "  GPU:         $GPU"
echo "  log:         $LOG_FILE"

exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host "$SERVE_HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --mem-fraction-static "$MEM_FRAC" \
    --context-length "$MAX_LEN" \
    2>&1 | tee "$LOG_FILE"
