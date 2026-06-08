#!/usr/bin/env bash
# Launch openai/gpt-oss-20b (MoE, MXFP4) on a single A100-40GB with sglang.
# strict mode is enabled WITHOUT -u so unbound vars in the cuda-nvcc
# conda activate hook don't trip us up
set -eo pipefail

# NOTE: don't name this var HOST — conda's cuda-nvcc activate hook
# overwrites $HOST with the GCC target triple (x86_64-conda-linux-gnu),
# which then propagates into the sglang --host flag and breaks bind.
MODEL_PATH="${MODEL_PATH:-/share5/users/xutao.ma/model/gpt-oss-20b}"
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
# ~13.8 GB MXFP4 weights + small active MoE -> fits on one 40GB card. TP=1.
GPU="${GPU:-1}"
TP="${TP:-1}"
MEM_FRAC="${MEM_FRAC:-0.9}"
MAX_LEN="${MAX_LEN:-40960}"
# gpt-oss separates chain-of-thought (analysis channel) from the final answer;
# the reasoning parser routes CoT to reasoning_content so the JSON we want
# lands in message.content.
REASONING_PARSER="${REASONING_PARSER:-gpt-oss}"
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
# cublas.h, ...); those live in the pip-installed nvidia/cu13 wheel.
NV_CU13="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13"
export CPATH="$NV_CU13/include:${CPATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NV_CU13/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NV_CU13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Launching sglang server"
echo "  model:       $MODEL_PATH"
echo "  served name: $SERVED_NAME"
echo "  host:port:   $SERVE_HOST:$PORT"
echo "  GPU:         $GPU  (tp=$TP)"
echo "  context:     $MAX_LEN"
echo "  reasoning:   $REASONING_PARSER"
echo "  log:         $LOG_FILE"

exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host "$SERVE_HOST" \
    --port "$PORT" \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRAC" \
    --context-length "$MAX_LEN" \
    --reasoning-parser "$REASONING_PARSER" \
    --trust-remote-code \
    2>&1 | tee "$LOG_FILE"
