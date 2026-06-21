#!/usr/bin/env bash
# =============================================================================
# run_gpt_oss_eval.sh  --  SINGLE entry script for the training platform.
# -----------------------------------------------------------------------------
# On ONE node with 8x H100 it:
#   1. resolves the repo root from THIS file's location (works at /share5/...,
#      /xutao/..., or any mount the platform hands us -- all paths relative);
#   2. activates the `verl` conda env the conda-free way (the env interpreter
#      directly -- the miniconda prefix is baked in absolutely, so we never
#      `conda activate`);
#   3. serves gpt-oss-20b on a LOCAL vLLM OpenAI endpoint (127.0.0.1), DP=#GPUs
#      (8), TP=1, with --reasoning-parser openai_gptoss (gpt-oss routes its
#      analysis CoT to reasoning_content, the JSON answer to content);
#   4. waits for readiness, then runs run_gpt_oss_selection.py against it to
#      score gpt-oss on the Template F hint-selection labels;
#   5. tears the server down on exit (success, failure, or Ctrl-C).
#
# Point the platform's entrypoint at THIS one file. The platform may append junk
# positional args; they are ignored. Override any knob below via env var.
#
#   results land in:  test_cite/gpt_oss_eval/results/<label_run>__gpt-oss-20b/
# =============================================================================
set -eo pipefail          # NOT -u: conda activate hooks reference unbound vars

# --- path bootstrap (relative to this file) ----------------------------------
SRC="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"        # .../test_cite/gpt_oss_eval
TEST_CITE="$(cd "$SCRIPT_DIR/.." && pwd)"           # .../test_cite
SELECTOR="$(cd "$TEST_CITE/.." && pwd)"             # .../hint_rl/selector
HINT_RL_HOME="$(cd "$SELECTOR/.." && pwd)"          # .../hint_rl
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"      # .../project
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"         # mount root (== /share5/users/xutao.ma)
VERL_HOME="$PROJECT_HOME/verl"

# =============================================================================
# CONFIG  -- override via env var; everything resolves on any mount
# =============================================================================
MODEL_PATH="${MODEL_PATH:-$BASE_HOME/model/gpt-oss-20b}"   # MXFP4 MoE weights
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"
PORT="${PORT:-30000}"
TP="${TP:-1}"                          # gpt-oss fits on one H100 -> TP=1
DP="${DP:-0}"                          # 0 => auto = #visible GPUs (8)
CONTEXT_LEN="${CONTEXT_LEN:-40960}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
REASONING_PARSER="${REASONING_PARSER:-openai_gptoss}"
API_KEY="${API_KEY:-EMPTY}"
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"

# --- driver knobs (passed through to run_gpt_oss_selection.py) ----------------
LABEL_RUN="${LABEL_RUN:-run_20260617_224209}"   # which labeled run to score against
N_SAMPLES="${N_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
ROW_WORKERS="${ROW_WORKERS:-32}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-16}"
STEPS="${STEPS:-}"                              # e.g. "1,2" ; empty = all
LIMIT="${LIMIT:-}"                              # e.g. "5" for a smoke test ; empty = all
# prompt source: local = rebuild from the EDITABLE prompt_template_F.py in this
# dir (default, so your prompt revisions drive the run); stored = the exact
# prompt Codex saw (apples-to-apples, ignores local edits).
PROMPT_SOURCE="${PROMPT_SOURCE:-local}"
# =============================================================================

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- non-relocatable miniconda: symlink the baked-in install prefix -----------
# conda.sh / shebangs / the editable verl .pth bake the ORIGINAL absolute prefix
# (/share5/users/xutao.ma). When the tree is mounted elsewhere, symlink it.
CONDA_INSTALL_PREFIX="/share5/users/xutao.ma"
_VERL_PYTHONPATH_FALLBACK=""
if [ ! -e "$CONDA_INSTALL_PREFIX" ]; then
  ( sudo mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && sudo ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null \
    || ( mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null \
    || true
  if [ -e "$CONDA_INSTALL_PREFIX" ]; then
    echo "[run] symlinked baked-in conda prefix $CONDA_INSTALL_PREFIX -> $BASE_HOME"
  else
    echo "[run] WARN: could not symlink $CONDA_INSTALL_PREFIX; PYTHONPATH fallback -> $VERL_HOME" >&2
    _VERL_PYTHONPATH_FALLBACK="$VERL_HOME"
  fi
fi

# --- python env activation (conda-free; verbatim from launch_gpt_oss_serve.sh) -
ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "FATAL: env python not found/executable: $ENV_PREFIX/bin/python" >&2
  echo "       (set CONDA_ROOT/CONDA_ENV to match your image)" >&2
  exit 1
}
export CONDA_PREFIX="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
PYTHON_BIN="$ENV_PREFIX/bin/python"
unset PYTHONPATH
[ -n "$_VERL_PYTHONPATH_FALLBACK" ] && export PYTHONPATH="$_VERL_PYTHONPATH_FALLBACK"
export PYTHONNOUSERSITE=1
CLEAN_LD=""
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  CLEAN_LD="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
    | grep -v -e '/opt/venv' -e '/usr/local/lib/python3' | paste -sd: -)"
fi
SITE="$("$ENV_PREFIX/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
SITE="${SITE:-$ENV_PREFIX/lib/python3.11/site-packages}"
NV_LIBS="$(find "$SITE/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${NV_LIBS:+:$NV_LIBS}${CLEAN_LD:+:$CLEAN_LD}"
CUDA_TK=""
for c in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12 \
         "$SITE/nvidia/cu13" "$SITE/nvidia/cuda_nvcc"; do
  [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA_TK="$c"; break; }
done
if [ -n "$CUDA_TK" ]; then
  export CUDA_HOME="$CUDA_TK"; export PATH="$CUDA_HOME/bin:$PATH"
  echo "[run] CUDA toolkit: $CUDA_HOME"
elif [ -d "$SITE/nvidia/cuda_runtime/lib" ]; then
  export CUDA_HOME="$SITE/nvidia/cuda_runtime"
fi
echo "[run] activated env (conda-free): $ENV_PREFIX  python=$PYTHON_BIN"

# vLLM: native Torch sampler (avoid the flashinfer JIT toolkit/torch mismatch).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1

# --- DP = #visible GPUs (auto when DP=0) -------------------------------------
if [ "$DP" -le 0 ] 2>/dev/null; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DP="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DP="$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)"
  else
    DP=8
  fi
fi

[ -d "$MODEL_PATH" ] || { echo "FATAL: MODEL_PATH not a dir: $MODEL_PATH" >&2; exit 1; }

echo "=================================================================="
echo " run_gpt_oss_eval  (serve + score, single node 8x H100)"
echo "   model       : $MODEL_PATH  (served as '$SERVED_NAME')"
echo "   topology    : dp=$DP tp=$TP  ctx=$CONTEXT_LEN mem=$MEM_FRAC max_num_seqs=$MAX_NUM_SEQS"
echo "   reasoning   : --reasoning-parser $REASONING_PARSER"
echo "   endpoint    : http://127.0.0.1:$PORT/v1  (local)"
echo "   label run   : $LABEL_RUN   (n=$N_SAMPLES temp=$TEMPERATURE max_tokens=$MAX_TOKENS)"
echo "   prompt src  : $PROMPT_SOURCE  ($SCRIPT_DIR/prompt_template_F.py)"
echo "=================================================================="

# --- launch vLLM, wait for readiness, run driver, tear down ------------------
SERVER_PID=""
cleanup() {
  echo ""
  echo "[run] tearing down server..."
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVER_LOG="$LOG_DIR/vllm_${SERVED_NAME}_${PORT}.log"
echo "[run] starting vLLM -> $SERVER_LOG"
API_KEY_ARG=()
[ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && API_KEY_ARG=(--api-key "$API_KEY")
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_FRAC" \
    --max-model-len "$CONTEXT_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --reasoning-parser "$REASONING_PARSER" \
    --trust-remote-code \
    "${API_KEY_ARG[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[run] waiting for readiness (DP=$DP load can take minutes)..."
AUTH_HDR=()
[ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && AUTH_HDR=(-H "Authorization: Bearer $API_KEY")
ready=0
for i in $(seq 1 240); do   # up to 20 min
  if curl -s "${AUTH_HDR[@]}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    ready=1; echo "[run] server up after ${i}x5s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] FATAL: server died during startup; tail of log:" >&2
    tail -n 40 "$SERVER_LOG" >&2; exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "[run] FATAL: not ready in time" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }

echo "[run] server READY; launching scorer."
DRIVER_ARGS=(
  --label-run "$LABEL_RUN"
  --base-url "http://127.0.0.1:${PORT}/v1"
  --api-key "$API_KEY"
  --model "$SERVED_NAME"
  -n "$N_SAMPLES"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --row-workers "$ROW_WORKERS"
  --sample-workers "$SAMPLE_WORKERS"
  --prompt-source "$PROMPT_SOURCE"
)
[ -n "$STEPS" ] && DRIVER_ARGS+=(--steps "$STEPS")
[ -n "$LIMIT" ] && DRIVER_ARGS+=(--limit "$LIMIT")

"$PYTHON_BIN" "$SCRIPT_DIR/run_gpt_oss_selection.py" "${DRIVER_ARGS[@]}"
RC=$?

echo "[run] scorer finished (rc=$RC). Server log: $SERVER_LOG"
exit $RC
