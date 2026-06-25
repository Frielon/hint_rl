#!/usr/bin/env bash
# =============================================================================
# run_multi_eval.sh  --  entry script for the MULTI-ROUND hint-selection eval.
#   1. resolves repo root from THIS file (one level below gpt_oss_eval);
#   2. activates the verl conda env the conda-free way;
#   3. serves gpt-oss-20b on a local vLLM OpenAI endpoint (DP=#GPUs, TP=1);
#   4. (re)builds the benchmark, then runs run_multi_selection.py against it;
#   5. tears the server down on exit.
# Override any knob via env var. Results land in:
#   test_cite/gpt_oss_eval/multi-cite-gpt-eval/results/multi__gpt-oss-20b__<ts>/
# =============================================================================
set -eo pipefail

SRC="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"        # .../multi-cite-gpt-eval
GPT_OSS_EVAL="$(cd "$SCRIPT_DIR/.." && pwd)"         # .../gpt_oss_eval
TEST_CITE="$(cd "$GPT_OSS_EVAL/.." && pwd)"          # .../test_cite
SELECTOR="$(cd "$TEST_CITE/.." && pwd)"             # .../hint_rl/selector
HINT_RL_HOME="$(cd "$SELECTOR/.." && pwd)"          # .../hint_rl
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"      # .../project
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"         # mount root

# ---- config (override via env) ----------------------------------------------
MODEL_PATH="${MODEL_PATH:-$BASE_HOME/model/gpt-oss-20b}"
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"
PORT="${PORT:-30000}"
TP="${TP:-1}"; DP="${DP:-0}"
CONTEXT_LEN="${CONTEXT_LEN:-40960}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
REASONING_PARSER="${REASONING_PARSER:-openai_gptoss}"
API_KEY="${API_KEY:-EMPTY}"
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"

LABEL_RUN="${LABEL_RUN:-run_20260617_224209}"
BENCH_SEED="${BENCH_SEED:-0}"
REBUILD_BENCH="${REBUILD_BENCH:-1}"               # 1 = rebuild benchmark.jsonl each run
N_SAMPLES="${N_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.3}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
ROW_WORKERS="${ROW_WORKERS:-32}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-16}"
STEPS="${STEPS:-}"
LIMIT="${LIMIT:-}"

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"

# ---- non-relocatable miniconda: symlink baked-in prefix if needed -----------
CONDA_INSTALL_PREFIX="/share5/users/xutao.ma"
_VERL_PYTHONPATH_FALLBACK=""
if [ ! -e "$CONDA_INSTALL_PREFIX" ]; then
  ( sudo mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && sudo ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null \
    || ( mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null || true
  [ -e "$CONDA_INSTALL_PREFIX" ] && echo "[run] symlinked $CONDA_INSTALL_PREFIX -> $BASE_HOME" \
    || { echo "[run] WARN: PYTHONPATH fallback -> $SELECTOR" >&2; _VERL_PYTHONPATH_FALLBACK="$SELECTOR"; }
fi

# ---- python env activation (conda-free) -------------------------------------
ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
[ -x "$ENV_PREFIX/bin/python" ] || { echo "FATAL: env python not found: $ENV_PREFIX/bin/python" >&2; exit 1; }
export CONDA_PREFIX="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
PYTHON_BIN="$ENV_PREFIX/bin/python"
unset PYTHONPATH
[ -n "$_VERL_PYTHONPATH_FALLBACK" ] && export PYTHONPATH="$_VERL_PYTHONPATH_FALLBACK"
export PYTHONNOUSERSITE=1
CLEAN_LD=""
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  CLEAN_LD="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -e '/opt/venv' -e '/usr/local/lib/python3' | paste -sd: -)"
fi
SITE="$("$ENV_PREFIX/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
SITE="${SITE:-$ENV_PREFIX/lib/python3.11/site-packages}"
NV_LIBS="$(find "$SITE/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${NV_LIBS:+:$NV_LIBS}${CLEAN_LD:+:$CLEAN_LD}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=1

if [ "$DP" -le 0 ] 2>/dev/null; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then DP="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
  elif command -v nvidia-smi >/dev/null 2>&1; then DP="$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)"
  else DP=8; fi
fi
[ -d "$MODEL_PATH" ] || { echo "FATAL: MODEL_PATH not a dir: $MODEL_PATH" >&2; exit 1; }

echo "=================================================================="
echo " run_multi_eval  (multi-round hint selection)"
echo "   model    : $MODEL_PATH  (served as '$SERVED_NAME')  dp=$DP tp=$TP"
echo "   endpoint : http://127.0.0.1:$PORT/v1"
echo "   label    : $LABEL_RUN   (n=$N_SAMPLES temp=$TEMPERATURE)"
echo "   prompt   : $SCRIPT_DIR/prompt_template_multiF.py"
echo "=================================================================="

# ---- build benchmark (before serving; cheap, no GPU) ------------------------
BENCH="$SCRIPT_DIR/benchmark.jsonl"
if [ "$REBUILD_BENCH" = "1" ] || [ ! -f "$BENCH" ]; then
  echo "[run] building benchmark -> $BENCH"
  "$PYTHON_BIN" "$SCRIPT_DIR/build_benchmark.py" --label-run "$LABEL_RUN" --seed "$BENCH_SEED" --out "$BENCH"
fi

# ---- launch vLLM, wait, run driver, tear down ------------------------------
SERVER_PID=""
cleanup() { echo ""; echo "[run] tearing down server..."; if [ -n "$SERVER_PID" ]; then
  kill "$SERVER_PID" 2>/dev/null || true; pkill -9 -P "$SERVER_PID" 2>/dev/null || true; kill -9 "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM

SERVER_LOG="$LOG_DIR/vllm_${SERVED_NAME}_${PORT}.log"
echo "[run] starting vLLM -> $SERVER_LOG"
API_KEY_ARG=(); [ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && API_KEY_ARG=(--api-key "$API_KEY")
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_FRAC" --max-model-len "$CONTEXT_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --reasoning-parser "$REASONING_PARSER" \
    --trust-remote-code "${API_KEY_ARG[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[run] waiting for readiness (DP=$DP load can take minutes)..."
AUTH_HDR=(); [ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && AUTH_HDR=(-H "Authorization: Bearer $API_KEY")
ready=0
for i in $(seq 1 240); do
  if curl -s "${AUTH_HDR[@]}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    ready=1; echo "[run] server up after ${i}x5s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[run] FATAL: server died; tail log:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "[run] FATAL: not ready in time" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }

echo "[run] server READY; launching multi-round scorer."
DRIVER_ARGS=(
  --benchmark "$BENCH"
  --base-url "http://127.0.0.1:${PORT}/v1" --api-key "$API_KEY" --model "$SERVED_NAME"
  -n "$N_SAMPLES" --temperature "$TEMPERATURE" --top-p "$TOP_P" --max-tokens "$MAX_TOKENS"
  --row-workers "$ROW_WORKERS" --sample-workers "$SAMPLE_WORKERS"
)
[ -n "$STEPS" ] && DRIVER_ARGS+=(--steps "$STEPS")
[ -n "$LIMIT" ] && DRIVER_ARGS+=(--limit "$LIMIT")
"$PYTHON_BIN" "$SCRIPT_DIR/run_multi_selection.py" "${DRIVER_ARGS[@]}"
RC=$?
echo "[run] scorer finished (rc=$RC). Server log: $SERVER_LOG"
exit $RC
