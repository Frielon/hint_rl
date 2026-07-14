#!/usr/bin/env bash
#
# run_eval_base.sh -- evaluate the BASE model
#   /share5/users/xutao.ma/model/Qwen2.5-7B-Instruct
# on the eval datasets: BARE single-turn, NO --hint-mode, box-scored with
# mathruler (directly comparable to GRPO / AutoHint numbers).
#
# The base model is already a plain HF checkpoint, so there is NO FSDP merge:
# it is served directly on a local vLLM OpenAI endpoint (DP=8, TP=1), and
# eval_ckpts.py rolls every problem out N times and grades the boxed answer.
#
# Datasets (all bare, hint_flag=0):
#   aime2025.parquet  aime2024.parquet  dapo_sample_hard_100.parquet
#
# All paths derive from this script's location. Override via env:
#   PORT CONTEXT_LEN MEM_FRAC MAX_NUM_SEQS                 (server)
#   N CHUNK CONCURRENCY TEMPERATURE TOP_P TOP_K MAX_TOKENS LIMIT   (eval)
#   CONDA_ROOT CONDA_ENV PYTHON_BIN                        (env)
#   BASE_MODEL           full path to the HF model dir
# ---------------------------------------------------------------------------
set -eo pipefail

# --- resolve repo paths relative to this script ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../eval
HINT_RL_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"                        # .../hint_rl
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"                      # .../project
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"                         # mount root
EVAL_PY="$SCRIPT_DIR/eval_ckpts.py"
VERL_HOME="${VERL_HOME:-$PROJECT_HOME/verl}"

# --- non-relocatable miniconda: symlink the baked-in install prefix ----------
CONDA_INSTALL_PREFIX="${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}"
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
  ( sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")" && sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}" ) 2>/dev/null \
    || ( mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")" && ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}" ) 2>/dev/null \
    || true
  if [ -e "${CONDA_INSTALL_PREFIX}" ]; then
    echo "[eval] symlinked baked-in conda prefix ${CONDA_INSTALL_PREFIX} -> ${BASE_HOME}"
  else
    echo "[eval] WARN: could not symlink ${CONDA_INSTALL_PREFIX}; adding ${VERL_HOME} to PYTHONPATH as fallback" >&2
    _VERL_PYTHONPATH_FALLBACK="${VERL_HOME}"
  fi
fi

# --- config (env-overridable) ---------------------------------------------
PORT="${PORT:-30000}"
CONTEXT_LEN="${CONTEXT_LEN:-36864}"
MEM_FRAC="${MEM_FRAC:-0.85}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
N="${N:-16}"
CHUNK="${CHUNK:-1}"
CONCURRENCY="${CONCURRENCY:-192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
LIMIT="${LIMIT:-}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- the BASE model to evaluate (plain HF dir, no FSDP merge) ---------------
BASE_MODEL="${BASE_MODEL:-/share5/users/xutao.ma/model/Qwen3-8B}"
LABEL="qwen3-8b"
SERVED_NAME="qwen3-8b"

# --- datasets (all BARE, box-scored like GRPO) -----------------------------
DATASET_DIR="$HINT_RL_HOME/dataset"
EVAL_SETS=(
  "$DATASET_DIR/hmmt_nov_2025.parquet"
  # "$DATASET_DIR/acereason_math_sample_1024.parquet"
  "$DATASET_DIR/aime2025.parquet"
  "$DATASET_DIR/aime2024.parquet"
  "$DATASET_DIR/dapo_sample_hard_100.parquet"
)

MERGED_DIR="${MERGED_DIR:-$SCRIPT_DIR/merged}"

# --- output ----------------------------------------------------------------
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/results/base_${RUN_TS}}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR" "$MERGED_DIR"

# --- python env activation (conda-free) ------------------------------------
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "$CONDA_ENV" ]; then
  ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
  [ -x "$ENV_PREFIX/bin/python" ] || {
    echo "FATAL: env python not found/executable: $ENV_PREFIX/bin/python" >&2
    echo "       (set CONDA_ROOT/CONDA_ENV to match your image, or CONDA_ENV='' to skip)" >&2
    exit 1
  }
  export CONDA_PREFIX="$ENV_PREFIX"
  export PATH="$ENV_PREFIX/bin:$PATH"
  [ "$PYTHON_BIN" = "python" ] && PYTHON_BIN="$ENV_PREFIX/bin/python"

  unset PYTHONPATH
  [ -n "${_VERL_PYTHONPATH_FALLBACK:-}" ] && export PYTHONPATH="${_VERL_PYTHONPATH_FALLBACK}"
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
    echo "[eval] CUDA toolkit: $CUDA_HOME"
  elif [ -d "$SITE/nvidia/cuda_runtime/lib" ]; then
    export CUDA_HOME="$SITE/nvidia/cuda_runtime"
  fi
  echo "[eval] activated env (conda-free): $ENV_PREFIX  python=$PYTHON_BIN"
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# --- detect GPU count -> DP ------------------------------------------------
detect_gpus() {
  if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .
  else echo 8; fi
}
DP="${DP:-$(detect_gpus)}"

# --- sanity checks ---------------------------------------------------------
[ -f "$EVAL_PY" ] || { echo "FATAL: eval script not found: $EVAL_PY" >&2; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "FATAL: base model dir not found: $BASE_MODEL (set BASE_MODEL)" >&2; exit 1; }
"$PYTHON_BIN" - "$VERL_HOME" <<'PYCHK' || { echo "FATAL: env $CONDA_ENV is missing required packages (see above)" >&2; exit 1; }
import importlib, sys
missing = []
for m in ("verl", "mathruler", "pandas", "openai", "httpx"):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({type(e).__name__}: {e})")
if missing:
    print("  missing/broken imports: " + "; ".join(missing), file=sys.stderr)
    print(f"  python = {sys.executable}", file=sys.stderr)
    sys.exit(1)
print("[eval] env check OK: verl + mathruler + pandas + openai + httpx import")
PYCHK

echo "=================================================================="
echo " repo root   : $HINT_RL_HOME"
echo " base model  : $BASE_MODEL"
echo " label       : $LABEL"
echo " topology    : dp=$DP tp=$TP"
echo " server      : ctx=$CONTEXT_LEN mem=$MEM_FRAC max_num_seqs=$MAX_NUM_SEQS port=$PORT"
echo " eval        : n=$N chunk=$CHUNK concurrency=$CONCURRENCY temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS${LIMIT:+ limit=$LIMIT}"
echo " datasets    : ${EVAL_SETS[*]}"
echo " out dir     : $OUT_BASE"
echo "=================================================================="

# --- helpers ---------------------------------------------------------------
SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  echo "[eval] stopping server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  kill -9 "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || break
    sleep 1
  done
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

# merge a verl FSDP actor dir -> HF model dir (idempotent). echoes the HF dir.
merge_fsdp() {
  local actor_dir="$1" target_dir="$2"
  if [ -f "$target_dir/config.json" ] && ls "$target_dir"/*.safetensors >/dev/null 2>&1; then
    echo "[eval] merged model already present: $target_dir" >&2
  else
    echo "[eval] merging FSDP checkpoint -> $target_dir" >&2
    "$PYTHON_BIN" -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$actor_dir" \
      --target_dir "$target_dir" >&2
  fi
  if [ -d "$actor_dir/huggingface" ]; then
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt \
             special_tokens_map.json added_tokens.json chat_template.jinja \
             generation_config.json; do
      [ -f "$actor_dir/huggingface/$f" ] && [ ! -f "$target_dir/$f" ] \
        && cp "$actor_dir/huggingface/$f" "$target_dir/" 2>/dev/null || true
    done
  fi
  echo "$target_dir"
}

# launch a vLLM server for $1 (served as $2), wait until ready.
serve() {
  local model_path="$1" served_name="$2"
  local server_log="$LOG_DIR/vllm_${served_name}_${PORT}.log"
  echo "[eval] starting vLLM server: $model_path  (served as $served_name) -> $server_log"
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --host 0.0.0.0 --port "$PORT" \
      --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM_FRAC" \
      --max-model-len "$CONTEXT_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --trust-remote-code \
      > "$server_log" 2>&1 &
  SERVER_PID=$!
  echo "[eval] waiting for server readiness (DP=$DP load can take minutes)..."
  local ready=0
  for i in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$served_name"; then
      ready=1; echo "[eval] server up after ${i}x5s"; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[eval] FATAL: server died during startup; tail of log:" >&2
      tail -n 40 "$server_log" >&2; exit 1
    fi
    sleep 5
  done
  [ "$ready" = 1 ] || { echo "[eval] FATAL: server not ready in time" >&2; tail -n 40 "$server_log" >&2; exit 1; }
}

# eval_ckpts.py on one dataset (bare, box-scored; no --hint-mode).
eval_one() {
  local served_name="$1" label="$2" dataset="$3" out_dir="$4"
  mkdir -p "$out_dir"
  local extra=()
  [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
  echo "[eval] --- $label  x  $(basename "$dataset") ---"
  "$PYTHON_BIN" "$EVAL_PY" \
      --dataset "$dataset" \
      --model "$served_name" \
      --model-label "$label" \
      --base-url "$BASE_URL" \
      --out-dir "$out_dir" \
      --n "$N" --chunk "$CHUNK" --concurrency "$CONCURRENCY" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      "${extra[@]}"
}

# --- run -------------------------------------------------------------------
echo ""
echo "##################################################################"
echo "# MODEL: $LABEL   ($BASE_MODEL)"
echo "##################################################################"
# base model is a plain HF dir -> serve directly, no FSDP merge
serve "$BASE_MODEL" "$SERVED_NAME"
for ds in "${EVAL_SETS[@]}"; do
  eval_one "$SERVED_NAME" "$LABEL" "$ds" "$OUT_BASE/$LABEL/$(basename "${ds%.parquet}")"
done
stop_server

# --- combined summary table ------------------------------------------------
echo ""
echo "=================================================================="
echo " COMBINED SUMMARY   (out: $OUT_BASE)"
echo "=================================================================="
"$PYTHON_BIN" - "$OUT_BASE" <<'PY'
import json, os, sys, glob
base = sys.argv[1]
rows = []
for sp in sorted(glob.glob(os.path.join(base, "*", "*", "_summary.json"))):
    d = json.load(open(sp))
    pk = d.get("pass_at_k", {})
    rows.append((
        d.get("model_label", "?"), d.get("dataset_name", "?"),
        d.get("acc_mean", 0.0), pk.get(f"pass@{d.get('n_requested_per_problem')}", pk.get("pass@128", 0.0)),
        d.get("format_rate", 0.0), d.get("hint_call_rate"), d.get("acc_box_only_mean", 0.0),
        d.get("n_problems", 0), d.get("n_samples", 0),
    ))
if not rows:
    print(" (no summaries found)"); sys.exit(0)
hdr = f"{'model':<22} {'dataset':<32} {'acc@n':>7} {'pass@N':>7} {'fmt':>5} {'hintcall':>8} {'box_only':>8}  n/prob"
print(hdr); print("-"*len(hdr))
for (ml, dn, acc, pN, fmt, hc, box, npb, ns) in rows:
    hc_s = f"{hc:.3f}" if hc is not None else "   -  "
    print(f"{ml:<22} {dn:<32} {acc:>7.4f} {pN:>7.3f} {fmt:>5.2f} {hc_s:>8} {box:>8.4f}  {npb}x{ns//max(npb,1)}")
print("-"*len(hdr))
PY
echo "=================================================================="
echo "[eval] done. per-run dirs + samples.jsonl under: $OUT_BASE"
