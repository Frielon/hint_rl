#!/usr/bin/env bash
#
# run_eval_native_len.sh -- measure the NATIVE generation length of the two
# untrained instruct models
#   /share5/users/xutao.ma/model/Qwen3-4B-Instruct-2507
#   /share5/users/xutao.ma/model/Olmo-3-7B-Instruct
# on the training dataset
#   dataset/dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8-single-turn.parquet
# BARE single-turn, NO --hint-mode, 1 sample per problem, max length 64k.
#
# Each model is a plain HF checkpoint (no FSDP merge): it is served on a local
# vLLM OpenAI endpoint (DP=<gpu count>, TP=1), eval_ckpts.py rolls every
# problem out once, and a post-pass re-tokenizes every completion with the
# model's own tokenizer to report the generation-length distribution
# (mean/p50/p90/p99/max + truncation rate).
#
# Context-window note: Qwen3-4B-Instruct-2507 is natively 262144 ctx, so it
# gets the full MAX_TOKENS=65536 (+2048 prompt headroom). Olmo-3-7B-Instruct
# is HARD-CAPPED at max_position_embeddings=65536 (YaRN), so its ctx is 65536
# and MAX_TOKENS=63488 -- the closest it can get to 64k while leaving prompt
# room (dataset prompts are <= ~500 tokens).
#
# Override via env:
#   PORT MEM_FRAC MAX_NUM_SEQS TP DP                       (server)
#   N CHUNK CONCURRENCY TEMPERATURE TOP_P TOP_K LIMIT      (eval)
#   CONDA_ROOT CONDA_ENV PYTHON_BIN                        (env)
#   OUT_BASE                                               (output)
#   ONLY_MODEL=<label>   run just that MODELS entry (for per-node parallelism)
#   SUMMARY_ONLY=1       print the combined table over an existing OUT_BASE
# Two-node parallel run (one model per node): submit
#   eval/launch_native_len_2node.sh  on BOTH pods of a 2-replica job.
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
MEM_FRAC="${MEM_FRAC:-0.85}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
N="${N:-1}"                             # 1 sample per problem
CHUNK="${CHUNK:-1}"
CONCURRENCY="${CONCURRENCY:-192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
LIMIT="${LIMIT:-}"
# 64k completions can exceed eval_ckpts.py's 200k-char store cap (~4 chars/tok)
# -> raise it so the length post-pass sees the FULL text, not a clipped one.
MAX_STORE_CHARS="${MAX_STORE_CHARS:-600000}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- the models to evaluate (plain HF dirs, no FSDP merge) ------------------
# Each entry: "<label>|<hf model dir>|<vllm max-model-len>|<max_tokens>"
MODELS=(
  "qwen3-4b-instruct-2507|/share5/users/xutao.ma/model/Qwen3-4B-Instruct-2507|67584|65536"
  "olmo-3-7b-instruct|/share5/users/xutao.ma/model/Olmo-3-7B-Instruct|65536|63488"
)

# ONLY_MODEL: restrict this invocation to a single MODELS entry by label.
# Used by launch_native_len_2node.sh to run one model per node in parallel
# (both nodes share OUT_BASE on the NFS, so the results still combine).
ONLY_MODEL="${ONLY_MODEL:-}"
if [ -n "$ONLY_MODEL" ]; then
  _sel=()
  for _m in "${MODELS[@]}"; do
    [ "${_m%%|*}" = "$ONLY_MODEL" ] && _sel+=("$_m")
  done
  [ "${#_sel[@]}" -gt 0 ] || { echo "FATAL: ONLY_MODEL=$ONLY_MODEL matches no MODELS label" >&2; exit 1; }
  MODELS=("${_sel[@]}")
fi

# SUMMARY_ONLY=1: skip serving/eval entirely and just print the combined
# summary table over whatever _summary.json/_length_stats.json already exist
# under OUT_BASE (the 2-node launcher's last finisher uses this).
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

# --- dataset (BARE, box-scored) --------------------------------------------
DATASET="$HINT_RL_HOME/dataset/dolci-instruct-rl-492-auto-hint-qwen3-4b-le1of8-single-turn.parquet"

# --- output ----------------------------------------------------------------
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/results/native_len_${RUN_TS}}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

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
[ -f "$DATASET" ] || { echo "FATAL: dataset not found: $DATASET" >&2; exit 1; }
for _m in "${MODELS[@]}"; do
  _dir="$(echo "$_m" | cut -d'|' -f2)"
  [ -f "$_dir/config.json" ] || { echo "FATAL: model dir not found: $_dir" >&2; exit 1; }
done
"$PYTHON_BIN" - <<'PYCHK' || { echo "FATAL: env $CONDA_ENV is missing required packages (see above)" >&2; exit 1; }
import importlib, sys
missing = []
for m in ("verl", "mathruler", "pandas", "openai", "httpx", "transformers"):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m} ({type(e).__name__}: {e})")
if missing:
    print("  missing/broken imports: " + "; ".join(missing), file=sys.stderr)
    print(f"  python = {sys.executable}", file=sys.stderr)
    sys.exit(1)
print("[eval] env check OK: verl + mathruler + pandas + openai + httpx + transformers import")
PYCHK

echo "=================================================================="
echo " repo root   : $HINT_RL_HOME"
for _m in "${MODELS[@]}"; do
  echo " model       : $(echo "$_m" | cut -d'|' -f1)  ($(echo "$_m" | cut -d'|' -f2))  ctx=$(echo "$_m" | cut -d'|' -f3) max_tokens=$(echo "$_m" | cut -d'|' -f4)"
done
echo " topology    : dp=$DP tp=$TP"
echo " server      : mem=$MEM_FRAC max_num_seqs=$MAX_NUM_SEQS port=$PORT"
echo " eval        : n=$N chunk=$CHUNK concurrency=$CONCURRENCY temp=$TEMPERATURE top_p=$TOP_P${LIMIT:+ limit=$LIMIT}"
echo " dataset     : $DATASET"
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

# launch a vLLM server for $1 (served as $2) with max-model-len $3, wait ready.
serve() {
  local model_path="$1" served_name="$2" context_len="$3"
  local server_log="$LOG_DIR/vllm_${served_name}_${PORT}.log"
  echo "[eval] starting vLLM server: $model_path  (served as $served_name, ctx=$context_len) -> $server_log"
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --host 0.0.0.0 --port "$PORT" \
      --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM_FRAC" \
      --max-model-len "$context_len" \
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

# eval_ckpts.py on the dataset (bare, box-scored; no --hint-mode).
eval_one() {
  local served_name="$1" label="$2" max_tokens="$3" out_dir="$4"
  mkdir -p "$out_dir"
  local extra=()
  [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
  echo "[eval] --- $label  x  $(basename "$DATASET")  (max_tokens=$max_tokens) ---"
  "$PYTHON_BIN" "$EVAL_PY" \
      --dataset "$DATASET" \
      --model "$served_name" \
      --model-label "$label" \
      --base-url "$BASE_URL" \
      --out-dir "$out_dir" \
      --n "$N" --chunk "$CHUNK" --concurrency "$CONCURRENCY" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
      --max-tokens "$max_tokens" \
      --max-store-chars "$MAX_STORE_CHARS" \
      "${extra[@]}"
}

# generation-length distribution: re-tokenize every completion with the
# model's own tokenizer (samples.jsonl stores text + finish_reason but no
# per-sample token count). Writes _length_stats.json next to samples.jsonl.
length_stats() {
  local samples="$1" model_dir="$2" label="$3" max_tokens="$4"
  "$PYTHON_BIN" - "$samples" "$model_dir" "$label" "$max_tokens" <<'PYLEN'
import json, os, sys
samples, model_dir, label, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
lens, trunc, accs = [], 0, []
with open(samples) as f:
    for line in f:
        r = json.loads(line)
        lens.append(len(tok.encode(r.get("text") or "", add_special_tokens=False)))
        trunc += int(r.get("finish_reason") == "length")
        accs.append(r.get("acc", 0.0))
if not lens:
    print(f"[len] {label}: no samples found in {samples}", file=sys.stderr); sys.exit(0)
sl = sorted(lens)
pct = lambda q: sl[min(len(sl) - 1, int(q * len(sl)))]
corr = [l for l, a in zip(lens, accs) if a >= 1.0]
wrong = [l for l, a in zip(lens, accs) if a < 1.0]
mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
stats = {
    "model_label": label,
    "n_samples": len(lens),
    "max_tokens": max_tokens,
    "mean": mean(lens), "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
    "min": sl[0], "max": sl[-1],
    "truncation_rate": trunc / len(lens),
    "n_truncated": trunc,
    "mean_correct": mean(corr), "n_correct": len(corr),
    "mean_incorrect": mean(wrong), "n_incorrect": len(wrong),
}
out = os.path.join(os.path.dirname(samples), "_length_stats.json")
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print(f"[len] {label}: n={stats['n_samples']}  mean={stats['mean']:.0f}  p50={stats['p50']}  "
      f"p90={stats['p90']}  p99={stats['p99']}  max={stats['max']}  "
      f"trunc={stats['n_truncated']} ({stats['truncation_rate']:.3f})")
print(f"[len] {label}: correct mean={stats['mean_correct']:.0f} (n={stats['n_correct']})  "
      f"incorrect mean={stats['mean_incorrect']:.0f} (n={stats['n_incorrect']})")
print(f"[len] stats -> {out}")
PYLEN
}

# --- run every model --------------------------------------------------------
DS_NAME="$(basename "${DATASET%.parquet}")"
if [ "$SUMMARY_ONLY" = "1" ]; then
  echo "[eval] SUMMARY_ONLY=1 -> skipping serving/eval, printing combined summary for $OUT_BASE"
else
  for _m in "${MODELS[@]}"; do
    IFS='|' read -r LABEL MODEL_DIR CONTEXT_LEN MAX_TOKENS <<< "$_m"
    echo ""
    echo "##################################################################"
    echo "# MODEL: $LABEL   ($MODEL_DIR)  ctx=$CONTEXT_LEN max_tokens=$MAX_TOKENS"
    echo "##################################################################"
    OUT_DIR="$OUT_BASE/$LABEL/$DS_NAME"
    serve "$MODEL_DIR" "$LABEL" "$CONTEXT_LEN"
    eval_one "$LABEL" "$LABEL" "$MAX_TOKENS" "$OUT_DIR"
    stop_server
    length_stats "$OUT_DIR/samples.jsonl" "$MODEL_DIR" "$LABEL" "$MAX_TOKENS"
  done
fi

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
    ls = {}
    lp = os.path.join(os.path.dirname(sp), "_length_stats.json")
    if os.path.exists(lp):
        ls = json.load(open(lp))
    rows.append((
        d.get("model_label", "?"),
        d.get("acc_mean", 0.0), d.get("format_rate", 0.0), d.get("truncation_rate", 0.0),
        ls.get("mean", 0.0), ls.get("p50", 0), ls.get("p90", 0), ls.get("p99", 0), ls.get("max", 0),
        d.get("n_samples", 0),
    ))
if not rows:
    print(" (no summaries found)"); sys.exit(0)
hdr = f"{'model':<26} {'acc':>6} {'fmt':>5} {'trunc':>6} {'len_mean':>8} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6}  n"
print(hdr); print("-" * len(hdr))
for (ml, acc, fmt, tr, lm, p50, p90, p99, mx, ns) in rows:
    print(f"{ml:<26} {acc:>6.4f} {fmt:>5.2f} {tr:>6.3f} {lm:>8.0f} {p50:>6} {p90:>6} {p99:>6} {mx:>6}  {ns}")
print("-" * len(hdr))
PY
echo "=================================================================="
echo "[eval] done. per-model samples.jsonl + _length_stats.json under: $OUT_BASE"
