#!/usr/bin/env bash
#
# run_eval_ckpts.sh -- evaluate the three checkpoints on 1 node of 8x H100.
#
# Each checkpoint is served once on a local vLLM OpenAI endpoint (DP=8 replicas,
# TP=1 -- the throughput-optimal layout for a 7B on 8 GPUs, same as the selector
# eval) and `eval_ckpts.py` rolls every problem out N times (default 128) and
# grades the boxed answer with mathruler (the SAME grader the training rewards
# use). Verl FSDP checkpoints are merged to HF first (model_merger).
#
#   1. baseline  Qwen2.5-7B-Instruct                  -> aime2025
#   2. GRPO      global_step_200 (merged)              -> aime2025
#   3. HPRL v3   0614-045934 / step_250 (merged)       -> aime2025-hint-mt (--hint-mode)
#   4. HPRL v3   0612-003103 / step_100 (merged)       -> aime2025 + aime2024 + dapo-100 (bare, no hint-mode)
# (all four run by default. Prior aime2024 / dapo_sample_hard_100 sets are
#  commented out -- only aime2025 / aime2025-hint-mt are active.)
#
# The HPRL eval runs in --hint-mode: its templated sets carry the budget-0
# hint-tool prompt, so an over-budget <hint_call/> is scored acc=0 (the box is
# not graded) -- byte-faithful to HintAgentLoop @ budget 0 + hint_reward.
#
# All paths derive from this script's location. Override any VAR via the env:
#   PORT CONTEXT_LEN MEM_FRAC MAX_NUM_SEQS    (server)
#   N CHUNK CONCURRENCY TEMPERATURE TOP_P TOP_K MAX_TOKENS LIMIT   (eval)
#   CONDA_ROOT CONDA_ENV PYTHON_BIN           (env)
#   MODELS  (subset of: baseline grpo hprl hprl2)  (which checkpoints to run)
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
# The conda env baked its ORIGINAL absolute install prefix (/share5/users/xutao.ma)
# into conda.sh, script shebangs, AND the EDITABLE verl install's .pth. When this
# tree is mounted elsewhere (e.g. /xutao in the job container) those paths don't
# resolve: the conda-free PYTHON_BIN below still works (it's a direct path), but
# `import verl` fails because the editable package lives at the baked-in prefix.
# Symlink the original prefix -> the current mount so they keep resolving --
# identical to run_hprl_qwen2.5_7b.sh. No-op where the prefix already exists
# (e.g. the dev box, already on /share5). Falls back to PYTHONPATH if no sudo.
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
CONTEXT_LEN="${CONTEXT_LEN:-16384}"     # prompt(2048)+response(8192) + headroom
MEM_FRAC="${MEM_FRAC:-0.85}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
N="${N:-128}"                           # rollouts per problem
# 1 sample/request: vLLM V1 returns identical completions for n>1 within one
# request (would bias pass@k); separate requests sample independently and the
# shared prompt is prefix-cached, so this costs ~nothing. Keep at 1.
CHUNK="${CHUNK:-1}"
CONCURRENCY="${CONCURRENCY:-192}"
TEMPERATURE="${TEMPERATURE:-1.0}"       # matches training rollout / val_kwargs
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"        # == training max_response_length
LIMIT="${LIMIT:-}"                      # e.g. LIMIT=3 for a quick smoke test
BASE_URL="http://127.0.0.1:${PORT}/v1"

# which checkpoints to run (space-separated subset of: baseline grpo hprl hprl2).
# Default runs all four: baseline + GRPO + hprl2 (0612/step100) on bare aime2025,
# and hprl (0614/step250) on aime2025-hint-mt (--hint-mode).
MODELS="${MODELS:-baseline grpo hprl hprl2}"

# --- paths to models / checkpoints / datasets ------------------------------
DATASET_DIR="$HINT_RL_HOME/dataset"
BASELINE_MODEL="${BASELINE_MODEL:-$BASE_HOME/model/Qwen2.5-7B-Instruct}"
GRPO_CKPT="${GRPO_CKPT:-$HINT_RL_HOME/ckpt/GRPO-Qwen2.5-7B-Instruct/GRPO-Qwen2.5-7B-Instruct-dapo-3139-20260613-131513/global_step_200/actor}"
HPRL_CKPT="${HPRL_CKPT:-$HINT_RL_HOME/ckpt/HPRL-Qwen2.5-7B-Instruct/HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260614-045934/global_step_250/actor}"
MERGED_DIR="${MERGED_DIR:-$SCRIPT_DIR/merged}"

# bare (single-turn) datasets for baseline + GRPO
# --- prior eval sets (commented out; re-enable to restore) ---
# BARE_SETS=("$DATASET_DIR/aime2024.parquet" "$DATASET_DIR/dapo_sample_hard_100.parquet")
BARE_SETS=("$DATASET_DIR/aime2025.parquet")
# hint-template (budget-0) datasets for HPRL
# --- prior eval sets (commented out; re-enable to restore) ---
# HINT_SETS=("$DATASET_DIR/aime2024-hint-mt.parquet" "$DATASET_DIR/dapo_sample_hard_100-hint-mt.parquet")
# aime2025-hint-mt.parquet built via:
#   script/hint_rl/prepare_eval_hint_template.py --in dataset/aime2025.parquet --out dataset/aime2025-hint-mt.parquet
HINT_SETS=("$DATASET_DIR/aime2025-hint-mt.parquet")
# bare (non-hint) datasets for the hprl2 checkpoint only (tested out-of-template,
# box-scored like baseline/grpo). Its own list so adding sets here does NOT change
# what baseline/grpo (BARE_SETS) run on.
HPRL2_SETS=("$DATASET_DIR/aime2024.parquet" "$DATASET_DIR/dapo_sample_hard_100.parquet")

# --- output ----------------------------------------------------------------
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/results/run_${RUN_TS}}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR" "$MERGED_DIR"

# --- python env activation (conda-free; verbatim from selector/run_eval_h100.sh) -
# The miniconda tree has its ORIGINAL install path baked in absolutely, so we do
# NOT `conda activate`; we point PYTHON_BIN at the env's own interpreter (a real
# ELF that runs from any mount) and wire up PATH / CONDA_PREFIX / LD_LIBRARY_PATH.
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"          # verl env: vllm 0.12 + mathruler + pandas + openai + verl
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

  # sanitize the container's leaked python3.12 venv (PYTHONPATH/LD), else our
  # env imports the wrong numpy/torch C-extensions and dies.
  unset PYTHONPATH
  # if the prefix symlink could not be made above, point at the verl repo so the
  # editable `import verl` still resolves on this mount.
  [ -n "${_VERL_PYTHONPATH_FALLBACK:-}" ] && export PYTHONPATH="${_VERL_PYTHONPATH_FALLBACK}"
  export PYTHONNOUSERSITE=1
  CLEAN_LD=""
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    CLEAN_LD="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
      | grep -v -e '/opt/venv' -e '/usr/local/lib/python3' | paste -sd: -)"
  fi
  # make the env's bundled CUDA-12/13 pip-wheel libs findable (torch/vllm need them).
  SITE="$("$ENV_PREFIX/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
  SITE="${SITE:-$ENV_PREFIX/lib/python3.11/site-packages}"
  NV_LIBS="$(find "$SITE/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: -)"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib${NV_LIBS:+:$NV_LIBS}${CLEAN_LD:+:$CLEAN_LD}"
  # a complete CUDA toolkit for any JIT/compile (vllm inductor); prefer system, else wheels.
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

# vllm: avoid the flashinfer-sampler JIT (toolkit/torch CUDA mismatch breaks it).
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
# Import each dep separately and report exactly which fail (and why) -- a single
# combined import hid that only `verl` (editable, at the baked-in prefix) was
# failing. verl is only needed to MERGE the FSDP ckpts (grpo/hprl); the eval
# client itself needs mathruler/pandas/openai/httpx.
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
    print(f"  (verl is an editable install; if it points at a stale path, the "
          f"prefix symlink or PYTHONPATH={sys.argv[1]} should fix it)", file=sys.stderr)
    sys.exit(1)
print("[eval] env check OK: verl + mathruler + pandas + openai + httpx import")
PYCHK

echo "=================================================================="
echo " repo root   : $HINT_RL_HOME"
echo " topology    : dp=$DP tp=$TP  (8x H100, replicas across GPUs)"
echo " server      : ctx=$CONTEXT_LEN mem=$MEM_FRAC max_num_seqs=$MAX_NUM_SEQS port=$PORT"
echo " eval        : n=$N chunk=$CHUNK concurrency=$CONCURRENCY temp=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS${LIMIT:+ limit=$LIMIT}"
echo " models      : $MODELS"
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
  # let the port free up before the next model's server binds it
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
  # ensure tokenizer / chat template are present (copy from the saved hf config
  # subdir if the merger didn't include them).
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
  for i in $(seq 1 240); do   # up to 20 min
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

# eval_ckpts.py on one dataset. args: served_name label hint_flag(0/1) dataset out_dir
eval_one() {
  local served_name="$1" label="$2" hint_flag="$3" dataset="$4" out_dir="$5"
  mkdir -p "$out_dir"
  local extra=()
  [ "$hint_flag" = "1" ] && extra+=(--hint-mode)
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

# serve one model and eval it over a list of datasets, then tear the server down.
# args: label served_name model_path hint_flag dataset...
run_model() {
  local label="$1" served_name="$2" model_path="$3" hint_flag="$4"; shift 4
  echo ""
  echo "##################################################################"
  echo "# MODEL: $label   ($model_path)"
  echo "##################################################################"
  serve "$model_path" "$served_name"
  local ds
  for ds in "$@"; do
    eval_one "$served_name" "$label" "$hint_flag" "$ds" "$OUT_BASE/$label/$(basename "${ds%.parquet}")"
  done
  stop_server
}

# --- run the requested checkpoints -----------------------------------------
for m in $MODELS; do
  case "$m" in
    # baseline)
    #   # base instruct model (already HF, no merge); bare aime2025 (BARE_SETS)
    #   run_model "baseline" "qwen2.5-7b-instruct" "$BASELINE_MODEL" 0 "${BARE_SETS[@]}"
    #   ;;
    # grpo)
    #   # GRPO baseline dapo-3139 @ step_200 (merged); bare aime2025 (BARE_SETS)
    #   GRPO_HF="$(merge_fsdp "$GRPO_CKPT" "$MERGED_DIR/GRPO-dapo-3139-step200")"
    #   run_model "grpo-step200" "grpo-step200" "$GRPO_HF" 0 "${BARE_SETS[@]}"
    #   ;;
    # hprl)
    #   # HPRL run v3 20260614-045934 @ step_250
    #   HPRL_CKPT="${HPRL_CKPT:-$HINT_RL_HOME/ckpt/HPRL-Qwen2.5-7B-Instruct/HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260614-045934/global_step_250/actor}"
    #   HPRL_HF="$(merge_fsdp "$HPRL_CKPT" "$MERGED_DIR/HPRL-v3-step250")"
    #   run_model "hprl-v3-step250" "hprl-step250" "$HPRL_HF" 1 "${HINT_SETS[@]}"
    #   ;;
    hprl2)
      # HPRL run v3 20260612-003103 @ step_100. Evaluated on the BARE (non-hint-call)
      # format (HPRL2_SETS = aime2025 + aime2024 + dapo_sample_hard_100), NO --hint-mode
      # -- NOT the budget-0 hint template the 0614 run above uses. Box-scored exactly
      # like baseline/grpo (directly comparable). Date-qualified names so its merged +
      # output dirs never collide with the other HPRL run's.
      HPRL2_CKPT="${HPRL2_CKPT:-$HINT_RL_HOME/ckpt/HPRL-Qwen2.5-7B-Instruct/HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260612-003103/global_step_100/actor}"
      HPRL2_HF="$(merge_fsdp "$HPRL2_CKPT" "$MERGED_DIR/HPRL-v3-0612-step100")"
      run_model "hprl-v3-0612-step100" "hprl-0612-step100" "$HPRL2_HF" 0 "${HPRL2_SETS[@]}"
      ;;
    *) echo "[eval] WARN: unknown model '$m' (want: baseline grpo hprl hprl2)" >&2 ;;
  esac
done

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
hdr = f"{'model':<18} {'dataset':<32} {'acc@n':>7} {'pass@N':>7} {'fmt':>5} {'hintcall':>8} {'box_only':>8}  n/prob"
print(hdr); print("-"*len(hdr))
for (ml, dn, acc, pN, fmt, hc, box, npb, ns) in rows:
    hc_s = f"{hc:.3f}" if hc is not None else "   -  "
    print(f"{ml:<18} {dn:<32} {acc:>7.4f} {pN:>7.3f} {fmt:>5.2f} {hc_s:>8} {box:>8.4f}  {npb}x{ns//max(npb,1)}")
print("-"*len(hdr))
print("acc@n = avg accuracy over n rollouts (official; HPRL: over-budget <hint_call/> -> 0).")
print("box_only = HPRL box accuracy ignoring the hint-call penalty (diagnostic).")
PY
echo "=================================================================="
echo "[eval] done. per-run dirs + samples.jsonl under: $OUT_BASE"
