#!/usr/bin/env bash
#
# Entry script for a web-platform job: run the hint-selection eval against a
# locally-served model on 8x H100 (or any GPU count).
#
# All repo paths are resolved RELATIVE to this script's location, so it works
# from any mount point. Cluster-specific values (model weights path, python,
# conda env) are taken from env vars with sensible defaults — set them at job
# submission time if the defaults don't match your image.
#
# Topology: gpt-oss-20b is a small MoE (~14 GB, 3.6B active). The throughput-
# optimal layout on 8 GPUs is data-parallel REPLICAS (DP=8, TP=1), NOT TP=8.
# One sglang server fronts the 8 replicas with a round-robin balancer.
#
# ---- overridable env vars -------------------------------------------------
#   ENGINE         inference engine: sglang | vllm  (default: sglang)
#   MODEL_PATH     path to the model weights        (REQUIRED if default absent)
#   SERVED_NAME    model id exposed by the server   (default: gpt-oss-20b)
#   MODELS_DIR     dir holding <SERVED_NAME>/        (default: <repo>/../../model)
#   DP             data-parallel replicas           (default: #GPUs)
#   TP             tensor-parallel size per replica (default: 1)
#   CONTEXT_LEN    server context length            (default: 40960)
#   MEM_FRAC       --mem-fraction-static            (default: 0.9)
#   PORT           server port                      (default: 30000)
#   N              samples per problem              (default: 16)
#   MAX_TOKENS     gen budget per sample            (default: 16000)
#   TEMPS          temperatures to sweep (space-sep) (default: 0.7 0.4 0.2 0.0)
#   MAX_NUM_SEQS   vllm max concurrent seqs/replica (default: 256; hybrid cap)
#   WORKERS        problems in flight (client)      (default: 2 * DP)
#   SAMPLE_WORKERS samples in flight per problem    (default: 16)
#   REASONING_PARSER reasoning parser               (default: gpt-oss)
#   PYTHON_BIN     python to use                    (default: env interpreter)
#   CONDA_ROOT     miniconda tree root              (default: <repo>/../../miniconda3)
#   CONDA_ENV      env name under $CONDA_ROOT/envs  (default: inference; '' skips)
#   OUT_DIR        base eval output dir             (default: <selector>/selector_<SERVED_NAME>)
#                  each run lands in <OUT_DIR>/run_<YYYYmmdd_HHMMSS>/
#   PROGRESS_LOG   live progress log file           (default: <run dir>/progress.log)
#   RL_ROLLOUTS    rollouts/step for RL projection  (default: 4096 = 256*16)
#   RL_CALLS_PER_ROLLOUT selector calls per rollout (default: 5)
#   RL_STEPS       RL steps for projection          (default: 500)
#   EVAL_EXTRA_ARGS extra args passed to the python eval
# ---------------------------------------------------------------------------
set -eo pipefail

# --- resolve repo paths relative to this script ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../selector
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_PY="$SCRIPT_DIR/run_hint_selection_model.py"
CODEX_DIR="$SCRIPT_DIR/selector_codex"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- config (env-overridable) ---------------------------------------------
# inference engine: sglang (default) or vllm. Both expose the same OpenAI
# /v1 API, so only the server launch command differs; the client/eval is
# engine-agnostic.
ENGINE="${ENGINE:-vllm}"
case "$ENGINE" in
  sglang|vllm) ;;
  *) echo "FATAL: ENGINE must be 'sglang' or 'vllm' (got '$ENGINE')" >&2; exit 1 ;;
esac
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/../../model}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/$SERVED_NAME}"
PORT="${PORT:-30000}"
CONTEXT_LEN="${CONTEXT_LEN:-40960}"
MEM_FRAC="${MEM_FRAC:-0.9}"
TP="${TP:-1}"
# default reasoning parser tracks the model family: gpt-oss splits a separate
# CoT channel and NEEDS a parser; Qwen3.x reasons inline and needs NONE (a
# CoT parser would mangle its answer). Use ${VAR-default} so an explicit
# REASONING_PARSER='' is still honored.
case "$SERVED_NAME" in
  *gpt-oss*|*gpt_oss*) _DEFAULT_PARSER=gpt-oss ;;
  *) _DEFAULT_PARSER= ;;
esac
REASONING_PARSER="${REASONING_PARSER-$_DEFAULT_PARSER}"
N="${N:-16}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-16}"
# max concurrent sequences per server replica (vllm --max-num-seqs). Qwen3.5/3.6
# are HYBRID (Mamba/GDN) models: each in-flight decode needs a Mamba state cache
# block, and only a few hundred fit alongside a 27B model. vllm's default (1024)
# exceeds that and CUDA-graph capture aborts ("max_num_seqs exceeds available
# Mamba cache blocks"). 256 is safely under the limit and far above the eval's
# real concurrency. Raise for pure-attention models (e.g. gpt-oss) if desired.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# default DP = number of visible GPUs
detect_gpus() {
  if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .
  else
    echo 8
  fi
}
DP="${DP:-$(detect_gpus)}"
WORKERS="${WORKERS:-$((2 * DP))}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/selector_$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9._-' '_')}"
# Temperature sweep: the eval runs once per value in TEMPS, all against the one
# shared server (loaded once). Each value writes to its own sibling run dir
#   $OUT_DIR/run_<SWEEP_TS>_t<TEMP>/
# holding everything for that run (samples, _summary.json, progress.log,
# estimate); --no-run-subdir stops the python side adding a second timestamp.
# Sibling dirs are directly discoverable by view_selector.py pointed at OUT_DIR.
# Set TEMPS="0.7" for a single run, or EVAL_EXTRA_ARGS="--limit 30" for a quick
# subset sweep. The shell owns the shared timestamp (one per invocation).
TEMPS="${TEMPS:-0.7 0.4 0.2 0.0}"
SWEEP_TS="$(date +%Y%m%d_%H%M%S)"

# --- python env activation (conda-free) ------------------------------------
# We deliberately do NOT use `conda activate`. The miniconda tree has its
# ORIGINAL install path baked in absolutely -- both in conda.sh and in the
# shebang of the `conda` launcher (#!/share5/.../miniconda3/bin/python). On a
# cluster that mounts the same tree at a different point (e.g. /xutao/...),
# that interpreter path doesn't exist, so the kernel can't even exec conda:
# "cannot execute: required file not found". The env's OWN python, however, is
# a real ELF binary that runs from any mount point -- so we activate the env
# manually by pointing PYTHON_BIN at it and wiring up PATH/CONDA_PREFIX.
#
# Resolution: <repo>/../../miniconda3/envs/<CONDA_ENV>, relative to this script
# (same layout as MODELS_DIR). Override CONDA_ROOT/CONDA_ENV if your image
# differs, or set CONDA_ENV='' to skip activation (env already on PATH).
CONDA_ROOT="${CONDA_ROOT:-$REPO_ROOT/../../miniconda3}"
# default env tracks the engine: sglang -> 'inference', vllm -> 'vllm'
_DEFAULT_ENV=inference; [ "$ENGINE" = vllm ] && _DEFAULT_ENV=vllm
CONDA_ENV="${CONDA_ENV-$_DEFAULT_ENV}"
if [ -n "$CONDA_ENV" ]; then
  ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
  [ -x "$ENV_PREFIX/bin/python" ] || {
    echo "FATAL: env python not found/executable: $ENV_PREFIX/bin/python" >&2
    echo "       (set CONDA_ROOT/CONDA_ENV to match your image, or CONDA_ENV='' to skip)" >&2
    exit 1
  }
  export CONDA_PREFIX="$ENV_PREFIX"
  export PATH="$ENV_PREFIX/bin:$PATH"
  # honor an explicit PYTHON_BIN override; otherwise use the env interpreter
  [ "$PYTHON_BIN" = "python" ] && PYTHON_BIN="$ENV_PREFIX/bin/python"

  # --- sanitize the container's leaked environment -------------------------
  # This image ships its own python3.12 venv (/opt/venv) and injects it via
  # PYTHONPATH + LD_LIBRARY_PATH. Left intact, our 3.11 env imports the 3.12
  # numpy/torch C-extensions and dies ("No module named numpy.core._multiarray
  # _umath"). Drop the inherited python paths so the env uses ONLY its own
  # site-packages and shared libs.
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  # strip the container's python3.12 lib dirs out of the inherited LD path
  CLEAN_LD=""
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    CLEAN_LD="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
      | grep -v -e '/opt/venv' -e '/usr/local/lib/python3' | paste -sd: -)"
  fi

  # --- make the env's bundled CUDA-12 libs findable ------------------------
  # torch/sgl_kernel/flashinfer ship their CUDA runtime as pip wheels under
  # site-packages/nvidia/<comp>/lib (libcudart.so.12, libnvrtc.so.12,
  # libcublas.so.12, ...). These are NOT on the default loader path, so
  # sgl_kernel's *.so (which NEED libnvrtc.so.12 / libcudart.so.12) fail to
  # dlopen unless we add them. Prepend them so the env's CUDA 12.8 libs win
  # over anything the container provides.
  # derive site-packages from the env's own python (py3.11 for inference,
  # py3.12 for vllm) instead of hardcoding the minor version.
  SITE="$("$ENV_PREFIX/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
  SITE="${SITE:-$ENV_PREFIX/lib/python3.11/site-packages}"
  NV_LIBS="$(find "$SITE/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: -)"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib${NV_LIBS:+:$NV_LIBS}${CLEAN_LD:+:$CLEAN_LD}"
  # CUDA_HOME must be a COMPLETE toolkit: flashinfer JIT (sglang) and
  # torch.compile/inductor (vllm; the Qwen3.5 GDN kernel triggers it) need
  # $CUDA_HOME/bin/nvcc, and sgl_kernel needs $CUDA_HOME/lib*/libcudart.so.
  # Prefer a system toolkit (the H100 container ships /usr/local/cuda), but
  # fall back to nvcc bundled in the env's pip wheels: newer cu13 wheels put a
  # whole toolkit under nvidia/cu13 (bin/nvcc + nvvm); older cu12 layout has
  # nvidia/cuda_nvcc. This lets vllm JIT even on boxes with no system toolkit.
  CUDA_TK=""
  for c in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12 \
           "$SITE/nvidia/cu13" "$SITE/nvidia/cuda_nvcc"; do
    [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA_TK="$c"; break; }
  done
  if [ -n "$CUDA_TK" ]; then
    export CUDA_HOME="$CUDA_TK"
    # make nvcc reachable on PATH too (inductor/flashinfer shell out to it)
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "[entry] CUDA toolkit (nvcc) for JIT/compile: $CUDA_HOME"
  else
    # No system nvcc -> flashinfer can't JIT. Fall back to backends that don't
    # need nvcc: triton (bundles its own compiler) for attention, and point
    # CUDA_HOME at the env's cuda_runtime so sgl_kernel still finds libcudart.
    [ -d "$SITE/nvidia/cuda_runtime/lib" ] && export CUDA_HOME="$SITE/nvidia/cuda_runtime"
    # the flashinfer-JIT fallback is sglang-specific (sgl_kernel); vllm picks
    # its own backend and 'triton' is not a valid VLLM_ATTENTION_BACKEND value.
    [ "$ENGINE" = sglang ] && : "${ATTENTION_BACKEND:=triton}"
    echo "[entry] WARNING: no system nvcc found; using triton attention backend (no JIT)." >&2
    echo "[entry]          CUDA_HOME=$CUDA_HOME (libcudart only)" >&2
  fi
  echo "[entry] activated env (conda-free): $ENV_PREFIX  python=$PYTHON_BIN"
fi

# --- flashinfer JIT header/lib paths --------------------------------------
# flashinfer may JIT-compile kernels and needs the CUDA headers/libs that the
# nvidia-*-cu12 wheels ship under site-packages/nvidia/<comp>/{include,lib}.
# Gather ALL of them (not just one) so nvcc/nvrtc can find cuda_runtime,
# nvrtc, cublas, etc. headers. Harmless no-op if the dirs don't exist.
if SITE="$($PYTHON_BIN -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null)" && [ -d "$SITE/nvidia" ]; then
  INC="$(find "$SITE/nvidia" -maxdepth 2 -name include -type d 2>/dev/null | paste -sd: -)"
  LIB="$(find "$SITE/nvidia" -maxdepth 2 -name lib     -type d 2>/dev/null | paste -sd: -)"
  [ -n "$INC" ] && export CPATH="${INC}${CPATH:+:$CPATH}"
  [ -n "$LIB" ] && export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

# --- sanity checks ---------------------------------------------------------
[ -f "$EVAL_PY" ]    || { echo "FATAL: eval script not found: $EVAL_PY" >&2; exit 1; }
[ -d "$CODEX_DIR" ]  || { echo "FATAL: codex reference dir not found: $CODEX_DIR" >&2; exit 1; }
[ -e "$MODEL_PATH" ] || { echo "FATAL: model not found: $MODEL_PATH (set MODEL_PATH or MODELS_DIR)" >&2; exit 1; }

echo "=================================================================="
echo " repo root      : $REPO_ROOT"
echo " engine         : $ENGINE"
echo " model          : $MODEL_PATH  (served as $SERVED_NAME)"
echo " topology       : dp=$DP tp=$TP  (replicas across GPUs)"
echo " context/mem    : $CONTEXT_LEN / $MEM_FRAC"
echo " eval           : n=$N max_tokens=$MAX_TOKENS workers=$WORKERS sample_workers=$SAMPLE_WORKERS"
echo " temp sweep     : $TEMPS"
echo " out dirs       : $OUT_DIR/run_${SWEEP_TS}_t<TEMP>"
echo " python         : $($PYTHON_BIN -c 'import sys;print(sys.executable)')"
echo "=================================================================="

# --- launch server, ensure teardown on exit -------------------------------
SERVER_LOG="$LOG_DIR/${ENGINE}_${SERVED_NAME}_${PORT}.log"
# --reasoning-parser only applies to models that split a CoT channel (gpt-oss);
# leave REASONING_PARSER empty for inline-reasoning models (e.g. Qwen3.5).
# Both engines accept --reasoning-parser, but the parser NAMES differ: sglang
# calls the gpt-oss parser 'gpt-oss', vllm calls it 'openai_gptoss'. Normalize
# the sglang-style name to the active engine so the same REASONING_PARSER value
# works for either.
if [ "$ENGINE" = vllm ]; then
  case "$REASONING_PARSER" in
    gpt-oss) REASONING_PARSER=openai_gptoss ;;
  esac
fi
REASONING_ARGS=()
[ -n "$REASONING_PARSER" ] && REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")
echo "[entry] starting $ENGINE server -> $SERVER_LOG  (attention=${ATTENTION_BACKEND:-default})"
if [ "$ENGINE" = sglang ]; then
  # sglang takes the attention backend as a CLI flag (set to 'triton'
  # automatically above when no nvcc is available so flashinfer never JITs;
  # override via ATTENTION_BACKEND).
  ATTN_ARGS=()
  [ -n "${ATTENTION_BACKEND:-}" ] && ATTN_ARGS=(--attention-backend "$ATTENTION_BACKEND")
  $PYTHON_BIN -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_NAME" \
      --host 0.0.0.0 --port "$PORT" \
      --dp-size "$DP" --tp-size "$TP" \
      --load-balance-method round_robin \
      --mem-fraction-static "$MEM_FRAC" \
      --context-length "$CONTEXT_LEN" \
      "${REASONING_ARGS[@]}" \
      "${ATTN_ARGS[@]}" \
      --trust-remote-code \
      > "$SERVER_LOG" 2>&1 &
else
  # vllm: equivalent flags (--model/--max-model-len/--gpu-memory-utilization/
  # --data-parallel-size/--tensor-parallel-size) and DP balancing is internal.
  # The attention backend is chosen via the VLLM_ATTENTION_BACKEND env var,
  # not a CLI flag.
  [ -n "${ATTENTION_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
  # Disable flashinfer's top-k/top-p sampler. It is NOT shipped prebuilt, so at
  # startup it JIT-compiles a CUDA kernel with nvcc -- which fails whenever the
  # nvcc toolkit version != the env's torch CUDA version (e.g. container CUDA
  # 12.9 vs the cu13 wheels here): "CUDA compiler and CUDA toolkit headers are
  # incompatible". Falling back to vllm's native Torch sampler avoids the JIT
  # entirely. Override by exporting VLLM_USE_FLASHINFER_SAMPLER=1 if your
  # toolkit/torch CUDA versions match and you want the faster path.
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_NAME" \
      --host 0.0.0.0 --port "$PORT" \
      --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM_FRAC" \
      --max-model-len "$CONTEXT_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      "${REASONING_ARGS[@]}" \
      --trust-remote-code \
      > "$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

cleanup() {
  echo "[entry] shutting down server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  # sglang spawns DP/scheduler children; kill the whole tree
  pkill -9 -P "$SERVER_PID" 2>/dev/null || true
  kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- wait for readiness ----------------------------------------------------
echo "[entry] waiting for server readiness (DP=$DP load can take minutes)..."
READY=0
for i in $(seq 1 240); do   # up to 20 min
  if curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    READY=1; echo "[entry] server up after ${i}x5s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[entry] FATAL: server process died during startup; tail of log:" >&2
    tail -n 40 "$SERVER_LOG" >&2; exit 1
  fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "[entry] FATAL: server not ready in time" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }

# --- RL projection config (override at submission to match your run) -------
RL_ROLLOUTS="${RL_ROLLOUTS:-4096}"           # e.g. 256 prompts * 16 samples
RL_CALLS_PER_ROLLOUT="${RL_CALLS_PER_ROLLOUT:-5}"
RL_STEPS="${RL_STEPS:-500}"

# --- temperature sweep: eval once per temperature on the shared server ------
# Headline question: does a lower temperature reduce the empty-final-channel
# parse failures? mean_parse_rate per temperature is the metric to watch.
SWEEP_SUMMARY="$OUT_DIR/run_${SWEEP_TS}_sweep_summary.txt"
mkdir -p "$OUT_DIR"
: > "$SWEEP_SUMMARY"
echo "[entry] temperature sweep over: $TEMPS  (n=$N per problem)"

for T in $TEMPS; do
  RUN_DIR="$OUT_DIR/run_${SWEEP_TS}_t${T}"
  PROGRESS_LOG="$RUN_DIR/progress.log"
  echo "[entry] ===== temperature=$T  ->  $RUN_DIR ====="
  echo "[entry] running eval (T=$T)... progress: $PROGRESS_LOG (tail -f to follow)"
  EVAL_START=$(date +%s)
  $PYTHON_BIN "$EVAL_PY" \
      --model "$SERVED_NAME" \
      --base-url "$BASE_URL" \
      --codex-dir "$CODEX_DIR" \
      --out-dir "$RUN_DIR" \
      --no-run-subdir \
      --progress-log "$PROGRESS_LOG" \
      --n "$N" \
      --temperature "$T" \
      --max-tokens "$MAX_TOKENS" \
      --workers "$WORKERS" \
      --sample-workers "$SAMPLE_WORKERS" \
      $EVAL_EXTRA_ARGS
  EVAL_WALL=$(( $(date +%s) - EVAL_START ))
  echo "[entry] T=$T complete in ${EVAL_WALL}s. summary: $RUN_DIR/_summary.json"

  # --- batch-time record + RL projection (per run) -------------------------
  # Uses the measured saturated throughput (tokens/s) from _summary.json to
  # extrapolate the selector inference cost of an RL run with this topology.
  ESTIMATE_FILE="$RUN_DIR/rl_time_estimate.txt"
  $PYTHON_BIN - "$RUN_DIR/_summary.json" "$EVAL_WALL" "$DP" "$TP" \
      "$RL_ROLLOUTS" "$RL_CALLS_PER_ROLLOUT" "$RL_STEPS" "$ESTIMATE_FILE" <<'PY'
import json, sys
summ_path, eval_wall, dp, tp, rollouts, calls, steps, out = sys.argv[1:9]
eval_wall=float(eval_wall); dp=int(dp); tp=int(tp)
rollouts=int(rollouts); calls=int(calls); steps=int(steps)
t = json.load(open(summ_path)).get("timing") or {}
tok_s = t.get("throughput_tokens_per_sec", 0.0)
tok_per_call = t.get("mean_completion_tokens_per_call", 0.0)
calls_per_step = rollouts * calls
tokens_per_step = calls_per_step * tok_per_call
sec_per_step = tokens_per_step / tok_s if tok_s else 0.0
total_calls = calls_per_step * steps
total_tokens = total_calls * tok_per_call
total_sec = sec_per_step * steps
def hms(s): return f"{int(s//3600)}h{int((s%3600)//60)}m" if s>=3600 else f"{int(s//60)}m{int(s%60)}s"
lines = [
  "=== selector inference timing (measured) ===",
  f"topology                : dp={dp} tp={tp}  (replicas={dp})",
  f"eval wall time          : {eval_wall:.0f}s",
  f"throughput (saturated)  : {tok_s:.0f} tok/s",
  f"mean tokens / call      : {tok_per_call:.0f}",
  "",
  "=== projected RL selector cost ===",
  f"config                  : {rollouts} rollouts x {calls} calls/rollout x {steps} steps",
  f"calls / step            : {calls_per_step}",
  f"tokens / step           : {tokens_per_step:,.0f}",
  f"selector time / step    : {sec_per_step:.0f}s  ({hms(sec_per_step)})",
  f"total selector calls    : {total_calls:,}",
  f"total selector tokens   : {total_tokens:,.0f}",
  f"TOTAL selector time     : {total_sec:.0f}s  ({hms(total_sec)})",
  "",
  "note: assumes the selector saturates this topology during RL (true if",
  "rollouts batch concurrently). If the selector shares GPUs with policy/train",
  "it is additive serially; on dedicated GPUs it overlaps up to max(step,sel).",
]
report="\n".join(lines)
print(report)
open(out,"w").write(report+"\n")
print(f"\n[entry] RL estimate written to {out}")
PY

  # --- append this temperature's quality metrics to the sweep summary ------
  $PYTHON_BIN - "$T" "$RUN_DIR/_summary.json" "$SWEEP_SUMMARY" <<'PY'
import json, sys
T, summ, out = sys.argv[1:4]
try:
    d = json.load(open(summ))
except Exception as e:
    line = f"T={T:>5}  (no summary: {e})"
else:
    line = (f"T={T:>5}  parse_rate={d.get('mean_parse_rate',0):.3f}"
            f"  self_consistency={d.get('mean_self_consistency',0):.3f}"
            f"  agree_codex={d.get('mean_agreement_with_codex',0):.3f}"
            f"  majority_agree={d.get('majority_agree_rate',0):.3f}"
            f"  mean_tokens={d.get('mean_completion_tokens',0):.0f}")
print("[sweep] " + line)
open(out, "a").write(line + "\n")
PY
done

# --- sweep comparison ------------------------------------------------------
echo "=================================================================="
echo " TEMPERATURE SWEEP SUMMARY  (parse_rate is the headline metric)"
echo "=================================================================="
cat "$SWEEP_SUMMARY"
echo "------------------------------------------------------------------"
echo " per-temp run dirs : $OUT_DIR/run_${SWEEP_TS}_t<TEMP>/"
echo " sweep summary file: $SWEEP_SUMMARY"
echo " view: python3 view_selector.py --dir $OUT_DIR   (pick each run from the dropdown)"
# cleanup runs via trap
