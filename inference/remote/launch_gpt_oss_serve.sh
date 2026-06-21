#!/usr/bin/env bash
# =============================================================================
# launch_gpt_oss_serve.sh
# -----------------------------------------------------------------------------
# SINGLE self-contained cluster launch file: serve gpt-oss-20b on a remote
# 8x H100 node and advertise the endpoint back to the dev box over the shared
# /share5 NFS. Point the cluster platform's entrypoint at THIS one file.
#
# Every serving parameter is HARDCODED in the CONFIG block below -- the platform
# does not need to inject any env var. The platform may append junk positional
# args (e.g. `.../launch_gpt_oss_serve.sh /xutao/ undefined`); they are ignored.
#
# What it does:
#   1. finds the repo root on WHATEVER mount this node sees (the tree may be at
#      /xutao/... in the job container, not /share5/...), and symlinks the
#      baked-in conda prefix so the env resolves;
#   2. activates the verl conda env the conda-free way (env interpreter directly,
#      not `conda activate` -- the miniconda prefix is baked in absolutely);
#   3. serves gpt-oss-20b on a vLLM OpenAI endpoint bound 0.0.0.0, DP=#GPUs,
#      TP=1, with --reasoning-parser openai_gptoss (routes the gpt-oss analysis
#      CoT to reasoning_content, the answer to content);
#   4. once READY, advertises every routable NIC + FQDN to ENDPOINT_FILE on the
#      shared NFS so the dev box auto-discovers it (multi-NIC, no IP copying);
#   5. holds the server in the foreground (keeps the pod alive); SIGTERM/Ctrl-C
#      tears it down and removes the advertisement.
#
# After it prints READY, test it from the dev box:
#   /share5/users/xutao.ma/project/hint_rl/inference/remote/call_remote.sh
# =============================================================================
set -eo pipefail          # NOT -u: conda activate hooks reference unbound vars

# --- path bootstrap: locate the repo root on this node's mount ---------------
# Resolved from this script's own location so it works at /share5/... (dev box)
# or /xutao/... (job container). Hardcoded fallback if BASH_SOURCE is unusable.
SRC="${BASH_SOURCE[0]:-/share5/users/xutao.ma/project/hint_rl/inference/remote/launch_gpt_oss_serve.sh}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"        # .../hint_rl/inference/remote
HINT_RL_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"      # .../hint_rl
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"       # .../project
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"          # mount root (== /share5/users/xutao.ma)
VERL_HOME="$PROJECT_HOME/verl"

# =============================================================================
# CONFIG  --  every parameter for gpt-oss-20b, hardcoded (edit here)
# =============================================================================
MODEL_PATH="$BASE_HOME/model/gpt-oss-20b"   # MXFP4 MoE weights (resolves on any mount)
SERVED_NAME="gpt-oss-20b"                    # model id the client calls
PORT=30000                                   # OpenAI server port (bound 0.0.0.0)
TP=1                                         # tensor-parallel (1: gpt-oss fits one H100)
DP=0                                         # data-parallel replicas; 0 => auto = #visible GPUs
CONTEXT_LEN=40960                            # max model len (selector default)
MEM_FRAC=0.90                                # gpu memory utilization
MAX_NUM_SEQS=512                             # gpt-oss is pure-attention; high concurrency
REASONING_PARSER="openai_gptoss"             # vLLM's gpt-oss CoT parser -> reasoning_content
API_KEY="EMPTY"                              # "EMPTY" = open endpoint (trusted fabric); else a bearer token
# Rendezvous file on the SHARED NFS -- the dev-box client reads this exact path.
# On the remote this resolves through BASE_HOME to the SAME NFS file the dev box
# sees at /share5/users/xutao.ma/project/hint_rl/inference/remote/remote_endpoint.env
ENDPOINT_FILE="$HINT_RL_HOME/inference/remote/remote_endpoint.env"
# conda env holding vllm (+ the gpt-oss serving support)
CONDA_ROOT="$BASE_HOME/miniconda3"
CONDA_ENV="verl"
# file bridge: let the dev box call this server over the SHARED NFS even when the
# pod has no routable NIC (k8s overlay-only, 10.233.x). fs_bridge.py (here) <->
# fs_proxy.py (dev box). ENABLE_BRIDGE=0 to skip. BRIDGE_DIR must be the same
# shared path the dev-box fs_call.sh uses (its default matches this).
ENABLE_BRIDGE=1
BRIDGE_DIR="$HINT_RL_HOME/inference/remote/bridge"
# =============================================================================

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# --- non-relocatable miniconda: symlink the baked-in install prefix ----------
# conda.sh / shebangs / the editable verl .pth bake the ORIGINAL absolute prefix
# (/share5/users/xutao.ma). When the tree is mounted elsewhere those don't
# resolve; symlink the original prefix -> this mount. No-op where it exists.
CONDA_INSTALL_PREFIX="/share5/users/xutao.ma"
_VERL_PYTHONPATH_FALLBACK=""
if [ ! -e "$CONDA_INSTALL_PREFIX" ]; then
  ( sudo mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && sudo ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null \
    || ( mkdir -p "$(dirname "$CONDA_INSTALL_PREFIX")" && ln -sfn "$BASE_HOME" "$CONDA_INSTALL_PREFIX" ) 2>/dev/null \
    || true
  if [ -e "$CONDA_INSTALL_PREFIX" ]; then
    echo "[launch] symlinked baked-in conda prefix $CONDA_INSTALL_PREFIX -> $BASE_HOME"
  else
    echo "[launch] WARN: could not symlink $CONDA_INSTALL_PREFIX; adding $VERL_HOME to PYTHONPATH as fallback" >&2
    _VERL_PYTHONPATH_FALLBACK="$VERL_HOME"
  fi
fi

# --- python env activation (conda-free; verbatim from run_eval_ckpt.sh) ------
ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
[ -x "$ENV_PREFIX/bin/python" ] || {
  echo "FATAL: env python not found/executable: $ENV_PREFIX/bin/python" >&2
  echo "       (check CONDA_ROOT/CONDA_ENV in the CONFIG block)" >&2
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
  echo "[launch] CUDA toolkit: $CUDA_HOME"
elif [ -d "$SITE/nvidia/cuda_runtime/lib" ]; then
  export CUDA_HOME="$SITE/nvidia/cuda_runtime"
fi
echo "[launch] activated env (conda-free): $ENV_PREFIX  python=$PYTHON_BIN"

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
REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")

# --- enumerate the addresses this node can be reached at ---------------------
# Advertise every non-loopback / non-link-local IPv4 (10.x fabric first), plus
# the FQDN. The client probes each and keeps the first that ANSWERS with our
# served name -- auto-selecting whichever fabric routes (the multi-NIC pitfall
# that silently broke the selector before), and a stray local server on the same
# port can't false-positive (the served name won't match).
collect_hosts() {
  hostname -I 2>/dev/null | tr ' ' '\n' \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
    | grep -v -e '^127\.' -e '^169\.254\.' \
    | awk '{ print (index($0,"10.")==1 ? 0 : 1), $0 }' \
    | sort -s -k1,1n | awk '{print $2}'
}
HOST_IPS="$(collect_hosts | paste -sd' ' -)"
FQDN="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo '')"
ADVERTISE_HOSTS="$HOST_IPS${FQDN:+ $FQDN}"

echo "=================================================================="
echo " launch_gpt_oss_serve  (single cluster launch file)"
echo "   node        : ${FQDN:-?}   ($(hostname -s 2>/dev/null))"
echo "   model       : $MODEL_PATH  (served as '$SERVED_NAME')"
echo "   topology    : dp=$DP tp=$TP  ctx=$CONTEXT_LEN mem=$MEM_FRAC max_num_seqs=$MAX_NUM_SEQS"
echo "   reasoning   : --reasoning-parser $REASONING_PARSER"
echo "   port        : $PORT  (bound 0.0.0.0)   api_key=$([ "$API_KEY" = EMPTY ] && echo '(none)' || echo '(set)')"
echo "   advertise   : $ADVERTISE_HOSTS"
echo "   endpoint    : $ENDPOINT_FILE  (shared NFS rendezvous)"
echo "=================================================================="

# --- launch + readiness + advertise + hold -----------------------------------
SERVER_PID=""
cleanup() {
  echo ""
  echo "[launch] tearing down..."
  rm -f "$ENDPOINT_FILE" 2>/dev/null || true
  [ -n "${BRIDGE_PID:-}" ] && { kill "$BRIDGE_PID" 2>/dev/null || true; }
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -9 -P "$SERVER_PID" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVER_LOG="$LOG_DIR/vllm_${SERVED_NAME}_${PORT}.log"
echo "[launch] starting vLLM -> $SERVER_LOG"
API_KEY_ARG=()
[ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && API_KEY_ARG=(--api-key "$API_KEY")
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_FRAC" \
    --max-model-len "$CONTEXT_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${REASONING_ARGS[@]}" \
    --trust-remote-code \
    "${API_KEY_ARG[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[launch] waiting for readiness (DP=$DP load can take minutes)..."
AUTH_HDR=()
[ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && AUTH_HDR=(-H "Authorization: Bearer $API_KEY")
ready=0
for i in $(seq 1 240); do   # up to 20 min
  if curl -s "${AUTH_HDR[@]}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    ready=1; echo "[launch] server up after ${i}x5s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[launch] FATAL: server died during startup; tail of log:" >&2
    tail -n 40 "$SERVER_LOG" >&2; exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "[launch] FATAL: not ready in time" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }

# advertise (sourceable by call_remote.sh -- no jq needed); only AFTER ready, so
# the file's presence means "server is up and probes will succeed".
umask 022
cat > "$ENDPOINT_FILE" <<EOF
# written by launch_gpt_oss_serve.sh on ${FQDN:-$(hostname)} at $(date -Is)
REMOTE_HOSTS="${ADVERTISE_HOSTS}"
REMOTE_PORT="${PORT}"
SERVED_NAME="${SERVED_NAME}"
API_KEY="${API_KEY}"
MODEL_PATH="${MODEL_PATH}"
FS_BRIDGE="${ENABLE_BRIDGE}"
BRIDGE_DIR="${BRIDGE_DIR}"
STARTED_AT="$(date -Is)"
EOF

# start the shared-filesystem bridge so the dev box can call us with NO network
# route to the pod (the usual case here: overlay-only pod). Background; cleanup()
# kills it; the vLLM server stays in the foreground.
BRIDGE_PID=""
if [ "$ENABLE_BRIDGE" = 1 ]; then
  mkdir -p "$BRIDGE_DIR/req" "$BRIDGE_DIR/resp"
  BRIDGE_LOG="$LOG_DIR/fs_bridge_${PORT}.log"
  echo "[launch] starting file bridge -> $BRIDGE_LOG  (shared dir: $BRIDGE_DIR)"
  "$PYTHON_BIN" "$SCRIPT_DIR/fs_bridge.py" \
      --dir "$BRIDGE_DIR" --target "http://127.0.0.1:$PORT" \
      --concurrency "$MAX_NUM_SEQS" > "$BRIDGE_LOG" 2>&1 &
  BRIDGE_PID=$!
fi

echo "=================================================================="
echo "[launch] READY. gpt-oss-20b advertised to: $ENDPOINT_FILE"
echo ""
echo "  Over the shared NFS (NO network route to the pod needed) -- on the dev box:"
echo "      /share5/users/xutao.ma/project/hint_rl/inference/remote/fs_call.sh"
echo ""
echo "  If this pod has a routable NIC, you can also connect directly:"
echo "      /share5/users/xutao.ma/project/hint_rl/inference/remote/call_remote.sh"
echo "  ...or target a specific NIC by hand:"
for h in $ADVERTISE_HOSTS; do
  echo "      BASE_URL=http://$h:$PORT/v1   (served model: $SERVED_NAME)"
done
echo ""
echo "  If NO NIC routes from the dev box, fall back to an SSH tunnel (on the dev box):"
echo "      ssh -N -L ${PORT}:127.0.0.1:${PORT} <user>@${FQDN:-<this-node>}"
echo "      BASE_URL=http://127.0.0.1:${PORT}/v1 <.../call_remote.sh>"
echo ""
echo "  Server log: $SERVER_LOG"
echo "=================================================================="

# hold the server in the foreground (keeps the pod alive); cleanup() on exit.
wait "$SERVER_PID"
