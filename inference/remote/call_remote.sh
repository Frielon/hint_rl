#!/usr/bin/env bash
#
# call_remote.sh -- connect to a model served on a REMOTE node (by launch_gpt_oss_serve.sh)
# and verify it from THIS (dev) box: discover the endpoint, probe every advertised
# NIC, smoke-test a chat completion, and optionally run the full eval_ckpts.py.
#
# Run THIS on the dev box after `launch_gpt_oss_serve.sh` is up on the remote node.
#
# Endpoint discovery (first match wins):
#   1. BASE_URL set in the env            -> used verbatim (e.g. an SSH tunnel:
#                                            BASE_URL=http://127.0.0.1:30000/v1).
#   2. REMOTE_HOST[:+PORT] set            -> http://REMOTE_HOST:PORT/v1.
#   3. ENDPOINT_FILE (shared-NFS rendezvous, written by launch_gpt_oss_serve.sh) -> probe
#      each advertised host:port, keep the FIRST that answers /v1/models with the
#      expected served name. This auto-selects the routable fabric and ignores
#      NICs the dev box can't reach (the multi-NIC pitfall we hit before).
#
# Usage:
#   ./call_remote.sh                 # discover + health-check + smoke-test
#   ./call_remote.sh --eval          # ...then run eval_ckpts.py (set DATASET/N)
#   BASE_URL=http://10.20.x.y:30000/v1 ./call_remote.sh
#   DATASET=/share5/users/xutao.ma/project/hint_rl/dataset/aime2025.parquet N=8 LIMIT=4 ./call_remote.sh --eval
# ---------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../inference/remote
HINT_RL_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"
EVAL_PY="$HINT_RL_HOME/eval/eval_ckpts.py"   # eval_ckpts.py stays in the eval/ harness

# --- config (env-overridable) ----------------------------------------------
ENDPOINT_FILE="${ENDPOINT_FILE:-$SCRIPT_DIR/remote_endpoint.env}"
PORT="${PORT:-30000}"                  # used only for the REMOTE_HOST form
API_KEY="${API_KEY:-}"                 # overrides the advertised key if set
PROBE_TIMEOUT="${PROBE_TIMEOUT:-5}"    # per-candidate /v1/models probe (s)
ENDPOINT_WAIT="${ENDPOINT_WAIT:-600}"  # wait this long for the rendezvous file (s)
SMOKE_PROMPT="${SMOKE_PROMPT:-What is 127*129? Put the final answer in \\boxed{}.}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-512}"   # headroom for reasoning models (gpt-oss)

RUN_EVAL=0
for a in "$@"; do
  case "$a" in
    --eval) RUN_EVAL=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[call] WARN: ignoring unknown arg '$a'" >&2 ;;
  esac
done

# --- discover candidate endpoints ------------------------------------------
# Preserve any env-provided values BEFORE sourcing the rendezvous file (sourcing
# would otherwise clobber API_KEY / SERVED_NAME with the file's values). Env wins.
_ENV_API_KEY="${API_KEY:-}"
_ENV_SERVED_NAME="${SERVED_NAME:-}"
SERVED_NAME="${SERVED_NAME:-}"
CANDIDATES=()   # full base URLs to try
if [ -n "${BASE_URL:-}" ]; then
  CANDIDATES=("$BASE_URL")
  echo "[call] using BASE_URL from env: $BASE_URL"
elif [ -n "${REMOTE_HOST:-}" ]; then
  CANDIDATES=("http://${REMOTE_HOST}:${PORT}/v1")
  echo "[call] using REMOTE_HOST from env: ${REMOTE_HOST}:${PORT}"
else
  # wait for the shared-NFS rendezvous file (launch_gpt_oss_serve.sh writes it only once
  # the server is actually ready), then read the advertised hosts.
  if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "[call] waiting up to ${ENDPOINT_WAIT}s for rendezvous file: $ENDPOINT_FILE"
    waited=0
    while [ ! -f "$ENDPOINT_FILE" ]; do
      [ "$waited" -ge "$ENDPOINT_WAIT" ] && {
        echo "FATAL: no rendezvous file after ${ENDPOINT_WAIT}s." >&2
        echo "       Is launch_gpt_oss_serve.sh running on the remote node? Or set BASE_URL/REMOTE_HOST." >&2
        exit 1; }
      sleep 5; waited=$((waited+5))
    done
  fi
  # shellcheck disable=SC1090
  . "$ENDPOINT_FILE"               # sets REMOTE_HOSTS / REMOTE_PORT / SERVED_NAME / API_KEY
  echo "[call] rendezvous: $ENDPOINT_FILE"
  echo "       served='$SERVED_NAME' port=$REMOTE_PORT hosts=[$REMOTE_HOSTS] started=$STARTED_AT"
  # env override wins over the file's values
  [ -n "$_ENV_API_KEY" ] && API_KEY="$_ENV_API_KEY"
  [ -n "$_ENV_SERVED_NAME" ] && SERVED_NAME="$_ENV_SERVED_NAME"
  for h in $REMOTE_HOSTS; do
    CANDIDATES+=("http://${h}:${REMOTE_PORT}/v1")
  done
fi
[ -z "${API_KEY:-}" ] && API_KEY="EMPTY"

# --- probe candidates: first to answer /v1/models (with served name) wins ---
AUTH_HDR=()
[ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ] && AUTH_HDR=(-H "Authorization: Bearer $API_KEY")

probe() {  # $1 = base url ; echoes the /v1/models body on success
  curl -sf --max-time "$PROBE_TIMEOUT" "${AUTH_HDR[@]}" "$1/models" 2>/dev/null
}

CHOSEN=""
echo "[call] probing ${#CANDIDATES[@]} candidate endpoint(s)..."
for url in "${CANDIDATES[@]}"; do
  body="$(probe "$url" || true)"
  if [ -n "$body" ]; then
    # if we know the served name, require it (guards against a stray local server
    # on the same port -- e.g. our own docker-bridge IP colliding).
    if [ -z "$SERVED_NAME" ] || printf '%s' "$body" | grep -q "$SERVED_NAME"; then
      CHOSEN="$url"
      echo "[call]   OK   $url"
      break
    else
      echo "[call]   skip $url  (answered, but served name '$SERVED_NAME' not present)"
    fi
  else
    echo "[call]   dead $url"
  fi
done

if [ -z "$CHOSEN" ]; then
  echo "" >&2
  echo "FATAL: no advertised endpoint is reachable from this box." >&2
  echo "  None of the remote NICs route here. Fall back to an SSH tunnel:" >&2
  FIRST_HOST="$(printf '%s' "${REMOTE_HOSTS:-<remote-node>}" | awk '{print $NF}')"
  echo "      ssh -N -L ${REMOTE_PORT:-$PORT}:127.0.0.1:${REMOTE_PORT:-$PORT} <user>@${FIRST_HOST}" >&2
  echo "  then re-run:  BASE_URL=http://127.0.0.1:${REMOTE_PORT:-$PORT}/v1 ./call_remote.sh" >&2
  exit 1
fi

# learn the served name from the endpoint if we didn't have it (BASE_URL path)
if [ -z "$SERVED_NAME" ]; then
  SERVED_NAME="$(probe "$CHOSEN" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  SERVED_NAME="${SERVED_NAME:-model}"
fi
echo "[call] connected: $CHOSEN   (served model: $SERVED_NAME)"

# --- smoke test: one chat completion ---------------------------------------
echo "[call] smoke-test chat completion..."
REQ=$(SMOKE_PROMPT="$SMOKE_PROMPT" SERVED_NAME="$SERVED_NAME" SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS" python3 - <<'PY'
import json, os
print(json.dumps({
    "model": os.environ["SERVED_NAME"],
    "messages": [{"role": "user", "content": os.environ["SMOKE_PROMPT"]}],
    "max_tokens": int(os.environ["SMOKE_MAX_TOKENS"]),
    "temperature": 0,
}))
PY
)
t0=$(date +%s.%N)
RESP="$(curl -sf --max-time 180 "${AUTH_HDR[@]}" -H 'Content-Type: application/json' \
        "$CHOSEN/chat/completions" -d "$REQ" || true)"
t1=$(date +%s.%N)
if [ -z "$RESP" ]; then
  echo "FATAL: chat completion request failed (endpoint answered /v1/models but not /v1/chat/completions)." >&2
  exit 1
fi
RESP="$RESP" DT="$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")" python3 - <<'PY'
import json, os
r = json.loads(os.environ["RESP"])
ch = r["choices"][0]
m = ch.get("message", {})
content = (m.get("content") or "").strip()
reasoning = (m.get("reasoning_content") or "").strip()   # gpt-oss CoT channel
usage = r.get("usage", {})
fin = ch.get("finish_reason", "?")
print("-" * 66)
if content:
    print(content[:1200])
elif reasoning:
    # reasoning model spent the budget thinking and never emitted a final answer
    print(f"(no final content; reasoning_content only{', finish=length -> raise SMOKE_MAX_TOKENS' if fin=='length' else ''})")
    print("  reasoning[:500]: " + reasoning[:500])
else:
    print("(empty response)")
print("-" * 66)
print(f"  latency = {os.environ['DT']}s   "
      f"completion_tokens = {usage.get('completion_tokens','?')}   "
      f"finish = {fin}{'   +reasoning_content' if reasoning else ''}")
print("[call] SMOKE TEST PASSED -- remote model is reachable and generating.")
PY

# --- optional: full eval over the remote endpoint --------------------------
if [ "$RUN_EVAL" = 1 ]; then
  DATASET="${DATASET:-$HINT_RL_HOME/dataset/aime2025.parquet}"
  [ -f "$DATASET" ] || { echo "FATAL: --eval needs a valid DATASET parquet (got: $DATASET)" >&2; exit 1; }
  N="${N:-128}"; CHUNK="${CHUNK:-1}"; CONCURRENCY="${CONCURRENCY:-192}"
  TEMPERATURE="${TEMPERATURE:-1.0}"; TOP_P="${TOP_P:-1.0}"; TOP_K="${TOP_K:--1}"
  MAX_TOKENS="${MAX_TOKENS:-8192}"; LIMIT="${LIMIT:-}"
  MODEL_LABEL="${MODEL_LABEL:-$SERVED_NAME}"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/remote_${RUN_TS}/$MODEL_LABEL/$(basename "${DATASET%.parquet}")}"
  mkdir -p "$OUT_DIR"

  # activate the verl env just for the eval client (openai/httpx/mathruler/pandas)
  CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"; CONDA_ENV="${CONDA_ENV:-verl}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  if [ -n "$CONDA_ENV" ]; then
    ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
    [ -x "$ENV_PREFIX/bin/python" ] || { echo "FATAL: env python not found: $ENV_PREFIX/bin/python" >&2; exit 1; }
    export CONDA_PREFIX="$ENV_PREFIX"; export PATH="$ENV_PREFIX/bin:$PATH"
    [ "$PYTHON_BIN" = "python" ] && PYTHON_BIN="$ENV_PREFIX/bin/python"
    unset PYTHONPATH; export PYTHONNOUSERSITE=1
  fi

  EXTRA=()
  [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
  [ "${HINT_MODE:-0}" = 1 ] && EXTRA+=(--hint-mode)
  echo "[call] running eval_ckpts.py: $(basename "$DATASET")  n=$N  -> $OUT_DIR"
  "$PYTHON_BIN" "$EVAL_PY" \
      --dataset "$DATASET" \
      --model "$SERVED_NAME" --model-label "$MODEL_LABEL" \
      --base-url "$CHOSEN" --api-key "$API_KEY" \
      --out-dir "$OUT_DIR" \
      --n "$N" --chunk "$CHUNK" --concurrency "$CONCURRENCY" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      "${EXTRA[@]}"
  echo "[call] eval done -> $OUT_DIR"
fi
