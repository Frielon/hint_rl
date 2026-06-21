#!/usr/bin/env bash
#
# fs_call.sh -- call the remote gpt-oss server over the SHARED-NFS file bridge,
# from the dev box, with NO network route to the pod. Pairs with the fs_bridge.py
# that launch_gpt_oss_serve.sh runs inside the pod.
#
# It starts a local OpenAI shim (fs_proxy.py) that proxies HTTP<->files, checks the
# pod bridge's heartbeat, smoke-tests one chat completion (which round-trips
# dev-box -> NFS -> pod -> vLLM -> NFS -> dev-box), and optionally runs the full
# eval_ckpts.py through the shim.
#
# Usage:
#   ./fs_call.sh                 # heartbeat + shim + smoke test
#   ./fs_call.sh --eval          # ... then eval_ckpts.py (set DATASET / N / LIMIT)
#   DATASET=/share5/users/xutao.ma/project/hint_rl/dataset/aime2025.parquet N=16 LIMIT=4 ./fs_call.sh --eval
# ---------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../inference/remote
HINT_RL_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"
EVAL_PY="$HINT_RL_HOME/eval/eval_ckpts.py"

# --- config (env-overridable) ----------------------------------------------
BRIDGE_DIR="${BRIDGE_DIR:-$SCRIPT_DIR/bridge}"   # shared NFS dir (matches the pod)
SHIM_HOST="${SHIM_HOST:-127.0.0.1}"
SHIM_PORT="${SHIM_PORT:-38900}"
SERVED_NAME="${SERVED_NAME:-gpt-oss-20b}"
SMOKE_PROMPT="${SMOKE_PROMPT:-What is 127*129? Put the final answer in \\boxed{}.}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-512}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-600}"            # generous: file round-trip + gen
RUN_EVAL=0
for a in "$@"; do case "$a" in --eval) RUN_EVAL=1 ;; -h|--help) sed -n '2,20p' "$0"; exit 0 ;; esac; done

# --- verl env (fs_proxy is stdlib; eval_ckpts needs openai/httpx/mathruler/pandas) -
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"; CONDA_ENV="${CONDA_ENV:-verl}"
ENV_PREFIX="$CONDA_ROOT/envs/$CONDA_ENV"
[ -x "$ENV_PREFIX/bin/python" ] || { echo "FATAL: env python not found: $ENV_PREFIX/bin/python" >&2; exit 1; }
export PATH="$ENV_PREFIX/bin:$PATH"; export CONDA_PREFIX="$ENV_PREFIX"
unset PYTHONPATH; export PYTHONNOUSERSITE=1
PYTHON_BIN="$ENV_PREFIX/bin/python"

mkdir -p "$BRIDGE_DIR/req" "$BRIDGE_DIR/resp" "$SCRIPT_DIR/logs"

# --- bridge heartbeat check ------------------------------------------------
echo "[fs] bridge dir   : $BRIDGE_DIR"
HB="$("$PYTHON_BIN" - "$BRIDGE_DIR" <<'PY'
import json, os, sys, time
d = sys.argv[1]
try:
    hb = json.load(open(os.path.join(d, "bridge.alive")))
    age = time.time() - float(hb.get("ts", 0))
    print(("OK " if age < 30 else "STALE ") + f"age={age:.0f}s pid={hb.get('pid')} inflight={hb.get('inflight')}")
except Exception as e:  # noqa: BLE001
    print(f"MISSING ({type(e).__name__})")
PY
)"
echo "[fs] bridge alive : $HB"
case "$HB" in
  OK*) ;;
  *) echo "[fs] WARN: pod bridge heartbeat not fresh. Is launch_gpt_oss_serve.sh (with the bridge) running in the pod, and is BRIDGE_DIR the same shared path?" >&2 ;;
esac

# --- start the local OpenAI shim -------------------------------------------
SHIM_LOG="$SCRIPT_DIR/logs/fs_proxy_${SHIM_PORT}.log"
"$PYTHON_BIN" "$SCRIPT_DIR/fs_proxy.py" --dir "$BRIDGE_DIR" --host "$SHIM_HOST" --port "$SHIM_PORT" \
    --timeout "$SMOKE_TIMEOUT" > "$SHIM_LOG" 2>&1 &
SHIM_PID=$!
cleanup() { kill "$SHIM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 50); do
  curl -sf "http://$SHIM_HOST:$SHIM_PORT/" >/dev/null 2>&1 && break
  kill -0 "$SHIM_PID" 2>/dev/null || { echo "FATAL: shim died:" >&2; cat "$SHIM_LOG" >&2; exit 1; }
  sleep 0.1
done
BASE_URL="http://$SHIM_HOST:$SHIM_PORT/v1"
echo "[fs] local shim   : $BASE_URL   (HTTP <-> files <-> pod vLLM)"

# --- smoke test (round-trips through the shared filesystem) -----------------
echo "[fs] smoke test (dev-box -> NFS -> pod -> vLLM -> NFS -> dev-box)..."
REQ="$(SMOKE_PROMPT="$SMOKE_PROMPT" SERVED_NAME="$SERVED_NAME" SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS" python3 - <<'PY'
import json, os
print(json.dumps({
    "model": os.environ["SERVED_NAME"],
    "messages": [{"role": "user", "content": os.environ["SMOKE_PROMPT"]}],
    "max_tokens": int(os.environ["SMOKE_MAX_TOKENS"]),
    "temperature": 0,
}))
PY
)"
t0=$(date +%s.%N)
RESP="$(curl -sf --max-time "$SMOKE_TIMEOUT" "$BASE_URL/chat/completions" \
        -H 'Content-Type: application/json' -d "$REQ" || true)"
t1=$(date +%s.%N)
[ -n "$RESP" ] || { echo "FATAL: no response over the file bridge (timeout). See $SHIM_LOG and the pod bridge log." >&2; exit 1; }
RESP="$RESP" DT="$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")" python3 - <<'PY'
import json, os
r = json.loads(os.environ["RESP"])
ch = r["choices"][0]; m = ch.get("message", {})
content = (m.get("content") or "").strip()
reasoning = (m.get("reasoning_content") or "").strip()
usage = r.get("usage", {}); fin = ch.get("finish_reason", "?")
print("-" * 66)
if content:
    print(content[:1200])
elif reasoning:
    print(f"(no final content; reasoning_content only{', finish=length -> raise SMOKE_MAX_TOKENS' if fin=='length' else ''})")
    print("  reasoning[:500]: " + reasoning[:500])
else:
    print("(empty response)")
print("-" * 66)
print(f"  round-trip = {os.environ['DT']}s   completion_tokens = {usage.get('completion_tokens','?')}   "
      f"finish = {fin}{'   +reasoning_content' if reasoning else ''}")
print("[fs] SMOKE TEST PASSED -- reached the remote model entirely over the shared filesystem.")
PY

# --- optional full eval through the shim -----------------------------------
if [ "$RUN_EVAL" = 1 ]; then
  DATASET="${DATASET:-$HINT_RL_HOME/dataset/aime2025.parquet}"
  [ -f "$DATASET" ] || { echo "FATAL: --eval needs a valid DATASET parquet (got: $DATASET)" >&2; exit 1; }
  N="${N:-128}"; CHUNK="${CHUNK:-1}"; CONCURRENCY="${CONCURRENCY:-128}"
  TEMPERATURE="${TEMPERATURE:-1.0}"; TOP_P="${TOP_P:-1.0}"; TOP_K="${TOP_K:--1}"
  MAX_TOKENS="${MAX_TOKENS:-8192}"; LIMIT="${LIMIT:-}"
  MODEL_LABEL="${MODEL_LABEL:-$SERVED_NAME}"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/fs_${RUN_TS}/$MODEL_LABEL/$(basename "${DATASET%.parquet}")}"
  mkdir -p "$OUT_DIR"
  EXTRA=(); [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT"); [ "${HINT_MODE:-0}" = 1 ] && EXTRA+=(--hint-mode)
  echo "[fs] eval over the file bridge: $(basename "$DATASET")  n=$N concurrency=$CONCURRENCY -> $OUT_DIR"
  echo "[fs]   (the shim's --timeout caps each request; raise SMOKE_TIMEOUT if NFS latency is high)"
  "$PYTHON_BIN" "$EVAL_PY" \
      --dataset "$DATASET" \
      --model "$SERVED_NAME" --model-label "$MODEL_LABEL" \
      --base-url "$BASE_URL" --api-key EMPTY \
      --out-dir "$OUT_DIR" \
      --n "$N" --chunk "$CHUNK" --concurrency "$CONCURRENCY" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" --timeout "$SMOKE_TIMEOUT" \
      "${EXTRA[@]}"
  echo "[fs] eval done -> $OUT_DIR"
fi
