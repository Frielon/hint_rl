#!/usr/bin/env bash
# =============================================================================
# launch_native_len_2node.sh -- TWO-NODE entrypoint for the native-generation-
# length test (run_eval_native_len.sh). Point the cluster platform at THIS one
# file and run it on BOTH pods of a 2-replica job (8 GPUs each).
#
# The test is embarrassingly parallel across the two models, so there is no Ray
# cluster and no cross-node serving: each pod claims ONE model via an atomic-
# mkdir rendezvous on the shared NFS and runs the whole serve->eval->length-
# stats pipeline for it locally (vLLM DP=#GPUs on that pod, port 30000 -- no
# cross-node port conflict). Both pods write into the SAME launch-stamped
# OUT_BASE on the NFS, and whichever pod finishes LAST prints the combined
# two-model summary table.
#
#   pod A -> qwen3-4b-instruct-2507  (ctx 67584, max_tokens 65536)
#   pod B -> olmo-3-7b-instruct      (ctx 65536, max_tokens 63488; 64k hard cap)
#
# Rendezvous: both pods derive one shared JOB_STAMP from an atomic mkdir under
# eval/results/ (keyed on MASTER_ADDR:MASTER_PORT when the platform injects
# them; TTL-rotated so a relaunch gets a fresh stamp). Model assignment is
# claim-based, not RANK-based, so it also works when the platform injects no
# rank env -- and running the script on >2 pods just idles the extras.
# NOTE (launch-stamp split-brain): a pod that starts > STAMP_TTL after its
# sibling would rotate the stamp and re-claim a fresh dir. If your platform can
# stagger pod starts by that much, either raise STAMP_TTL or pin JOB_STAMP in
# the job env (both pods then skip the rendezvous entirely).
#
# Platform junk positional args are ignored. Overrides via env:
#   JOB_STAMP     pin the shared stamp/output dir name (skips rendezvous)
#   STAMP_TTL     stamp reuse window in seconds (default 900)
#   plus everything run_eval_native_len.sh accepts (N, TEMPERATURE, PORT, ...).
# Output: eval/results/native_len_<stamp>/<model>/<dataset>/
# Per-pod console logs: eval/logs/native_len_<stamp>_<model>.log (shared NFS).
# =============================================================================
set -eo pipefail

SRC="${BASH_SOURCE[0]:-/share5/users/xutao.ma/project/hint_rl/eval/launch_native_len_2node.sh}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"          # .../eval  (on this node's mount)
RUN_SH="${RUN_SH:-$SCRIPT_DIR/run_eval_native_len.sh}"
[ -f "$RUN_SH" ] || { echo "[launch] FATAL: $RUN_SH not found" >&2; exit 1; }

# labels must match the MODELS entries in run_eval_native_len.sh
LABELS=(
  # "qwen3-4b-instruct-2507"
  "olmo-3-7b-instruct"
)

# --- shared launch stamp: atomic mkdir on the NFS, one winner per job --------
PST_TZ="${PST_TZ:-America/Los_Angeles}"
pst_now() { TZ="$PST_TZ" date "$@"; }
STAMP_ROOT="${STAMP_ROOT:-$SCRIPT_DIR/results}"
STAMP_TTL="${STAMP_TTL:-900}"
mkdir -p "$STAMP_ROOT"
if [ -z "${JOB_STAMP:-}" ]; then
  _job_key="${MASTER_ADDR:-x}_${MASTER_PORT:-30000}"
  _stamp_dir="$STAMP_ROOT/.stamp.native_len.${_job_key}"
  if [ -f "$_stamp_dir/ts" ]; then
    _age=$(( $(date +%s) - $(stat -c %Y "$_stamp_dir/ts" 2>/dev/null || echo 0) ))
    if [ "$_age" -gt "$STAMP_TTL" ]; then
      mv "$_stamp_dir" "${_stamp_dir}.old.$(date +%s).$$" 2>/dev/null || true
    fi
  fi
  if mkdir "$_stamp_dir" 2>/dev/null; then             # atomic on shared FS -> one winner
    pst_now '+%Y%m%d_%H%M%S' > "$_stamp_dir/ts"
  fi
  JOB_STAMP=""
  for _ in $(seq 1 40); do                             # wait <=20s for the winner's stamp
    [ -s "$_stamp_dir/ts" ] && { JOB_STAMP="$(cat "$_stamp_dir/ts")"; break; }
    sleep 0.5
  done
  [ -n "$JOB_STAMP" ] || JOB_STAMP="$(pst_now '+%Y%m%d_%H%M%S')"   # fallback: own stamp
fi

OUT_BASE="$STAMP_ROOT/native_len_${JOB_STAMP}"
CLAIMS_DIR="$OUT_BASE/.claims"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$CLAIMS_DIR" "$LOG_DIR"

# --- claim ONE model (atomic mkdir; first come, first served) ----------------
MY_LABEL=""
for _lbl in "${LABELS[@]}"; do
  if mkdir "$CLAIMS_DIR/$_lbl" 2>/dev/null; then
    MY_LABEL="$_lbl"
    hostname > "$CLAIMS_DIR/$_lbl/host" 2>/dev/null || true
    break
  fi
done
if [ -z "$MY_LABEL" ]; then
  echo "[launch] $(hostname): all models already claimed under $OUT_BASE -- extra pod, idling out."
  echo "[launch] (relaunching? wait STAMP_TTL=${STAMP_TTL}s for a fresh stamp, or set JOB_STAMP.)"
  exit 0
fi

POD_LOG="$LOG_DIR/native_len_${JOB_STAMP}_${MY_LABEL}.log"
echo "[launch] $(hostname): claimed model '$MY_LABEL'  stamp=$JOB_STAMP"
echo "[launch]   out : $OUT_BASE"
echo "[launch]   log : $POD_LOG"

# --- run the single-model eval on THIS pod -----------------------------------
ONLY_MODEL="$MY_LABEL" OUT_BASE="$OUT_BASE" bash "$RUN_SH" 2>&1 | tee "$POD_LOG"

touch "$OUT_BASE/.done.$MY_LABEL"
echo "[launch] $(hostname): model '$MY_LABEL' DONE."

# --- last finisher prints the combined two-model summary ---------------------
# Poll briefly for the sibling's done marker (covers a near-tie + NFS attribute
# lag); the earlier finisher gives up and lets the later one print.
_all=0
for _ in $(seq 1 30); do
  _all=1
  for _lbl in "${LABELS[@]}"; do
    [ -e "$OUT_BASE/.done.$_lbl" ] || _all=0
  done
  [ "$_all" = 1 ] && break
  sleep 3
done
if [ "$_all" = 1 ]; then
  echo "[launch] all models done -> combined summary:"
  SUMMARY_ONLY=1 OUT_BASE="$OUT_BASE" bash "$RUN_SH"
else
  echo "[launch] sibling model still running; the last finisher will print the combined summary."
  echo "[launch] (or run later:  SUMMARY_ONLY=1 OUT_BASE=$OUT_BASE bash $RUN_SH)"
fi
