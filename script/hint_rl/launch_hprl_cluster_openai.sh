#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster_openai.sh -- HPRL cluster entrypoint with the OPENAI API
# as the frozen hint selector. NO selector pods.
#
# Run this on EVERY pod (head + workers), exactly like launch_hprl_cluster.sh.
# The difference vs that script: there is no selector/training node split --
# EVERY pod is a training node (they all form the Ray cluster and run
# TRAIN_SCRIPT via ray_cluster_launch.sh), and every hint call goes to
# api.openai.com instead of self-hosted gpt-oss-20b vLLM servers. So the whole
# selector serving machinery (per-pod vLLM bring-up, endpoint rendezvous dir,
# fabric-NIC advertisement) is gone; what remains is:
#
#   * env: SELECTOR_API_MODE=openai + the model/call params, read by
#     hint_selector.HintSelector.from_env() on the agent-loop workers (the run
#     script forwards them through the Ray runtime env). Reasoning models
#     (gpt-5*/o*) automatically use max_completion_tokens + reasoning_effort
#     and drop temperature/top_p (the API rejects non-default values).
#   * key: OPENAI_API_KEY from the environment, else sourced from
#     script/hint_rl/api_keys.sh (shared FS -> same file on every pod).
#   * a fail-fast reachability probe of ${SELECTOR_OPENAI_BASE_URL}/models from
#     every pod -- the direct analogue of the selector endpoint probe: a fabric
#     with no API egress would otherwise silently degrade EVERY hint call to a
#     no-op for the whole run. If the pods need an egress proxy, set
#     OPENAI_PROXY (or HTTPS_PROXY) in the job env; it is used by the probe and
#     forwarded to the workers by the run script.
#
# Model default: gpt-5-mini at reasoning_effort=low -- the best model of the
# offline multi-round selector eval (selector/test_cite/openai-models,
# 2026-07-22: agree_merged 0.873 / newly_recall 0.869 / citation verbatim 0.927
# vs the gpt-oss-20b reference's 0.835 / 0.688 / 0.646).
#
# COST NOTE: every wrong-answer turn of every rollout makes one API call whose
# prompt embeds the full student trace (can be 10k+ tokens late in a rollout).
# Expect very roughly $0.002-0.01 per call at gpt-5-mini prices; each worker
# process logs cumulative prompt/completion tokens every 200 calls
# ("HintSelector[...]" lines), and the per-call selector dump
# (HPRL_SELECTOR_DUMP_DIR) keeps the exact record for post-hoc accounting.
#
# Required env (injected by the PyTorchJob): MASTER_ADDR, MASTER_PORT and, for
# multi-pod jobs, WORLD_SIZE (= TRAIN_NNODES; there are no selector pods to
# subtract). RANK is only used to label the per-pod check log.
# Key overrides:
#   TRAIN_SCRIPT       training script the Ray head runs (default: the same
#                      sync default as launch_hprl_cluster.sh).
#   RESUME_EXP_NAME    existing exp_name to RESUME IN PLACE (default: the
#                      dolci-rl-zero-10324 run of 2026-07-24; empty = fresh run).
#   SELECTOR_MODEL     OpenAI model (default gpt-5-mini).
#   SELECTOR_*         call params (see below).
#   HPRL_AUTO_HINT_PROGRESS_MESSAGE
#                      cumulative completed-steps prefix in hint turns
#                      (default true for this launcher; false restores hint-only).
#   OPENAI_PROXY       egress proxy URL for api.openai.com (optional).
# =============================================================================
set -euo pipefail

# --- paths (mirrors launch_hprl_cluster.sh) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # script/hint_rl
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}
RAY_LAUNCH=${RAY_LAUNCH:-"${SCRIPT_DIR}/../ray_cluster_launch.sh"}    # script/ray_cluster_launch.sh
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_auto_hint_qwen3_8b_base.sh"}

# --- resume-in-place: this launch CONTINUES an existing run ------------------
# Pinning exp_name to the prior run (instead of letting the run wrapper stamp a
# fresh one) is the whole mechanism:
#   * CKPTS_DIR resolves to that run's ckpt dir, so verl's default
#     RESUME_MODE=auto loads the newest global_step_N there (340 at edit time)
#     -- weights, optimizer AND dataloader state (data.pt). Deliberately NOT
#     pinned via RESUME_FROM_PATH: auto tracks latest_checkpointed_iteration,
#     so a crash+relaunch of this launcher resumes from wherever the run got
#     to instead of rewinding to a hardcoded step.
#   * EXP_LOG_DIR resolves to that run's logs dir, so the budget ratchet's
#     budget_state.json -- which lives OUTSIDE the ckpt -- is found by
#     BudgetManager and the online-raised budgets carry over; the run script
#     rotates the prior metrics jsonl aside (verl's FileLogger would truncate
#     it) and stamps its own console log / src snapshot, so nothing collides.
# TRAIN_FILE must match the ckpt's dataset (the resumed dataloader requires
# it); the wrapper's dolci-rl-zero-10324 default already does. wandb gets a
# NEW run with the SAME name whose step axis continues at the resume step.
# Set RESUME_EXP_NAME= (empty) to launch a fresh stamped run as before.
RESUME_EXP_NAME=${RESUME_EXP_NAME-}
if [ -n "${RESUME_EXP_NAME}" ]; then
    export exp_name="${RESUME_EXP_NAME}"
fi

# --- conda env --------------------------------------------------------------
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- cluster topology: ALL pods are training pods ---------------------------
# No selector split to validate; TRAIN_NNODES defaults to the job's WORLD_SIZE
# (and vice versa) so a single knob sets both. A mismatch still aborts so a
# job-spec replica count that disagrees with an explicit TRAIN_NNODES fails at
# launch, not mid-run.
TRAIN_NNODES=${TRAIN_NNODES:-${WORLD_SIZE:-4}}
WORLD_SIZE=${WORLD_SIZE:-${TRAIN_NNODES}}
if [ "${TRAIN_NNODES}" -ne "${WORLD_SIZE}" ]; then
    echo "[launch] FATAL: TRAIN_NNODES(${TRAIN_NNODES}) != WORLD_SIZE(${WORLD_SIZE})." >&2
    echo "         With the OpenAI-API selector every pod trains; set the job's" >&2
    echo "         replica count to TRAIN_NNODES (or drop the TRAIN_NNODES override)." >&2
    exit 1
fi
# node rank, from whichever the launcher set -- only used to label the check log.
RANK=${RANK:-${NODE_RANK:-${GROUP_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-na}}}}}

# --- OPENAI key: env wins, else the shared api_keys.sh ----------------------
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "${SCRIPT_DIR}/api_keys.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/api_keys.sh"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[launch] FATAL: OPENAI_API_KEY is unset and ${SCRIPT_DIR}/api_keys.sh did not provide it." >&2
    exit 1
fi
export OPENAI_API_KEY

# --- optional egress proxy for api.openai.com -------------------------------
# OPENAI_PROXY populates HTTPS_PROXY/https_proxy (unless already set); the
# probe below uses it, and the run script forwards any set proxy vars into the
# workers' runtime env.
if [ -n "${OPENAI_PROXY:-}" ]; then
    export HTTPS_PROXY=${HTTPS_PROXY:-"${OPENAI_PROXY}"}
    export https_proxy=${https_proxy:-"${OPENAI_PROXY}"}
fi

# --- selector CALL params (read by hint_selector.HintSelector.from_env) ----
# Exported here so the training head forwards them into the rollout runtime env.
export SELECTOR_API_MODE=openai
export SELECTOR_OPENAI_BASE_URL=${SELECTOR_OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export SELECTOR_MODEL=${SELECTOR_MODEL:-"gpt-5-mini"}
export SELECTOR_REASONING_EFFORT=${SELECTOR_REASONING_EFFORT:-low}
case "${SELECTOR_REASONING_EFFORT}" in
    minimal | low | medium | high) ;;
    *)
        echo "[launch] FATAL: invalid SELECTOR_REASONING_EFFORT=${SELECTOR_REASONING_EFFORT@Q}." >&2
        echo "         Expected one of: minimal, low, medium, high." >&2
        exit 1
        ;;
esac
# temperature/top_p only reach CHAT models (gpt-4.1-* etc.); reasoning models
# ignore them. 0.3/0.95 = the offline eval's settings.
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.3}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-0.95}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}
# The API answers in seconds, not minutes -- 300s already covers a pathological
# tail; 5 retries (with the client's exponential backoff) rides out 429 bursts.
export SELECTOR_REQUEST_TIMEOUT_S=${SELECTOR_REQUEST_TIMEOUT_S:-300}
export SELECTOR_MAX_RETRIES=${SELECTOR_MAX_RETRIES:-5}
# In-flight cap PER agent-loop worker process (rate-limit hygiene). Effective
# cluster-wide concurrency ~= this x #agent-loop workers; raise it if rollout
# stalls on hint calls and the org's RPM/TPM headroom allows.
export SELECTOR_MAX_CONCURRENCY=${SELECTOR_MAX_CONCURRENCY:-16}
# This launcher uses the progress-aware selector prompt, so default its matching
# rollout-message mode on. The run scripts now default it on as well; the export
# here is kept so an explicit =false override still flows through unchanged.
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}

# --- pre-launch cleanup toggle ----------------------------------------------
# Reap orphaned vLLM ROLLOUT workers left by a SIGKILLed prior run before this
# pod binds its ports (same EADDRINUSE story as launch_hprl_cluster.sh; there
# is no selector api_server here, so only the rollout patterns are reaped).
HPRL_SKIP_PRELAUNCH_CLEAN=${HPRL_SKIP_PRELAUNCH_CLEAN:-0}

# --- dedicated selector-check log (one file per pod, same layout as the vLLM
# launcher: logs/selector_check/<stamp>_<label>/rank<N>.log, Pacific time, one
# shared stamp per launch claimed via atomic mkdir) -----------------------------
PST_TZ=${PST_TZ:-"America/Los_Angeles"}
pst_now() { TZ="${PST_TZ}" date "$@"; }

SELECTOR_CHECK_ROOT=${SELECTOR_CHECK_ROOT:-"${HINT_RL_HOME}/logs/selector_check"}
SELECTOR_STAMP_TTL=${SELECTOR_STAMP_TTL:-120}
_job_key="${MASTER_ADDR:-x}_${MASTER_PORT:-30000}"
_dir_label="${SELECTOR_EXP_TAG:-${_job_key}}"
_stamp_dir="${SELECTOR_CHECK_ROOT}/.stamp.${_job_key}"
mkdir -p "${SELECTOR_CHECK_ROOT}"
if [ -f "${_stamp_dir}/ts" ]; then
    _age=$(( $(date +%s) - $(stat -c %Y "${_stamp_dir}/ts" 2>/dev/null || echo 0) ))
    if [ "${_age}" -gt "${SELECTOR_STAMP_TTL}" ]; then
        mv "${_stamp_dir}" "${_stamp_dir}.old.$(date +%s).$$" 2>/dev/null || true
    fi
fi
if mkdir "${_stamp_dir}" 2>/dev/null; then            # atomic on shared FS -> exactly one winner
    pst_now '+%Y%m%d-%H%M%S_%Z' > "${_stamp_dir}/ts"
fi
JOB_STAMP=""
for _ in $(seq 1 40); do                              # wait <=20s for the winner's stamp
    [ -s "${_stamp_dir}/ts" ] && { JOB_STAMP="$(cat "${_stamp_dir}/ts")"; break; }
    sleep 0.5
done
[ -z "${JOB_STAMP}" ] && JOB_STAMP="$(pst_now '+%Y%m%d-%H%M%S_%Z')"   # fallback: own stamp
SELECTOR_CHECK_DIR=${SELECTOR_CHECK_DIR:-"${SELECTOR_CHECK_ROOT}/${JOB_STAMP}_${_dir_label}"}
mkdir -p "${SELECTOR_CHECK_DIR}"
SELECTOR_CHECK_LOG=${SELECTOR_CHECK_LOG:-"${SELECTOR_CHECK_DIR}/rank${RANK}.log"}
: > "${SELECTOR_CHECK_LOG}"   # truncate at launch
slog() {
    local msg="$*"
    echo "${msg}"
    echo "[$(pst_now '+%Y-%m-%d %H:%M:%S %Z')] [rank ${RANK}/$(hostname)] ${msg}" >> "${SELECTOR_CHECK_LOG}"
}

slog "[launch] rank=${RANK}/${WORLD_SIZE}  selector=OPENAI API (no selector pods)  train_nnodes=${TRAIN_NNODES}  check_log=${SELECTOR_CHECK_LOG}"
slog "[launch]   model    : ${SELECTOR_MODEL}  (effort=${SELECTOR_REASONING_EFFORT} max_concurrency=${SELECTOR_MAX_CONCURRENCY}/worker progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE})"
slog "[launch]   endpoint : ${SELECTOR_OPENAI_BASE_URL}  (key=...${OPENAI_API_KEY: -4}${HTTPS_PROXY:+  proxy=${HTTPS_PROXY}})"
slog "[launch]   resume   : ${RESUME_EXP_NAME:-<fresh run>}"

hprl_reap_stale_vllm() {
    [ "${HPRL_SKIP_PRELAUNCH_CLEAN:-0}" = "1" ] && return 0
    local p
    for p in "$@"; do
        if pkill -9 -f "${p}" 2>/dev/null; then
            slog "[launch]   reaped stale procs matching: ${p}"
        fi
    done
    sleep 2   # let the kernel release the freed ports before we bind
}

# --- fail-fast API reachability check ---------------------------------------
# Same rationale as the selector-endpoint probe in launch_hprl_cluster.sh: a
# pod with no API egress (or a bad key) degrades EVERY hint call to a no-op for
# the whole run -- invisibly, because HintSelector falls back gracefully. Probe
# /models from THIS pod now so it surfaces at launch. A remote API is up or it
# isn't -- no model-load wait, so far fewer retries than the vLLM probe. Set
# SELECTOR_REQUIRE_REACHABLE=0 to downgrade the abort to a warning.
SELECTOR_REQUIRE_REACHABLE=${SELECTOR_REQUIRE_REACHABLE:-1}
SELECTOR_PROBE_RETRIES=${SELECTOR_PROBE_RETRIES:-3}
_probe_url="${SELECTOR_OPENAI_BASE_URL%/}/models"
if ! command -v curl >/dev/null 2>&1; then
    slog "[launch] WARN: curl not found; skipping the OpenAI API reachability probe."
else
    ok=0
    for _ in $(seq 1 "${SELECTOR_PROBE_RETRIES}"); do
        if curl -fsS --max-time 10 -H "Authorization: Bearer ${OPENAI_API_KEY}" \
                "${_probe_url}" >/dev/null 2>&1; then ok=1; break; fi
        sleep 5
    done
    if [ "${ok}" = "1" ]; then
        slog "[launch] selector check passed: ${_probe_url} reachable from this pod."
    else
        slog "[launch] FATAL: ${_probe_url} unreachable from this pod (curl failed)."
        slog "         Hints would silently degrade to no-ops for the entire run."
        slog "         Check API egress from the training fabric (set OPENAI_PROXY if it"
        slog "         needs a proxy) and that OPENAI_API_KEY is valid."
        if [ "${SELECTOR_REQUIRE_REACHABLE}" = "1" ]; then exit 1; fi
        slog "[launch] SELECTOR_REQUIRE_REACHABLE=0 -> continuing despite unreachable API."
    fi
fi

# Free rollout-vLLM ports from a SIGKILLed prior run before `ray start` brings
# this node up (rollout-worker patterns only; no selector api_server exists here).
hprl_reap_stale_vllm EngineCore WorkerProc multiproc_executor vLLMHttpServer \
    "vllm.v1.engine" "vllm.v1.executor"

# ALL pods join the Ray cluster; ray_cluster_launch.sh auto-detects the head
# (MASTER_ADDR), waits for NNODES, then runs TRAIN_SCRIPT (which `ray job
# submit`s the job and forwards the SELECTOR_*/OPENAI_API_KEY env into the
# rollout runtime env).
export NNODES=${TRAIN_NNODES}
export TRAIN_SCRIPT
exec "${RAY_LAUNCH}"
