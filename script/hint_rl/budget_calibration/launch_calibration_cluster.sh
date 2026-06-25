#!/usr/bin/env bash
# =============================================================================
# launch_calibration_cluster.sh -- one entrypoint for the OFFLINE budget
# calibration (calibrate_budget.py), modeled on launch_hprl_cluster.sh.
#
# Run this on EVERY pod. It splits the cluster by node rank into:
#
#   * SELECTOR nodes (the last SELECTOR_NNODES pods, default 4): each serves
#     ${SELECTOR_MODEL_PATH} (gpt-oss-20b) via vLLM (single-node DP=GPUs, TP=1) and
#     publishes its fabric endpoint to the selector rendezvous. IDENTICAL to the
#     selector node in launch_hprl_cluster.sh.
#
#   * POLICY nodes (the first POLICY_NNODES pods, default 4): each serves
#     ${POLICY_MODEL_PATH} (Qwen2.5-7B-Instruct) via vLLM and publishes its fabric
#     endpoint to the policy rendezvous.
#       - rank 0 is ALSO the DRIVER: it serves Qwen in the BACKGROUND, collects ALL
#         policy + selector endpoints, probes them, then runs calibrate_budget.py
#         LOAD-BALANCED across every policy + selector endpoint (the Endpoint client
#         round-robins comma-separated URLs). On exit it stops its background server.
#       - ranks 1..POLICY_NNODES-1 serve Qwen in the FOREGROUND (publish, then block).
#
# So with the default 4+4 split: 4 nodes (8x H100 each) serve gpt-oss-20b, 4 nodes
# serve Qwen, and rank 0 drives the calibration across all 8 endpoints. Result: a
# budget-state JSON (default budget_state_calibrated.json) usable as the init budget.
# calibrate_budget.py mid-saves the aggregate every --save-every problems and is
# resumable (per-problem cache), so a relaunch continues where it left off.
#
# Required env (PyTorchJob): MASTER_ADDR, MASTER_PORT, WORLD_SIZE, per-node RANK.
# Key overrides: SELECTOR_NNODES (4) / POLICY_NNODES (rest), SELECTOR_HOST /
#   POLICY_HOST (comma-separated hosts -> skip rendezvous), POLICY_* / SELECTOR_* / CAL_*.
# =============================================================================
set -euo pipefail

# --- paths (mirrors launch_hprl_cluster.sh) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# this launcher lives in script/hint_rl/budget_calibration/, so the project root
# (HINT_RL_HOME, which holds dataset/ and logs/) is three levels up, not two.
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

# --- conda env (verl holds vllm + calibrate_budget deps) -------------------
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- cluster topology (default 4 policy + 4 selector = 8 nodes) -------------
# SELECTOR_NNODES selector pods + POLICY_NNODES policy pods. WORLD_SIZE (injected)
# must equal the sum. The driver is policy rank 0; >=1 policy + >=1 selector required.
SELECTOR_NNODES=${SELECTOR_NNODES:-4}
if [ -z "${POLICY_NNODES:-}" ]; then
    POLICY_NNODES=$(( ${WORLD_SIZE:-8} - SELECTOR_NNODES ))
fi
WORLD_SIZE=${WORLD_SIZE:-$((POLICY_NNODES + SELECTOR_NNODES))}
if [ "$((POLICY_NNODES + SELECTOR_NNODES))" -ne "${WORLD_SIZE}" ]; then
    echo "[launch] FATAL: POLICY_NNODES(${POLICY_NNODES}) + SELECTOR_NNODES(${SELECTOR_NNODES}) != WORLD_SIZE(${WORLD_SIZE})." >&2
    exit 1
fi
if [ "${POLICY_NNODES}" -lt 1 ] || [ "${SELECTOR_NNODES}" -lt 1 ]; then
    echo "[launch] FATAL: need >= 1 policy pod and >= 1 selector pod (got policy=${POLICY_NNODES} selector=${SELECTOR_NNODES})." >&2
    exit 1
fi
RANK=${RANK:-${NODE_RANK:-${GROUP_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-}}}}}
if [ -z "${RANK}" ]; then
    echo "[launch] FATAL: node RANK is unset. Set RANK (or NODE_RANK), or set SELECTOR_HOST/POLICY_HOST." >&2
    exit 1
fi
# Policy occupies ranks [0..POLICY_NNODES-1] (rank 0 = driver); selector the last ones.
SELECTOR_FIRST_RANK=${SELECTOR_FIRST_RANK:-$((WORLD_SIZE - SELECTOR_NNODES))}
IS_SELECTOR_NODE=0; [ "${RANK}" -ge "${SELECTOR_FIRST_RANK}" ] && IS_SELECTOR_NODE=1
IS_DRIVER=0; [ "${RANK}" -eq 0 ] && IS_DRIVER=1

# --- selector SERVING config (one independent server per pod) --------------
SELECTOR_MODEL_PATH=${SELECTOR_MODEL_PATH:-"${BASE_HOME}/model/gpt-oss-20b"}
SELECTOR_SERVED_NAME=${SELECTOR_SERVED_NAME:-"gpt-oss-20b"}
SELECTOR_PORT=${SELECTOR_PORT:-30000}
SELECTOR_GPUS_PER_NODE=${SELECTOR_GPUS_PER_NODE:-8}
SELECTOR_TP=${SELECTOR_TP:-1}
SELECTOR_DP=${SELECTOR_DP:-$(( SELECTOR_GPUS_PER_NODE / SELECTOR_TP ))}
SELECTOR_MEM_FRAC=${SELECTOR_MEM_FRAC:-0.9}
SELECTOR_CONTEXT_LEN=${SELECTOR_CONTEXT_LEN:-40960}
SELECTOR_MAX_NUM_SEQS=${SELECTOR_MAX_NUM_SEQS:-512}
SELECTOR_REASONING_PARSER=${SELECTOR_REASONING_PARSER:-openai_gptoss}
export SELECTOR_MODEL=${SELECTOR_MODEL:-"${SELECTOR_SERVED_NAME}"}
export SELECTOR_API_KEY=${SELECTOR_API_KEY:-"EMPTY"}
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.7}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-1.0}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}

# --- policy SERVING config (the model being calibrated) --------------------
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
POLICY_SERVED_NAME=${POLICY_SERVED_NAME:-"Qwen2.5-7B-Instruct"}
POLICY_PORT=${POLICY_PORT:-8000}
POLICY_GPUS_PER_NODE=${POLICY_GPUS_PER_NODE:-8}
POLICY_TP=${POLICY_TP:-1}
POLICY_DP=${POLICY_DP:-$(( POLICY_GPUS_PER_NODE / POLICY_TP ))}
POLICY_MEM_FRAC=${POLICY_MEM_FRAC:-0.9}
POLICY_CONTEXT_LEN=${POLICY_CONTEXT_LEN:-32768}
POLICY_MAX_NUM_SEQS=${POLICY_MAX_NUM_SEQS:-512}
export POLICY_TEMPERATURE=${POLICY_TEMPERATURE:-1.0}
export POLICY_TOP_P=${POLICY_TOP_P:-1.0}
export POLICY_MAX_TOKENS=${POLICY_MAX_TOKENS:-4096}

# --- calibration params ----------------------------------------------------
CAL_DATA=${CAL_DATA:-"${HINT_RL_HOME}/dataset/dapo-3139-auto-hint.parquet"}
CAL_OUT=${CAL_OUT:-"${SCRIPT_DIR}/budget_state_calibrated.json"}
CAL_BUDGET=${CAL_BUDGET:-10}
CAL_N=${CAL_N:-32}
CAL_RANK=${CAL_RANK:-4}
# Concurrency (async): PROBLEMS processed at once x ROLLOUTS in flight per problem.
# Total concurrent rollouts = product (default 8 x 32 = 256, to feed 4+4 servers). On the
# 4th correct, a problem ABORTS its in-flight rollouts (true early stop). Lower rollout-workers
# to cut abort waste; raise problem-workers for more concurrency.
CAL_PROBLEM_WORKERS=${CAL_PROBLEM_WORKERS:-8}
CAL_ROLLOUT_WORKERS=${CAL_ROLLOUT_WORKERS:-32}
CAL_SAVE_EVERY=${CAL_SAVE_EVERY:-50}
CAL_LIMIT=${CAL_LIMIT:-}
CAL_CLAMP_MAX=${CAL_CLAMP_MAX:-}
# Early-stop a problem once it has this many CORRECT rollouts (empty -> calibrate_budget
# default = CAL_RANK; set 0 to disable and always run all CAL_N). Big speedup on easy problems.
CAL_STOP_AT_CORRECT=${CAL_STOP_AT_CORRECT:-}
# Seed already-done problems from an external budget JSON (e.g. budget_state_done_so_far.json):
# those pids are skipped. The per-problem cache (<CAL_OUT>.cache) ALSO auto-resumes regardless.
CAL_RESUME_FROM=${CAL_RESUME_FROM:-}
# Re-derive already-done problems from a rollouts DUMP dir (<CAL_OUT>.rollouts): re-aggregates
# each problem's dumped rollouts and skips them. Use to recover done problems from transcripts
# after clearing the cache (e.g. for a uniform early-stop run).
CAL_RESUME_FROM_ROLLOUTS=${CAL_RESUME_FROM_ROLLOUTS:-}
# Dump EVERY rollout's full transcript (one <pid>.jsonl per problem: outcome + the whole
# conversation incl. injected hints). ON by default -> <CAL_OUT>.rollouts; set
# CAL_DUMP_ROLLOUTS="" to disable. LARGE on the full set (CAL_N x problems multi-turn rollouts).
CAL_DUMP_ROLLOUTS=${CAL_DUMP_ROLLOUTS:-"${CAL_OUT}.rollouts"}

HPRL_SKIP_PRELAUNCH_CLEAN=${HPRL_SKIP_PRELAUNCH_CLEAN:-0}

# --- per-launch logs + endpoint rendezvous (shared-FS) ---------------------
PST_TZ=${PST_TZ:-"America/Los_Angeles"}
pst_now() { TZ="${PST_TZ}" date "$@"; }
LOG_ROOT=${LOG_ROOT:-"${HINT_RL_HOME}/logs/calibration"}
_job_key="${MASTER_ADDR:-x}_${MASTER_PORT:-30000}"
_stamp_dir="${LOG_ROOT}/.stamp.${_job_key}"
mkdir -p "${LOG_ROOT}"
# Rotate a STALE stamp from a previous launch so each relaunch gets a FRESH run_dir +
# rendezvous dir (else a relaunch reuses the old endpoint files -> stale-endpoint race).
# Pods of the SAME launch start < TTL apart and share the stamp.
CAL_STAMP_TTL=${CAL_STAMP_TTL:-120}
if [ -f "${_stamp_dir}/ts" ]; then
    _age=$(( $(date +%s) - $(stat -c %Y "${_stamp_dir}/ts" 2>/dev/null || echo 0) ))
    [ "${_age}" -gt "${CAL_STAMP_TTL}" ] && mv "${_stamp_dir}" "${_stamp_dir}.old.$(date +%s).$$" 2>/dev/null || true
fi
if mkdir "${_stamp_dir}" 2>/dev/null; then pst_now '+%Y%m%d-%H%M%S' > "${_stamp_dir}/ts"; fi
JOB_STAMP=""
for _ in $(seq 1 40); do
    [ -s "${_stamp_dir}/ts" ] && { JOB_STAMP="$(cat "${_stamp_dir}/ts")"; break; }
    sleep 0.5
done
[ -z "${JOB_STAMP}" ] && JOB_STAMP="$(pst_now '+%Y%m%d-%H%M%S')"
RUN_DIR=${RUN_DIR:-"${LOG_ROOT}/${JOB_STAMP}_${_job_key}"}
mkdir -p "${RUN_DIR}"
RDV_SELECTOR=${SELECTOR_ENDPOINT_DIR:-"${LOG_ROOT}/.selector_endpoints.${_job_key}.${JOB_STAMP}"}
RDV_POLICY=${POLICY_ENDPOINT_DIR:-"${LOG_ROOT}/.policy_endpoints.${_job_key}.${JOB_STAMP}"}
mkdir -p "${RDV_SELECTOR}" "${RDV_POLICY}"
CHECK_LOG="${RUN_DIR}/rank${RANK}.log"
: > "${CHECK_LOG}"
slog() { echo "$*"; echo "[$(pst_now '+%F %T %Z')] [rank ${RANK}/$(hostname)] $*" >> "${CHECK_LOG}"; }
slog "[launch] rank=${RANK}/${WORLD_SIZE}  policy=[0..$((SELECTOR_FIRST_RANK-1))] (driver=0)  selector=[${SELECTOR_FIRST_RANK}..$((WORLD_SIZE-1))]  run_dir=${RUN_DIR}"

hprl_reap_stale_vllm() {
    [ "${HPRL_SKIP_PRELAUNCH_CLEAN:-0}" = "1" ] && return 0
    local p
    for p in "$@"; do
        if pkill -9 -f "${p}" 2>/dev/null; then slog "[launch]   reaped stale procs matching: ${p}"; fi
    done
    sleep 2
}

# Resolve the routable fabric address to advertise (not the default-route NIC).
resolve_fabric_ip() {
    local pref="${1}" ip self=""
    [ -n "${pref}" ] && for ip in $(hostname -I 2>/dev/null); do
        case "${ip}" in "${pref}"*) self="${ip}"; break ;; esac
    done
    [ -z "${self}" ] && self="$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)"
    [ -z "${self}" ] && self="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "${self:-127.0.0.1}"
}
FABRIC_PREFIX=${SELECTOR_FABRIC_PREFIX-"10.20."}

# Collect endpoint files for a rank range from a rendezvous dir into a bash array.
collect_endpoints() {  # <rdv_dir> <first_rank> <last_rank> ; sets COLLECTED=()
    local rdv="$1" first="$2" last="$3" R f
    COLLECTED=()
    for R in $(seq "${first}" "${last}"); do
        f="${rdv}/${R}"
        for _i in $(seq 1 240); do [ -s "${f}" ] && break; sleep 5; done   # up to ~20 min
        [ -s "${f}" ] || { slog "[launch] FATAL: endpoint for rank ${R} never published (${f})"; exit 1; }
        COLLECTED+=("$(cat "${f}")")
    done
}

# Probe an OpenAI /v1/models endpoint until it answers (vLLM may still be loading).
probe_endpoint() {  # <url> <label>
    local u="$1" label="$2" i
    for i in $(seq 1 120); do   # up to ~10 min
        if curl -fsS --max-time 5 -H "Authorization: Bearer EMPTY" "${u}/models" >/dev/null 2>&1; then
            slog "[launch]   ${label} OK: ${u}"; return 0
        fi
        sleep 5
    done
    slog "[launch] FATAL: ${label} unreachable after wait: ${u}"; return 1
}

if [ "${IS_SELECTOR_NODE}" = "1" ]; then
    # =================== SELECTOR NODE (gpt-oss-20b) =========================
    SELF_IP="${SELECTOR_ADVERTISE_IP:-$(resolve_fabric_ip "${FABRIC_PREFIX}")}"
    SELECTOR_URL="http://${SELF_IP}:${SELECTOR_PORT}/v1"
    echo "${SELECTOR_URL}" > "${RDV_SELECTOR}/${RANK}"   # publish early
    slog "[launch] SELECTOR node: serving ${SELECTOR_SERVED_NAME} (dp=${SELECTOR_DP} tp=${SELECTOR_TP}) -> ${SELECTOR_URL}"
    slog "[launch]   model: ${SELECTOR_MODEL_PATH}  all_ips: $(hostname -I 2>/dev/null)"

    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    REASONING_ARGS=()
    [ -n "${SELECTOR_REASONING_PARSER}" ] && REASONING_ARGS=(--reasoning-parser "${SELECTOR_REASONING_PARSER}")
    hprl_reap_stale_vllm "vllm.entrypoints.openai.api_server"
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${SELECTOR_MODEL_PATH}" \
        --served-model-name "${SELECTOR_SERVED_NAME}" \
        --host 0.0.0.0 --port "${SELECTOR_PORT}" \
        --data-parallel-size "${SELECTOR_DP}" \
        --tensor-parallel-size "${SELECTOR_TP}" \
        --gpu-memory-utilization "${SELECTOR_MEM_FRAC}" \
        --max-model-len "${SELECTOR_CONTEXT_LEN}" \
        --max-num-seqs "${SELECTOR_MAX_NUM_SEQS}" \
        "${REASONING_ARGS[@]}" \
        --trust-remote-code
fi

# =================== POLICY NODE (ranks 0..POLICY_NNODES-1) ==================
export VLLM_USE_V1=${VLLM_USE_V1:-1}
SELF_IP="${POLICY_ADVERTISE_IP:-$(resolve_fabric_ip "${FABRIC_PREFIX}")}"
echo "http://${SELF_IP}:${POLICY_PORT}/v1" > "${RDV_POLICY}/${RANK}"   # publish early
POLICY_SERVE_LOG="${RUN_DIR}/policy_serve.rank${RANK}.log"
POLICY_VLLM_CMD=(python -m vllm.entrypoints.openai.api_server
    --model "${POLICY_MODEL_PATH}"
    --served-model-name "${POLICY_SERVED_NAME}"
    --host 0.0.0.0 --port "${POLICY_PORT}"
    --data-parallel-size "${POLICY_DP}"
    --tensor-parallel-size "${POLICY_TP}"
    --gpu-memory-utilization "${POLICY_MEM_FRAC}"
    --max-model-len "${POLICY_CONTEXT_LEN}"
    --max-num-seqs "${POLICY_MAX_NUM_SEQS}"
    --trust-remote-code)
hprl_reap_stale_vllm "vllm.entrypoints.openai.api_server" EngineCore WorkerProc

if [ "${IS_DRIVER}" != "1" ]; then
    # ---- POLICY SERVE-ONLY (ranks 1..POLICY_NNODES-1): publish + serve (block) ----
    slog "[launch] POLICY node: serving ${POLICY_SERVED_NAME} (dp=${POLICY_DP} tp=${POLICY_TP}) -> http://${SELF_IP}:${POLICY_PORT}/v1"
    exec "${POLICY_VLLM_CMD[@]}" > "${POLICY_SERVE_LOG}" 2>&1
fi

# =================== DRIVER (policy rank 0): serve Qwen bg + calibrate =======
slog "[launch] DRIVER node: serving ${POLICY_SERVED_NAME} in background (log ${POLICY_SERVE_LOG})"
"${POLICY_VLLM_CMD[@]}" > "${POLICY_SERVE_LOG}" 2>&1 &
POLICY_PID=$!
cleanup() { slog "[launch] stopping local policy server (pid ${POLICY_PID})"; kill "${POLICY_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Collect every policy endpoint (own = local 127.0.0.1, others via fabric) + selectors.
policy_urls=("http://127.0.0.1:${POLICY_PORT}/v1")
if [ -n "${POLICY_HOST:-}" ]; then
    IFS=',' read -ra hosts <<< "${POLICY_HOST}"; for h in "${hosts[@]}"; do policy_urls+=("http://${h}:${POLICY_PORT}/v1"); done
elif [ "${POLICY_NNODES}" -gt 1 ]; then
    slog "[launch] driver: collecting $((POLICY_NNODES - 1)) peer policy endpoint(s) from ${RDV_POLICY} ..."
    collect_endpoints "${RDV_POLICY}" 1 $((POLICY_NNODES - 1)); policy_urls+=("${COLLECTED[@]}")
fi
if [ -n "${SELECTOR_HOST:-}" ]; then
    selector_urls=(); IFS=',' read -ra hosts <<< "${SELECTOR_HOST}"; for h in "${hosts[@]}"; do selector_urls+=("http://${h}:${SELECTOR_PORT}/v1"); done
else
    slog "[launch] driver: collecting ${SELECTOR_NNODES} selector endpoint(s) from ${RDV_SELECTOR} ..."
    collect_endpoints "${RDV_SELECTOR}" "${SELECTOR_FIRST_RANK}" $((WORLD_SIZE - 1)); selector_urls=("${COLLECTED[@]}")
fi
POLICY_BASE_URLS="$(IFS=,; echo "${policy_urls[*]}")"
SELECTOR_BASE_URLS="$(IFS=,; echo "${selector_urls[*]}")"
export SELECTOR_BASE_URLS SELECTOR_BASE_URL="${selector_urls[0]}"
slog "[launch] driver: POLICY_BASE_URLS=${POLICY_BASE_URLS}"
slog "[launch] driver: SELECTOR_BASE_URLS=${SELECTOR_BASE_URLS}"

# Wait for ALL endpoints' /v1/models (local policy + peers + selectors).
probe_endpoint "http://127.0.0.1:${POLICY_PORT}/v1" "policy(local)" || exit 1
for u in "${policy_urls[@]:1}"; do probe_endpoint "${u}" "policy" || exit 1; done
for u in "${selector_urls[@]}"; do probe_endpoint "${u}" "selector" || exit 1; done

# Run the calibration (load-balanced across all policy + selector endpoints).
CAL_LOG="${RUN_DIR}/calibrate.log"
slog "[launch] driver: starting calibration -> ${CAL_OUT}  (log ${CAL_LOG})"
slog "[launch]   probe: budget ${CAL_BUDGET}, n ${CAL_N}, rank ${CAL_RANK}, problem_workers ${CAL_PROBLEM_WORKERS} x rollout_workers ${CAL_ROLLOUT_WORKERS}, save_every ${CAL_SAVE_EVERY}${CAL_LIMIT:+, limit ${CAL_LIMIT}}"
cd "${SCRIPT_DIR}"
set +e
python calibrate_budget.py \
    --data "${CAL_DATA}" \
    --out "${CAL_OUT}" \
    --budget "${CAL_BUDGET}" --n "${CAL_N}" --rank "${CAL_RANK}" \
    --problem-workers "${CAL_PROBLEM_WORKERS}" --rollout-workers "${CAL_ROLLOUT_WORKERS}" \
    --save-every "${CAL_SAVE_EVERY}" \
    ${CAL_LIMIT:+--limit "${CAL_LIMIT}"} \
    ${CAL_CLAMP_MAX:+--clamp-max "${CAL_CLAMP_MAX}"} \
    ${CAL_RESUME_FROM:+--resume-from "${CAL_RESUME_FROM}"} \
    ${CAL_RESUME_FROM_ROLLOUTS:+--resume-from-rollouts "${CAL_RESUME_FROM_ROLLOUTS}"} \
    ${CAL_DUMP_ROLLOUTS:+--dump-rollouts "${CAL_DUMP_ROLLOUTS}"} \
    ${CAL_STOP_AT_CORRECT:+--stop-at-correct "${CAL_STOP_AT_CORRECT}"} \
    --policy-base-url "${POLICY_BASE_URLS}" \
    --policy-model "${POLICY_SERVED_NAME}" \
    --policy-temperature "${POLICY_TEMPERATURE}" \
    --policy-top-p "${POLICY_TOP_P}" \
    --policy-max-tokens "${POLICY_MAX_TOKENS}" \
    --selector-base-urls "${SELECTOR_BASE_URLS}" \
    --selector-model "${SELECTOR_MODEL}" \
    --selector-temperature "${SELECTOR_TEMPERATURE}" \
    --selector-top-p "${SELECTOR_TOP_P}" \
    --selector-max-tokens "${SELECTOR_MAX_TOKENS}" \
    2>&1 | tee "${CAL_LOG}"
rc=${PIPESTATUS[0]}
set -e
slog "[launch] driver: calibration finished (rc=${rc}) -> ${CAL_OUT}"
exit "${rc}"
