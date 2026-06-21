#!/usr/bin/env bash
# =============================================================================
# launch_hprl_cluster.sh -- one entrypoint for the whole HPRL job.
#
# Run this on EVERY pod (head + workers), exactly like ray_cluster_launch.sh.
# It splits the cluster by node rank:
#
#   * SELECTOR nodes (the last SELECTOR_NNODES pods): each serves
#     ${SELECTOR_MODEL_PATH} (default /share5/users/xutao.ma/model/gpt-oss-20b)
#     via vLLM as a frozen hint selector -- an OpenAI-compatible /v1 endpoint.
#     gpt-oss-20b is a small MoE, so each pod runs an INDEPENDENT single-node
#     data-parallel server (DP = GPUs/pod, TP=1) -- NO cross-node DP coordination
#     (vLLM's multi-node DP handshake is fragile across pods). The training side
#     load-balances client-side across all selector endpoints with failover
#     (hint_selector.HintSelector over SELECTOR_BASE_URLS), so there is no proxy
#     single-point-of-failure and a slow/down selector pod is simply skipped.
#
#   * TRAINING nodes (the remaining pods): form the Ray cluster and run
#     run_hprl_qwen2.5_7b.sh via ray_cluster_launch.sh. The Ray head submits the
#     job with the selector endpoints + all selector call params forwarded into
#     the rollout workers' runtime env.
#
# Each selector pod publishes its own endpoint to a per-job shared-FS rendezvous
# DIRECTORY (one file per selector rank); the training pods collect them into the
# comma-separated SELECTOR_BASE_URLS. (Set SELECTOR_HOST to a comma-separated host
# list to skip the rendezvous.) Endpoints are published early -- before the model
# loads -- so training proceeds in parallel; HintSelector retries/fails over and
# degrades gracefully (unaided) if a call still fails.
#
# Required env (injected by the PyTorchJob): MASTER_ADDR, MASTER_PORT,
#   WORLD_SIZE, and a per-node RANK (RANK / NODE_RANK / GROUP_RANK / *_COMM_WORLD_RANK).
# Key overrides:
#   TRAIN_NNODES      # of training pods (default: WORLD_SIZE - SELECTOR_NNODES).
#   SELECTOR_NNODES   # of pods serving the selector (default: 2). Set BOTH to
#                     pick any split (e.g. TRAIN_NNODES=2 SELECTOR_NNODES=1);
#                     WORLD_SIZE must equal the sum (validated at launch).
#   SELECTOR_HOST     comma-separated selector host/IPs (skips rendezvous).
#   SELECTOR_*        serving + call params (see below).
# =============================================================================
set -euo pipefail

# --- paths (mirrors run_hprl_qwen2.5_7b.sh) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # script/hint_rl
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}
RAY_LAUNCH=${RAY_LAUNCH:-"${SCRIPT_DIR}/../ray_cluster_launch.sh"}    # script/ray_cluster_launch.sh
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_hprl_qwen2.5_7b.sh"}

# --- conda env (verl holds BOTH training and the vllm used to serve gpt-oss) -
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- cluster topology ------------------------------------------------------
# Specify the split EXPLICITLY: TRAIN_NNODES training pods + SELECTOR_NNODES
# selector pods (any combination, e.g. 4+2, 2+1, 1+1). WORLD_SIZE (injected by
# the PyTorchJob) must equal their sum -- mismatches abort so a job-spec replica
# count that disagrees with the requested split fails at launch, not mid-run.
# If TRAIN_NNODES is unset it is derived as WORLD_SIZE - SELECTOR_NNODES (the
# original behavior); if WORLD_SIZE is unset it is derived as the sum.
SELECTOR_NNODES=${SELECTOR_NNODES:-2}
if [ -z "${TRAIN_NNODES:-}" ]; then
    TRAIN_NNODES=$(( ${WORLD_SIZE:-6} - SELECTOR_NNODES ))
fi
WORLD_SIZE=${WORLD_SIZE:-$((TRAIN_NNODES + SELECTOR_NNODES))}
if [ "$((TRAIN_NNODES + SELECTOR_NNODES))" -ne "${WORLD_SIZE}" ]; then
    echo "[launch] FATAL: TRAIN_NNODES(${TRAIN_NNODES}) + SELECTOR_NNODES(${SELECTOR_NNODES}) != WORLD_SIZE(${WORLD_SIZE})." >&2
    echo "         Set the job's replica count to the sum, or fix the split." >&2
    exit 1
fi
if [ "${TRAIN_NNODES}" -lt 1 ] || [ "${SELECTOR_NNODES}" -lt 1 ]; then
    echo "[launch] FATAL: need >= 1 training pod and >= 1 selector pod (got train=${TRAIN_NNODES} selector=${SELECTOR_NNODES})." >&2
    exit 1
fi
# node rank, from whichever the launcher set.
RANK=${RANK:-${NODE_RANK:-${GROUP_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-}}}}}
if [ -z "${RANK}" ]; then
    echo "[launch] FATAL: node RANK is unset. Set RANK (or NODE_RANK), or set" >&2
    echo "         SELECTOR_HOST and run the selector pods at the last SELECTOR_NNODES ranks." >&2
    exit 1
fi
# The selector occupies the LAST SELECTOR_NNODES ranks (all symmetric, independent
# servers); the first selector rank must be != 0 (rank 0 = MASTER_ADDR = Ray head).
SELECTOR_FIRST_RANK=${SELECTOR_FIRST_RANK:-$((WORLD_SIZE - SELECTOR_NNODES))}
IS_SELECTOR_NODE=0; [ "${RANK}" -ge "${SELECTOR_FIRST_RANK}" ] && IS_SELECTOR_NODE=1

# --- selector SERVING config (one INDEPENDENT single-node server per pod) ---
SELECTOR_MODEL_PATH=${SELECTOR_MODEL_PATH:-"${BASE_HOME}/model/gpt-oss-20b"}
SELECTOR_SERVED_NAME=${SELECTOR_SERVED_NAME:-"gpt-oss-20b"}
SELECTOR_PORT=${SELECTOR_PORT:-30000}
SELECTOR_GPUS_PER_NODE=${SELECTOR_GPUS_PER_NODE:-8}   # H100s per selector pod
SELECTOR_TP=${SELECTOR_TP:-1}
# Per-node data-parallel replicas (all local to this pod -> no cross-node DP).
SELECTOR_DP=${SELECTOR_DP:-$(( SELECTOR_GPUS_PER_NODE / SELECTOR_TP ))}
SELECTOR_MEM_FRAC=${SELECTOR_MEM_FRAC:-0.9}
SELECTOR_CONTEXT_LEN=${SELECTOR_CONTEXT_LEN:-40960}
SELECTOR_MAX_NUM_SEQS=${SELECTOR_MAX_NUM_SEQS:-512}      # gpt-oss is pure-attention; raise for throughput
SELECTOR_REASONING_PARSER=${SELECTOR_REASONING_PARSER:-openai_gptoss}  # vllm's gpt-oss CoT parser

# --- selector CALL params (read by hint_selector.HintSelector.from_env) ----
# Exported here so the training head forwards them into the rollout runtime env.
export SELECTOR_MODEL=${SELECTOR_MODEL:-"${SELECTOR_SERVED_NAME}"}
export SELECTOR_API_KEY=${SELECTOR_API_KEY:-"EMPTY"}
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.7}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-1.0}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}
export SELECTOR_REQUEST_TIMEOUT_S=${SELECTOR_REQUEST_TIMEOUT_S:-600}
export SELECTOR_MAX_RETRIES=${SELECTOR_MAX_RETRIES:-3}

# --- pre-launch cleanup toggle -------------------------------------------------
# hprl_reap_stale_vllm() (below) pkills orphaned vLLM workers left by a SIGKILLed
# prior run before this pod binds its ports, preventing the EADDRINUSE bring-up
# failure (run v3-20260611-150247 -> 173707). Set to 1 to skip the reap entirely
# (e.g. when you KNOW the nodes are clean, or to avoid touching co-located procs).
HPRL_SKIP_PRELAUNCH_CLEAN=${HPRL_SKIP_PRELAUNCH_CLEAN:-0}

# NOTE: the per-launch selector-endpoint rendezvous DIR (RDV_DIR) is defined
# below, AFTER JOB_STAMP -- it is keyed on the shared per-launch stamp so a
# relaunch never reads the previous launch's stale endpoint files.

# --- dedicated selector-check log (one file per pod, so you can just `cat` it) -
# Records this pod's advertised fabric IP (selector pods) and the reachability
# probe results (training pods). Separate from the Ray console log so a silent
# selector outage is easy to spot. Layout (timestamps are Pacific time):
#   logs/selector_check/<YYYYmmdd-HHMMSS_ZONE>_<job>/rank<N>.log
# All pods of one job share ONE timestamped dir: the first pod to win an atomic
# `mkdir` writes the shared stamp, the rest read it (so we don't get 6 dirs that
# differ by the few seconds between pod starts).
#
# The dir LABEL defaults to MASTER_ADDR (the k8s PyTorchJob name, e.g. ...-v2-...),
# NOT the training script's exp_name -- this launcher runs on every pod (incl. the
# selector pods that never run run_hprl), so it cannot see exp_name. To tag the
# dir with the experiment (e.g. v3), export SELECTOR_EXP_TAG in the JOB SPEC env.
#
# The shared stamp is keyed on the (unchanged) job name, so a RELAUNCH of the same
# PyTorchJob would otherwise reuse the previous launch's stamp (stale timestamp).
# We rotate any stamp older than SELECTOR_STAMP_TTL so each relaunch gets a FRESH
# time. TTL must exceed the few-second pod-start spread within one launch but be
# below the gap between launches; 120s fits both.
PST_TZ=${PST_TZ:-"America/Los_Angeles"}
pst_now() { TZ="${PST_TZ}" date "$@"; }   # all check-log timestamps in Pacific time

SELECTOR_CHECK_ROOT=${SELECTOR_CHECK_ROOT:-"${HINT_RL_HOME}/logs/selector_check"}
SELECTOR_STAMP_TTL=${SELECTOR_STAMP_TTL:-120}
_job_key="${MASTER_ADDR:-x}_${MASTER_PORT:-30000}"
_dir_label="${SELECTOR_EXP_TAG:-${_job_key}}"
_stamp_dir="${SELECTOR_CHECK_ROOT}/.stamp.${_job_key}"
mkdir -p "${SELECTOR_CHECK_ROOT}"
# Rotate a STALE stamp left by a previous launch so this launch re-stamps fresh.
# `mv` is atomic: if pods race, one wins the rename and the rest find it gone and
# fall through to the mkdir claim below. Pods of the SAME launch start < TTL apart
# and so never rotate each other's fresh stamp.
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

# --- per-LAUNCH selector-endpoint rendezvous DIR (one file per selector rank) -
# Keyed on the shared per-launch JOB_STAMP (agreed by ALL pods above, before the
# selector/training split) in addition to the job name. This isolates each launch:
# a relaunch of the same PyTorchJob gets a fresh dir, so a fast-starting training
# pod can no longer collect the PREVIOUS launch's stale endpoint files (which point
# to now-dead selector IPs) before this launch's selectors publish. Override with
# SELECTOR_ENDPOINT_DIR to pin an exact path.
RDV_DIR=${SELECTOR_ENDPOINT_DIR:-"${HINT_RL_HOME}/logs/.selector_endpoints.${_job_key}.${JOB_STAMP}"}
mkdir -p "${RDV_DIR}"
SELECTOR_CHECK_LOG=${SELECTOR_CHECK_LOG:-"${SELECTOR_CHECK_DIR}/rank${RANK}.log"}
: > "${SELECTOR_CHECK_LOG}"   # truncate at launch
# slog: print to stdout AND append (with Pacific timestamp + rank) to the check log.
slog() {
    local msg="$*"
    echo "${msg}"
    echo "[$(pst_now '+%Y-%m-%d %H:%M:%S %Z')] [rank ${RANK}/$(hostname)] ${msg}" >> "${SELECTOR_CHECK_LOG}"
}

slog "[launch] rank=${RANK}/${WORLD_SIZE}  selector=[${SELECTOR_FIRST_RANK}..$((WORLD_SIZE-1))]  train_nnodes=${TRAIN_NNODES}  check_log=${SELECTOR_CHECK_LOG}"

# Reap orphaned vLLM workers from a SIGKILLed prior run before THIS pod binds any
# port. A run killed by eviction/preemption/OOM does not reap its distributed vLLM
# children, so they keep their TCP ports bound; on hostNetwork pods the next
# process to bind the same port dies with EADDRINUSE (seen: run v3-20260611-150247
# -> 173707, vLLM EngineCore "address already in use"). At pod startup nothing of
# THIS run has bound yet, so any match is a stale orphan -> safe to -9. Pure local
# pkill: no Ray dependency, runs before `ray start`/vLLM, and each branch passes
# only the patterns for the role it is about to start (so a training pod never
# kills a co-located selector's api_server, and vice versa). Disable with
# HPRL_SKIP_PRELAUNCH_CLEAN=1.
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

if [ "${IS_SELECTOR_NODE}" = "1" ]; then
    # =================== SELECTOR NODE (independent gpt-oss-20b server) =======
    # Advertise the IP on the SAME fabric the training pods (the Ray cluster) use,
    # NOT just the first `hostname -I` address. These pods have MANY interfaces --
    # a 10.20.x host fabric (where Ray binds & training reaches us), several
    # 192.168.x pod/RDMA NICs, and a 10.233.x k8s service net. The 192.168.x and
    # 10.233.x nets are NOT routable pod-to-pod (proven: curl to 192.168.x times
    # out, curl to our 10.20.x is routable). Neither `hostname -I | first` NOR the
    # default route works here: on these pods the DEFAULT ROUTE egresses via a
    # 192.168.x NIC (so two earlier runs advertised 192.168.x and EVERY hint call
    # failed). The only reliable signal is the fabric subnet PREFIX: pick the
    # `hostname -I` address that starts with SELECTOR_FABRIC_PREFIX (the Ray
    # fabric, default "10.20."). Overrides: SELECTOR_ADVERTISE_IP=<ip> forces an
    # exact IP; SELECTOR_FABRIC_PREFIX="" disables prefix matching.
    SELECTOR_FABRIC_PREFIX=${SELECTOR_FABRIC_PREFIX-"10.20."}
    SELF_IP="${SELECTOR_ADVERTISE_IP:-}"
    if [ -z "${SELF_IP}" ] && [ -n "${SELECTOR_FABRIC_PREFIX}" ]; then
        for _ip in $(hostname -I 2>/dev/null); do
            case "${_ip}" in "${SELECTOR_FABRIC_PREFIX}"*) SELF_IP="${_ip}"; break ;; esac
        done
    fi
    [ -z "${SELF_IP}" ] && { SELF_IP="$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)" || SELF_IP=""; }
    [ -z "${SELF_IP}" ] && { SELF_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || SELF_IP=""; }
    SELF_IP="${SELF_IP:-127.0.0.1}"
    slog "[launch]   self_ip  : ${SELF_IP}  (fabric IP via prefix='${SELECTOR_FABRIC_PREFIX}')"
    slog "[launch]   all_ips  : $(hostname -I 2>/dev/null)"
    SELECTOR_URL="http://${SELF_IP}:${SELECTOR_PORT}/v1"
    echo "${SELECTOR_URL}" > "${RDV_DIR}/${RANK}"   # publish early -> training starts in parallel
    slog "[launch] SELECTOR node (rank ${RANK}): serving ${SELECTOR_SERVED_NAME} (dp=${SELECTOR_DP} tp=${SELECTOR_TP})"
    slog "[launch]   model    : ${SELECTOR_MODEL_PATH}"
    slog "[launch]   endpoint : ${SELECTOR_URL}  (published -> ${RDV_DIR}/${RANK})"

    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    REASONING_ARGS=()
    [ -n "${SELECTOR_REASONING_PARSER}" ] && REASONING_ARGS=(--reasoning-parser "${SELECTOR_REASONING_PARSER}")

    # Free the selector port from a stale api_server before we re-bind it.
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
else
    # =================== TRAINING NODE (Ray + run_hprl) ===================
    urls=()
    if [ -n "${SELECTOR_HOST:-}" ]; then
        IFS=',' read -ra hosts <<< "${SELECTOR_HOST}"
        for h in "${hosts[@]}"; do urls+=("http://${h}:${SELECTOR_PORT}/v1"); done
    else
        echo "[launch] training node: collecting ${SELECTOR_NNODES} selector endpoint(s) from ${RDV_DIR} ..."
        for R in $(seq "${SELECTOR_FIRST_RANK}" $((WORLD_SIZE - 1))); do
            f="${RDV_DIR}/${R}"
            for _i in $(seq 1 240); do        # up to ~20 min per endpoint
                if [ -s "${f}" ]; then break; fi
                # Late-pod fallback: a pod that starts >SELECTOR_STAMP_TTL after
                # the launch wave rotates to a FRESH stamp, so its RDV_DIR can
                # never receive the wave's endpoint files (the selectors published
                # to the wave's dir) -> 20min FATAL -> restart -> fresh stamp
                # again: a permanent crashloop that holds the cluster at N-1/N
                # nodes (seen: v6 rank1, stamp 204114 vs wave 203828, 20260611).
                # After 5 min of an empty own dir (selectors publish within ~1 min
                # of pod start, long before model load), adopt this rank's file
                # from the NEWEST sibling rdv dir of the same job key; the
                # reachability probe below rejects dead/stale adoptions.
                if [ "${_i}" -ge 60 ]; then
                    for _d in $(ls -dt "${HINT_RL_HOME}/logs/.selector_endpoints.${_job_key}."* 2>/dev/null); do
                        [ "${_d}" = "${RDV_DIR}" ] && continue
                        if [ -s "${_d}/${R}" ]; then
                            { cp "${_d}/${R}" "${f}" 2>/dev/null && \
                                slog "[launch]   adopted rank-${R} selector endpoint from sibling launch dir ${_d##*/}"; } || true
                            break
                        fi
                    done
                fi
                sleep 5
            done
            [ -s "${f}" ] || { echo "[launch] FATAL: selector endpoint for rank ${R} never published (${f})" >&2; exit 1; }
            urls+=("$(cat "${f}")")
        done
    fi
    SELECTOR_BASE_URLS="$(IFS=,; echo "${urls[*]}")"
    export SELECTOR_BASE_URLS
    export SELECTOR_BASE_URL="${urls[0]}"   # back-compat (first endpoint)
    echo "[launch] training node: SELECTOR_BASE_URLS=${SELECTOR_BASE_URLS}  model=${SELECTOR_MODEL}"

    # --- fail-fast selector reachability check --------------------------------
    # A silent selector outage (e.g. an endpoint advertised on an unroutable NIC)
    # degrades EVERY hint call to a no-op for the whole run -- invisibly, because
    # HintSelector falls back gracefully. Probe each endpoint's /v1/models now so
    # the failure surfaces at launch instead of after hours of dead-hint training.
    # Set SELECTOR_REQUIRE_REACHABLE=0 to downgrade the abort to a warning.
    SELECTOR_REQUIRE_REACHABLE=${SELECTOR_REQUIRE_REACHABLE:-1}
    SELECTOR_PROBE_RETRIES=${SELECTOR_PROBE_RETRIES:-60}   # ~5 min (model may still be loading)
    slog "[launch] training node: probing ${#urls[@]} selector endpoint(s) -> ${SELECTOR_BASE_URLS}"
    if ! command -v curl >/dev/null 2>&1; then
        slog "[launch] WARN: curl not found; skipping selector reachability probe."
    else
        n_ok=0
        for u in "${urls[@]}"; do
            ok=0
            for _ in $(seq 1 "${SELECTOR_PROBE_RETRIES}"); do
                if curl -fsS --max-time 5 -H "Authorization: Bearer ${SELECTOR_API_KEY}" \
                        "${u}/models" >/dev/null 2>&1; then ok=1; break; fi
                sleep 5
            done
            if [ "${ok}" = "1" ]; then
                slog "[launch]   selector OK    : ${u}"; n_ok=$((n_ok + 1))
            else
                slog "[launch]   selector UNREACHABLE: ${u}  (curl ${u}/models failed)"
            fi
        done
        if [ "${n_ok}" -eq 0 ]; then
            slog "[launch] FATAL: no selector endpoint reachable from this training node."
            slog "         Hints would silently degrade to no-ops for the entire run."
            slog "         Check the selector advertised the fabric IP (self_ip above) and is serving."
            if [ "${SELECTOR_REQUIRE_REACHABLE}" = "1" ]; then exit 1; fi
            slog "[launch] SELECTOR_REQUIRE_REACHABLE=0 -> continuing despite unreachable selector."
        else
            slog "[launch] selector check passed: ${n_ok}/${#urls[@]} endpoint(s) reachable."
        fi
    fi

    # Free rollout-vLLM ports from a SIGKILLed prior run before `ray start` brings
    # this node up. Rollout-worker patterns only -- never api_server, so a selector
    # co-located via hostNetwork is left untouched.
    hprl_reap_stale_vllm EngineCore WorkerProc multiproc_executor vLLMHttpServer \
        "vllm.v1.engine" "vllm.v1.executor"

    # Only the TRAIN_NNODES nodes join the Ray cluster (the selector pods are out).
    export NNODES=${TRAIN_NNODES}
    export TRAIN_SCRIPT
    # ray_cluster_launch.sh: ray start (head/worker auto-detected via MASTER_ADDR),
    # head waits for NNODES, then runs TRAIN_SCRIPT (run_hprl), which `ray job
    # submit`s the job and forwards the SELECTOR_* env into the rollout runtime env.
    exec "${RAY_LAUNCH}"
fi
