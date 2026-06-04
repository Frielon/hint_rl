#!/usr/bin/env bash
# =============================================================================
# Ray Cluster Auto-Launch & Job Submission (multi-node PyTorchJob)
# =============================================================================
# Which training script the head runs is set by the TRAIN_SCRIPT variable below
# (default: run_drgrpo_qwen2.5_7b_npu.sh), or overridden via the TRAIN_SCRIPT env
# var. Positional args are ignored on purpose -- the PyTorchJob wrapper appends
# its own junk args (`... /xutao/ undefined`).
#
# Run this on EVERY pod of the PyTorchJob (head + workers). It:
#   1. activates the verl conda env (same path logic as run_dapo_*.sh),
#   2. waits for the master pod's DNS name to resolve,
#   3. auto-detects whether this pod is the head (MASTER_ADDR -> a local IP),
#   4. head : `ray start --head`, waits for all NNODES to join, then runs the
#             training script (run_dapo_*.sh, which `ray job submit`s the job),
#      worker: `ray start --block --address <head>:<port>` and stays alive.
#
# Point the web service / PyTorchJob entrypoint at THIS script instead of
# run_dapo_qwen2.5_7b_npu.sh.
#
# Required env (injected by the PyTorchJob):
#   MASTER_ADDR  - hostname/IP of the head pod
#   MASTER_PORT  - port for the Ray head GCS server
# Optional env:
#   NNODES              - total pods (default: WORLD_SIZE, else 2)
#   N_GPUS_PER_NODE     - GPUs per pod (default: 8)
#   RAY_DASHBOARD_PORT  - head dashboard port (default: 8265; must match
#                         RAY_ADDRESS in run_dapo_*.sh)
#   NODE_WAIT_TIMEOUT   - minutes to wait for cluster/DNS (default: 180)
#   POLL_INTERVAL       - minutes between cluster status checks (default: 1)
#   TRAIN_SCRIPT        - training script the head runs (default: the sibling
#                         run_drgrpo_qwen2.5_7b_npu.sh).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths + conda env (mirrors run_dapo_qwen2.5_7b_npu.sh so head and workers
# share one activation path; the conda prefix symlink fix is needed on every
# node, not just the head).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}

# conda is not relocatable: the install prefix is baked into conda.sh and every
# script shebang. Symlink it to the current mount point so those paths resolve.
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

RAY_BIN=${RAY_BIN:-"${CONDA_HOME}/envs/${CONDA_ENV}/bin/ray"}

# Training script the head runs. Set it here (or override via the TRAIN_SCRIPT
# env var). We deliberately do NOT read positional args: the PyTorchJob wrapper
# invokes this script as `ray_cluster_launch.sh /xutao/ undefined`, so $1/$2 are
# platform junk, not ours. A bare filename (no slash) is resolved against this
# script's dir.
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_dapo_qwen2.5_7b_npu.sh"}
# TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${SCRIPT_DIR}/run_drgrpo_qwen2.5_7b_npu.sh"}
if [[ "${TRAIN_SCRIPT}" != */* ]]; then
    TRAIN_SCRIPT="${SCRIPT_DIR}/${TRAIN_SCRIPT}"
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate / default cluster env
# ---------------------------------------------------------------------------
for var in MASTER_ADDR MASTER_PORT; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] Required environment variable ${var} is not set." >&2
        exit 1
    fi
done

NNODES=${NNODES:-${WORLD_SIZE:-2}}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
NODE_WAIT_TIMEOUT=${NODE_WAIT_TIMEOUT:-180}   # minutes
POLL_INTERVAL=${POLL_INTERVAL:-1}             # minutes

NODE_WAIT_TIMEOUT_SEC=$((NODE_WAIT_TIMEOUT * 60))
POLL_INTERVAL_SEC=$((POLL_INTERVAL * 60))

echo "============================================="
echo " Ray Cluster Launch"
echo "   MASTER_ADDR        : ${MASTER_ADDR}"
echo "   MASTER_PORT        : ${MASTER_PORT}"
echo "   NNODES             : ${NNODES}"
echo "   N_GPUS_PER_NODE    : ${N_GPUS_PER_NODE}"
echo "   RAY_DASHBOARD_PORT : ${RAY_DASHBOARD_PORT}"
echo "   RAY_BIN            : ${RAY_BIN}"
echo "   TRAIN_SCRIPT       : ${TRAIN_SCRIPT}"
echo "============================================="

# ---------------------------------------------------------------------------
# Wait for MASTER_ADDR to be DNS-resolvable (k8s pod DNS can lag pod start).
# This is what was failing before: `nslookup can't resolve ...master-0`.
# ---------------------------------------------------------------------------
resolve_host() {
    getent hosts "$1" >/dev/null 2>&1 \
        || nslookup "$1" >/dev/null 2>&1 \
        || python3 -c "import socket,sys; socket.getaddrinfo(sys.argv[1], None)" "$1" >/dev/null 2>&1
}

if [[ "${MASTER_ADDR}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "[INFO] MASTER_ADDR is an IP (${MASTER_ADDR}); skipping DNS wait."
else
    echo "[INFO] Waiting for DNS resolution of ${MASTER_ADDR}..."
    elapsed=0
    while ! resolve_host "${MASTER_ADDR}"; do
        if [[ "${elapsed}" -ge "${NODE_WAIT_TIMEOUT_SEC}" ]]; then
            echo "[ERROR] Timed out resolving ${MASTER_ADDR} after $((elapsed / 60))min." >&2
            exit 1
        fi
        echo "[INFO] DNS not ready, retry in ${POLL_INTERVAL}min (elapsed: $((elapsed / 60))min)"
        sleep "${POLL_INTERVAL_SEC}"
        elapsed=$((elapsed + POLL_INTERVAL_SEC))
    done
    echo "[INFO] Resolved ${MASTER_ADDR} -> $(getent hosts "${MASTER_ADDR}" 2>/dev/null | awk '{print $1}')"
fi

# ---------------------------------------------------------------------------
# Head detection: does MASTER_ADDR map to one of this pod's IPs?
# ---------------------------------------------------------------------------
is_head_node() {
    local local_ips resolved_ip
    local_ips=$(hostname -I 2>/dev/null || echo "")

    [[ "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "127.0.0.1" ]] && return 0

    for ip in ${local_ips}; do
        [[ "${MASTER_ADDR}" == "${ip}" ]] && return 0
    done
    [[ "${MASTER_ADDR}" == "$(hostname -s 2>/dev/null)" ]] && return 0
    [[ "${MASTER_ADDR}" == "$(hostname -f 2>/dev/null)" ]] && return 0

    resolved_ip=$(getent hosts "${MASTER_ADDR}" 2>/dev/null | awk '{print $1}' || echo "")
    [[ -z "${resolved_ip}" ]] && resolved_ip=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))" 2>/dev/null || echo "")
    if [[ -n "${resolved_ip}" ]]; then
        for ip in ${local_ips}; do
            [[ "${resolved_ip}" == "${ip}" ]] && { echo "[INFO] ${MASTER_ADDR} -> ${resolved_ip} matches a local IP."; return 0; }
        done
    fi
    return 1
}

echo "[INFO] Stopping any pre-existing Ray on this node..."
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
sleep 2

if is_head_node; then
    echo "[INFO] HEAD node. Starting Ray head on :${MASTER_PORT} (dashboard :${RAY_DASHBOARD_PORT})..."
    "${RAY_BIN}" start --head \
        --port="${MASTER_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --num-gpus="${N_GPUS_PER_NODE}"
    sleep 10

    echo "[INFO] Waiting for ${NNODES} nodes to join (timeout ${NODE_WAIT_TIMEOUT}min)..."
    elapsed=0
    while true; do
        node_count=$(python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    print(sum(1 for n in ray.nodes() if n['Alive']))
except Exception:
    print(0)
" 2>/dev/null)
        echo "[INFO] Nodes ready: ${node_count}/${NNODES} (elapsed: $((elapsed / 60))min)"
        [[ "${node_count}" -ge "${NNODES}" ]] && { echo "[INFO] All ${NNODES} nodes joined."; break; }
        if [[ "${elapsed}" -ge "${NODE_WAIT_TIMEOUT_SEC}" ]]; then
            echo "[ERROR] Timed out: only ${node_count}/${NNODES} nodes joined." >&2
            "${RAY_BIN}" status 2>&1 || true
            exit 1
        fi
        sleep "${POLL_INTERVAL_SEC}"
        elapsed=$((elapsed + POLL_INTERVAL_SEC))
    done

    echo "============================================="
    "${RAY_BIN}" status
    echo "============================================="

    # Cluster is up; run the training script. It `ray job submit`s to the
    # dashboard on localhost:${RAY_DASHBOARD_PORT} and dispatches across all nodes.
    echo "[INFO] Launching training: ${TRAIN_SCRIPT}"
    exec /bin/bash "${TRAIN_SCRIPT}"
else
    echo "[INFO] WORKER node. Waiting for head ${MASTER_ADDR}:${MASTER_PORT} to be reachable..."
    elapsed=0
    while ! timeout 5 bash -c "echo >/dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; do
        if [[ "${elapsed}" -ge "${NODE_WAIT_TIMEOUT_SEC}" ]]; then
            echo "[ERROR] Timed out waiting for head ${MASTER_ADDR}:${MASTER_PORT}." >&2
            exit 1
        fi
        echo "[INFO] Head not reachable yet, retry in ${POLL_INTERVAL}min (elapsed: $((elapsed / 60))min)"
        sleep "${POLL_INTERVAL_SEC}"
        elapsed=$((elapsed + POLL_INTERVAL_SEC))
    done
    echo "[INFO] Connecting to head and blocking (keeps the worker pod alive)..."
    # --block must stay in the foreground: if it exits, the pod dies.
    exec "${RAY_BIN}" start --block \
        --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus="${N_GPUS_PER_NODE}"
fi
