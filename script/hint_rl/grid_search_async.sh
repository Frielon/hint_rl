#!/usr/bin/env bash
# =============================================================================
# grid_search_async.sh -- sequential grid search over the fully-async pipeline
# knobs. Runs ON THE RAY HEAD in the TRAIN_SCRIPT slot (cluster already up):
# each trial is one bounded `run_hprl_async.sh` launch (= one ray job) with a
# different config; the Ray cluster + selector pods stay up across trials.
#
# Platform entry: submit launch_hprl_cluster_async_grid.sh (4 replicas).
#
# Swept axes (all per-ray-job settings; override the lists via env):
#   SPLITS          TRAINER:ROLLOUT node splits. "auto" = every t:r with
#                   t+r == NNODES, t,r >= 1  (NNODES=3 -> "1:2 2:1").
#   STALENESS_LIST  async_training.staleness_threshold values.
#   PARTIAL_LIST    async_training.partial_rollout values.
# NOT sweepable here: SELECTOR_NNODES -- the selector/training pod split is
# fixed when the pods start. To scan it (e.g. 1+3 vs 2+2 on 4 pods), submit the
# platform job once per value; this driver adapts to whatever NNODES it gets.
#
# Each trial: exactly TRIAL_STEPS param versions (default 10; enforced via
# TOTAL_TRAINING_STEPS -> rollout.total_rollout_steps, with TRIAL_EPOCHS as a
# generous secondary ceiling), validation + checkpointing + selector dumps OFF,
# wall-clock backstop TRIAL_TIMEOUT_S, own exp dir under logs/. Metrics
# appended to ${GRID_DIR}/results.tsv:
#   versions        param versions completed (1 == one sync-equivalent step ==
#                   64 prompts x 8 responses at trigger=1/require=1)
#   vph_overall     versions / total wall-clock hour
#   vph_steady      rate over the trial's 2nd half (excludes vLLM/NCCL warmup)
#                   -- the PRIMARY ranking metric
#   trainer_idle / rollouter_idle
#                   mean fully_async idle ratios over the 2nd half (rebalance
#                   signal: high rollouter idle -> shift a node to the trainer)
#
# DRY_RUN=1 prints the trial plan and exits without launching anything.
# =============================================================================
set -uo pipefail  # deliberately NO -e: a failed trial must not kill the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
RUN_SCRIPT=${RUN_SCRIPT:-"${SCRIPT_DIR}/run_hprl_async.sh"}
NNODES=${NNODES:-3}   # training pods in the Ray cluster (exported by the launch chain)
RAY_DASH=${RAY_ADDRESS:-http://localhost:8265}

SPLITS=${SPLITS:-auto}
STALENESS_LIST=${STALENESS_LIST:-"0.3 0.5"}
PARTIAL_LIST=${PARTIAL_LIST:-"True False"}
TRIAL_STEPS=${TRIAL_STEPS:-10}             # param versions per trial (the hard bound)
TRIAL_EPOCHS=${TRIAL_EPOCHS:-4}            # secondary ceiling; dapo-512 x4 = 32 versions max
TRIAL_TIMEOUT_S=${TRIAL_TIMEOUT_S:-9000}   # 2.5 h backstop per trial
SETTLE_S=${SETTLE_S:-60}                   # post-trial drain before the next launch
SAMPLE_EVERY_S=${SAMPLE_EVERY_S:-60}
DRY_RUN=${DRY_RUN:-0}

if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "[grid] FATAL: RUN_SCRIPT not found: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [ "${SPLITS}" = "auto" ]; then
    SPLITS=""
    for ((t = 1; t < NNODES; t++)); do SPLITS+="${t}:$((NNODES - t)) "; done
fi
if [ -z "${SPLITS// /}" ]; then
    echo "[grid] FATAL: no valid trainer:rollout split for NNODES=${NNODES} (need >= 2 training pods)." >&2
    exit 1
fi

GRID_ID=${GRID_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
GRID_DIR=${GRID_DIR:-"${HINT_RL_HOME}/logs/async_grid_${GRID_ID}"}
LOG_DIR="${HINT_RL_HOME}/logs"
mkdir -p "${GRID_DIR}"
RESULTS="${GRID_DIR}/results.tsv"
printf 'trial\ttrainer:rollout\tstaleness\tpartial\tstatus\tversions\telapsed_min\tvph_overall\tvph_steady\ttrainer_idle\trollouter_idle\n' > "${RESULTS}"

say() { echo "[grid $(date +%H:%M:%S)] $*"; }

# Stop every RUNNING/PENDING ray job on the cluster (only our trials run here).
# Needed because killing the `ray job submit` CLIENT (timeout) does NOT stop
# the server-side job.
stop_ray_jobs() {
    python3 - "${RAY_DASH}" <<'PY' 2>/dev/null
import json, sys, urllib.request
base = sys.argv[1].rstrip('/') + '/api/jobs/'
try:
    jobs = json.load(urllib.request.urlopen(base, timeout=30))
except Exception:
    sys.exit(0)
for j in jobs:
    if j.get('status') in ('RUNNING', 'PENDING'):
        sid = j.get('submission_id')
        if sid:
            try:
                urllib.request.urlopen(
                    urllib.request.Request(base + sid + '/stop', method='POST'), timeout=30)
                print('[grid] stopped ray job', sid)
            except Exception as exc:
                print('[grid] stop failed for', sid, exc)
PY
}
trap 'stop_ray_jobs >/dev/null 2>&1' EXIT

# Reap orphaned rollout-vLLM workers on EVERY training node between trials. A
# trial that dies hard (OOM, SIGKILL) leaves detached vLLM children holding
# their torch.distributed ports AND GPU memory; stop_ray_jobs + settle does not
# collect them, so the next trial dies at vLLM startup with EADDRINUSE (seen:
# grid 20260720-141549, st1 OOM -> st3 dead in 4.5 min). Same patterns as the
# launcher's per-pod pre-launch reap -- rollout-worker procs only, never
# api_server, and the selector pods are not in the Ray cluster, so a co-located
# or separate selector is untouched. Scheduled per-node via NodeAffinity.
reap_stale_vllm_cluster() {
    python3 - <<'PY' 2>/dev/null
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
try:
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
except Exception:
    raise SystemExit(0)

PATTERNS = ["EngineCore", "WorkerProc", "multiproc_executor", "vLLMHttpServer",
            "vllm.v1.engine", "vllm.v1.executor"]

@ray.remote(num_cpus=0)
def _reap(patterns):
    import socket, subprocess
    killed = [p for p in patterns
              if subprocess.run(["pkill", "-9", "-f", p],
                                capture_output=True).returncode == 0]
    return socket.gethostname(), killed

futs = [_reap.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n["NodeID"], soft=False)).remote(PATTERNS)
        for n in ray.nodes() if n.get("Alive")]
try:
    for host, killed in ray.get(futs, timeout=120):
        if killed:
            print(f"[grid] reaped stale vllm procs on {host}: {' '.join(killed)}")
except Exception as exc:
    print(f"[grid] WARN: cluster reap incomplete: {exc}")
PY
    sleep 2   # let the kernel release the freed ports
}

# trial_metrics <jsonl> <progress> <elapsed_s> -> the 6 metric columns as TSV
trial_metrics() {
    python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys
jsonl, progress, elapsed = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
if os.path.exists(jsonl):
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
# Versions completed = highest step logged, NOT the line count: the fully-async
# jsonl carries one rollouter row AND one trainer row per version (line-count
# double-counted, e.g. grid 155006: 22 rows = 10 versions + step-0 rows).
steps = [r.get('step') for r in rows if isinstance(r.get('step'), int)]
n = max(steps) if steps else 0

def mean_key(suffix):
    tail = rows[max(0, len(rows) // 2):]
    # metrics live under the file logger's nested 'data' dict, not the top level
    vals = [v for r in tail for k, v in (r.get('data') or r).items()
            if k.endswith(suffix) and isinstance(v, (int, float))]
    return f'{sum(vals) / len(vals):.3f}' if vals else 'NA'

vph = n / elapsed * 3600 if elapsed > 0 else 0.0
steady = ''
if os.path.exists(progress):
    samples = []
    with open(progress) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                try:
                    samples.append((int(p[0]), int(p[1])))
                except Exception:
                    pass
    if len(samples) >= 4:
        mid, last = samples[len(samples) // 2], samples[-1]
        dt, dn = last[0] - mid[0], last[1] - mid[1]
        if dt > 0 and dn >= 0:
            # sampler counts jsonl LINES; convert to versions via this trial's
            # observed lines-per-version ratio (2 in fully-async mode)
            per_version = len(rows) / n if n else 1.0
            steady = f'{dn / dt * 3600 / per_version:.2f}'
print(f"{n}\t{elapsed / 60:.1f}\t{vph:.2f}\t{steady or 'NA'}\t"
      f"{mean_key('trainer/idle_ratio')}\t{mean_key('rollouter/idle_ratio')}")
PY
}

say "grid: splits [${SPLITS% }] x staleness [${STALENESS_LIST}] x partial [${PARTIAL_LIST}]"
say "NNODES=${NNODES} training pods; ${TRIAL_STEPS} steps/trial (epoch ceiling ${TRIAL_EPOCHS}), ${TRIAL_TIMEOUT_S}s backstop; results -> ${RESULTS}"

for split in ${SPLITS}; do
  T=${split%%:*}; R=${split##*:}
  for stal in ${STALENESS_LIST}; do
    for part in ${PARTIAL_LIST}; do
      name="t${T}r${R}_st${stal}_pr${part}"
      exp="async-grid-${GRID_ID}-${name}"
      jsonl="${LOG_DIR}/${exp}/${exp}.jsonl"
      progress="${GRID_DIR}/${name}.progress"

      if [ "${DRY_RUN}" != "0" ]; then
          say "DRY_RUN ${name}: TRAINER_NNODES=${T} ROLLOUT_NNODES=${R}" \
              "STALENESS_THRESHOLD=${stal} PARTIAL_ROLLOUT=${part} exp_name=${exp}"
          continue
      fi

      say "=== trial ${name} (exp ${exp}) ==="
      stop_ray_jobs
      reap_stale_vllm_cluster
      : > "${progress}"
      ( while sleep "${SAMPLE_EVERY_S}"; do
            echo "$(date +%s) $(wc -l < "${jsonl}" 2>/dev/null || echo 0)"
        done >> "${progress}" ) &
      sampler=$!
      t0=$(date +%s)
      timeout --kill-after=120 "${TRIAL_TIMEOUT_S}" \
          env TRAINER_NNODES="${T}" ROLLOUT_NNODES="${R}" \
              STALENESS_THRESHOLD="${stal}" PARTIAL_ROLLOUT="${part}" \
              TOTAL_TRAINING_STEPS="${TRIAL_STEPS}" \
              TOTAL_EPOCHS="${TRIAL_EPOCHS}" VAL_BEFORE_TRAIN=False \
              TEST_FREQ=-1 SAVE_FREQ=-1 HPRL_DUMP_SELECTOR=0 \
              project_name='HPRL-async-grid' wandb_project='hint_rl-async-grid' \
              exp_name="${exp}" \
          bash "${RUN_SCRIPT}" > "${GRID_DIR}/${name}.console.log" 2>&1
      rc=$?
      t1=$(date +%s)
      kill "${sampler}" 2>/dev/null
      wait "${sampler}" 2>/dev/null
      stop_ray_jobs
      sleep "${SETTLE_S}"

      case ${rc} in
          0)       status=done ;;
          124|137) status=timeout ;;
          *)       status="fail(${rc})" ;;
      esac
      metrics=$(trial_metrics "${jsonl}" "${progress}" "$((t1 - t0))")
      printf '%s\t%s:%s\t%s\t%s\t%s\t%s\n' \
          "${name}" "${T}" "${R}" "${stal}" "${part}" "${status}" "${metrics}" >> "${RESULTS}"
      say "trial ${name}: ${status}, $(echo "${metrics}" | cut -f1) versions" \
          "in $(echo "${metrics}" | cut -f2) min (vph_steady $(echo "${metrics}" | cut -f4))"
    done
  done
done

if [ "${DRY_RUN}" != "0" ]; then
    say "dry run complete (nothing launched)"
    exit 0
fi

say "grid complete. ranked by steady versions/hour (fallback: overall):"
{ head -1 "${RESULTS}"
  tail -n +2 "${RESULTS}" | sort -t$'\t' -k9,9gr -k8,8gr
} | column -t -s$'\t' | tee "${GRID_DIR}/results_ranked.txt"
best=$(tail -n +2 "${RESULTS}" | sort -t$'\t' -k9,9gr -k8,8gr | head -1 | cut -f1-4)
say "best config: ${best}"
