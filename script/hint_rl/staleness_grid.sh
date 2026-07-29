#!/usr/bin/env bash
# =============================================================================
# staleness_grid.sh -- SYNC-vs-ASYNC config benchmark from a trained ckpt:
#   trial 0        : SYNC baseline (colocated hybrid engine), ALL NNODES pods
#                    training -- the architecture the step-320 ckpt was made on.
#   trials 1..K    : FULLY-ASYNC, fixed split 1 trainer : (NNODES-1) rollout,
#                    sweeping async_training.staleness_threshold over
#                    STALENESS_LIST (default 0.2 .. 2).
#
# Runs ON THE RAY HEAD in the TRAIN_SCRIPT slot (cluster already up), exactly
# like grid_search_async.sh: each trial is one bounded ray job; the Ray cluster
# (and, with the OpenAI selector mode, nothing else) stays up across trials.
#
# Platform entry: submit launch_hprl_cluster_openai_staleness_grid.sh
# (N replicas, N >= 2; the OpenAI API is the selector so ALL pods train).
# Works identically behind the classic selector-pod launcher:
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster.sh
#
# START POLICY -- every trial starts from the SAME trained policy:
#   BASE_CKPT (default: the Qwen3-8B-Base auto-hint run's global_step_320) is
#   MERGED once (idempotently) to a HF dir and passed as MODEL_PATH. Merging --
#   not trainer.resume_from_path -- because the FSDP shards are saved at the
#   source run's actor world size (32 = 4 nodes x 8): the 1-node async trainer
#   (world 8) could not load them, and a fresh optimizer/dataloader is the
#   right equal footing for a throughput benchmark anyway. The merged dir goes
#   under eval/merged/ with the eval harness's label so later evals reuse it;
#   it inherits the eos-widened generation_config ([151643,151645]) from
#   actor/huggingface, and the stale max_new_tokens=2048 is stripped (same
#   repair as eval/run_eval_ckpt.sh -- vLLM would otherwise cap generation).
#
# Trial config parity (science knobs identical across ALL trials, matching the
# run that produced the ckpt -- run_auto_hint_qwen3_8b_base.sh):
#   * sync  trial -> run_auto_hint_qwen3_8b_base.sh (the wrapper itself; its
#     MODEL_PATH guard skips the eos-derived-dir build since we pass the merged
#     dir, which already carries the fixed generation_config).
#   * async trials -> run_hprl_async.sh with the qwen3 deltas exported:
#     max_response_length=30720 (32k-native model; async default 28000 is the
#     olmo value) and HPRL_OVERLONG_PENALTY_TYPE=value (the qwen3/sync routing;
#     async's post_hoc default is the olmo wrapper's choice).
#   Architecture-inherent diffs stay: budget-grouped sampling is sync-only
#   (async streams, no generation batch), and each trial re-seeds budgets from
#   the parquet (fresh BUDGET_STATE_PATH per exp) -- equal footing, not the
#   320-step ratcheted state.
#
# Staleness range note: verl only asserts staleness_threshold >= 0; values > 1
# are legal (the rollouter may run (1+s)*B samples ahead; observed generation
# concurrency = s*B exactly -- the cap minus the version in training). The
# 20260722 sweep (0.2..2 at 1:4) showed steady versions/hour saturating toward
# the ~152 s/version trainer-compute floor (~24 vph ceiling), with s=2 already
# at ~94%; the default list now probes the remaining tail from 2 up to 3,
# where production should meet the trainer floor and the curve flatline.
#
# Each trial: TRIAL_STEPS param versions / sync steps (both = 64 prompts x 8
# responses -> versions/hour is directly comparable across modes), validation +
# checkpointing + selector dumps OFF, TRIAL_TIMEOUT_S wall-clock backstop.
# Metrics appended to ${GRID_DIR}/results.tsv (same columns as the async grid;
# trainer/rollouter idle are NA for the sync trial -- no fully_async metrics).
#
# DRY_RUN=1 prints the trial plan (and the merge that would run) and exits.
# =============================================================================
set -uo pipefail  # deliberately NO -e: a failed trial must not kill the sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
SYNC_RUN_SCRIPT=${SYNC_RUN_SCRIPT:-"${SCRIPT_DIR}/run_auto_hint_qwen3_8b_base.sh"}
ASYNC_RUN_SCRIPT=${ASYNC_RUN_SCRIPT:-"${SCRIPT_DIR}/run_hprl_async.sh"}
NNODES=${NNODES:-4}   # training pods in the Ray cluster (exported by the launch chain)
RAY_DASH=${RAY_ADDRESS:-http://localhost:8265}

# --- the trained policy every trial starts from ------------------------------
BASE_CKPT=${BASE_CKPT:-"${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen3-8B-Base/HPRL-AutoHint-Qwen3-8B-Base-dapo-20260720-235159/global_step_320"}
# eval-harness label for this run/step (eval/run_eval_ckpt.sh CKPT_SPECS) so
# the merged dir is shared with -- or reused from -- checkpoint evals.
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-"${HINT_RL_HOME}/eval/merged/hprl-qwen3-base-512-step320"}

# --- sweep config ------------------------------------------------------------
# Sync default OFF since the 20260722 sweep: on a 5-pod cluster the sync trial
# dies on batch divisibility (5x8 GPUs / sp4 = 10 DP ranks, 512 % 10 != 0), and
# the 4-node sync baseline is measured separately (~9.3 vph, ~388 s/step; run
# HPRL-AutoHint-Qwen3-8B-Base-dapo-20260723-100408 steps 321-332).
RUN_SYNC=${RUN_SYNC:-0}                       # 1 = run the sync baseline (needs DP-divisible batch)
STALENESS_LIST=${STALENESS_LIST:-"2 2.2 2.4 2.6 2.8 3"}     # "" = skip async
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}      # fixed (partial=False is dead at trigger=1)
TRAINER_NNODES=${TRAINER_NNODES:-1}           # async split: 1 trainer : rest rollout
ROLLOUT_NNODES=${ROLLOUT_NNODES:-$((NNODES - TRAINER_NNODES))}
TRIAL_STEPS=${TRIAL_STEPS:-16}                # sync steps / param versions per trial
TRIAL_EPOCHS=${TRIAL_EPOCHS:-4}               # secondary ceiling; dapo-512 x4 = 32 versions
# Backstop sized for the SLOWEST trial at 16 steps: the 07-20 sweep's worst
# overall rate was ~7 versions/h -> ~2.3 h for 16, plus bring-up headroom.
TRIAL_TIMEOUT_S=${TRIAL_TIMEOUT_S:-12600}     # 3.5 h backstop per trial
SETTLE_S=${SETTLE_S:-60}                      # post-trial drain before the next launch
SAMPLE_EVERY_S=${SAMPLE_EVERY_S:-60}
DRY_RUN=${DRY_RUN:-0}

# qwen3 deltas for the ASYNC trials (see header; the sync wrapper sets its own).
QWEN3_MAX_RESPONSE_LENGTH=${QWEN3_MAX_RESPONSE_LENGTH:-30720}
QWEN3_OVERLONG_TYPE=${QWEN3_OVERLONG_TYPE:-value}

for f in "${SYNC_RUN_SCRIPT}" "${ASYNC_RUN_SCRIPT}"; do
    [ -f "${f}" ] || { echo "[stgrid] FATAL: run script not found: ${f}" >&2; exit 1; }
done
if [ -n "${STALENESS_LIST// /}" ] && { [ "${TRAINER_NNODES}" -lt 1 ] || [ "${ROLLOUT_NNODES}" -lt 1 ]; }; then
    echo "[stgrid] FATAL: async trials need >=1 trainer and >=1 rollout node" \
         "(got ${TRAINER_NNODES}:${ROLLOUT_NNODES} of NNODES=${NNODES})." >&2
    exit 1
fi

GRID_ID=${GRID_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
GRID_DIR=${GRID_DIR:-"${HINT_RL_HOME}/logs/staleness_grid_${GRID_ID}"}
LOG_DIR="${HINT_RL_HOME}/logs"
mkdir -p "${GRID_DIR}"
RESULTS="${GRID_DIR}/results.tsv"
printf 'trial\ttrainer:rollout\tstaleness\tpartial\tstatus\tversions\telapsed_min\tvph_overall\tvph_steady\ttrainer_idle\trollouter_idle\n' > "${RESULTS}"

say() { echo "[stgrid $(date +%H:%M:%S)] $*"; }

# --- one-time merge of BASE_CKPT -> MERGED_MODEL_DIR (idempotent) ------------
# Compact copy of eval/run_eval_ckpt.sh merge_fsdp: model_merger, then the
# tokenizer/generation_config sidecars from actor/huggingface, then strip the
# training-default max_new_tokens that would cap vLLM generation at 2048.
ensure_merged_model() {
    if [ -f "${MERGED_MODEL_DIR}/config.json" ] && ls "${MERGED_MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
        say "merged model already present: ${MERGED_MODEL_DIR}"
    else
        [ -d "${BASE_CKPT}/actor" ] || { say "FATAL: no actor/ under BASE_CKPT=${BASE_CKPT}"; exit 1; }
        if [ "${DRY_RUN}" != "0" ]; then
            say "DRY_RUN: would merge ${BASE_CKPT}/actor -> ${MERGED_MODEL_DIR}"
            return 0
        fi
        say "merging ${BASE_CKPT}/actor -> ${MERGED_MODEL_DIR} (one-time; a few minutes)"
        if ! python3 -m verl.model_merger merge --backend fsdp \
                --local_dir "${BASE_CKPT}/actor" --target_dir "${MERGED_MODEL_DIR}"; then
            say "FATAL: verl.model_merger failed"; exit 1
        fi
    fi
    [ "${DRY_RUN}" != "0" ] && return 0
    if [ -d "${BASE_CKPT}/actor/huggingface" ]; then
        for f in tokenizer.json tokenizer_config.json vocab.json merges.txt \
                 special_tokens_map.json added_tokens.json chat_template.jinja \
                 generation_config.json; do
            [ -f "${BASE_CKPT}/actor/huggingface/${f}" ] && [ ! -f "${MERGED_MODEL_DIR}/${f}" ] \
                && cp "${BASE_CKPT}/actor/huggingface/${f}" "${MERGED_MODEL_DIR}/" 2>/dev/null || true
        done
    fi
    if [ -f "${MERGED_MODEL_DIR}/generation_config.json" ]; then
        python3 - "${MERGED_MODEL_DIR}/generation_config.json" <<'PYGC'
import json, sys
p = sys.argv[1]
with open(p) as f: g = json.load(f)
if g.pop("max_new_tokens", None) is not None or g.pop("max_length", None) is not None:
    with open(p, "w") as f: json.dump(g, f, indent=2)
    print(f"[stgrid] stripped max_new_tokens/max_length from {p}")
PYGC
    fi
}

# --- ray-job stop + cluster-wide stale-vLLM reap (verbatim from the async grid)
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
                print('[stgrid] stopped ray job', sid)
            except Exception as exc:
                print('[stgrid] stop failed for', sid, exc)
PY
}
trap 'stop_ray_jobs >/dev/null 2>&1' EXIT

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
            print(f"[stgrid] reaped stale vllm procs on {host}: {' '.join(killed)}")
except Exception as exc:
    print(f"[stgrid] WARN: cluster reap incomplete: {exc}")
PY
    sleep 2   # let the kernel release the freed ports
}

# trial_metrics <jsonl> <progress> <elapsed_s> -> the 6 metric columns as TSV.
# Works for both modes: versions = highest logged step; the fully-async jsonl
# has ~2 rows/version (rollouter + trainer), the sync one ~1 row/step -- the
# steady-rate math converts via the observed rows-per-version ratio. The idle
# columns come from fully_async/* keys and are NA on the sync trial.
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
steps = [r.get('step') for r in rows if isinstance(r.get('step'), int)]
n = max(steps) if steps else 0

def mean_key(suffix):
    tail = rows[max(0, len(rows) // 2):]
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
            per_version = len(rows) / n if n else 1.0
            steady = f'{dn / dt * 3600 / per_version:.2f}'
print(f"{n}\t{elapsed / 60:.1f}\t{vph:.2f}\t{steady or 'NA'}\t"
      f"{mean_key('trainer/idle_ratio')}\t{mean_key('rollouter/idle_ratio')}")
PY
}

# run_trial <name> <split-label> <staleness-label> <partial-label> <script> [VAR=VAL ...]
run_trial() {
    local name="$1" split="$2" stal="$3" part="$4" script="$5"; shift 5
    local exp="stgrid-${GRID_ID}-${name}"
    local jsonl="${LOG_DIR}/${exp}/${exp}.jsonl"
    local progress="${GRID_DIR}/${name}.progress"

    if [ "${DRY_RUN}" != "0" ]; then
        say "DRY_RUN ${name}: $(basename "${script}") $* exp_name=${exp}"
        return 0
    fi

    say "=== trial ${name} (exp ${exp}) ==="
    stop_ray_jobs
    reap_stale_vllm_cluster
    : > "${progress}"
    ( while sleep "${SAMPLE_EVERY_S}"; do
          echo "$(date +%s) $(wc -l < "${jsonl}" 2>/dev/null || echo 0)"
      done >> "${progress}" ) &
    local sampler=$!
    local t0 t1 rc attempt
    for attempt in 1 2; do
        t0=$(date +%s)
        timeout --kill-after=120 "${TRIAL_TIMEOUT_S}" \
            env "$@" \
                MODEL_PATH="${MERGED_MODEL_DIR}" \
                TOTAL_TRAINING_STEPS="${TRIAL_STEPS}" \
                TOTAL_EPOCHS="${TRIAL_EPOCHS}" VAL_BEFORE_TRAIN=False \
                TEST_FREQ=-1 SAVE_FREQ=-1 HPRL_DUMP_SELECTOR=0 \
                RESUME_FROM_PATH= \
                project_name='HPRL-staleness-grid' wandb_project='hint_rl-async-grid' \
                exp_name="${exp}" \
            bash "${script}" > "${GRID_DIR}/${name}.console.log" 2>&1
        rc=$?
        t1=$(date +%s)
        # Launch-race guard: a stale-vLLM EADDRINUSE killed st1.2 of the 20260722
        # sweep 3 min in, before any step ran. A nonzero exit inside 10 min is a
        # bring-up failure, not a result -- deep-clean and retry ONCE.
        if [ "${rc}" -ne 0 ] && [ $((t1 - t0)) -lt 600 ] && [ "${attempt}" -eq 1 ]; then
            say "trial ${name}: rc=${rc} after $((t1 - t0))s (launch race?); cleaning + retrying once"
            mv "${GRID_DIR}/${name}.console.log" "${GRID_DIR}/${name}.console.attempt1.log" 2>/dev/null
            stop_ray_jobs
            reap_stale_vllm_cluster
            sleep 10
            : > "${progress}"
            continue
        fi
        break
    done
    kill "${sampler}" 2>/dev/null
    wait "${sampler}" 2>/dev/null
    stop_ray_jobs
    sleep "${SETTLE_S}"

    local status
    case ${rc} in
        0)       status=done ;;
        124|137) status=timeout ;;
        *)       status="fail(${rc})" ;;
    esac
    local metrics
    metrics=$(trial_metrics "${jsonl}" "${progress}" "$((t1 - t0))")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${split}" "${stal}" "${part}" "${status}" "${metrics}" >> "${RESULTS}"
    say "trial ${name}: ${status}, $(echo "${metrics}" | cut -f1) versions" \
        "in $(echo "${metrics}" | cut -f2) min (vph_steady $(echo "${metrics}" | cut -f4))"
}

say "base ckpt : ${BASE_CKPT}"
say "merged -> : ${MERGED_MODEL_DIR}"
say "plan      : sync=${RUN_SYNC} (all ${NNODES} nodes train)  async staleness [${STALENESS_LIST}]" \
    "@ ${TRAINER_NNODES}:${ROLLOUT_NNODES} partial=${PARTIAL_ROLLOUT}"
say "trials    : ${TRIAL_STEPS} steps each (epoch ceiling ${TRIAL_EPOCHS}), ${TRIAL_TIMEOUT_S}s backstop; results -> ${RESULTS}"

ensure_merged_model

# --- trial 0: SYNC baseline (colocated; all NNODES nodes are trainer+rollout) -
if [ "${RUN_SYNC}" != "0" ]; then
    run_trial "sync" "sync:${NNODES}" "-" "-" "${SYNC_RUN_SCRIPT}" \
        NNODES="${NNODES}"
fi

# --- trials 1..K: ASYNC staleness sweep at the fixed 1:(N-1) split -----------
for stal in ${STALENESS_LIST}; do
    run_trial "st${stal}" "${TRAINER_NNODES}:${ROLLOUT_NNODES}" "${stal}" "${PARTIAL_ROLLOUT}" \
        "${ASYNC_RUN_SCRIPT}" \
        TRAINER_NNODES="${TRAINER_NNODES}" ROLLOUT_NNODES="${ROLLOUT_NNODES}" \
        STALENESS_THRESHOLD="${stal}" PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT}" \
        max_response_length="${QWEN3_MAX_RESPONSE_LENGTH}" \
        HPRL_OVERLONG_PENALTY_TYPE="${QWEN3_OVERLONG_TYPE}"
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
