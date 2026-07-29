#!/usr/bin/env bash
# =============================================================================
# run_openai_multi_eval.sh  --  score CLOSED OpenAI models on the MULTI-ROUND
# hint-selection + citation task, using the SAME prompt + benchmark as
# gpt_oss_eval/multi-cite-gpt-eval/, reporting the SAME _summary.json stats.
# -----------------------------------------------------------------------------
# No server to launch -- we hit api.openai.com directly. Reuses the static
# benchmark.jsonl already built under multi-cite-gpt-eval/ (apples-to-apples
# with the gpt-oss reference run). Loops the model list, one
# multi_results/multi__<model>__<ts>/ per model, then prints a compare table.
#
#   MODELS="gpt-4o-mini gpt-4.1-mini" N_SAMPLES=16 LIMIT=5 bash run_openai_multi_eval.sh
# =============================================================================
set -eo pipefail

SRC="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"          # .../test_cite/openai-models
TEST_CITE="$(cd "$SCRIPT_DIR/.." && pwd)"
GPT_OSS="$TEST_CITE/gpt_oss_eval"
MULTI="$GPT_OSS/multi-cite-gpt-eval"
SELECTOR="$(cd "$TEST_CITE/.." && pwd)"
HINT_RL_HOME="$(cd "$SELECTOR/.." && pwd)"
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"

# --- config (override via env) -----------------------------------------------
MODELS="${MODELS:-gpt-5.4-mini, gpt-5.4-nano, gpt-5-mini}"
MODELS="${MODELS//,/ }"                  # tolerate comma-separated lists ("a, b" -> "a b")
BENCHMARK="${BENCHMARK:-$MULTI/benchmark.jsonl}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.3}"       # ignored for reasoning models (gpt-5*)
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
ROW_WORKERS="${ROW_WORKERS:-8}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-4}"
STEPS="${STEPS:-}"
LIMIT="${LIMIT:-}"
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"
# =============================================================================

PYTHON_BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
[ -x "$PYTHON_BIN" ] || { echo "FATAL: env python not found: $PYTHON_BIN (set CONDA_ROOT/CONDA_ENV)" >&2; exit 1; }
[ -f "$BENCHMARK" ] || { echo "FATAL: benchmark not found: $BENCHMARK (run multi-cite-gpt-eval/build_benchmark.py)" >&2; exit 1; }

if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$SCRIPT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env.sh"
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "FATAL: OPENAI_API_KEY unset (put it in $SCRIPT_DIR/env.sh)" >&2; exit 1; }

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"

echo "=================================================================="
echo " run_openai_multi_eval  (multi-round; closed OpenAI models)"
echo "   models     : $MODELS"
echo "   benchmark  : $BENCHMARK"
echo "   sampling   : n=$N_SAMPLES temp=$TEMPERATURE max_tokens=$MAX_TOKENS"
echo "   reasoning  : effort=$REASONING_EFFORT (gpt-5*/o* only)"
echo "   prompt     : ${PROMPT_FILE:-$SCRIPT_DIR/prompt_template_multiF.py (local default)}"
echo "   parser     : ${PARSER_FILE:-driver default (run_hint_selection_model)}"
echo "   steps/limit: '${STEPS:-all}' / '${LIMIT:-all}'"
echo "=================================================================="

export OPENAI_API_KEY BENCHMARK N_SAMPLES TEMPERATURE TOP_P MAX_TOKENS \
       REASONING_EFFORT ROW_WORKERS SAMPLE_WORKERS STEPS LIMIT PROMPT_FILE PARSER_FILE

# PARALLEL_MODELS=1 (default): all models run concurrently, each to its own log
# (their stdout would interleave, so we don't tee to the console -- tail the logs).
# PARALLEL_MODELS=0: one model at a time, streamed to the console.
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
overall_rc=0
declare -A MODEL_PID

if [ "$PARALLEL_MODELS" = "1" ]; then
  echo ""
  echo "[run] launching all models in PARALLEL (row_workers=$ROW_WORKERS, "
  echo "      sample_workers=$SAMPLE_WORKERS => up to N_models*$((ROW_WORKERS*SAMPLE_WORKERS)) in-flight calls)."
  echo "      Watch progress with:  tail -f $LOG_DIR/multi_<model>.log"
  for MODEL in $MODELS; do
    MODEL_LOG="$LOG_DIR/multi_${MODEL//\//_}.log"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_openai_multi_selection.py" --model "$MODEL" > "$MODEL_LOG" 2>&1 &
    MODEL_PID[$MODEL]=$!
    echo "  $MODEL -> pid ${MODEL_PID[$MODEL]}   ($MODEL_LOG)"
  done
  for MODEL in $MODELS; do
    if wait "${MODEL_PID[$MODEL]}"; then
      echo "[run] $MODEL done (log: $LOG_DIR/multi_${MODEL//\//_}.log)"
    else
      echo "[run] WARN: $MODEL failed; see $LOG_DIR/multi_${MODEL//\//_}.log" >&2
      overall_rc=1
    fi
  done
else
  for MODEL in $MODELS; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[run] scoring model: $MODEL"
    echo "------------------------------------------------------------------"
    MODEL_LOG="$LOG_DIR/multi_${MODEL//\//_}.log"
    if "$PYTHON_BIN" "$SCRIPT_DIR/run_openai_multi_selection.py" --model "$MODEL" 2>&1 | tee "$MODEL_LOG"; then
      echo "[run] $MODEL done (log: $MODEL_LOG)"
    else
      rc=${PIPESTATUS[0]}
      echo "[run] WARN: $MODEL failed (rc=$rc); continuing. Log: $MODEL_LOG" >&2
      overall_rc=1
    fi
  done
fi

echo ""
echo "[run] all models done; building comparison table."
"$PYTHON_BIN" "$SCRIPT_DIR/compare_multi_runs.py" || true

exit $overall_rc
