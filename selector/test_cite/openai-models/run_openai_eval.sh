#!/usr/bin/env bash
# =============================================================================
# run_openai_eval.sh  --  score CLOSED OpenAI models on the Template F
# hint-selection + citation task, against the same Codex labels as gpt_oss_eval.
# -----------------------------------------------------------------------------
# No server to launch (unlike gpt_oss_eval) -- we hit api.openai.com directly.
# This just loops the model list and calls run_openai_selection.py per model,
# writing one results/<label>__<model>__<ts>/ per model, then prints a compare
# table across the runs it produced.
#
#   1. resolves paths relative to THIS file (works on any mount);
#   2. picks the verl conda env's python (has openai + tqdm), conda-free;
#   3. sources env.sh for OPENAI_API_KEY (or inherit it from the environment);
#   4. runs each model in MODELS sequentially (rows/samples are concurrent);
#   5. compares the produced summaries.
#
# Override any knob via env var, e.g.:
#   MODELS="gpt-4o-mini gpt-4.1-mini"  N_SAMPLES=16  LIMIT=5  bash run_openai_eval.sh
# =============================================================================
set -eo pipefail

SRC="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd)"          # .../test_cite/openai-models
TEST_CITE="$(cd "$SCRIPT_DIR/.." && pwd)"
SELECTOR="$(cd "$TEST_CITE/.." && pwd)"
HINT_RL_HOME="$(cd "$SELECTOR/.." && pwd)"
PROJECT_HOME="$(cd "$HINT_RL_HOME/.." && pwd)"
BASE_HOME="$(cd "$PROJECT_HOME/.." && pwd)"            # == /share5/users/xutao.ma

# --- config (override via env) -----------------------------------------------
# de-duped from the request (gpt-4.1-nano was listed twice).
MODELS="${MODELS:-gpt-5-nano gpt-4.1-nano gpt-4.1-mini gpt-4o-mini}"
MODELS="${MODELS//,/ }"                  # tolerate comma-separated lists ("a, b" -> "a b")
LABEL_RUN="${LABEL_RUN:-run_20260617_224209}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.3}"       # ignored for reasoning models (gpt-5*)
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"   # gpt-5*/o* only
ROW_WORKERS="${ROW_WORKERS:-8}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-4}"
STEPS="${STEPS:-}"                      # e.g. "1,2" ; empty = all
LIMIT="${LIMIT:-}"                      # e.g. "5" smoke ; empty = all 100/step
PROMPT_SOURCE="${PROMPT_SOURCE:-local}"
CONDA_ROOT="${CONDA_ROOT:-$BASE_HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl}"
# =============================================================================

# --- python from the verl env (conda-free) -----------------------------------
PYTHON_BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
[ -x "$PYTHON_BIN" ] || { echo "FATAL: env python not found: $PYTHON_BIN (set CONDA_ROOT/CONDA_ENV)" >&2; exit 1; }

# --- API key: env.sh (if present) else inherited environment ------------------
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$SCRIPT_DIR/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env.sh"
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "FATAL: OPENAI_API_KEY unset (put it in $SCRIPT_DIR/env.sh)" >&2; exit 1; }

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"

echo "=================================================================="
echo " run_openai_eval  (closed OpenAI models, api.openai.com)"
echo "   models     : $MODELS"
echo "   label run  : $LABEL_RUN   (n=$N_SAMPLES temp=$TEMPERATURE max_tokens=$MAX_TOKENS)"
echo "   reasoning  : effort=$REASONING_EFFORT (gpt-5*/o* only)"
echo "   prompt src : $PROMPT_SOURCE  ($TEST_CITE/gpt_oss_eval/prompt_template_F.py)"
echo "   steps/limit: '${STEPS:-all}' / '${LIMIT:-all}'"
echo "   python     : $PYTHON_BIN"
echo "=================================================================="

export OPENAI_API_KEY LABEL_RUN N_SAMPLES TEMPERATURE TOP_P MAX_TOKENS \
       REASONING_EFFORT ROW_WORKERS SAMPLE_WORKERS STEPS LIMIT PROMPT_SOURCE

overall_rc=0
for MODEL in $MODELS; do
  echo ""
  echo "------------------------------------------------------------------"
  echo "[run] scoring model: $MODEL"
  echo "------------------------------------------------------------------"
  MODEL_LOG="$LOG_DIR/${MODEL//\//_}.log"
  if "$PYTHON_BIN" "$SCRIPT_DIR/run_openai_selection.py" --model "$MODEL" 2>&1 | tee "$MODEL_LOG"; then
    echo "[run] $MODEL done (log: $MODEL_LOG)"
  else
    rc=${PIPESTATUS[0]}
    echo "[run] WARN: $MODEL failed (rc=$rc); continuing. Log: $MODEL_LOG" >&2
    overall_rc=1
  fi
done

echo ""
echo "[run] all models done; building comparison table."
"$PYTHON_BIN" "$SCRIPT_DIR/compare_runs.py" || true

exit $overall_rc
