#!/usr/bin/env bash
set -xeuo pipefail

# ---------------------------------------------------------------------------
# Hint Penalized RL (HPRL) -- multi-turn GRPO with the hint tool.
#
# This is the multi-turn tool-use sibling of run_grpo_qwen2.5_7b_npu.sh. The
# algorithm is unchanged (GRPO: group-std-normalized advantage, clip-higher, no
# KL). What is added is the agent loop + the stateful `request_hint` tool:
#
#   * rollout.mode=async + multi_turn.enable=True  -> the agent loop.
#   * multi_turn.tool_config_path=hint_tool_config.yaml  -> declares the
#     `request_hint` tool (hint_tool.HintTool), which routes each hint call to
#     a frozen selector model and records the applied hints as per-rollout state.
#   * per-row agent_name="tool_agent" (added by prepare_hint_data.py) routes
#     training prompts through the tool-calling loop. The val set (aime2024) has
#     NO agent_name, so validation stays single-turn / unaided.
#   * reward: hint_reward.compute_score (outcome reward * blank hint penalty),
#     with HintRewardManager merging the applied-hints state into extra_info.
#
# Prereqs:
#   1. Build the multi-turn parquet:   python prepare_hint_data.py
#   2. Serve the frozen selector model on an OpenAI-compatible endpoint and set
#      SELECTOR_BASE_URL / SELECTOR_MODEL below (default sglang @ :30000).
#
# All paths derive from this script's location; override any VAR via the env.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# this script lives at <HINT_RL_HOME>/script/hint_rl/<this>
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}

# Activate the verl conda environment
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"verl"}

# miniconda is not relocatable; symlink the original prefix to the current mount
# so baked-in absolute paths keep resolving when the tree is mounted elsewhere.
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-/share5/users/xutao.ma}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo mkdir -p "$(dirname "${CONDA_INSTALL_PREFIX}")"
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Load secrets (wandb key, ...) from .envrc.
if [ -f "${HINT_RL_HOME}/.envrc" ]; then
    source "${HINT_RL_HOME}/.envrc"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-${wandb_key:-}}"

# project_name / exp_name overridable via env so the auto-hint wrapper
# (run_auto_hint_qwen2.5_7b.sh) can label its runs distinctly.
project_name=${project_name:-'HPRL-Qwen2.5-7B-Instruct'}
exp_name=${exp_name:-"HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
wandb_project=${wandb_project:-"hint_rl"}

# ---- GRPO algorithm (identical to the plain GRPO run) ---------------------
adv_estimator=grpo
norm_adv_by_std_in_grpo=True
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

# ---- lengths: multi-turn trajectories accumulate tokens across turns ------
# Env-overridable so a per-model wrapper can shrink them: prompt+response is the
# vLLM max_model_len budget and must fit the model's max_position_embeddings.
# The async server pins max_model_len to max_position_embeddings and CLAMPS each
# request to it, so on a 32k-native model (Qwen3-8B-Base; Olmo3 is 64k yarn) the
# 2048+32768 default leaves a phantom 2048 the engine can never generate -- the
# qwen3 wrapper exports max_response_length=30720 to keep the budget real.
max_prompt_length=${max_prompt_length:-2048}
# bumped from 8192: prompt + N*(assistant turn + hint tool result).
max_response_length=${max_response_length:-32768}

# ---- multi-turn / hint knobs ----------------------------------------------
# Cap on assistant turns; must be >= the largest per-problem hint budget B_q.
max_turns=${max_turns:-10}
# Hint mechanism: the policy emits the sentinel <hint_call/> and the custom agent
# loop (hint_agent_loop.HintAgentLoop) detects it, calls the selector, and injects
# the hint as the next USER message. Registered via this agent-loop config; the
# hermes request_hint tool is NOT loaded (no tool schema in the prompt).
AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-"${SCRIPT_DIR}/hint_agent_config.yaml"}
# Hints can be a full sentence or two; allow more than the 256-token default.
max_tool_response_length=${max_tool_response_length:-2048}

# ---- frozen selector model (OpenAI-compatible endpoint) -------------------
# All of these are read by hint_selector.HintSelector.from_env() in the agent
# loop and forwarded into the Ray job below so the rollout workers can reach the
# selector. launch_hprl_cluster.sh sets SELECTOR_BASE_URL (the served gpt-oss-20b
# endpoint) and the call params; the defaults here let the script also run solo.
export SELECTOR_BASE_URL=${SELECTOR_BASE_URL:-"http://localhost:30000/v1"}
# Comma-separated list of INDEPENDENT selector endpoints (set by
# launch_hprl_cluster.sh). HintSelector load-balances + fails over across them;
# empty -> falls back to the single SELECTOR_BASE_URL above.
export SELECTOR_BASE_URLS=${SELECTOR_BASE_URLS:-""}
export SELECTOR_MODEL=${SELECTOR_MODEL:-"gpt-oss-20b"}
export SELECTOR_API_KEY=${SELECTOR_API_KEY:-"EMPTY"}
export SELECTOR_TEMPERATURE=${SELECTOR_TEMPERATURE:-0.7}
export SELECTOR_TOP_P=${SELECTOR_TOP_P:-1.0}
export SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-16000}
export SELECTOR_REQUEST_TIMEOUT_S=${SELECTOR_REQUEST_TIMEOUT_S:-600}
export SELECTOR_MAX_RETRIES=${SELECTOR_MAX_RETRIES:-3}
# ---- selector API mode: local vLLM endpoints (default) vs the REAL OpenAI API.
# SELECTOR_API_MODE=openai (set by launch_hprl_cluster_openai.sh, which also
# picks SELECTOR_MODEL) sends every hint call to SELECTOR_OPENAI_BASE_URL with
# OPENAI_API_KEY -- no selector pods at all. The SELECTOR_BASE_URL(S) pair above
# is ignored in that mode. Reasoning models (gpt-5*/o*) automatically switch to
# max_completion_tokens + SELECTOR_REASONING_EFFORT and drop temperature/top_p
# (the API rejects non-default values). SELECTOR_MAX_CONCURRENCY caps in-flight
# calls PER agent-loop worker process in openai mode only (rate-limit hygiene);
# local vLLM keeps the uncapped fan-out it wants for continuous batching.
export SELECTOR_API_MODE=${SELECTOR_API_MODE:-local}
export SELECTOR_OPENAI_BASE_URL=${SELECTOR_OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export SELECTOR_REASONING_EFFORT=${SELECTOR_REASONING_EFFORT:-low}
export SELECTOR_MAX_CONCURRENCY=${SELECTOR_MAX_CONCURRENCY:-16}
export OPENAI_API_KEY=${OPENAI_API_KEY:-}

# Cluster
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=64

# Ray
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${VERL_HOME}"}

# Paths
MODEL_PATH=${MODEL_PATH:-"${BASE_HOME}/model/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${HINT_RL_HOME}/ckpt/${project_name}/${exp_name}"}

# ---- AUTO-HINT (push-hint) mode -- ON BY DEFAULT --------------------------------
# HPRL_AUTO_HINT=true (default) makes this an auto-hint run: the policy runs the plain
# single-turn prompt and the LOOP injects a selector hint on a wrong answer
# (auto_hint_agent_loop.AutoHintAgentLoop, routed by the train parquet's
# agent_name="auto_hint"), with the trainer-side verified-prefix gradient mask
# (data.hprl.auto_hint.enable, passed to the job below). This block pins the matching
# train + (bare, prompt-matched) val files, the per-hint penalty, and DISABLES the
# <hint_call/>-specific reward terms (this mode emits no <hint_call/>). Each `:=` is a
# DEFAULT -- any value you export still wins. Set HPRL_AUTO_HINT=false to run the
# legacy <hint_call/> job instead (the hint_call defaults further below then apply).
# MUST precede the TRAIN_FILE / HINT_* / HPRL_KPACK_* defaults so these win.
HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
# Include all selector-verified progress and previously delivered hints before
# each new hint. On by default everywhere (the v2 recap injection eliminates the
# post-hint restart drift); set =false to restore the prior hint-only message.
HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
# Prune the X.0 step-guidance hints from each pool before it reaches the selector, so the
# rollout presents the SAME substep-only pools the offline selector eval (multi-cite-gpt-eval)
# was built and scored on (eval/train parity). Default off -> the full pool (X.0 included).
HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-false}
# STEP-LEVEL advantage (auto-hint only): replace GRPO's scalar advantage with the
# value-based per-segment one (step_advantage.py) and SKIP the verified-prefix mask.
# Default off (the mask runs). Set HPRL_STEP_ADV=true to switch the auto-hint job to it.
HPRL_STEP_ADV=${HPRL_STEP_ADV:-false}
# Uniform multiplier on the (small) value-based advantages -- raise to ~5-10 to match
# GRPO's gradient magnitude without retuning the LR. 1.0 == the raw step-adv formula.
# When HPRL_STEP_ADV_NORM=true, this is the TARGET std instead (1.0 -> unit).
HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
# GRPO-style per-group normalization: divide each group's advantages by their std so the
# (small) raw value-based advantages become ~unit scale adaptively -- the fix for a too-small
# gradient. true is recommended when enabling step-adv; false = the plain adv_scale multiply.
HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-false}
# Over-turn-length penalty (auto-hint + step-adv + per-turn cap). Subtract this ABSOLUTE
# value from every per-turn-cap truncation tail BEFORE the per-group normalize, so it does
# NOT ride the value-baseline brake a_I=penalty*(fc/d-1) that collapses to ~0 once a group
# co-truncates -- the reason the truncation penalty vanishes exactly when truncation is
# widespread. Raw units (~total_penalty/K); ~0.1 lands near unit scale after normalize.
# SCORED groups only (no-correct/all-truncate groups stay zeroed). 0 = off (default).
HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0}
# Overlong penalty ROUTING (auto-hint + step-adv; only matters when HPRL_OVERLONG_PENALTY > 0):
#   post_hoc (default) -- subtract P_over from each per-turn-cap truncation tail AFTER the
#     advantage is assigned (the original behavior); only the truncated rows move down.
#   value -- fold P_over into the VALUE recursion instead (truncated-at-k rollouts carry reward
#     r_k - P_over), lowering V[k] so every NON-truncated rollout anchored there is LIFTED to a
#     POSITIVE advantage -- a "do the within-length thing" reward, not just a "don't truncate"
#     push; the non-truncate<->truncate gap stays a co-truncation-proof P_over. Keep P_over small
#     (it also positively reinforces a concise-but-wrong first turn).
HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-post_hoc}
# WHOLE-TURN step-adv (auto-hint + step-adv). Score each turn with ONE advantage --
# A = V[s_end] + hint_penalty - V[s_start] (= r_se + V[se+1] - V[ss] for a failed turn,
# V[se] - V[ss] otherwise) -- instead of the verified-prefix a_C / failed-tail a_I split.
# Boundary-free (no fuzzy-quote dependence); the whole turn is scored together. Default
# false -> the a_C/a_I split. Only meaningful when HPRL_STEP_ADV=true.
HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-false}
# Per-turn generation-length cap (auto-hint only). When > 0, each assistant turn is
# bounded to min(this, remaining global room) tokens; a turn that hits the cap without an
# EOS (global budget remaining) is scored as failing its next hint step (whole turn = a_I,
# length_truncated -> acc=0) and the rollout terminates. 0 (default) -> off (global
# max_response_length only). Mind that max_response_length must cover ~(#turns x cap).
HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-0}
case "${HPRL_AUTO_HINT}" in
  true | True | 1 | yes | on)
    # hint-wise re-seeded initial budgets (budget_calibration/apply_budget_state.py from
    # budget_calibration/budget_state_hint_wise.json;
    # problems absent there keep their original baked budget). Override to dapo-3139-auto-hint.parquet
    # for the original (#major-steps) budgets.
    : "${TRAIN_FILE:=${HINT_RL_HOME}/dataset/dapo-3139-auto-hint-hintwise.parquet}"
    # bare (plain-prompt, no agent_name) eval sets -> prompt-matched unaided eval.
    : "${TEST_FILE:=${HINT_RL_HOME}/dataset/aime2024.parquet}"
    : "${HARD_TEST_FILE:=${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet}"
    : "${AIME2025_FILE:=${HINT_RL_HOME}/dataset/aime2025.parquet}"
    : "${HINT_STRATEGY:=hint}"           # per-hint penalty (selector gives one substep hint/round)
    : "${HINT_CALL_REWARD:=0.0}"         # no <hint_call/> emission -> no anti-suppression bonus
    : "${HINT_SHAPE_COEFF:=0.0}"         # no front-loading for the shape term to act on
    : "${NO_HINT_PENALTY_FACTOR:=0.0}"   # the LOOP, not the policy, decides hint availability
    : "${HINT_FINALIZE_INCORRECT:=false}"
    : "${HPRL_KPACK_ENABLE:=false}"      # k-pack off for auto-hint by default (set true to combine)
    ;;
esac

TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-3139-hint-verl-mt-clean.parquet"}
# Templated eval sets: aime2024-hint-mt / dapo_sample_hard_100-hint-mt carry the
# SAME hint-tool template as training (agent_name="hint_agent", full <hint_call/>
# system instruction + budget-0 user reminder), so validation is prompt-matched to
# training instead of running on the bare single-turn prompt. Budget is 0 -- these
# problems have no curated hint pool -- so the eval is unaided and the agent loop
# never contacts the selector (the len(applied)>=budget branch short-circuits).
# Built by prepare_eval_hint_template.py from dataset/{aime2024,dapo_sample_hard_100}
# .parquet (the originals are kept untouched for the non-HPRL baselines). Set
# TEST_FILE back to the bare .parquet for an out-of-template unaided eval.
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024-hint-mt.parquet"}
# Second held-out eval: 100 hard DAPO problems. Its problem_ids have ZERO overlap
# with the 3740 training rows (and with the ratchet table, so HintBudgetDataset
# re-renders it at the baked budget 0). Its data_source is "math_dapo", so verl
# reports it separately as val-core/math_dapo/* (aime2024 stays val-core/aime2024/*).
# verl evaluates each val file independently.
HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100-hint-mt.parquet"}
# Third held-out eval: AIME 2025 (data_source "aime2025", reported separately as
# val-core/aime2025/*). Templated (-hint-mt) here for the <hint_call/> job; the
# auto-hint block above pins the bare dataset/aime2025.parquet instead.
AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025-hint-mt.parquet"}
# All validation files as a hydra list (RLHFDataset concatenates them, each row
# keeping its own data_source for per-set metrics). Override VAL_FILES to change.
VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}']"}

# HPRL reward function (outcome reward minus the summed hint penalty).
REWARD_FN_PATH=${REWARD_FN_PATH:-"${SCRIPT_DIR}/hint_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}
# HPRL reward manager: merges per-rollout applied-hints state into extra_info.
REWARD_MGR_PATH=${REWARD_MGR_PATH:-"${SCRIPT_DIR}/hint_reward_manager.py"}
REWARD_MGR_CLASS=${REWARD_MGR_CLASS:-"HintRewardManager"}
# Hint-penalty knobs (hint_penalty.py). Passed as reward_kwargs so they retune
# WITHOUT regenerating the dataset: total penalty across all hints of a problem,
# the per-difficulty-level multiplier (harder = HARD_FACTOR x), and the difficulty
# assigned to the X.0 guidance hint.
HINT_PENALTY_TOTAL=${HINT_PENALTY_TOTAL:-1.0}
HINT_PENALTY_HARD_FACTOR=${HINT_PENALTY_HARD_FACTOR:-1.5}
HINT_GUIDANCE_DIFFICULTY=${HINT_GUIDANCE_DIFFICULTY:-easy}
# Make every X.0 GUIDANCE hint free (penalty 0): its weight is dropped and the step's
# penalty is borne entirely by the substep hints (the pool total stays HINT_PENALTY_TOTAL).
# Applies to the per-hint reward AND the step-adv r(h). false = the X.0 hint is priced
# normally (HINT_GUIDANCE_DIFFICULTY). (reward_kwargs.hint_guidance_free.)
HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-false}

# "No hint available" penalty: each pool-exhausted <hint_call/> (the loop had no
# candidate left to serve -- common terminal state of cumulative step-exclude) costs
# this factor x the MINIMUM major-step penalty in the pool (factor 0.1, min step
# penalty 0.2 -> 0.02 per call). Summed over the rollout's exhausted calls and
# subtracted from the CORRECT reward (shares the correct_floor). 0 disables it.
# (hint_reward.compute_score reward_kwargs.no_hint_penalty_factor.)
NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.1}

# ---- effort-shaping penalty (premature/shallow hint calls) ----------------
# Fixes the front-loading pathology: with a timing-blind hint penalty the policy
# reasons for a few tokens, calls a hint, repeats -- and only reasons hard once
# the budget is spent. This term (hint_reward.effort_shortfall_penalty) charges,
# per APPLIED hint, coeff * relu(mean_turn_len - pre_call_reasoning_len)/mean_turn_len,
# summed over calls -- i.e. a hint emitted after a near-empty turn (vs the rollout's
# own mean turn length) is penalized; one emitted after a real struggle is ~free.
# Subtracted from the CORRECT reward only (incorrect stays at -1). 0 disables it.
# Start small (~0.3) and watch rollouts for filler-padding before raising.
HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}

# ---- hint-call reward (anti-suppression bonus) ----------------------------
# One-off bonus ADDED to the INCORRECT reward when a failing rollout RECEIVED a
# hint at least once (correct rollouts are untouched -- it does not stack on a
# solve). BINARY: one applied hint earns it; extra hints add nothing, so it can't
# be spammed. Gates on an APPLIED hint, not a bare <hint_call/> emission -- a call
# the selector failed to serve conveyed nothing to the policy and would pay out
# during a selector outage. Counters the GRPO hint-suppression pathology (unhinted-
# correct out-ranks hinted-correct -> the policy stops calling hints) by keeping a
# positive gradient on hint use among the rollouts that fail anyway. 0 disables it.
# (hint_reward.compute_score reward_kwargs.hint_call_reward.)
HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}

# ---- hint-selection + penalty strategy ------------------------------------
# Selects HOW a hint call is answered and penalized. The same value is given to
# BOTH the agent loop (data.hprl.strategy -> HintAgentLoop) and the reward
# (reward_kwargs.hint_strategy -> hint_reward.compute_score); they MUST agree.
#   hint        : the selector reveals ONE hint within the major step it identifies;
#                 that single hint is excluded from the next call; the reward
#                 subtracts the per-hint difficulty weight (the default behavior).
#   major_step  : the selector identifies the major step and ALL of that step's
#                 hints are injected at once; the WHOLE step goes into the rollout
#                 state and is excluded from the next call; the reward subtracts the
#                 per-step penalty directly (hint_penalty.applied_step_penalty).
HINT_STRATEGY=${HINT_STRATEGY:-major_step}

# ---- major-step exclusion mode (selector prompt construction) -------------
# Only meaningful with HINT_STRATEGY=major_step. Chooses which already-revealed
# major steps are dropped from the selector's candidate pool before the next call
# (agent-loop side only -- no effect on the reward/penalty):
#   applied     : drop only the steps actually revealed so far (default).
#   cumulative  : drop EVERY step id <= the latest revealed step -- once step 5 is
#                 revealed, steps 1..5 are removed and only 6, 7, ... remain, forcing
#                 the selector strictly forward (no re-offering an earlier step).
HINT_STEP_EXCLUDE_MODE=${HINT_STEP_EXCLUDE_MODE:-cumulative}

# ---- finalize-incorrect (score wrong answers as correct-with-more-hints) ---
# When a rollout ends on an answer turn (no <hint_call/>), the agent loop grades it; if
# WRONG, it makes ONE final selector call to find the hint the student still needs, and
# the reward scores the rollout as correct_reward - penalty(used) - penalty(stuck-hint k
# .. last) - no_hint_penalty (acc stays 0). Given to BOTH the agent loop
# (data.hprl.finalize_incorrect) and the reward (reward_kwargs.finalize_incorrect); they
# MUST agree. Adds a blocking selector call per INCORRECT rollout. false -> off.
HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-true}

# The directory holding hint_tool.py must be importable by the rollout workers
# (the tool config's class_name "hint_tool.HintTool" is resolved with importlib).
TOOL_PYTHONPATH="${SCRIPT_DIR}"

# Local logs
RUN_ID=${RUN_ID:-"$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${HINT_RL_HOME}/logs"}
EXP_LOG_DIR=${EXP_LOG_DIR:-"${LOG_DIR}/${exp_name}"}
LOG_FILE=${LOG_FILE:-"${EXP_LOG_DIR}/${exp_name}.jsonl"}
CONSOLE_LOG=${CONSOLE_LOG:-"${EXP_LOG_DIR}/${exp_name}.${RUN_ID}.console.log"}
mkdir -p "${EXP_LOG_DIR}"
# Resume-in-place (exp_name pinned to a prior run) re-enters an existing exp
# dir, and verl's FileLogger opens VERL_FILE_LOGGER_PATH with mode "wb" --
# which would TRUNCATE the prior segment's metrics. Rotate it aside first
# (this script runs on the Ray head only, so no cross-pod race); cat the
# segments in stamp order to rebuild the full curve. Fresh runs never hit
# this: their stamped exp_name makes a brand-new dir.
if [ -f "${LOG_FILE}" ]; then
    mv "${LOG_FILE}" "${LOG_FILE%.jsonl}.pre-${RUN_ID}.jsonl"
fi

# ---- HPRL dynamic-budget ratchet (paper Section 7) ------------------------
# Master switch + per-experiment budget-state store (written by the trainer
# ratchet, read by the dynamic-budget dataset). HPRL_ENABLE=false -> ordinary
# multi-turn GRPO (HintBudgetDataset + HPRLRayPPOTrainer become no-ops).
HPRL_ENABLE=${HPRL_ENABLE:-True}
BUDGET_STATE_PATH=${BUDGET_STATE_PATH:-"${EXP_LOG_DIR}/budget_state.json"}
# ---- resume control (verl checkpoint) ------------------------------------
# SINGLE KNOB: set RESUME_FROM_PATH=<...>/global_step_N to fork/continue from a SPECIFIC
# checkpoint (verl reads N from the path and loads actor/ + data.pt = weights, optimizer,
# dataloader); RESUME_MODE then flips to resume_path automatically. Leave it null (default)
# for a fresh start -- RESUME_MODE=auto then resumes the LATEST ckpt in THIS run's own
# default_local_dir (none for a new exp_name, so it trains from the base model).
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
if [ "${RESUME_FROM_PATH}" != "null" ] && [ -n "${RESUME_FROM_PATH}" ]; then
    RESUME_MODE=${RESUME_MODE:-resume_path}
else
    RESUME_MODE=${RESUME_MODE:-auto}
fi
HPRL_MIN_BUDGET=${HPRL_MIN_BUDGET:-0}
HPRL_DECREMENT=${HPRL_DECREMENT:-1}
HPRL_DEFAULT_BUDGET=${HPRL_DEFAULT_BUDGET:-6}
# Single-pack ratchet rule: "downward" (strictly down; default) or "adaptive" (raise B_q
# by 1 when NO rollout is correct; set B_q to the N/2-th smallest correct hint count when
# OVER HALF are correct; else hold). See budget_manager.compute_adaptive_budget.
HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-downward}
# Ceiling on B_q (bounds the adaptive rule's upward branch). Defaults to the turn cap.
HPRL_MAX_BUDGET=${HPRL_MAX_BUDGET:-6}
# false -> ONE-SIDED (raise-only) ratchet: budget DECREASES are vetoed and held at the
# current B_q (BudgetUpdate.rule gains "_held"; wandb hprl/num_decrease_held counts them)
# while the adaptive raise-on-zero-correct branch keeps working. With ratchet_mode=downward
# or the k-pack probe (which only lower) false freezes budgets entirely. Default true ->
# the unchanged two-sided/downward behavior. See budget_manager.BudgetManager.
HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-true}

# k-pack counterfactual-probe ratchet ("double-rollout" / k-pack). OFF by default ->
# the single-pack downward ratchet runs unchanged. When on, EVERY problem's rollout.n
# rollouts are split into k packs of rollout.n/k, each forced to a different budget
# B,B-1,..,B-k+1 (each its own GRPO group); the ratchet pools their successes and snaps
# B to the smallest B' with >= require_successes correct rollouts at <= B' hints. The
# per-step rollout TOTAL is unchanged (rollout.n is divided by k internally). REQUIRES
# rollout.n divisible by k. See config/hprl_trainer.yaml.
HPRL_KPACK_ENABLE=${HPRL_KPACK_ENABLE:-false}
HPRL_KPACK_K=${HPRL_KPACK_K:-2}
HPRL_KPACK_REQUIRE_SUCCESSES=${HPRL_KPACK_REQUIRE_SUCCESSES:-2}
# scale ppo_mini_batch_size by k so the PPO mini-batch sample count is unchanged.
HPRL_KPACK_SCALE_MINI_BATCH=${HPRL_KPACK_SCALE_MINI_BATCH:-true}

# ---- budget-grouped data sampling (load-balance the generation step) ----------
# The stock uniform sampler mixes budget-0 (one unaided turn) and high-budget (many
# selector rounds) problems in one step; async multi-turn rollout ends a step on its
# SLOWEST rollout, so the high-budget problems straggle and the finished GPUs idle.
# When ON, each epoch orders problems by their CURRENT (ratcheted) budget B_q and packs
# same-budget problems into the same step -> a step's rollouts run ~equal hint rounds
# and finish together. Each batch stays exactly train_batch_size and the per-epoch
# problem set is unchanged (a random remainder < batch_size is dropped, as with the
# stock sampler + drop_last). ON by default for HPRL runs; set HPRL_BUDGET_SAMPLING=false
# for the stock uniform random sampler. See budget_sampler.BudgetGroupedSampler.
HPRL_BUDGET_SAMPLING=${HPRL_BUDGET_SAMPLING:-true}
# randomize the order the homogeneous batches run within an epoch (each batch stays
# homogeneous; only the step sequence is shuffled) so every step is still a random
# budget level. false -> ascending-budget order (a fixed easy->hard ramp per epoch).
HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER=${HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER:-true}
# NOTE: HPRL_AUTO_HINT (the push-hint mode, ON by default) + its coordinated
# train/val/reward defaults are set EARLY, in the "Paths" section above (they must
# precede the TRAIN_FILE / HINT_* / HPRL_KPACK_* defaults so they can win).

# Archive a verbatim snapshot of the WHOLE hint_rl script folder alongside the
# run's logs -- not just this launcher, but every HPRL source it pulls in (agent
# loop, reward fn, selector, penalty, budget manager, configs, ...). This makes
# each run reproducible from its own log dir even after the source tree changes.
# __pycache__ is excluded; the folder includes this script itself.
SRC_SNAPSHOT_DIR="${EXP_LOG_DIR}/hint_rl_src.${RUN_ID}"
mkdir -p "${SRC_SNAPSHOT_DIR}"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "${SCRIPT_DIR}/" "${SRC_SNAPSHOT_DIR}/"
else
    cp -r "${SCRIPT_DIR}/." "${SRC_SNAPSHOT_DIR}/"
    find "${SRC_SNAPSHOT_DIR}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "${SRC_SNAPSHOT_DIR}" -name '*.pyc' -delete 2>/dev/null || true
fi
echo "[run_hprl] archived hint_rl source snapshot -> ${SRC_SNAPSHOT_DIR}"

# Algorithm sampling
temperature=1.0
top_p=1.0
top_k=-1

# Performance
sp_size=4
use_dynamic_bsz=True
# --- update-phase speed (lever 4) ------------------------------------------------
# The step is ~86% generation and ~10% actor update; these two knobs trim the update
# (and the ref/rollout log-prob) passes. Both RAISE peak GPU memory during the update,
# so each is env-gated for an easy revert if the first step OOMs.
#
# (a) micro-batch packing: with use_dynamic_bsz the per-GPU token budget below caps how
#     many tokens are packed per forward/backward micro-batch. It was one full-length
#     sequence's worth ((prompt+resp)/sp_size); PPO_TOKEN_MULT packs that many sequences
#     per micro-batch -> ~1/MULT as many micro-batches. Gradient checkpointing bounds the
#     activation memory. Lower PPO_TOKEN_MULT (or set 1) if the update OOMs.
PPO_TOKEN_MULT=${PPO_TOKEN_MULT:-2}
actor_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$((PPO_TOKEN_MULT * (max_prompt_length + max_response_length) / sp_size))
# (b) offload: the actor params+optimizer are FSDP-sharded across all 32 GPUs (~a few GB
#     each), so offloading them to CPU every step costs a load/reload for no real memory
#     saving. vLLM sleeps during the update (async colocated mode), so keeping them
#     resident does not contend with the KV cache. OFFLOAD=True reverts to CPU offload.
offload=${OFFLOAD:-False}
gen_tp=1

# Per-run Ray runtime env. Env vars from this shell do NOT propagate to a
# submitted job, so wandb key, file-logger path, the selector endpoint, and the
# PYTHONPATH for the hint tool must all be injected here.
# DEBUG: dump EVERY hint-selection call (the exact `trace` build_trace produced
# -- the selector's whole view of student progress -- plus the candidate steps
# and the selector's parsed + raw output). Use it to verify on a LIVE run whether
# build_trace carries the student's reasoning or only the injected hints. Files
# land in <run dir>/selector_calls/. ON by default while we debug the blind-trace
# bug; set HPRL_DUMP_SELECTOR=0 to disable (it writes a lot of data over a full
# run, so turn it off once a step or two has been captured).
HPRL_DUMP_SELECTOR=${HPRL_DUMP_SELECTOR:-1}
HPRL_SELECTOR_DUMP_DIR=${HPRL_SELECTOR_DUMP_DIR:-""}
if [ "${HPRL_DUMP_SELECTOR}" != "0" ] && [ -z "${HPRL_SELECTOR_DUMP_DIR}" ]; then
  HPRL_SELECTOR_DUMP_DIR="${EXP_LOG_DIR}/selector_calls"
fi

# Optional HARD step bound (same convention as run_hprl_async.sh: 0/empty = off
# -> trainer.total_training_steps=null, so the total_epochs bound alone applies).
# Used by staleness_grid.sh to run short fixed-length benchmark trials.
if ! [ "${TOTAL_TRAINING_STEPS:-0}" -gt 0 ] 2>/dev/null; then
    TOTAL_TRAINING_STEPS=null
fi

# Egress-proxy forwarding (openai selector mode on a fabric that needs one):
# any proxy var set at submit time is forwarded verbatim into the workers'
# runtime env. Conditional lines -- injecting an EMPTY proxy var could confuse
# httpx/vLLM networking in the (default) proxy-less local mode.
SELECTOR_PROXY_ENV=""
for _pv in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [ -n "${!_pv:-}" ]; then
        SELECTOR_PROXY_ENV="${SELECTOR_PROXY_ENV}  ${_pv}: \"${!_pv}\""$'\n'
    fi
done

RUNTIME_ENV_RUN=${RUNTIME_ENV_RUN:-"${LOG_DIR}/.runtime_env.${RUN_ID}.yaml"}
( umask 077
  cat > "${RUNTIME_ENV_RUN}" <<EOF
working_dir: "${WORKING_DIR}"
excludes: ["/.git/"]
env_vars:
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  VLLM_USE_V1: "1"
  VERL_FILE_LOGGER_PATH: "${LOG_FILE}"
  WANDB_API_KEY: "${WANDB_API_KEY}"
  HINT_RL_HOME: "${HINT_RL_HOME}"
  SELECTOR_BASE_URL: "${SELECTOR_BASE_URL}"
  SELECTOR_BASE_URLS: "${SELECTOR_BASE_URLS}"
  SELECTOR_MODEL: "${SELECTOR_MODEL}"
  SELECTOR_API_KEY: "${SELECTOR_API_KEY}"
  SELECTOR_TEMPERATURE: "${SELECTOR_TEMPERATURE}"
  SELECTOR_TOP_P: "${SELECTOR_TOP_P}"
  SELECTOR_MAX_TOKENS: "${SELECTOR_MAX_TOKENS}"
  SELECTOR_REQUEST_TIMEOUT_S: "${SELECTOR_REQUEST_TIMEOUT_S}"
  SELECTOR_MAX_RETRIES: "${SELECTOR_MAX_RETRIES}"
  SELECTOR_API_MODE: "${SELECTOR_API_MODE}"
  SELECTOR_OPENAI_BASE_URL: "${SELECTOR_OPENAI_BASE_URL}"
  SELECTOR_REASONING_EFFORT: "${SELECTOR_REASONING_EFFORT}"
  SELECTOR_MAX_CONCURRENCY: "${SELECTOR_MAX_CONCURRENCY}"
  OPENAI_API_KEY: "${OPENAI_API_KEY}"
${SELECTOR_PROXY_ENV}  HPRL_SELECTOR_DUMP_DIR: "${HPRL_SELECTOR_DUMP_DIR}"
  # make hint_tool.py / hint_reward_manager.py importable in the job env.
  PYTHONPATH: "${TOOL_PYTHONPATH}"
# NOTE: do NOT add a runtime_env pip block on this air-gapped fabric. mathruler
# and openai (imported by custom_reward.py / the selector client) are already baked
# into the verl conda env, so a pip install here is redundant AND fatal: with no
# PyPI egress Ray hangs forever in "Runtime env is setting up" and never reaches
# main_hprl. Symptom: run v3-20260611-195503 stuck ~22 min at that line. If you
# ever need a new dep, install it into the env/image, not via runtime_env pip.
EOF
)

# Submit to the Ray cluster's job server. The cluster must already be up.
# HPRL entry: verl's TaskRunner/run_ppo with the trainer swapped for
# HPRLRayPPOTrainer (config = config/hprl_trainer.yaml = ppo_trainer + data.hprl).
# Launched by file path; cwd is the Ray working-dir (= verl repo root) so the
# config's `searchpath: file://verl/trainer/config` resolves the base config.
ray job submit --runtime-env="${RUNTIME_ENV_RUN}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 "${SCRIPT_DIR}/main_hprl.py" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.hprl.enable=${HPRL_ENABLE} \
    data.hprl.budget_state_path="${BUDGET_STATE_PATH}" \
    data.hprl.min_budget=${HPRL_MIN_BUDGET} \
    data.hprl.decrement=${HPRL_DECREMENT} \
    data.hprl.default_budget=${HPRL_DEFAULT_BUDGET} \
    data.hprl.ratchet_mode=${HPRL_RATCHET_MODE} \
    data.hprl.max_budget=${HPRL_MAX_BUDGET} \
    data.hprl.allow_decrease=${HPRL_ALLOW_DECREASE} \
    data.hprl.strategy=${HINT_STRATEGY} \
    data.hprl.step_exclude_mode=${HINT_STEP_EXCLUDE_MODE} \
    data.hprl.finalize_incorrect=${HINT_FINALIZE_INCORRECT} \
    data.hprl.kpack.enable=${HPRL_KPACK_ENABLE} \
    data.hprl.kpack.k=${HPRL_KPACK_K} \
    data.hprl.kpack.require_successes=${HPRL_KPACK_REQUIRE_SUCCESSES} \
    data.hprl.kpack.scale_mini_batch=${HPRL_KPACK_SCALE_MINI_BATCH} \
    data.hprl.budget_sampling.enable=${HPRL_BUDGET_SAMPLING} \
    data.hprl.budget_sampling.shuffle_batch_order=${HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER} \
    data.hprl.auto_hint.enable=${HPRL_AUTO_HINT} \
    data.hprl.auto_hint.fuzzy_threshold=${HPRL_AUTO_HINT_FUZZY} \
    data.hprl.auto_hint.progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} \
    data.hprl.auto_hint.prune_guidance=${HPRL_PRUNE_GUIDANCE} \
    data.hprl.auto_hint.max_turn_tokens=${HPRL_MAX_TURN_TOKENS} \
    data.hprl.auto_hint.step_adv.enable=${HPRL_STEP_ADV} \
    data.hprl.auto_hint.step_adv.adv_scale=${HPRL_STEP_ADV_SCALE} \
    data.hprl.auto_hint.step_adv.normalize=${HPRL_STEP_ADV_NORM} \
    data.hprl.auto_hint.step_adv.overlong_penalty=${HPRL_OVERLONG_PENALTY} \
    data.hprl.auto_hint.step_adv.overlong_penalty_type=${HPRL_OVERLONG_PENALTY_TYPE} \
    data.hprl.auto_hint.step_adv.whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${HPRL_LR_SCHEDULER:-constant} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager="${REWARD_MGR_CLASS}" \
    reward_model.reward_loop_source=importlib \
    reward_model.reward_loop_module_path="${REWARD_MGR_PATH}" \
    reward_model.reward_loop_class_name="${REWARD_MGR_CLASS}" \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    +custom_reward_function.reward_kwargs.correct_reward=1.0 \
    +custom_reward_function.reward_kwargs.incorrect_reward=0.0 \
    +custom_reward_function.reward_kwargs.format_reward=0.0 \
    +custom_reward_function.reward_kwargs.hint_call_reward=${HINT_CALL_REWARD} \
    +custom_reward_function.reward_kwargs.hint_penalty_total=${HINT_PENALTY_TOTAL} \
    +custom_reward_function.reward_kwargs.hint_penalty_hard_factor=${HINT_PENALTY_HARD_FACTOR} \
    +custom_reward_function.reward_kwargs.hint_guidance_difficulty=${HINT_GUIDANCE_DIFFICULTY} \
    +custom_reward_function.reward_kwargs.hint_guidance_free=${HINT_GUIDANCE_FREE} \
    +custom_reward_function.reward_kwargs.hint_strategy=${HINT_STRATEGY} \
    +custom_reward_function.reward_kwargs.hint_shape_coeff=${HINT_SHAPE_COEFF} \
    +custom_reward_function.reward_kwargs.no_hint_penalty_factor=${NO_HINT_PENALTY_FACTOR} \
    +custom_reward_function.reward_kwargs.finalize_incorrect=${HINT_FINALIZE_INCORRECT} \
    trainer.logger="['console','file','wandb']" \
    trainer.project_name="${wandb_project}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-20} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.max_actor_ckpt_to_keep=100 \
    trainer.total_epochs=${TOTAL_EPOCHS:-100} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${LOG_DIR}/${exp_name}/rollouts" \
    trainer.validation_data_dir="${LOG_DIR}/${exp_name}/val_rollouts" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.resume_from_path=${RESUME_FROM_PATH} \
    trainer.device=cuda \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    2>&1 | tee "${CONSOLE_LOG}"
