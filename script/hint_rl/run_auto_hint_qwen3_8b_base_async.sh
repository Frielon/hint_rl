#!/usr/bin/env bash
#
# AUTO-HINT (push-hint) HPRL run for Qwen3-8B-Base -- FULLY-ASYNC edition.
#
# The async sibling of run_auto_hint_qwen3_8b_base.sh, modeled on
# run_auto_hint_olmo3_7b_instruct_async.sh: every auto-hint SCIENCE knob below
# is IDENTICAL to the sync qwen3 wrapper (same data, reward terms, step-level
# advantage config incl. its value-routed overlong penalty, ratchet policy);
# what changes is the architecture underneath -- run_hprl_async.sh splits the
# training pods into a streaming ROLLOUT pool and a TRAINER pool that overlap
# in time (verl fully_async_policy), instead of the colocated synchronous
# trainer gated on its slowest rollout each step.
#
# Async knob defaults = the 20260722 staleness sweep's production pick:
#   1 trainer : N-1 rollout nodes at staleness_threshold=2 -- ~22 versions/hour
#   steady on 1:4, ~94% of the ~24 vph trainer-compute ceiling (~152 s/version),
#   vs the 4-node sync baseline's 9.3 vph. The 2..3 sweep is still probing the
#   remaining tail; bump STALENESS_THRESHOLD if it moves the knee.
#
# Qwen3-8B-Base mechanical deltas (full derivations in the sync wrapper header):
#   1. eos -- the base checkpoint stops only on <|endoftext|> (151643) while its
#      chat template closes turns with <|im_end|> (151645); unfixed, every turn
#      runs to the turn cap and the loop scores it length_truncated. Fix: the
#      same idempotent eos-derived-dir build as the sync wrapper (below).
#   2. context -- 32768-native (no rope scaling): export max_response_length=
#      30720 so prompt(2048)+response = exactly 32768. verl's async server
#      pins vLLM max_model_len to max_position_embeddings (vllm_async_server
#      .py:818), so a response budget above 30720 is FATAL, not just wasteful:
#      run ...-20260723-215956 launched with 34768 and died at v17 the moment
#      one rollout's accumulated context crossed 32768 (vLLM "no room to
#      generate" ValueError kills the whole Rollouter and the job reports
#      SUCCEEDED). Keep prompt+response <= 32768 on this model.
#
# START / RESUME:
#   default            RESUME the 20260731-020431 run from its global_step_200
#                      ckpt: ${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen3-8B-Base-
#                      async/HPRL-AutoHint-Qwen3-8B-Base-async-dolci-rl-zero-
#                      10324-20260731-020431/global_step_200 (HOME-relative so
#                      the same default resolves on the dev box and on the
#                      pods' project mount). trainer.resume_mode=resume_path
#                      restores weights + optimizer + dataloader AND the step
#                      counter: both async actors parse the 200 out of the dir
#                      name, so param version / sync step CONTINUE AT 200 and
#                      new ckpts land in this launch's own freshly-stamped exp
#                      dir (the source run's global_step_300..700 are never
#                      touched). The ckpt fits this file's defaults: its actor
#                      shards are world 8 = TRAINER_NNODES=1, and TRAIN_FILE
#                      already IS that run's dataset -- keep both in sync if
#                      you override either. The eos-derived-dir build below
#                      still runs: MODEL_PATH must stay the -hprl dir
#                      (tokenizer + vLLM init read it; the ckpt supplies only
#                      the weights). NOTE steps 0..200 were trained under
#                      bypass_mode=True + clip_ratio_low=0.8; the current
#                      defaults continue on the decoupled corrector
#                      (ROLLOUT_CORR_BYPASS=False, clip 0.2) -- a deliberate
#                      regime change at the resume boundary. Budget ratchet
#                      state is NOT part of the ckpt -- see "Budget state"
#                      below.
#   RESUME_FROM_PATH=  exported EMPTY -> the previous FRESH-run behavior:
#                      pristine base model via the idempotent eos-derived dir
#                      build (per-file symlinks + the patched
#                      generation_config.json with eos [151643,151645]),
#                      trainer.resume_mode=auto -- which still picks up THIS
#                      experiment's own latest ckpt if the same exp_name is
#                      relaunched.
#   RESUME_FROM_PATH=<other .../global_step_N dir>  resume that ckpt instead.
#                      Constraints: (a) the TRAINER pool must match the
#                      ckpt's saved actor world size -- the sync qwen3 run's
#                      shards are world 32 (4 nodes x 8), so resuming them
#                      needs TRAINER_NNODES=4 (which starves rollout on a
#                      5-pod cluster); (b) TRAIN_FILE must match the ckpt's
#                      dataset (the restored dataloader requires it; and a
#                      SYNC-made dataloader state restarts the epoch stream --
#                      sync steps batches, async streams single samples);
#                      (c) a resumed async run loses queue-in-flight samples
#                      (upstream limitation, logged by the Rollouter).
#                      To continue a ckpt's POLICY on a mismatched world size
#                      instead, merge it to HF format and pass it as
#                      MODEL_PATH (fresh optimizer) -- e.g. the already-merged
#                      eval/merged/hprl-qwen3-base-512-step320 is how the
#                      staleness grid starts its trials from step 320.
#   MODEL_PATH=...     you own the eos fix (the derived-dir build is skipped).
#
# Budget state: default is a FRESH budget_state.json (parquet seeds; the
# raise-only adaptive ratchet re-fits online). The ckpt resume above does NOT
# restore it: the ratchet lives outside global_step_N, and the source run kept
# no step-200 snapshot -- only its end-of-run (~step 700) logs/<exp>/
# budget_state.json survives, whose raise-only budgets are >= the step-200
# ones. To carry a prior run's budgets anyway, point BUDGET_STATE_PATH at a
# COPY of its budget_state.json (the trainer writes to it -- never share the
# original).
#
# NOT available in async mode: HPRL_KPACK_ENABLE (hard error in the base
# script) and HPRL_BUDGET_SAMPLING (forced off -- the streaming rollout pool
# has no generation batch to budget-group; the queue absorbs the variance).
#
# Cluster launch (OpenAI selector mode -- every pod trains, no selector pods):
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster_openai_async.sh
# or behind the classic selector-pod launcher:
#   TRAIN_SCRIPT=<this file> bash launch_hprl_cluster.sh
#
# First-launch checklist (this is the cluster smoke for the 2026-07-23
# extra_info-restore fix): the rollouter prints "[HPRL ASYNC] restoring N
# dataset non-tensor keys" once, step-1 logs show step-adv scored>0 /
# zeroed<64, nonzero actor/pg_loss, and the hprl/* metric block is present.
#
# Override any knob via env, e.g.:
#   STALENESS_THRESHOLD=2.4 TRAINER_NNODES=1 bash run_auto_hint_qwen3_8b_base_async.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
BASE_HOME=${BASE_HOME:-"$(cd "${HINT_RL_HOME}/../.." && pwd)"}

# --- base model: Qwen3-8B-Base via the derived -hprl dir (see header pt. 1) ---
# Verbatim from run_auto_hint_qwen3_8b_base.sh; skipped entirely when
# MODEL_PATH is already set (you then own the eos fix). The kept
# max_new_tokens=2048 is pristine-parity and harmless in training: verl sets
# explicit per-request budgets; only bare `vllm serve` reads it.
if [ -z "${MODEL_PATH:-}" ]; then
    QWEN3_SRC_DIR=${QWEN3_SRC_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base"}
    QWEN3_HPRL_DIR=${QWEN3_HPRL_DIR:-"${BASE_HOME}/model/Qwen3-8B-Base-hprl"}
    if [ ! -f "${QWEN3_SRC_DIR}/config.json" ]; then
        echo "[run_auto_hint_qwen3_async] ERROR: pristine model dir not found: ${QWEN3_SRC_DIR}" >&2
        exit 1
    fi
    mkdir -p "${QWEN3_HPRL_DIR}"
    # Symlink every pristine file except generation_config.json; re-run safe
    # (links already pointing at the right target are left untouched), and a
    # file added upstream later gets linked on the next launch.
    for f in "${QWEN3_SRC_DIR}"/*; do
        b="$(basename "${f}")"
        if [ "${b}" = "generation_config.json" ]; then
            continue
        fi
        if [ "$(readlink "${QWEN3_HPRL_DIR}/${b}" 2>/dev/null || true)" != "${f}" ]; then
            ln -sfn "${f}" "${QWEN3_HPRL_DIR}/${b}"
        fi
    done
    # Pristine file + eos_token_id widened to [<|endoftext|>, <|im_end|>].
    cat > "${QWEN3_HPRL_DIR}/generation_config.json.tmp" <<'EOF'
{
  "bos_token_id": 151643,
  "do_sample": false,
  "eos_token_id": [151643, 151645],
  "max_new_tokens": 2048,
  "transformers_version": "4.37.0"
}
EOF
    mv -f "${QWEN3_HPRL_DIR}/generation_config.json.tmp" "${QWEN3_HPRL_DIR}/generation_config.json"
    export MODEL_PATH="${QWEN3_HPRL_DIR}"
fi

# --- context budget: fit the model's native 32768 (header delta 2) -------------
export max_response_length=${max_response_length:-30720}

# --- async pipeline knobs: the staleness-sweep production pick -----------------
# NNODES is exported by the cluster launcher (total training pods); 5 is the
# base script's default for a bare head-node launch.
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-5} - TRAINER_NNODES ))}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-1.9}
export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
# DECOUPLED off-policy correction (NOT the stock bypass mode): recompute
# old_log_probs on the trainer as an on-policy proximal anchor + seq-TIS
# weights vs the rollout logprobs. Bypass mode's clip-vs-stale-policy ratio
# is a one-way rectifier for rare suppressed tokens (negative-advantage grad
# clips to zero, positive side stays live) -- the 'assistant'-tic resurgence
# that collapsed the 0723/0724/0729 dolci async runs while their sync twins
# held it extinct (devlog 2026-07-30). Watch rollout_corr/* + actor/entropy
# (both absent under bypass) and the bare-'assistant' rollout fraction.
export ROLLOUT_CORR_BYPASS=${ROLLOUT_CORR_BYPASS:-True}
# Widened lower PPO clip: bound 1-0.95=0.05, so negative-advantage gradients
# on already-suppressed tokens stay live instead of clipping to zero (the
# rectifier above). Base script default is 0.2 (bound 0.8).
export clip_ratio_low=${clip_ratio_low:-0.2}

# --- PPO update geometry + optimizer: defaults = the shared run_hprl_async.sh
#     values, surfaced here for per-run sweeps. rollout.n is the GRPO group
#     size (the budget ratchet's N/2-th-smallest fallback and the step-adv
#     value fit read the group -- change with care); train_prompt_mini_bsz
#     prompts = one optimizer step = one param version at trigger=1 (changing
#     it changes what a "step" means vs the resumed s200 numbering). ---
export actor_lr=${actor_lr:-1e-6}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-64}
export n_resp_per_prompt=${n_resp_per_prompt:-16}

# --- turn on the auto-hint mechanism (identical to the sync wrapper) -----------
export HPRL_AUTO_HINT=${HPRL_AUTO_HINT:-true}
export HPRL_AUTO_HINT_FUZZY=${HPRL_AUTO_HINT_FUZZY:-0.8}
export HPRL_AUTO_HINT_PROGRESS_MESSAGE=${HPRL_AUTO_HINT_PROGRESS_MESSAGE:-true}
# Prune X.0 step-guidance hints from the pool before the selector sees it
# (eval/train parity with the offline selector eval).
export HPRL_PRUNE_GUIDANCE=${HPRL_PRUNE_GUIDANCE:-true}

# --- train parquet: dapo-512 with hint-wise calibrated initial budgets (seeds
#     only for this model -- the raise-only ratchet re-fits online). Use
#     dataset/dapo-3139-auto-hint.parquet for the full pool. ---
export TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dolci-rl-zero-10324-auto-hint.parquet"}
# --- trainer resume: DEFAULT = the 20260731-020431 run's global_step_200;
#     weights+optimizer+dataloader restore there and the step counter
#     continues at 200 (header "START / RESUME"). ${VAR-...} (not :-) on
#     purpose: export RESUME_FROM_PATH= (empty) to opt back into a fresh run;
#     any other .../global_step_N obeys the header's world-size / dataset
#     constraints -- or merge the ckpt and pass MODEL_PATH instead. ---
QWEN3_ASYNC_S200_CKPT="${HINT_RL_HOME}/ckpt/HPRL-AutoHint-Qwen3-8B-Base-async/HPRL-AutoHint-Qwen3-8B-Base-async-dolci-rl-zero-10324-20260731-020431/global_step_200"
export RESUME_FROM_PATH=${RESUME_FROM_PATH-}
if [ -n "${RESUME_FROM_PATH}" ] && [ ! -d "${RESUME_FROM_PATH}/actor" ]; then
    echo "[run_auto_hint_qwen3_async] ERROR: RESUME_FROM_PATH has no actor/ ckpt: ${RESUME_FROM_PATH}" >&2
    exit 1
fi

# --- validation: the BARE (plain-prompt, no agent_name) eval sets, matched to
#     auto-hint training. Runs on the ROLLOUT pool every trainer.test_freq
#     param versions. ---
export TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024.parquet"}
export HARD_TEST_FILE=${HARD_TEST_FILE:-"${HINT_RL_HOME}/dataset/dapo_sample_hard_100.parquet"}
export AIME2025_FILE=${AIME2025_FILE:-"${HINT_RL_HOME}/dataset/aime2025.parquet"}
export HMMT_FILE=${HMMT_FILE:-"${HINT_RL_HOME}/dataset/hmmt_nov_2025.parquet"}
export VAL_FILES=${VAL_FILES:-"['${TEST_FILE}','${HARD_TEST_FILE}','${AIME2025_FILE}','${HMMT_FILE}']"}

# --- reward: per-hint penalty; the <hint_call/>-specific terms disabled --------
export HINT_STRATEGY=${HINT_STRATEGY:-hint}
export HINT_CALL_REWARD=${HINT_CALL_REWARD:-0.0}
export HINT_SHAPE_COEFF=${HINT_SHAPE_COEFF:-0.0}
export NO_HINT_PENALTY_FACTOR=${NO_HINT_PENALTY_FACTOR:-0.0}
export HINT_FINALIZE_INCORRECT=${HINT_FINALIZE_INCORRECT:-false}
# Free X.0 guidance hint (weight dropped, step penalty borne by the substeps).
export HINT_GUIDANCE_FREE=${HINT_GUIDANCE_FREE:-true}

# --- STEP-LEVEL advantage (identical to the sync wrapper) ----------------------
export HPRL_STEP_ADV=${HPRL_STEP_ADV:-true}
export HPRL_STEP_ADV_SCALE=${HPRL_STEP_ADV_SCALE:-1.0}
export HPRL_STEP_ADV_NORM=${HPRL_STEP_ADV_NORM:-true}
export HPRL_OVERLONG_PENALTY=${HPRL_OVERLONG_PENALTY:-0.1}
# value = fold P_over into the value recursion (the qwen3/sync routing; the
# base async script's own default is post_hoc, the olmo wrapper's choice).
export HPRL_OVERLONG_PENALTY_TYPE=${HPRL_OVERLONG_PENALTY_TYPE:-value}
export HPRL_STEP_ADV_WHOLE_TURN=${HPRL_STEP_ADV_WHOLE_TURN:-true}

# --- actor LR schedule: CONSTANT (no decay), as in the sync wrapper ------------
export HPRL_LR_SCHEDULER=${HPRL_LR_SCHEDULER:-constant}

# --- PER-TURN generation-length cap (identical to the sync wrapper); with the
#     eos fix this is the backstop against base-model rambling, not the norm. ---
export HPRL_MAX_TURN_TOKENS=${HPRL_MAX_TURN_TOKENS:-8192}

# --- k-pack / budget-grouped sampling: architecture-inherently sync-only;
#     pinned off (the base script errors/forces respectively). ---
export HPRL_KPACK_ENABLE=false
export HPRL_BUDGET_SAMPLING=false

# --- budget ratchet: ADAPTIVE, RAISE-ONLY (identical to the sync wrapper) ------
export HPRL_RATCHET_MODE=${HPRL_RATCHET_MODE:-adaptive}
export HPRL_ALLOW_DECREASE=${HPRL_ALLOW_DECREASE:-false}

# --- distinct run labels in wandb ----------------------------------------------
export project_name=${project_name:-'HPRL-AutoHint-Qwen3-8B-Base-async'}
export exp_name=${exp_name:-"HPRL-AutoHint-Qwen3-8B-Base-async-dolci-rl-zero-10324-r16-$(TZ='America/Los_Angeles' date +%Y%m%d-%H%M%S)"}

echo "[run_auto_hint_qwen3_async] auto-hint FULLY-ASYNC mode ON  (model=${MODEL_PATH})"
echo "[run_auto_hint_qwen3_async]   base-model deltas: eos=[151643,151645] via ${MODEL_PATH##*/}, max_response_length=${max_response_length}"
echo "[run_auto_hint_qwen3_async]   TRAIN_FILE=${TRAIN_FILE}"
echo "[run_auto_hint_qwen3_async]   resume_from=${RESUME_FROM_PATH:-<fresh>}"
echo "[run_auto_hint_qwen3_async]   split trainer:rollout=${TRAINER_NNODES}:${ROLLOUT_NNODES} nodes  staleness=${STALENESS_THRESHOLD} trigger=${TRIGGER_PARAMETER_SYNC_STEP} require=${REQUIRE_BATCHES} partial_rollout=${PARTIAL_ROLLOUT} rollout_corr_bypass=${ROLLOUT_CORR_BYPASS} clip_ratio_low=${clip_ratio_low}"
echo "[run_auto_hint_qwen3_async]   lr=${actor_lr} mini_bsz=${train_prompt_mini_bsz} rollout_n=${n_resp_per_prompt}"
echo "[run_auto_hint_qwen3_async]   strategy=${HINT_STRATEGY} call_reward=${HINT_CALL_REWARD} shape=${HINT_SHAPE_COEFF} ratchet=${HPRL_RATCHET_MODE} allow_decrease=${HPRL_ALLOW_DECREASE}"
echo "[run_auto_hint_qwen3_async]   step_adv=${HPRL_STEP_ADV} step_adv_scale=${HPRL_STEP_ADV_SCALE} step_adv_norm=${HPRL_STEP_ADV_NORM} whole_turn=${HPRL_STEP_ADV_WHOLE_TURN} overlong_penalty=${HPRL_OVERLONG_PENALTY} overlong_type=${HPRL_OVERLONG_PENALTY_TYPE} guidance_free=${HINT_GUIDANCE_FREE} prune_guidance=${HPRL_PRUNE_GUIDANCE} progress_message=${HPRL_AUTO_HINT_PROGRESS_MESSAGE} max_turn_tokens=${HPRL_MAX_TURN_TOKENS} lr_sched=${HPRL_LR_SCHEDULER}"
exec bash "${SCRIPT_DIR}/run_hprl_async.sh" "$@"
