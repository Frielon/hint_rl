#!/usr/bin/env bash
# =============================================================================
# run_auto_hint_olmo3_7b_instruct_async_restart_test.sh -- 1+1-NODE SMOKE of
# the olmo async-restart wrapper, primarily for the wandb rollout-mirroring
# path (data.hprl.wandb_save_rollouts -- devlog 2026-08-12). THIN DELTA: this
# exports the test overrides, then execs the REAL wrapper, so every training
# knob stays at parity and future wrapper changes flow through automatically
# (an earlier version of this file was a full copy, which drifts).
#
# Deltas vs run_auto_hint_olmo3_7b_instruct_async_restart.sh:
#   * split = 1 trainer : 1 rollout node (2-pod job). Trainer dp = 8 GPU/sp2
#     = 4; the 64-prompt x n8 = 512-row step geometry still divides dp, so no
#     other knob has to move. Throughput knobs stay at 2:4 parity -- a 1:1
#     smoke is simply slower (rollout-bound single vLLM node).
#   * exp_name = FIXED test label (...-TEST-1p1), NOT a real exp. Resuming a
#     real run here CANNOT work: those ckpts were saved at actor world 16
#     (2x8) and auto-resume onto 1x8 dies at bring-up -- and a run that got
#     further would write test ckpts/dumps and ratchet budget_state.json
#     INSIDE the real experiment. Re-running THIS test auto-resumes the test
#     exp's own world-8 state (if a ckpt was saved); for a clean slate delete
#     logs/<test exp> and ckpt/<project_name>/<test exp>.
#   * bounded: TOTAL_TRAINING_STEPS=10 -> the Rollouter streams exactly
#     10 x 64 groups and the job STOPS CLEANLY (queue sentinel), which also
#     exercises the shutdown wandb sweep (tail-file upload). Set =0 to run
#     unbounded (epoch semantics).
#   * VAL_BEFORE_TRAIN=False -- reach step 1 (first uploads) fast on the lone
#     rollout node; flip to True to also smoke the val_rollouts upload path.
#   * HPRL_WANDB_SAVE_ROLLOUTS defaults ON here too, so the smoke works even
#     if launched outside launch_hprl_cluster_openai_async_restart_olmo_test.sh.
#
# CATCH-UP (resume) PATH CHECK -- the fit-start pass uploads whatever already
# sits in the dump dirs, so pre-seed a few files before launching:
#   D=${HINT_RL_HOME}/logs/HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async-restart-dolci-10k-TEST-1p1/rollouts
#   mkdir -p "$D" && cp ${HINT_RL_HOME}/logs/<real exp>/rollouts/{1,2,3}.jsonl "$D"/
# -> the seeded files should appear under the NEW wandb run's Files tab within
# the first minute of trainer fit, before step 1 completes.
#
# Launch (2-pod platform job, entry on EVERY pod):
#   launch_hprl_cluster_openai_async_restart_olmo_test.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1+1 split (NNODES injected by the launcher/platform; 2-pod default) -------
export TRAINER_NNODES=${TRAINER_NNODES:-1}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-$(( ${NNODES:-2} - TRAINER_NNODES ))}

# --- own labels: never collide with (or auto-resume) a real experiment ---------
export exp_name=${exp_name:-"HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-async-restart-dolci-10k-TEST-1p1"}

# --- smoke shape ----------------------------------------------------------------
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export HPRL_WANDB_SAVE_ROLLOUTS=${HPRL_WANDB_SAVE_ROLLOUTS:-true}

exec bash "${SCRIPT_DIR}/run_auto_hint_olmo3_7b_instruct_async_restart.sh" "$@"
