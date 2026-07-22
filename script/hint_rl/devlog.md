# HPRL dev log

Running engineering log. Two parts: **TODO / Planned** (ideas not yet built) directly
below, then the **Done log** (shipped work, newest entry on top — append a new
`## YYYY-MM-DD` section per working session). Move an item from TODO to a dated Done
entry once it lands.

---

# TODO / Planned (not yet done)

_(none open — the step-level advantage calculation landed 2026-06-25; see the Done log.)_

---

# Done log

## 2026-07-15 — FULLY-ASYNC training mode (disaggregated Rollouter/Trainer pools on verl `fully_async_policy`), flag-gated — new entry point + launch scripts; NOT yet cluster-validated

HPRL can now train on verl's fully-async architecture: the training pods split into a ROLLOUT pool
(vLLM replicas + the agent loops, STREAMING one prompt-group — a problem's `rollout.n` trajectories,
one GRPO group under a shared uid — at a time into a MessageQueue) and a TRAINER pool (FSDP actor
pulling `require_batches × ppo_mini_batch_size` groups per iteration, NCCL-pushing weights back to
the replicas every `trigger_parameter_sync_step` iterations, sub-second for a 7B), overlapping in
wall-clock instead of alternating. (The two step-adv pricing bugs in the entry below were found by
this port's config-level dry run.)

**Why.** The sync trainer is structurally serial AND tail-bound. Measured over all 171 steps of
`logs/HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-dapo-20260705-095232`: `gen` = **71.6%** of the 318 s
mean step (227.6 s; update_actor 20.8%, old_log_prob 6.1%) — almost exactly the 70% verl quotes for
DAPO-32B — and within each gen phase the slowest rollout runs **3.58×** the mean (per-rollout gen
mean 45.9 s, max 162 s; 8.3 turns/rollout, each waiting on blocking selector calls), so finished
GPUs idle behind one straggler. Upstream benchmarks: 2.35–2.67× (pure math 7B), **1.55–1.6×** on
multi-turn ReTool — the closest analog to the hint loop — so ~1.5× is the realistic target here.

**How the pieces map (no verl core edits — the project rule holds).**
* Trainer: `hprl_fully_async._HPRLFullyAsyncTrainerImpl(HPRLRayPPOTrainer, <FullyAsyncTrainer body>)`
  — a cooperative-MRO subclass. verl's async trainer inherits `RayPPOTrainer` through
  `SeparateRayPPOTrainer`, and its `fit_step` template calls `self._update_actor(batch)` AFTER
  merging GRPO advantages + the reward extras (`acc`/`num_hints`) into the batch — exactly the
  contract the HPRL `_update_actor` override (verified-prefix mask / step-adv + budget ratchet) and
  `_log_rollout_data` (raw-token dumps + hint columns) already code against, so both land UNCHANGED.
  `fit` is re-overridden async (HPRL's sync `fit` would shadow the queue loop; it now guards k-pack
  and delegates). Ray mechanics: an `@ray.remote`-decorated class cannot be subclassed; the
  undecorated body is `<ActorClass>.__ray_metadata__.modified_class` (verified subclass+re-decorate
  on ray 2.55.1) — the one private-API concession, confined to `hprl_fully_async.py`. NB ray's
  tracing injection wraps every method (`__wrapped__`); identity checks in tests must unwrap.
* Rollouter: runs STOCK. `HintBudgetDataset` rides `data.custom_cls` (the Rollouter builds datasets
  via `create_rl_dataset`), the auto-hint rollout rides the per-row `agent_name` (preserved because
  `multi_turn.enable=True` — `prepare_single_generation_data` only force-overwrites it otherwise),
  and the streaming reward rides the same `RewardLoopManager` the sync job already used.
* Agent loop: **zero changes.** `AutoHintAgentLoop` already accumulates `response_logprobs`
  (padding the injected hint turns) and merges `min/max_global_steps` from the server's
  extra_fields — exactly what `assemble_batch_from_rollout_samples` requires; and PARTIAL ROLLOUT is
  invisible to it by design (`FullyAsyncLLMServerClient.generate` re-issues an aborted request with
  the accumulated tokens; the per-turn `max_tokens` COPY the loop passes composes with the client's
  decrement-on-resume).
* Budget ratchet: crosses the actor boundary through the existing shared-FS JSON (atomic
  `os.replace` writer on the trainer actor, mtime-cached reader in the dataset) — same mechanism as
  sync, now with the pipeline's bounded staleness lag instead of the step boundary.
* Off-policy correctness: trajectories train under the ROLLOUT policy's own logprobs
  (`rollout.calculate_log_probs` + `actor.use_rollout_log_probs` +
  `algorithm.rollout_correction.bypass_mode`, all base-config defaults), so the PPO ratio accounts
  for the staleness; `old_log_prob` leaves the trainer's critical path entirely.

**Equivalence bookkeeping.** Defaults `require_batches(1) × ppo_mini(64) × trigger(1)` = 64 prompts
per weight sync == one sync-job step (`train_prompt_bsz=64`, mini==batch → one optimizer step), so a
"param version" corresponds 1:1 to a sync global step: `test_freq`/`save_freq`/`total_epochs` and
checkpoint numbering keep their meanings. Resource split default 2 trainer : 2 rollout nodes (the
benchmarked 16:16 for a 7B multi-turn job); staleness_threshold 0.5, partial_rollout True. Rebalance
from `fully_async/{rollouter,trainer}/idle_ratio` in wandb.

**Config quirk.** verl's `fully_async_ppo_trainer.yaml` declares its own `hydra.searchpath`, which
hydra rejects in any non-primary config — so `config/hprl_fully_async_trainer.yaml` pulls
`ppo_trainer` directly and INLINES the async block verbatim (re-diff the upstream file on a verl
bump). The `data.hprl` block mirrors `hprl_trainer.yaml` (keep the two in sync).

**Not supported (guarded loudly at launch, `main_hprl_async._apply_hprl_async_guards` + the run
script):** k-pack (`HPRL_KPACK_ENABLE=true` → hard error — the probe rewrites `rollout.n` and
expands the gen batch inside the sync trainer's `_get_gen_batch`, which the queue path never runs)
and budget-grouped sampling (forced false — streaming has no generation BATCH to keep
budget-homogeneous; the queue absorbs the per-sample duration variance the sampler existed to
remove).

**Launch.** `TRAIN_SCRIPT=<...>/run_auto_hint_olmo3_7b_instruct_async.sh bash launch_hprl_cluster.sh`
(the launcher itself is unchanged; selector pods unaffected — the agent loops reach them over HTTP
exactly as before). Every auto-hint science knob in the async wrapper is IDENTICAL to the sync one.

**Verification (no cluster run yet — flag honestly).** 27-check harness
(scratch `test_async_port.py`): MRO resolution (each method unwraps to the intended
implementation; `super()._update_actor` from HPRL lands on `RayPPOTrainer._update_actor`), full
hydra compose with the production override set, both guards, the step-adv penalty regression;
plus `bash -n`, `py_compile`, and an end-to-end dry run of the wrapper that assembled the complete
`ray job submit` command and failed only at the absent local cluster. First real run should watch:
the two idle ratios, `fully_async/count/stale_trajectory_processed` / `partial_ratio`, and
`hprl/budget_mean` continuity vs the sync run.

### Files touched (all NEW; plus the bug fixes logged separately below)
- `hprl_fully_async.py` — the trainer/task-runner subclasses (the `__ray_metadata__.modified_class` mechanics live here)
- `main_hprl_async.py` — async entry point (mirrors `fully_async_main.main` + the HPRL guards)
- `config/hprl_fully_async_trainer.yaml` — ppo_trainer + inlined async defaults + the `data.hprl` mirror
- `run_hprl_async.sh` — base run script (async fork of `run_hprl_qwen2.5_7b.sh`: node-split + `async_training.*` knobs, `train_batch_size=0`/`gen_batch_size=1`, `hybrid_engine=False`)
- `run_auto_hint_olmo3_7b_instruct_async.sh` — the Olmo auto-hint launch wrapper (science knobs identical to the sync wrapper)

## 2026-07-15 — step-adv r(h) was DECOUPLED from the reward/loop: two pricing bugs fixed (migrated reward node + pruned-order state parity)

Found while porting HPRL onto verl's fully-async trainer (compose-level dry run of the launch
config). The step-adv machinery's design invariant is that its per-hint reward `r(h)` equals the
REWARD's per-hint penalty, indexed in the SAME pool order the LOOP used to record states. Both
halves of that invariant were silently broken — in **every step-adv run to date**.

**Bug 1 — `_step_adv_penalty_cfg` read a config node that no longer exists at runtime.** verl's
`migrate_legacy_reward_impl` (run by `main_hprl` BEFORE the trainer is built) MOVES the launch-time
`custom_reward_function` node to `reward.custom_reward_function` and DELETES the top-level one. The
getter read only the legacy location → always `None` → **code defaults**
`(total=1.8, hard_factor=1.5, guidance='moderate', guidance_free=False)` instead of the launch
kwargs `(1.0, 1.5, 'easy', guidance_free=True)`. Under `normalize=true` the uniform 1.8→1.0 scaling
washes out (per-group std is scale-invariant), but `guidance_free` changes the RELATIVE weights —
X.0 hints were FREE in the reward yet PRICED in the value recursion. Fix: read the migrated node
first, legacy as fallback (the pre-migration shape only exists in the launch argv).

**Bug 2 — with `prune_guidance=true`, the trainer priced states over the UNPRUNED pool order.** The
loop computes every recorded state (`state_start`/`state_end`, `fail_state = order.index(hid)`, and
the solving turn's `K`) over the PRUNED order (X.0 dropped — `AutoHintAgentLoop._pool`, 2026-06-2x),
but `_step_adv_penalty_vec` built `order = pending_hint_ids(pool, [])` from the RAW `extra_info`
pool. `pending_hint_ids` does NOT filter X.0, so `pv[k]` was the k-th UNPRUNED hint's weight: every
state at/after an X.0 entry read a WRONG (generally earlier) hint's penalty, and `K` overshot the
loop's terminal state by #major-steps. The value recursion itself runs in loop (pruned) coordinates
(`final_states`, `F_k`, `D_k` all consistent; `V` telescopes flat above the loop's max state since
no fails land there), so the damage was confined to the r(h) ladder — scrambled per-state penalties.
NASTY INTERACTION: fixing Bug 1 alone makes Bug 2 WORSE — `guidance_free=True` zeroes the X.0 slots
of the unpruned vector, so a pruned state colliding with an X.0 slot reads `r = 0.0` → failing that
hint becomes FREE (`a_I = 0`). Fix: `_step_adv_penalty_vec(..., prune_guidance=)` mirrors
`data.hprl.auto_hint.prune_guidance` and applies `prune_hint_pool` before `pending_hint_ids` — now
`pv == [pens[h] for h in loop_order]`, `K == len(loop_order)`, and `sum(pv) == total_penalty`
(guidance share redistributed). Flag off → raw order, byte-identical to before.

**Affected runs.** Bug 1: all step-adv runs. Bug 2: step-adv runs with `prune_guidance=true` — the
whole Olmo campaign (wrapper defaults BOTH on: 224821, 20260705-095232, 20260714-000914). The
2026-07-14 root-cause analysis (length-POSITIVE reward) is unaffected: V stays monotone and progress
stays paid regardless of which weights fill the ladder; only per-state penalty magnitudes shift.

**Tests** (suite 37 → **39/39**): `test_step_adv_penalty_cfg_reads_migrated_reward_node` (migrated
shape wins over stale legacy; legacy fallback works; bare config → code defaults) and
`test_step_adv_penalty_vec_matches_loop_order_under_prune` (pruned parity: K and per-state weights
match the loop order exactly, no X.0 zero leaks in, total preserved; flag-off keeps the raw order).
Both import the trainer lazily and skip loudly outside the verl env (suite stays dependency-light).

### Files touched
- `hprl_ray_trainer.py` — `_step_adv_penalty_cfg` reads `reward.custom_reward_function` first (legacy fallback); `_step_adv_penalty_vec(..., prune_guidance=False)` prunes the pool to the loop's order; `_hprl_apply_step_advantage` reads `auto_hint.prune_guidance` and passes it through; import `prune_hint_pool`
- `test_auto_hint.py` — the two regression tests above (+ `_PRUNE_POOL` fixture, `_import_hprl_trainer` skip helper)

## 2026-07-14 — over-long penalty: VALUE-INTEGRATED routing (`overlong_penalty_type=post_hoc|value`), flag-gated — RESULT: does NOT arrest the truncation; root cause is a length-POSITIVE reward

A second routing for the over-turn-length surcharge built 2026-07-01. That entry fixed the
*brake-collapse* half (an ABSOLUTE `P_over` survives `fc/d→1`); it leaves the other half untouched.
Being applied AFTER the advantages are assigned, post-hoc only ever moves the TRUNCATED rows DOWN —
every non-truncated row keeps its value-based advantage, which for a within-length turn failing at
`se=0` is `a_I = penalty[0]·(F_0/D_0 − 1) ≈ 0` once co-failure is common. So the group carries **no
positive signal for "produce a within-length turn"**, only a penalty for not doing so.

**The value mode (math).** Route the surcharge through the VALUE recursion instead: a rollout truncated
at state `k` carries reward `r_k − P_over` at its failed step, so with `T_k ≡ #truncated-at-k` (⊆ `F_k`,
since a per-turn cut is already coded `fail@pre_state`, 2026-06-29):

    V[k] = V[k+1] + (F_k·r_k − T_k·P_over) / D_k

Only `V[0..k]` drop (a truncated row has `V_i = k`, so it never enters `F_j`/`D_j` for `j>k`; a
first-turn cut at `se=0` lowers `V[0]` alone). With NO mean-centering that shift is not absorbed:

    non-truncated row at state k :  A += T_k·P_over / D_k          (LIFTED, → positive)
    truncated row at state k     :  A  = r_k(1 − F_k/D_k) − P_over(1 − T_k/D_k)
    gap(non-truncate, truncate)  =  P_over                          (EXACTLY, ∀ T_k/D_k)

i.e. the same absolute, brake-proof gap post-hoc gives, but positioned so the within-length rows land
ABOVE zero instead of at it — a "do the within-length thing" reward, not only a "don't truncate" push.
The rarer the survivor the bigger its lift (`T_k/D_k → 1`). Trade-off taken knowingly: it also
positively reinforces a concise-but-WRONG turn, so keep `P_over` small.

**Unchanged.** Scored-groups-only (no-correct groups still `continue`d + zeroed → no penalty either way,
same deliberate choice as 2026-07-01); `whole_turn` and split both work (the truncation segment carries
`boundary==ts`, so `−P_over` on the LAST segment's `r_se` lands on the whole turn in both); `normalize`
still reads it into the group std (it is in the raw advantage pre-std). `post_hoc` is byte-identical at
the default.

**Wiring (flag-gated as usual).** `data.hprl.auto_hint.step_adv.overlong_penalty_type` (yaml **default
`post_hoc`** = the 2026-07-01 behaviour) → trainer reads `sa.get("overlong_penalty_type")` → passed into
`apply_step_level_advantages`, which builds per-group `trunc_counts` (`T_k` from each truncated row's LAST
segment `se`) and threads `overlong_penalty`/`row_truncated` into `compute_state_values` /
`assign_row_advantages`; the post-hoc subtraction block is SKIPPED in value mode (no double-count).
Magnitude stays `HPRL_OVERLONG_PENALTY`. Env `HPRL_OVERLONG_PENALTY_TYPE` (default `post_hoc`) in
`run_hprl_qwen2.5_7b.sh` + job arg, exported + echoed in `run_auto_hint_olmo3_7b_instruct.sh`. New metric
`step_adv/overlong_value_mode`; trainer log gains `ov=<mode>(rows=N)`. Test
`test_apply_step_level_advantages_overlong_value_mode` (lift == `T_0·P_over/D_0`; the wrong-but-within-
length row flips negative→POSITIVE; gap == `P_over`; ≠ the naive post-hoc `a0 − P_over`; post_hoc leaves
the non-truncated row untouched) — suite 37/37.

**RESULT — it does NOT arrest the truncation.** Run
`logs/HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-dapo-20260714-000914` (dapo-512, FRESH start, 82 steps,
`max_turn_tokens=4096`, `max_response_length=28000`, whole_turn, normalize, `P_over=0.1`,
`overlong_penalty_type=value`, `lr_scheduler_type=constant`). Mode confirmed LIVE:
`overlong_value_mode=1.0` every step, `overlong_rows`≈79/step, `overlong_tokens`≈325k/step. Yet per-turn
truncation = **18.15%** (7621/41984; 100% per-turn, global=0; reconstruction validated EXACT against
`hprl/turn_truncated` at every step) and RISING: 16.6% → 18.8% → 19.0% (early/mid/late thirds). Turn mix
t1 37.2% / t2 43.6% / t3 12.4% (t1+t2 = 80.8%). Telling split: **turn-1 rate FALLS 7.5→5.8% while turn-2
rate RISES 5.7→9.8%** — the penalty bites where the truncated turn IS the whole rollout, and the inflation
migrates to the hinted turns. (vs the 20260705 post_hoc run: 15.6%, t1-dominant — NOT a clean A/B, that
one forked a mid-training ckpt, this one is fresh.)

**Why — the root cause is not the penalty, it is a length-POSITIVE reward (GRPO baseline).**
`logs/GRPO-Olmo-3-7B-Instruct-SFT-dapo-512-20260704-172025` is single-turn (`multi_turn.enable=False`) at
the SAME 4096 budget with reward `{correct_reward:0.9, format_reward:0.1, incorrect_reward:0.0}` — **no
over-long penalty of any kind** — and resolves length for free: `clip_ratio` 5.3% → ~0% (3% late),
`response_length/mean` 857 → ~590. Its reward is length-neutral and **success-GATED**: extra tokens have
no upside, only truncation risk, so the optimum is "solve as concisely as possible" and length collapses.
step_adv's reward is the opposite — it pays for the STATE reached (`V[se]−V[ss]`, `V` monotone), and
reaching a higher state costs tokens, **including on rollouts that ultimately FAIL**. Two failing
rollouts: GRPO scores both `0.0` (ties them — no length signal); step_adv scores the one that got further
HIGHER, i.e. pays for its extra tokens. Measured: step-adv all-turn mean length 789 → 1361, first-turn p90
2703 → 3934, incorrect first-turn p90 = 4096 (AT the cap), incorrect rows lengthen (×1.17) and reach
higher states (0.25→0.29); GRPO's incorrect rows do NOT lengthen (×1.05) and its correct rows COMPRESS
(×0.60). Note step-adv accuracy is HIGHER (32→57% vs GRPO 2→12%) — it is not truncating because it fails
more. Same root in entropy: GRPO 0.594→0.278 (−0.264 over its first 82 steps) vs step-adv 0.628→0.582
(−0.046) — a broad partial-progress target never concentrates the policy, and with `normalize` + hints
(`groups_zeroed→0`) every group keeps emitting a unit-scale gradient, so nothing anneals. GRPO's own late
3% is the signature of a success-gated coupling: drifts up with accuracy, then plateaus, because failed
length is never paid.

**Fix direction (next; supersedes "bigger / graded penalty").** A truncation-boundary penalty of ANY
shape (post_hoc, value, or the still-unbuilt `L_buffer` ramp) only touches tokens AT the cap, while the
inflation lives in the 500→3000-token body BELOW it, where progress is bought with free tokens. Make the
progress credit length-aware instead — fix the turn's TOTAL credit and charge for the tokens:

    A'_j = A_turn / L_turn − λ      ⟹   Σ_{j∈turn} A'_j = A_turn − λ·L_turn

so reaching `se` pays `A_turn` regardless of how many tokens it took (genuine progress still rewarded,
B-at-state-2 still beats A-at-state-0) and every token costs `λ` → the shortest path to each state wins;
a padding token adds 0 to `A_turn` and −λ to the cost. NB a FLAT `A_j − λ` is NOT sufficient under
`whole_turn`+`token-mean`: the positive part scales with `L` too, so any turn with `A_turn > λ` still
profits from padding. Keep the overlong knobs as off-by-default — they are symptom-level.

### Files touched
- `step_advantage.py` — `compute_state_values(..., trunc_counts=None, overlong_penalty=0.0)` (the `−T_k·P_over/D_k` term); `assign_row_advantages(..., overlong_penalty=0.0, row_truncated=False)` (`−P_over` on the LAST segment's `r_se`; both modes); `apply_step_level_advantages(..., overlong_penalty_type="post_hoc")` — mode routing, per-group `trunc_counts`, post-hoc block gated; `step_adv/overlong_value_mode` stat; module header + docstrings
- `hprl_ray_trainer.py` — read `step_adv.overlong_penalty_type`, pass through, log `ov=%s(rows=%d)`
- `config/hprl_trainer.yaml` — `overlong_penalty_type: post_hoc` + doc
- `run_hprl_qwen2.5_7b.sh` — `HPRL_OVERLONG_PENALTY_TYPE` env (default `post_hoc`) + job arg
- `run_auto_hint_olmo3_7b_instruct.sh` — export knob + echo (`overlong_type=`)
- `test_auto_hint.py` — `test_apply_step_level_advantages_overlong_value_mode`

## 2026-07-05 — auto-hint step-adv: WHOLE-TURN advantage mode (score each turn as one macro-action, boundary-free) — flag-gated

A second step-adv scoring mode, alternative to the `a_C`/`a_I` per-segment SPLIT (2026-06-25).
Instead of cutting each turn at the selector-verified boundary into a non-negative verified prefix
`a_C` and a non-positive failed tail `a_I`, score the ENTIRE turn `[turn_start, turn_end)` with ONE
value — the TD form of the whole turn as a macro-action:

    A = V(s_end) + hint_penalty − V(s_start)
      = r_se + V[se+1] − V[ss]     (FAILED turn; == a_C + a_I telescoped)
      = V[se]         − V[ss]      (solve / no-fail turn; no hint given)

**Interpretation (the one design choice).** `s_start = S_ss`; `hint_penalty = r_se = −penalty[se]`
(the injected hint's cost); `s_end = S_{se+1}` — the POST-HINT state (the turn's own tokens reach the
verified `S_se`, fail step `se`, then the injected hint completes it, landing at `S_{se+1}`, exactly
where the next turn resumes). Using the post-step value `V[se+1]` (NOT `V[se]`) is what keeps this a
proper advantage: `V` is built by the backward recursion `V[k]=V[k+1]+F_k·r_k/D_k` precisely so that
per-step advantages sum to ~0 across the group — `F_k·(r_k+V[k+1]−V[k]) + (D_k−F_k)·(V[k+1]−V[k]) = 0`
— and that GRPO-relative baselining only holds with `V[se+1]`. So whole-turn = `a_C + a_I` telescoped,
written uniformly over the whole span. (If one ever wanted the literal verified state `s_end = S_se`
instead, it is a one-line `V[se+1]→V[se]` flip; not what we want — it double-prices the hint.)

**What changes vs the split — and what doesn't.** Solve / no-fail turns are IDENTICAL (they already
carry `boundary == turn_end`, so `[ts,b) == [ts,te)` and both modes write `V[se]−V[ss]`). The ONLY
difference is a genuine progress-making FAILED turn (`turn_start < boundary < turn_end`): the SPLIT
gives its verified-prefix tokens `V[se]−V[ss]` and its tail `r_se+V[se+1]−V[se]`; whole-turn gives
EVERY token of the turn the single combined value `r_se+V[se+1]−V[ss]`. **Boundary-free** — it reads
only `(ss, se, is_fail)`, never the fuzzy verified-prefix boundary, so the gradient has ZERO dependence
on the selector citation-locate (`locate_quote_end`) accuracy; the whole turn is reinforced/penalized
together. Motivation: remove the fuzzy-quote boundary as a per-token noise source.

**The loop is unchanged.** `AutoHintAgentLoop` still records the SAME `step_adv_turns=[ts,boundary,te,
ss,se,is_fail]` segments (the boundary is still computed — it drives the legacy MASK mode and the
rollout dumps); whole-turn merely ignores column `boundary` when assigning. Purely a trainer-side switch.

**`HPRL_OVERLONG_PENALTY` still works — IDENTICALLY (asked & verified).** The over-turn-length penalty
(2026-07-01) touches ONLY rows flagged `turn_truncated=1`, and the per-turn-cap path always records its
truncation segment with `boundary == turn_start` (2026-06-29). That single fact makes the two modes
COINCIDE on exactly those rows: `boundary==ts` ⟹ the split's `a_C` is empty and `a_I` already spans the
whole turn, and with `ss==se` both compute the same base `r_se+V[se+1]−V[se]`; the penalty then subtracts
`P_over` from `[boundary,turn_end) = [ts,te)` = the whole turn in both. Confirmed empirically (3-rollout
group, one progress-fail-then-truncate row): identical penalty delta (`−0.5` over the truncated span),
identical truncated-turn value pre/post; the only place whole-turn differs (a PRIOR progress turn's
prefix) is not a truncation segment, and the penalty leaves it untouched. Same for `normalize` (P_over
still subtracted pre-std) and `zero_if_no_correct` (no-correct/all-truncate groups still zeroed first).
So the Olmo run's `HPRL_OVERLONG_PENALTY=0.1` behaves the same with whole-turn on.

**Wiring (flag-gated as usual).** New knob `data.hprl.auto_hint.step_adv.whole_turn` (yaml **default
`false`** → the `a_C`/`a_I` split runs) → `HPRLRayPPOTrainer._hprl_apply_step_advantage` reads
`sa.get("whole_turn")` and passes `whole_turn=` into `step_advantage.apply_step_level_advantages` →
`assign_row_advantages` branches (default keyword arg, so every other caller is byte-identical). Env
`HPRL_STEP_ADV_WHOLE_TURN` in `run_hprl_qwen2.5_7b.sh` (default `false`) + job arg. **`run_auto_hint_-
olmo3_7b_instruct.sh` defaults it to `true`** — whole-turn is now that run's default step-adv mode (as
`HPRL_ALLOW_DECREASE=false` is there); `HPRL_STEP_ADV_WHOLE_TURN=false` restores the split. Only
meaningful when `HPRL_STEP_ADV=true`. The trainer's step-adv log line gains `wt=<0|1>`.

**Tests.** New `test_apply_step_level_advantages_whole_turn` on the 2026-06-25 3-rollout worked example:
the whole failed turn shares one value (its prefix is no longer 0 as in the split), asserted against the
split on the two verified-prefix sub-ranges; solve / `boundary==ts` turns match the split exactly.
Existing truncation/overlong tests pass unchanged (their segments carry `boundary==ts`, where the modes
coincide). Suite 36/36.

### Files touched
- `step_advantage.py` — `assign_row_advantages(..., whole_turn=False)` branch (stats refactored into a `_tally` closure); threaded through `apply_step_level_advantages`; module header + docstring notes
- `hprl_ray_trainer.py` — read `step_adv.whole_turn`, pass through, log `wt=%d`
- `config/hprl_trainer.yaml` — `data.hprl.auto_hint.step_adv.whole_turn: false` + doc
- `run_hprl_qwen2.5_7b.sh` — `HPRL_STEP_ADV_WHOLE_TURN` env (default false) + job arg
- `run_auto_hint_olmo3_7b_instruct.sh` — export knob (default **true** for this run) + echo line
- `test_auto_hint.py` — `test_apply_step_level_advantages_whole_turn`

## 2026-07-01 (b) — `HPRL_ALLOW_DECREASE`: raise-only budget ratchet (decreases vetoed at the manager)

Follow-up lever from the truncation-tax diagnosis below (`hprl/budget_mean` compounding
4.09 → 1.97 fed the tax): a knob that keeps the adaptive rule's RAISE-on-zero-correct branch
but disables every budget-LOWERING outcome, so B_q can only go up (or hold).

**Mechanism.** `BudgetManager(allow_decrease=False)` applies a post-rule veto
(`_hold_decrease`) in `update_group` / `update_group_kpack`: any update whose `new_budget`
lands below the budget the group ran under is HELD at that budget; the `BudgetUpdate` keeps
the rule's decision stats (n_correct, pivot/min hint counts) and its `rule` gains a `"_held"`
suffix (`pivot_set_held`, ...). One mechanism covers all three rules — adaptive keeps only
the raise branch; `downward` and the k-pack probe (pure decreases) freeze entirely. Raises
and holds pass through; `allow_decrease=true` (default) is byte-identical to before. NOT
persisted in the state-JSON meta (same as `ratchet_mode`/`max_budget`), so flipping the env
on a resumed run takes effect.

**Wiring (flag-gated as usual).** `data.hprl.allow_decrease` (yaml, default `true`) →
`HPRLRayPPOTrainer._hprl_budget_mgr` → manager; env `HPRL_ALLOW_DECREASE` in
`run_hprl_qwen2.5_7b.sh` (default `true`). `run_auto_hint_olmo3_7b_instruct.sh` defaults it
to **`false`** — raise-only is now that run's default; `HPRL_ALLOW_DECREASE=true` restores
the two-sided rule. New wandb scalar `hprl/num_decrease_held` counts vetoed decreases per
step (0 unless the guard is on).

**Validation.** `python budget_manager.py --selftest` (raise passes through / pivot_set held
with stats kept / kpack + downward frozen / default-mode unchanged) + new
`test_auto_hint.py::test_allow_decrease_false_is_raise_only`; full suite 35/35. `bash -n` on
both launch scripts.

### Files touched
- `budget_manager.py` — `allow_decrease` ctor knob + `_hold_decrease` veto + selftests
- `hprl_ray_trainer.py` — read `data.hprl.allow_decrease`, include it in the ratchet log line
- `config/hprl_trainer.yaml` — `data.hprl.allow_decrease: true` (+ doc)
- `hint_budget_callback.py` — `hprl/num_decrease_held` metric
- `run_hprl_qwen2.5_7b.sh` — `HPRL_ALLOW_DECREASE` env (default `true`) + hydra override
- `run_auto_hint_olmo3_7b_instruct.sh` — defaults `HPRL_ALLOW_DECREASE=false` (raise-only) + echo
- `test_auto_hint.py` — `test_allow_decrease_false_is_raise_only`

## 2026-07-01 — FINDING: auto-hint reward decline is a length-TRUNCATION TAX; step_adv's over-long `a_I` penalty self-extinguishes (co-truncation at `se=0` dilutes its own brake)

Diagnostic of `logs/HPRL-AutoHint-Olmo-3-7B-Instruct-SFT-dapo-20260630-224821` (dapo-512,
`total_epochs=100` → ~8 steps/epoch, `rollout.n=8`, step_adv ON with `normalize=true` /
`adv_scale=1.0`, `loss_agg_mode=token-mean`, `max_turn_tokens=16384`,
`max_response_length=34768`, `incorrect_reward=0.0`). Symptom: scalar train reward
(`critic/rewards/mean`) peaks ~0.376 @ ep6, sags to ~0.336 @ ep8 then hovers ~0.36; `correct_frac`
peaks ~0.499 @ ep4–6 → ~0.42 @ ep8–12. Question: budget ratchet vs length truncation vs policy
degradation. **Verdict: length-truncation tax. NOT the budget, NOT sustained policy degradation.**

**Policy is improving, not degrading (three independent signals).**
- `val_acc` (unaided pass@1; `val-aux/*/length_truncated == 0` at every step) rises 0.058 → 0.19 @ ep6,
  dips to 0.128 @ ep8, then RECOVERS to 0.19–0.21 by ep9–12 (ends at/above the ep6 peak).
- `step_adv/value_s0_mean` = `V[0]` rises MONOTONICALLY 0.337 → 0.620, no ep8 dip.
- Within-budget-bucket reward and "finished-rollout" correctness (`correct/(1−trunc)`) both rise; the
  ep7–8 wobble is a transient blip that heals. `value_s0` grows a +0.20 gap ABOVE `correct_frac` by
  ep12 = potential-vs-realized, i.e. the truncation tax made visible from a second computation.

**Budget decrease is NOT the trigger (refuted).** `hprl/budget_mean` falls monotonically 4.09 → 1.97
across epochs (ratchet compounding), INCLUDING ep0–6 where reward was RISING. Raw
`corr(reward, budget) = −0.92` (lower budget ↔ higher reward). A monotone variable can't cause a
reversal at ep8. Budget's role is INDIRECT: fewer hints → model self-solves → longer turn-1 CoT →
more truncation.

**Length truncation IS the sustained driver.** `hprl/length_truncated_frac` climbs 0.3% (ep0) → 3.3%
(ep6) → 5.4% (ep8) → 10.3% (ep12); `response_length/mean` 4.8k → 11.8k; `clip_ratio` 0 → 0.02. Each
truncated rollout floors to `acc=0`. Clincher: val (0% truncation) keeps rising while train reward
stalls — val↑/train↓ can ONLY be the training-specific truncation. NB the scalar reward is a
MONITORING signal; the gradient uses step_adv (whose `value_s0` rises), so part of the "reward
decrease" is a metric artifact, not what is optimized.

**The step_adv over-long penalty SELF-EXTINGUISHES (the core mechanistic finding).** A per-turn cut is
coded `fail@pre_state`, whole turn as `a_I` (see 2026-06-29 entry). By the backward-value recursion this
is EXACT: `a_I = penalty[se]·(fail_count[se]/d_se − 1) ≤ 0`, i.e. `|a_I| = penalty[se]·brake`,
`brake ≡ 1 − fc/d`. Reconstructed from `rollouts/<step>.jsonl` (`step_adv_turns` + `acc` +
`length_truncated`, grouped by `input`; per-turn cut ⟺ last seg `b==ts, is_fail=1, te−ts≈16384`,
validated EXACT vs the `turn_truncated` metric — 55/55 @ step84):
- **brake fades as truncation spreads:** ~0.44 (rare, ep6/12) → ~0.22 (common, ep10–13).
  `corr(per-turn%, eff_brake) = −0.35`; `zero_if_no_correct`-zeroed share of per-turn cuts rises to ~28%.
- **it's MUTUAL-INFLATION, not skill loss:** decomposing `fc` at `se`,
  `corr(per-turn%, truncated-fail/d) = +0.84` but `corr(per-turn%, genuine-fail/d) = −0.27` (flat). The
  `#correct/group` drop (−0.59) is truncated-rollouts-counted-incorrect, not lower skill (val/`value_s0`
  rise). Truncated rollouts are what land in `fc`, and they push `fc/d → 1`, killing each other's brake.
- **why they collide:** 93.4% of per-turn truncations land on `se=0` — the FIRST unaided turn hits
  16384 before boxing. 47% of truncating groups have ≥2 co-truncations, and 94% of those share the
  same `se`. Co-located `fail@0` within a group is exactly the mutual-inflation condition.
- **plus two leak holes:** `zero_if_no_correct` zeros all-fail groups (→ `a_I=0`); and global-ceiling
  LATER-turn cuts (~11–19% of truncations, RISING as rollouts lengthen toward 34768) record no segment,
  so their tail keeps advantage 0 — fully unpenalized.

**Root-cause chain.** budget ratchet strips hints → turn-1 unaided CoT inflates → hits the 16384
per-turn cap at `se=0` → coded `fail@0`, clustered within group → group-relative `a_I` brake collapses
(+ `zero_if_no_correct`) → penalty can't arrest length growth → more truncation. Self-reinforcing; a
turn-1 length problem masquerading as a reward decline.

**Implication / fix direction.** A GROUP-RELATIVE penalty structurally cannot self-regulate truncation
— the offenders (co-truncated `fail@0` rollouts) dilute their own punishment, and it is weakest exactly
where truncation is worst. Want an ABSOLUTE per-token soft-overlong penalty (DAPO-style ramp over the
last `L_buffer` tokens before the cap), magnitude independent of group composition; OR, since
val/`value_s0` say the length growth is genuinely productive, raise the caps. This is NOT a
loss-normalization issue — the run already uses `token-mean` (the Dr-GRPO/DAPO no-length-norm
aggregation); flipping that lever does nothing here. Do NOT use DAPO over-long MASKING (it would delete
the negative signal that does exist). Artifacts in the run dir: `reward_decline_analysis.png`,
`truncation_brake_evolution.png`, `perturn_penalty_evolution.png`.

**Implemented (2026-07-01): absolute over-turn-length penalty, flag-gated, default off.** New knob
`data.hprl.auto_hint.step_adv.overlong_penalty` (env `HPRL_OVERLONG_PENALTY`, default `0` = off). In
`step_advantage.apply_step_level_advantages`, AFTER pass-1 assigns the raw value-based advantages and
BEFORE `_group_token_std`/normalize, subtract an absolute `P_over` from every per-turn-cap truncation
tail — the row's LAST segment for rows with `turn_truncated=1` (the trainer reads `turn_truncated` from
`non_tensor_batch` → `turn_truncated_per_row`). Because it is subtracted PRE-normalize it is read into
the group std and scaled into the unit range ("inside-normalization", so raw ~`penalty/K`; ~`0.1` lands
near unit); because it is an ABSOLUTE term (not `penalty·(fc/d−1)`) it does NOT ride the brake, so it
survives a co-truncating group (`fc/d→1`). The injection sits after the `zero_if_no_correct` `continue`,
so **no-correct / all-truncate groups stay zeroed — no penalty there (deliberate design choice)**.
New metrics `step_adv/overlong_rows`, `step_adv/overlong_tokens`. Test
`test_apply_step_level_advantages_overlong_penalty` (subtracts exactly `P_over` off the truncated tail
only; no-correct group untouched; std widens under normalize) — suite 34/34. Design notes: chose
inside-normalize (subtract pre-norm) over a post-norm absolute floor — folds P_over into the group std
so it self-limits and stays in-distribution, at the cost of collapsing on the fully-degenerate
all-same/all-truncate group (which we accept, since those are the no-correct groups we leave zeroed).
NOT YET DONE: the `L_buffer` ramp (penalty is currently flat over the whole truncated turn, not
concentrated on the last tokens before the cap). To enable: `HPRL_OVERLONG_PENALTY=0.1`, watch
`length_truncated_frac`/`turn_truncated_frac` fall and `response_length/mean` stabilize.

## 2026-06-29 — auto-hint PER-TURN generation-length cap: an over-long turn fails its next hint step (whole-span `a_I`); the global ceiling stays neutral

Adds an independent per-turn token cap to the AUTO-HINT rollout. Before this, only the
GLOBAL response budget bounded generation, and it effectively bounded just the FIRST turn:
the rollout `sampling_params` carries **no `max_tokens`** (`agent_loop.py:496-502`), so
`vllm_async_server.generate` takes its fallback `min(response_length, prompt_length +
response_length − len(prompt_ids))` (`:482-494`) — on turn 1 the short prompt makes that
`= response_length` (the full 16384), but on later turns it is only the *remaining context
room*, so a turn could eat whatever was left of the global budget with no per-turn limit of
its own.

**Feasibility (checked first).** A per-turn cap is just a per-request `max_tokens`:
`generate()` honors an explicit `sampling_params["max_tokens"]` if present (`:483-484`) and
otherwise uses the fallback, clamping only DOWN to context room (`:498`). Same per-turn
override pattern the base `ToolAgentLoop` already uses for `stop_token_ids`. The catch:
`stop_reason` can't tell a length cut from an EOS — vLLM reports both as `"completed"`
(`vllm_async_server.py:577`) — so a cut must be detected by **token count**, not stop reason.

**Mechanism (`auto_hint_agent_loop._handle_generating_state`).** New knob
`data.hprl.auto_hint.max_turn_tokens` (env `HPRL_MAX_TURN_TOKENS`, **default 0 = off**). When
> 0, each turn is generated with `max_tokens = min(max_turn_tokens, room)` where `room =
response_length − tokens-used-so-far`, passed as a **copy** of `sampling_params` (which also
stops the loop's prior in-place mutation of the rollout-shared dict that `generate()` pops).

**Two length cuts, by which limit binds** — decided by the pure, unit-tested
`step_advantage.classify_length_cut(n_tokens, max_turn_tokens, room)` (single source of
truth; the loop calls it instead of duplicating the inline checks):
- **`per_turn`** — the per-turn cap is the binding limit (`max_turn_tokens <= room`) and the
  turn ran to it. Treated as FAILING to finish its next hint step: `length_truncated=1`
  (reward floor → `acc=0` → incorrect) + a `turn_truncated=1` extra-field flag, and in step-adv mode
  one whole-span segment `[turn_start, turn_start, turn_end, pre_state, pre_state,
  is_fail=1]` — `boundary == turn_start` so the WHOLE turn is the `a_I` tail (no verified
  `a_C`), failing the next pending hint `h_{k+1}` at the rollout's current state
  `S_k = pre_state` (so `h_{k+1} ∈ H_i`). Then terminate — **no** selector call, **no**
  injection. This GENERALIZES the 2026-06-26 first-turn-global case (which hardcoded
  `pre_state 0`) to any turn at any state.
- **`global`** — the global ceiling bound the turn (cap off, or `room < max_turn_tokens` so it
  ran out of TOTAL budget). UNCHANGED behavior: `length_truncated=1`; in step-adv a FIRST-turn
  cut is charged as failing `h_0` at `S_0`, a LATER-turn cut records no segment → its tail
  keeps **advantage 0** (the ceiling is not the turn's fault).

**Boundary (the tie `room == max_turn_tokens`).** The per-turn branch is checked first and
wins the tie, so a turn cut at length `== max_turn_tokens` is PUNISHED even when it coincides
with the global ceiling; only a *strictly smaller* global room (`room < max_turn_tokens`) is
the neutral, advantage-0 global cut. (Discriminator rule: `max_turn_tokens <= room` → per-turn
binds.) Cap-off is byte-for-byte the old behavior — `classify_length_cut(n, 0, room)` returns
`"global"` iff `n >= room`, identical to the old `len(response_mask) >= response_length`.

**All-truncated GROUP → zero gradient.** If every rollout in a GRPO group truncates, all are
`length_truncated → acc=0 → correct_per_row=False`, so `has_correct=False` and (with the
DEFAULT `zero_if_no_correct=true`, overridden nowhere) the group is zeroed in
`apply_step_level_advantages` BEFORE any per-segment advantage is assigned — no `a_I` tail
reaches the rows. (Special case of the existing all-incorrect-group zeroing.)

**Truncation metrics.** verl's `response_length/clip_ratio` is
`mean(total_response_tokens == max_response_length)` (`metric_utils.py:236`) — it counts only
rollouts that filled the WHOLE global budget, so a per-turn cut (which terminates the rollout
EARLY, below the global ceiling) is invisible to it and it UNDER-reports truncation once the
cap is on. The reliable signals are the per-rollout flags surfaced through
`hint_budget_callback`: `hprl/length_truncated_frac` (mean of `length_truncated` = the TRUE
truncation ratio, both cuts) and `hprl/turn_truncated_frac` (mean of the new `turn_truncated`
= the per-turn-cap subset). The per-turn flag was made an EXTRA-FIELD (not an
`agent_data.metrics["turn_length_truncated"]` counter, which `AgentLoopMetrics` strips on
`model_dump`, same fate as the hint_select timer) so it survives to `non_tensor_batch`.

**Tuning interactions (documented in the configs, not bugs).** The global
`max_response_length` must cover ~`#turns × max_turn_tokens` + injected-hint tokens, else the
global ceiling bites first (e.g. cap 4096 → ~4 turns under 16384; raise the global budget to
suit). The cap also bounds the SOLVING turn, so keep it ≥ a realistic single-step solution
length. Scoped to `AutoHintAgentLoop` only (the legacy `<hint_call/>` loop is untouched).

**Verified.** 33/33 `test_auto_hint.py`, incl. four new tests:
`test_classify_length_cut_per_turn_vs_global` (the discriminator incl. the tie
`classify_length_cut(4096,4096,4096) == "per_turn"`), `test_per_turn_truncation_segment_whole_turn_a_i`
(whole-turn `a_I` failing `h_{k+1}` at `pre_state=2`: `a_I=−0.1` on every turn token, `h_3 ∈ H`),
`test_all_truncated_group_zeroed` (mixed truncation shapes, no correct → all rows 0), and
`test_truncation_frac_metrics` (drives the real `hprl_update_budgets`: `length_truncated_frac`
0.5 / `turn_truncated_frac` 0.25, and absent keys skip cleanly).

### Files touched
- `auto_hint_agent_loop.py` — `max_turn_tokens` parsed in `__init__`; `_handle_generating_state` applies the per-turn `max_tokens` and routes the cut through `classify_length_cut` (per-turn punish vs global neutral); `turn_truncated` extra-field flag (setdefault 0 + set 1 on a per-turn cut)
- `step_advantage.py` — new pure `classify_length_cut(n_tokens, max_turn_tokens, room)` discriminator (per_turn / global / None; per-turn wins the tie)
- `hint_budget_callback.py` — `hprl/length_truncated_frac` (true truncation ratio, both cuts) + `hprl/turn_truncated_frac` (per-turn subset), mirroring `hint_budget_exceeded_frac`
- `config/hprl_trainer.yaml` — `data.hprl.auto_hint.max_turn_tokens: 0` (+ rationale)
- `run_hprl_qwen2.5_7b.sh` — `HPRL_MAX_TURN_TOKENS` default + Hydra override
- `run_auto_hint_olmo3_7b_instruct.sh` — `HPRL_MAX_TURN_TOKENS` export (defaults to **8192** for this run — opts in; overridable) + echo. Note vs the global `max_response_length=34768` this leaves room for only ~4 turns before the global ceiling; raise `max_response_length` for more
- `test_auto_hint.py` — `test_classify_length_cut_per_turn_vs_global`, `test_per_turn_truncation_segment_whole_turn_a_i`, `test_all_truncated_group_zeroed`

## 2026-06-27 — selector citation locator: ambiguous quote now anchors on the FIRST matched sentence

A correctness fix to where a selector citation is located in the student trace. The
auto-hint loop uses `selector_multi.locate_quote_end(quote, text)` to find the char
offset where a completed-hint's verbatim `quote` ENDS, and that offset bounds the
verified-prefix gradient mask / step-adv segment (`auto_hint_agent_loop._verified_boundary`,
also the `cite_found_rate` metric). The quote is matched against the WHOLE turn text (no
sentence segmentation), so a quote can align to more than one sentence.

**The bug.** When a quote matched several places, the result depended on the match tier:
the exact-substring tier (`text.find`) returned the FIRST occurrence, but the
whitespace-normalized and loose/fuzzy tiers both went through `_raw_match_end`, which
scanned ALL of difflib's matching blocks and returned `max(..., key=lambda b: b.b + b.size)`
— the FURTHEST-into-text block end. So a stray late-matching token (e.g. the quote's last
word reappearing in a later sentence) could drag the verified-prefix boundary forward into
a sentence the student hadn't actually reached, over-extending the positively-masked prefix.

**The fix.** `_raw_match_end` now anchors on `difflib.SequenceMatcher.find_longest_match`,
which returns the EARLIEST longest common run (ties broken to the earliest text position),
and computes the end as that run's text position extended by the quote chars trailing the
run (`m.b + (len(quote) - m.a)`, clamped to `len(text)`). When a quote matches multiple
sentences the end now lands in the FIRST one, consistent with the exact tier, and a stray
late token can no longer pull it forward. All three tiers of `locate_quote_end` are now
uniformly first-occurrence.

**Trade-off.** In the fuzzy tier the end is now `run_start + remaining-quote-len` rather
than the last literally-matched char, so a quote whose TAIL paraphrases heavily can
overshoot by a few chars. Harmless directionally — this only bounds the verified prefix and
the mask is meant to err conservative (`auto_hint_agent_loop.py:90-91`).

**Unchanged on purpose.** `_verified_boundary` still takes the MAX end across *different*
completed hints — that's the furthest prefix verified by distinct hints (each legitimately
completing a different part of the turn), not multiple matches of one citation.

**Validation.** Spot-checked first-occurrence semantics across exact-repeat (tier 1),
whitespace-diff repeat (tier 2), and fuzzy-with-stray-late-token (tier 3): all three now
bound at the first matched sentence.

### Files touched
- `selector_multi.py` — `_raw_match_end` rewritten to first-occurrence (`find_longest_match`); `locate_quote_end` docstring tier bullet updated

## 2026-06-26 — auto-hint step-adv: a FIRST-turn length-cap truncation is now scored as a failed first step (whole-span `a_I`)

Closes a step-adv gap on the response-length cap. In AUTO-HINT step-adv mode (`HPRL_STEP_ADV`,
the 2026-06-25 step-level advantage work) the length cap in
`auto_hint_agent_loop._handle_assistant_response` terminated the rollout **without recording
any `step_adv_turns` segment** ("a capped turn is the ending turn, nothing more to inject").
When that cap fires on the **first** turn the rollout is left with an EMPTY turns list → in
`step_advantage.py` the whole row is zeroed (`final_state = 0`, no fails) → **zero gradient,
and the over-long first turn escapes penalty entirely** (`h_0` never enters any `H_i`, so it
doesn't even depress `V[0]`). A run-on first turn was free.

**Fix (`auto_hint_agent_loop.py`).** When step-adv is on AND this is the first turn
(`agent_data.assistant_turns == 1`, already incremented at the top of the handler), the cap
branch records one whole-span failed segment `[turn_start, turn_start, turn_end, 0, 0,
is_fail=1]`:
- `boundary == turn_start` → the WHOLE response is the `a_I` tail, no verified-prefix `a_C`
  credit (nothing is ever selector-verified on a first turn).
- `state_start == state_end == 0` with `is_fail` → charged with FAILING `h_0` at `S_0`:
  end-state `S_0` (not even reaching the first step) and `h_0 ∈ H_i`.
So every response token of the truncated first turn gets `a_I = r_0 + V[1] − V[0] ≤ 0`, and
the rollout now counts in `F_0` (depresses `V[0]`) — aligned with the intent that **one turn's
generation should fit within the length**.

**Scope: first turn ONLY.** A non-first-turn length cap is unchanged — it records no segment,
so its truncated tail keeps **advantage 0** as before (the earlier turns' `a_C`/`a_I` segments
already scored the rollout). Rationale: a first turn that overruns made no selector-verified
progress, so "failed the first step" is unambiguous; a later turn may have made real—but
unverified—progress before overrunning, so we leave it at 0 rather than relabel the whole turn
a failure. `length_truncated = 1` (the reward floor + metrics) is still set in both cases; this
only ADDS the step-adv label, and step-adv off → the branch is skipped (legacy mask path intact).

**Verified.** 29/29 `test_auto_hint.py`. Simulated the emitted segment through the real
`apply_step_level_advantages` (a first-turn-truncated rollout + one correct sibling, K=3,
penalty 0.6 → `V=[0.7,1,1,1]`): the truncated row gets a uniform `a_I = −0.3` on every token
(no zero-grad tokens), end-state `S_0`, and `h_0` counted in `H` pulls `V[0]=0.7 < V[1]=1.0`.

Files: `auto_hint_agent_loop.py`.

## 2026-06-25 (b) — auto-hint selector: X.0-guidance pruning (eval/train parity) + viewer shows the selector prompt & raw output

Two small, related changes to the AUTO-HINT selector path.

**1. Prune the X.0 step-guidance hints before the selector sees the pool (flag-gated).**
The offline selector eval (`selector/test_cite/gpt_oss_eval/multi-cite-gpt-eval`) was built
and scored on **substep-only** pools: `build_benchmark.py` runs every pool through
`run_gpt_oss_selection.prune_hint_pool`, which drops each hint whose `type` is
`step_guidence_hint` **or** whose `hint_id` ends in `.0`, and strips the per-hint `type`
field (the benchmark stores the pruned pool, commented `# pruned (no x.0, no type)`). But the
TRAIN/rollout path (`selector_multi.render_hints_with_status`) rendered the **full** pool —
X.0 guidance hints included — so the selector was trained on pools it was never evaluated on.
- Ported `prune_hint_pool` into `selector_multi.py` (verbatim logic, stdlib-only) so the two
  can't drift.
- `auto_hint_agent_loop.py`: added one chokepoint `self._pool(agent_data)` that reads `hints`
  and prunes when enabled, and routed **both** pool reads (the step-adv solving-turn order +
  the selector call) through it — so the selector prompt, the pending/exhaustion check, and the
  step-adv hint **order/state indices** all see ONE consistently-pruned pool (pruning only the
  prompt would desync the step-adv states and exhaustion logic).
- Hyperparameter: `data.hprl.auto_hint.prune_guidance` (env `HPRL_PRUNE_GUIDANCE`), wired
  through `run_hprl_qwen2.5_7b.sh` + `config/hprl_trainer.yaml`. **Default off** (full pool, as
  before); **on by default in `run_auto_hint_qwen2.5_7b.sh`** since parity is that run's point.
- Note: with pruning on, `HINT_GUIDANCE_FREE` (zero the X.0 penalty) is effectively redundant
  on this path — the selector can no longer apply an X.0 hint at all.
- Test: `test_prune_hint_pool_drops_x0_and_type` (29/29 pass).

**2. Rollout viewer: show the selector's PROMPT and RAW OUTPUT for an injected-hint turn.**
The auto-hint per-call debug dump (`auto_hint_agent_loop._dump`) recorded only the parsed
`selection`, `problem`, and `trace` — not the raw selector output, not the prompt, and not a
`messages` list. So (a) the viewer had nothing to show, and (b) `build_selector_index.py`
(which content-keys on the record's assistant `messages`) couldn't even join auto-hint records
(it saw an empty trace). Fixed end to end:
- `hint_selector.select_multi` now also returns the **exact prompt** it sent (4-tuple
  `selection, raw, err, prompt`).
- `auto_hint_agent_loop._dump` now writes `selector_prompt`, `selector_raw`, and `messages`
  (assistant turns through the calling turn, system content elided — for index keying), at all
  three call sites (selector-failure, step-adv terminal label, normal hint).
- `tools/rollout_viewer.py`: `api_selector_call` surfaces `selector_prompt` / `mode` /
  `completed_status_before`; `renderSelectorCall` adds open-by-default **"full prompt sent to
  selector"** and **"raw selector output"** panels, renders the multi-round `completed_hints`
  (hint_id + quote), and hides empty detail panels.
- **Caveat:** records dumped before this change have none of these fields (raw output is not
  recoverable from the parsed `selection`), so the feature lights up only for dumps written by
  the patched code — restart the run / start a fresh one, then rebuild
  `selector_index.sqlite` via `tools/build_selector_index.py --run-dir logs/<EXP>`.

Files: `selector_multi.py`, `auto_hint_agent_loop.py`, `hint_selector.py`,
`config/hprl_trainer.yaml`, `run_hprl_qwen2.5_7b.sh`, `run_auto_hint_qwen2.5_7b.sh`,
`tools/rollout_viewer.py`, `test_auto_hint.py`.

## 2026-06-25 — STEP-LEVEL advantage calculation (auto-hint): value-based per-segment advantage replacing GRPO's scalar

Implements the TODO. In AUTO-HINT mode the selector already tells us, per turn, how far
the student got (the contiguous count of completed hints = the STATE reached) and which
hint it FAILED. This turns each rollout into a walk over **K+1 states** `S_0..S_K`
(`S_k` = "the first k pool hints are done") and lets us assign a **per-token, value-based
advantage** instead of broadcasting one GRPO scalar across the whole rollout. Flag-gated
`data.hprl.auto_hint.step_adv.enable` (env `HPRL_STEP_ADV`, **default off** → the existing
verified-prefix MASK runs unchanged); the two are **mutually exclusive** (step-adv gives the
unverified tail a negative advantage instead of masking it away, so the mask is skipped when
step-adv is on).

**The model (`step_advantage.py`, pure / duck-typed torch+numpy).** Each turn splits over its
RESPONSE tokens into:
- `a_C` = the selector-VERIFIED prefix `[turn_start, boundary)` advancing `S_ss → S_se`
  (`ss` = state at turn start, `se` = verified state end); reward 0.
- `a_I` = the unverified tail `[boundary, turn_end)` that attempted hint `se` and failed
  (the loop then injects it); reward `r_se = −penalty(h_se) < 0`.
The CORRECT rollout's ending turn is all-`a_C` reaching `S_K` (no `a_I`). Over a problem's N
rollouts (one GRPO group), with `V_i` = verified final state (K if solved) and `H_i` = its
failed-hint set:
```
V[K] = terminal_value (=1)
V[k] = V[k+1] + F_k·r_k / D_k     (k = K−1 .. 0)
  F_k = #{i : k ∈ H_i}   (failed hint k)
  D_k = #{i : V_i ≥ k}    (reached state k = took the S_k→S_{k+1} transition)
A(a_C) = V[se] − V[ss]            ≥ 0
A(a_I) = r_se + V[se+1] − V[se] = r_se·(1 − F_se/D_se)  ≤ 0
```
So self-achieving a step that OFTEN trips others (`F_k/D_k` high) is rewarded and failing a
step others clear is penalized — the GRPO-relative intuition, **per step**. The per-segment
advantages **telescope** exactly to `(rollout return − V[0])`, i.e. it is a proper TD
decomposition (verified: a 1-hint solve sums to `r_0 + 1 − V[0]`). An **all-incorrect group
is zeroed** (`zero_if_no_correct`, default true): without a solve the `V[K]=1` anchor is
unreached and the positive `a_C` pulls would chase a goal nobody hit.

**The `E_i` convention (denominator).** The TODO's `D_k = Σ_i 1_{E_i > k}` uses `E_i` = the
**post-selector-check** state: an incorrect rollout whose verified state is `m` has `E_i = m+1`
(with `h_{m+1}` added to `H_i`). Under that convention `1_{E_i > k} = 1_{m ≥ k}`, so the
denominator counts every rollout that **reached** state k (took the `S_k→S_{k+1}` transition) —
including the one that failed `k+1` terminally (it's in `H_i`, so it must be in `D_k` too, else
`A(a_I) > 0` rewards a failed step). The impl carries the verified state `m` directly
(`final_states[i]`) and computes `D_k = #{m_i ≥ k}` — identical to `Σ_i 1_{E_i > k}`. This makes
`F_k ≤ D_k` (so `V` non-decreasing, `A(a_I) ≤ 0`) and self-counts the failing rollout; with
`zero_if_no_correct` a scored group always has a solver crossing every `k<K`, so `D_k ≥ 1`.

**State accounting (`auto_hint_agent_loop.py`).** `state_start = prefix_state(pool_order,
completed)` = the contiguous-completed-prefix count BEFORE the turn (the multi-round selector
only ever completes the next pending hints, so `completed` is a contiguous prefix). A failed
turn's `state_end = pool index of the selector's selected hint` (the first still-pending one)
— robust without re-deriving from quotes. The given hint advances the state by 1 (next turn's
`state_start = state_end + 1`), so a re-done given step is never re-credited; the boundary
(`_verified_boundary`, same fuzzy `completed_hints`-quote locator the mask uses) splits
`a_C`/`a_I`. Per turn the loop records `step_adv_turns = [ts, boundary, te, state_start,
state_end, is_fail]` (response-relative token coords). **Terminal labeling** (per the TODO):
when a wrong rollout's budget is spent, the loop now routes to the selector to LABEL the last
turn (identify its failed step) but injects **no** hint and charges **no** applied-hint
penalty (so `num_hints`/the ratchet are unchanged) — one extra selector call per over-budget
incorrect rollout (incl. budget-0 problems, a real selector-load cost; an all-incorrect group
is zeroed anyway, so some of those labels are "wasted" — noted). Pool-exhaustion / selector
failure record a zero-advantage no-fail segment (never guess a boundary on an outage).

**Trainer (`hprl_ray_trainer.py`).** `_update_actor` dispatches: `step_adv.enable` →
`_hprl_apply_step_advantage` (else the mask). It runs in the SAME hook as the mask — by then
the batch carries the GRPO `advantages`/`returns` (the SAME tensor; verl GRPO returns
`scores, scores`) and the assistant-only `response_mask`. It groups rows by `uid` (= the GRPO
group), reads per-rollout `step_adv_turns` + `acc` (correctness) + `extra_info`, builds the
per-hint penalty vector in **pool order** from `extra_info.hint_full` with the **same**
`hint_penalty_total`/`hard_factor`/`guidance` knobs `hint_reward.compute_score` prices hints
with (so `r` matches the reward's penalty), computes `V`, and **overwrites** the per-token
`advantages`/`returns` IN PLACE (zeroing each row first, so the stale GRPO scalar and the
masked injected-hint tokens go to 0). The actor consumes `advantages` as-is (verl whitens
only inside the per-estimator `compute_advantage`, which this overrides; the policy loss does
not re-normalize) — verified. `adv_scale` (default 1.0) uniformly scales the small raw values
(per-hint penalties ~`total_penalty/K` ≈ 0.06 for a 13-hint pool, ≪ GRPO's ~unit advantages)
to a usable gradient magnitude without retuning the LR.

**Metrics:** `step_adv/{groups_total,groups_scored,groups_zeroed,rows_scored,tokens_assigned,
adv_pos_mean,adv_neg_mean,value_s0_mean}` (+ the auto-hint `cite_found_rate`, folded in as
before). `value_s0_mean` is a clean problem-hardness readout (1 = trivial, lower = harder).

**Tests** (`test_auto_hint.py`, **26/26**): `prefix_state` (contiguous + robust to a stray
out-of-order id); `compute_state_values` on a worked 3-rollout example (`V=[0.7,0.9,0.9,29/30,
29/30,1]`) and the all-fail telescoping `V[k]=1−Σp[k:]`; `apply_step_level_advantages`
end-to-end (every segment's exact value, `returns` mirrors `advantages`, stale values
overwritten, `adv_scale` exact 5×); all-incorrect zeroing and `zero_if_no_correct=false`.
Also an offline integration run of the REAL `HPRLRayPPOTrainer._hprl_apply_step_advantage`
against a real dataset `extra_info` (K=13 pool, penalties from `hint_full`) on torch tensors
with `returns is advantages` — correct a_C/a_I, in-place propagation, uid grouping.

**Run:** `HPRL_STEP_ADV=true bash run_hprl_qwen2.5_7b.sh` (auto-hint only). `HPRL_STEP_ADV_SCALE`
(default 1.0) tunes the gradient magnitude. Both flow as `data.hprl.auto_hint.step_adv.*`
hydra overrides. Not yet run live.

**Follow-up — per-group advantage normalization (`step_adv.normalize`, default false).** The
raw value-based advantages are small (~`total_penalty/K`, std ≈ 0.04 on real data) → a too-small
gradient. `normalize=true` (env `HPRL_STEP_ADV_NORM`) divides each GRPO group's per-token
advantages by the group's advantage std (over its trained tokens), bringing them to ~unit scale
ADAPTIVELY per group; `adv_scale` then sets the TARGET std (1.0 → unit). NO mean-centering — the
value function V already baselines the advantages (their token-mean is ~0), and centering again
would distort the `a_C≥0 / a_I≤0` sign. A near-zero-std (degenerate) group is left unnormalized
(`_NORM_EPS=1e-6`, the divide-by-~0 blow-up guard). New metrics
`step_adv/{group_std_mean,norm_factor_mean}`. Verified on real data: raw std 0.041 → 1.008 at
scale 1 (factor ≈ 24×), → 3.02 at scale 3. `run_auto_hint_qwen2.5_7b.sh` defaults it ON (step-adv
is on there).

**Follow-up — free X.0 guidance hint (`hint_guidance_free`, default false).** New reward kwarg
(env `HINT_GUIDANCE_FREE`) making every `<step_id>.0` GUIDANCE hint cost 0: in
`hint_penalty.compute_hint_penalties` the guidance hint gets zero WEIGHT, so the step's penalty is
borne entirely by its substep hints and `sum(penalties)` STAYS `total_penalty` (the cost is
redistributed onto the revealing substeps, not dropped). Threaded through `applied_penalty` /
`penalty_from_k` / `hint_reward.hint_penalty` / `compute_score` (with string coercion), and through
the step-adv path (`_step_adv_penalty_cfg`/`_vec`) so the value-based `r(h)` matches the reward — a
guidance step then has `r=0`, so self-achieving or being given it is value-neutral. Only zeroed when
the step has substeps to absorb it (no guidance-only steps exist in the curated pools: 0/2069).
Verified on real data: x.0 → 0, total preserved at 0.8, substep `1.1` 0.0436→0.0623.

### Files touched this session
- `step_advantage.py` *(new)* — `prefix_state`, `compute_state_values` (the backward V
  recursion), `assign_row_advantages` (per-row a_C/a_I token write), `apply_step_level_advantages`
  (group-by-uid orchestrator + stats), `_group_token_std` (per-group normalization); pure,
  duck-typed torch/numpy
- `hint_penalty.py` — `guidance_free` in `compute_hint_penalties` / `applied_penalty` / `penalty_from_k`
- `hint_reward.py` — `hint_guidance_free` kwarg (coerced) → `hint_penalty` + `penalty_from_k`
- `auto_hint_agent_loop.py` — `step_adv_enable` flag; per-turn `step_adv_turns` recording
  (solving / hinted / terminal-label / pool-exhausted / selector-fail segments); budget-spent
  → terminal selector LABEL (no hint given); `_record_step_adv_turn`; import `prefix_state`
- `hprl_ray_trainer.py` — `_hprl_apply_step_advantage` + `_step_adv_penalty_cfg` /
  `_step_adv_penalty_vec` / `_coerce_turns`; `_update_actor` mask-vs-step-adv dispatch;
  `step_adv_turns` added to the rollout-log columns; `_truthy` helper
- `config/hprl_trainer.yaml` — `data.hprl.auto_hint.step_adv` block (enable / terminal_value /
  zero_if_no_correct / adv_scale)
- `run_hprl_qwen2.5_7b.sh` — `HPRL_STEP_ADV` + `HPRL_STEP_ADV_SCALE` + `HINT_GUIDANCE_FREE` knobs
- `test_auto_hint.py` — step-adv unit tests (prefix_state, values, advantage assignment, scale,
  zeroing) + guidance-free (x.0→0, total preserved)

## 2026-06-22 — AUTO-HINT (push-hint) mode + verified-prefix gradient mask

New, flag-gated mechanism alongside the `<hint_call/>` path. The policy is **no longer
taught to ask for hints**: it runs the ordinary single-turn math prompt (the exact format
of `dataset/dapo-3139-single-turn.parquet`), and the **loop** decides when to hint.

**Rollout (`auto_hint_agent_loop.AutoHintAgentLoop`, agent_name=`auto_hint`).** Per turn:
generate → grade the boxed answer (same mathruler grader the reward uses) → **correct →
stop**; **wrong & injections < budget B_q →** ask the frozen selector for the next hint,
inject it as a user message (masked 0), continue. Stops on correct, budget spent, or
pending-hint pool exhausted. Injected hints are recorded into `applied_hints` (the same
key HintAgentLoop uses), so the reward penalty + budget ratchet are unchanged. No
`<hint_call/>`, so no over-budget protocol violation, no box-then-call, no front-loading.

**Selector → multi-round Template F** (`selector_multi.py`, vendored from
`selector/test_cite/.../multi-cite-gpt-eval/prompt_template_multiF.py`). The WHOLE pool is
rendered with a per-hint `status` (`completed` | `pending`): hints already given/verified
are `completed` (never re-selected); the selector picks the next `pending` hint and reports
in `completed_hints` any pending hint the student newly achieved that round, each with a
verbatim `quote`. `HintSelector.select_multi(problem, trace, pool, completed)` added (shares
the round-robin/failover `_complete` with the single-pick `select`). A rollout tracks its
`completed` set across rounds = given hints ∪ self-achieved cited hints.

**Reward** = the existing `hint_reward.compute_score`, run with the `<hint_call/>`-specific
terms OFF (`hint_call_reward=0`, `hint_shape_coeff=0`, `no_hint_penalty_factor=0`,
`finalize_incorrect=false`), so it reduces to `correct ? max(correct+format−penalty, floor)
: incorrect`. Per-hint penalty (`strategy=hint`), since the multi-round selector gives one
substep hint per round.

**Verified-prefix gradient mask** (the novel part; `auto_hint_mask.py`, applied in
`HPRLRayPPOTrainer._update_actor` before the actor update, gated on
`data.hprl.auto_hint.enable`):
- **advantage ≤ 0 → train on ALL model tokens** (a worse-than-average rollout is pushed
  away in full);
- **advantage > 0 → per the user's split:** a turn that was **followed by a hint injection**
  is trained only up to its **last selector-verified sentence** (fuzzy-locate each
  `completed_hints` quote in that turn via `locate_quote_end`, keep tokens up to the furthest
  match, **disable the unverified tail**); the **ending turn** (correct answer, or budget
  reached — never followed by a hint) is **always fully promoted**. So a first-try-correct
  rollout *is* the ending turn → full promotion (no empty-prefix degenerate case), and no
  extra "verify-on-correct" selector call is needed.

Mechanics: the rollout records compact `disable_spans` = `[start,end)` response-token ranges
(only the unverified tails). These flatten to `non_tensor_batch` (agent_loop.py:1000-1006);
the trainer zeroes them out of `batch.batch["response_mask"]` (the assistant-only loss mask,
agent_loop.py:484) for positive-advantage rows only — verified against the verl flow
(`_update_actor` receives the batch with `advantages` + `response_mask` already computed;
modifying `response_mask` propagates into every micro-batch loss). `use_kl_in_reward=False`,
so the per-sequence advantage sign read (masked mean) is exact.

**Metrics:** `auto_hint/{rows_with_spans, rows_pos_masked, mask_tokens_dropped}` (mask effect)
and **`auto_hint/cite_found_rate`** (+ `cite_quotes_found`/`_total`) — of the selector's
`completed_hints` quotes, the fraction that fuzzy-locate in the trace it was shown (selector-
citation honesty; a low rate means the mask anchors on fabricated/paraphrased quotes). Counted
per rollout in the loop (against the full `build_trace`), aggregated in
`HPRLRayPPOTrainer._auto_hint_cite_stats`, surfaced to wandb via `_update_actor`.

**Data.** `dataset/dapo-3139-auto-hint.parquet` (`build_auto_hint_data.py`): the plain
single-turn prompts joined by `problem_id` with the hint pool / `hint_full` / budget from
`dapo-3139-hint-verl-mt-clean` (same 3139 set, 3139/3139 overlap). `extra_info.hprl_auto_hint`
makes `HintBudgetDataset` update only the loop's budget, never the (hint-agnostic) prompt.
`prepare_hint_data.py --mode auto_hint` builds the same from a fresh pre-upgrade source.

**Adaptive ratchet option** (`data.hprl.ratchet_mode`, default `downward`; env
`HPRL_RATCHET_MODE`). The existing downward ratchet is strictly monotone-down and, in
auto-hint, very aggressive (a single 0-hint first-try solve snaps B_q→0). `ratchet_mode=adaptive`
(`budget_manager.compute_adaptive_budget`) is two-sided: **no correct rollout → RAISE B_q by 1**
(clamped to `data.hprl.max_budget`, default = turn cap), **over half correct (2C>N) → set B_q
to the (N/2)-th smallest correct hint count** (no decrement), else hold. Applies to the
single-pack path (k-pack-probed problems still use the k-pack rule). Self-tests in
`test_auto_hint.py`.

**Run:** auto-hint is now the **DEFAULT** — `bash run_hprl_qwen2.5_7b.sh` runs it
(`HPRL_AUTO_HINT=true`, set in the Paths section, which pins the auto-hint train/val files
+ per-hint penalty + disabled `<hint_call/>`-reward terms + `data.hprl.auto_hint.enable=true`
via `:=` defaults you can still override). `run_auto_hint_qwen2.5_7b.sh` is now just a
distinctly-named alias. **To run the legacy `<hint_call/>` job: `HPRL_AUTO_HINT=false`**
(restores the `<hint_call/>` train file, `major_step`, finalize-incorrect, k-pack; the mask
no-ops since those rollouts carry no `disable_spans`). k-pack defaulted OFF under auto-hint.

**Tests:** `test_auto_hint.py` — 11/11 (mask transform sign-gating/clamping, the fuzzy
locator across exact/whitespace/unicode-LaTeX/no-match, status rendering, pool exhaustion).
Verified the run-script var resolution: default → auto-hint, `HPRL_AUTO_HINT=false` → the
exact legacy `<hint_call/>` settings, user env overrides win.

### Files touched this session
- `selector_multi.py` *(new)* — vendored multi-round Template F prompt + `render_hints_with_status` / `pending_hint_ids` + fuzzy `locate_quote_end`
- `auto_hint_mask.py` *(new)* — pure positive-advantage span-zeroing transform (duck-typed torch/numpy)
- `auto_hint_agent_loop.py` *(new)* — `AutoHintAgentLoop` push-hint rollout
- `build_auto_hint_data.py` *(new)* — join plain prompts + hint payload → auto-hint parquet
- `run_auto_hint_qwen2.5_7b.sh` *(new)* — auto-hint run wrapper
- `test_auto_hint.py` *(new)* — mask + locator unit tests
- `hint_selector.py` — `select_multi` + shared `_complete`
- `hprl_ray_trainer.py` — `_hprl_apply_auto_hint_mask` in `_update_actor`; `disable_spans` in the rollout dump
- `hint_dataset.py` — `_update_auto_hint_budget` (budget-only, no prompt re-render)
- `budget_manager.py` — `compute_adaptive_budget` (two-sided rule) + `BudgetManager` `ratchet_mode`/`max_budget`
- `prepare_hint_data.py` — `--mode auto_hint` (plain prompt)
- `hint_agent_config.yaml` — register `auto_hint`
- `config/hprl_trainer.yaml` — `data.hprl.auto_hint` block
- `run_hprl_qwen2.5_7b.sh` — `HPRL_AUTO_HINT` flags (default off), overridable run name

---

## 2026-06-14 — `v4_cite` selector prompt adopted; selector dump made self-contained; reward reconfigured (~2× compression) for the next run

**Selector prompt → `v4_cite` (Template E).** Swapped `utils.selector_prompt` from `v2_final_gate` to
`v4_cite` (the Round-3 winner, devlog 2026-06-11 entry): inserted the **"Cite your evidence"** workflow
section (a verbatim student-quote citation for every earlier candidate step) and a `completed_steps` array
at the head of the `<output>` JSON; module docstring updated to match. Rendered template verified
byte-identical to Template E in `selector/prompt_improvement_progress.md`. **Drop-in**: the extra
`completed_steps` field is parsed-but-IGNORED by the loop (the guard that would consume it is the TODO
above). Real-failure-case numbers it buys: unearned final-step reveal **9.4%→6.4%** @T0.7 / **7.0%→4.2%**
@T0.1 (→3.0%/0.8% once the enforcement guard lands).

**Parser unaffected — verified, no change.** `parse_output` keeps the new key on the happy path
(json.loads of the whole block) and every consumer reads fields by name, so `completed_steps` is inert
downstream. `_hard_parse`'s target keys (`major_step_id`/`hint_id`/`hint`/`reasoning_*`/`confidence_*`) are
**disjoint** from `completed_steps`' keys (`step_id`/`quote`/`why`) → no false matches; scanning the whole
completion it still recovers the real scalar fields even when the (now leading) `completed_steps` array is
the malformed part — confirmed by a test forcing the hard path with a raw newline + unescaped quote inside
a `quote` value (recovered `major_step_id`/`hint_id`/`hint` correctly). `max_tokens=16000` leaves ample
headroom for the longer output. Noted but NOT applied (rare, ≤~1%, no crash): tighten `_as_selection_dict`
to a selection-shaped-dict check so the degraded-parse fallback can't latch onto a nested `completed_steps`
entry.

**Selector dump now self-contained.** Added `"problem"` to `_dump_selector_call`'s record
(`hint_agent_loop.py`), beside `trace`/`candidate_hints_str` — the three inputs to `selector_prompt()`. A
dump row now re-renders the EXACT prompt the selector saw
(`selector_prompt(row["problem"], row["trace"], row["candidate_hints_str"])`), so `v4_cite` citations can
be audited offline against live calls (`selector_raw`/`selection` carry `completed_steps`; `trace` retains
the `[hint given]` markers for `analyze_citations.student_only`). Dump stays gated on
`HPRL_SELECTOR_DUMP_DIR` (on by default → `${EXP_LOG_DIR}/selector_calls`).

**Reward reconfigured for the next run** (`run_hprl_qwen2.5_7b.sh`, vs the `20260612-171235` snapshot) —
the whole reward compressed ~2× and k-pack flipped on:
- `correct_reward 1.0→0.9`, `incorrect_reward −1.0→0.0`, `hint_penalty_total 1.8→0.8`,
  `hint_shape_coeff 0.3→0.15`, `hint_call_reward 0.1→0.05`; `n_resp_per_prompt 16→32`;
  `HPRL_KPACK_ENABLE=true` (k=2, require_successes=2, scale_mini_batch=true).
- **Score landscape** (format=0.1): correct/0-hint **1.0**, correct/full-budget **0.20** (= floor),
  incorrect/+hint 0.15, incorrect/no-hint 0.10, over-budget·malformed 0.0. Group span **2.1→1.0**.
- **Analysis (under `norm_adv_by_std_in_grpo=False`).** Std-norm is off, so the span compression
  ≈**halves the advantage/gradient** → gentler updates (plausibly intended after the prior
  over-call/abstention collapse). `incorrect_reward=0.0` is a no-op *alone* under critic-free GRPO (only
  within-group spread matters); its effect is purely via the span. Bonus/penalty ratio ≈ preserved
  (0.056→0.0625); the absolute hint-call pull is halved → mildly *less* over-call pressure. The
  correct-branch room `= correct+format−floor = 0.80` exactly equals `penalty_total=0.8`, so the penalty
  sits **exactly at the floor-saturation knee** — any `penalty_total≥0.8` is inert, and the solve-vs-fail
  margin is now the hardcoded floor `+0.05` (was an un-pinned 0.10 when penalty 1.8 < the old 1.85 knee).
  `budget_exceeded_reward` (unset) inherits `incorrect_reward` → floor now **0.0** (was −1.0); the
  0.10-below-a-normal-failure deterrent is preserved in relative terms. `n 16→32` keeps 16 rollouts/pack
  under k=2 at the cost of **2× rollout generation**; `ppo_mini_batch` auto-scales 32→64.

**Decisions taken (discussed, no change made).**
- **Keep the floor `+0.05`** (declined shrinking to `+0.03`): at `penalty_total=0.8` it is nearly inert
  (only un-clamps 0.02 of already-unused room) and it IS the anti-suppression safety margin — wrong lever
  to trim in a std-off, already-halved-gradient regime. Matched levers instead: `correct_reward` (branch
  width) for more frugality bite, `hint_call_reward` (incorrect branch) for less over-call.
- **Keep `norm_adv_by_std_in_grpo=False`.** Std-on rescales every group to unit variance, which promotes
  the only signal varying in all-wrong groups — the **hint-call bonus** (span ≤0.15) — up to parity with
  the outcome signal from mixed groups, i.e. it amplifies the exact gradient behind over-calling.
  Compounding risks here: no `filter_groups` (we run `use_dynamic_bsz`=token batching, NOT DAPO dynamic
  *sampling*) to drop degenerate groups, and no KL (`kl_coef=0`) to dampen the `std→0` blow-up; it would
  also nullify the deliberate scale compression. Revisit only if easy/hard prompts are seen contributing
  ~zero gradient, and only paired with `filter_groups`.

**Follow-ups (not done).** (1) the citation-enforcement guard (see TODO); (2) optional `_as_selection_dict`
shape guard; (3) stale comment at `run_hprl_qwen2.5_7b.sh` ~L162 ("Subtracted from the CORRECT reward only
(incorrect stays at -1)") — `incorrect_reward` is now 0.0, comment wrong. Next: a live run with `v4_cite` +
the reconfigured reward + k-pack.


## 2026-06-13 (i) — k-pack redesigned: split each problem's n rollouts into k packs (supersedes (h))

**Why.** The (h) build realized k-pack as cross-problem SUBSTITUTION (probe packs displaced
already-solved rows; gated by last-step success; capped by `max_probe`). On review the user
specified a cleaner shape: **keep total rollouts at `train_batch × rollout.n`, probe EVERY
problem, and split each problem's own `n` rollouts into `k` packs of `n/k`** at budgets
`B, B−1, … B−k+1`. No gating, no substitution, no dropped problems.

**Mechanics (the crux: verl repeats every prompt row uniformly by `rollout.n` and groups by
`uid`).** To get `k` groups of `n/k` per problem, the pre-repeat batch must be `N×k` rows and
the repeat factor `n/k`:
- `fit()` override → `_hprl_apply_kpack_split_config` (once, before `super().fit()`): validate
  `rollout.n % k == 0` (**the user's requested check — raises `ValueError` otherwise**), set
  `rollout.n → n/k`, and scale `actor.ppo_mini_batch_size × k` (default) so the PPO mini-batch
  SAMPLE count (`ppo_mini_batch_size × rollout.n`, ray_trainer.py:1311) is unchanged — the update
  is byte-identical, only GRPO grouping/budgets differ. Verified safe: the rollout ENGINE never
  reads `rollout.n` (agent-loop = one request per repeated row; grep of `verl/workers/rollout` +
  `agent_loop` is clean), GRPO ignores `num_repeat` (groups by `uid`), and validation uses
  `val_kwargs.n`.
- `_get_gen_batch` → `_hprl_expand_kpacks` now GROWS `N → N×k` (every row gains `k−1` variants at
  `clamp(B−1)…clamp(B−k+1)`); after the `÷k` repeat the post-repeat count is `N×k×(n/k) = N×n`,
  identical to a non-kpack run, so the no-auto-pad train-path constraint (memory
  `verl-no-train-path-autopad`) is satisfied without padding. Floor clamp collapses redundant
  sub-budgets (a problem already at the floor → repeated `min_budget` packs; the callback then sees
  1 distinct budget → falls back to the single-pack downward rule).

**Removed (the substitution machinery):** `budget_manager.plan_kpack_substitution` + its self-tests,
`BudgetManager.record_stats/get_stats/_stats` (+ the `stats` JSON key), the callback's `record_stats`
call, and the `gate_min_correct` / `max_probe_problems` knobs (config + `HPRL_KPACK_*` env). Added the
`scale_mini_batch` knob (`HPRL_KPACK_SCALE_MINI_BATCH`, default true). **Kept:** `compute_kpack_budget`
(the pooled `require_successes`-th-smallest rule), `update_group_kpack`, `kpack_expand.render_variant_rows`,
and the callback's pool-by-`problem_id` + `>1-distinct-budget → k-pack rule` dispatch (`current_budget =
max` over packs).

**Tests.** `python budget_manager.py --selftest` and `python test_kpack_expansion.py` (now: prompt
re-render, variant build with source rows untouched, the `N→N×k` split + floor clamp, callback dispatch)
both green; and `test_kpack_real_verl.py` (verl env python) drives the real `fit`-config split +
`_get_gen_batch` on a real `DataProto` and asserts the headline invariant — **total rollouts after the
repeat == `N × the ORIGINAL rollout.n`** (`128` for `N=4, n=32`), plus `rollout.n 32→16`,
`ppo_mini_batch 32→64`, the `n%k` `ValueError`, every problem spanning `{B,B−1}`, and the no-`uid`
validation no-op. Launch wiring unchanged (`HPRL_KPACK_*` → `data.hprl.kpack.*` hydra CLI overrides).
Still pending: a full live cluster TRAINING run with `HPRL_KPACK_ENABLE=true` (needs `rollout.n` a
multiple of `k`, e.g. 32 & k=2 → packs of 16).

**Follow-up fix (first cluster launch).** The first `HPRL_KPACK_ENABLE=true` launch crashed at step 1
in verl's `_write_generations` (the OPTIONAL rollout-generation dump, `rollout_data_dir`) with
`IndexError`: it builds `gts` via `for item in batch`, which integer-indexes until a non_tensor
column runs out — so any non_tensor key SHORTER than the tensor batch makes `gts` short and the dump
dies. Training itself is unaffected (GRPO groups by `uid` and the trained tensors are full length —
both verified `N×n`); only the dump's full-non_tensor scan trips. Offline repro of the k-pack
expand→repeat→union path is clean (all `N×n`), so the short key is reward/agent-loop/env specific and
not reproducible offline. Made `_log_rollout_data` self-healing: it logs any length-mismatched
non_tensor key (names the culprit) and DROPS it just for the dump, then guards the whole dump in
try/except so it can never crash training. Verified against real verl that a short key makes `for
item in batch` stop early and that dropping mismatched keys restores full iteration
(`test_kpack_real_verl.py`). Next launch's log line "rollout dump: non_tensor keys with len != batch
len" will name the key for a root fix; immediate workaround is to unset `rollout_data_dir`.

**k-pack training CONFIRMED working on the cluster (run `…-20260614-004248`).** Step 1 logged healthy
metrics — `hprl/n_problems:128`, `hprl/kpack_num_probed:90`, `hprl/kpack_num_ratcheted:5`,
`hprl/budget_mean:2.82` (min 0/max 5), normal `pg_loss`/`grad_norm` — and rollout dumps for steps
1–10 wrote fine. The only failure is the rollout dump at a LATER (resumed) step: a STRUCTURAL
non_tensor key (NOT a reward key — the dumped columns are all per-rollout reward keys) goes short
data-dependently. The crashed job was a RESUME (ray session 07:41) that ran PRE-fix code (the
traceback shows the un-wrapped `_log_rollout_data` super() call propagating uncaught). Also added a
`_shutdown_dump_executor` override (backstop): verl re-raises a failed background dump there too
(ray_trainer.py:1399/1758/1770), OUTSIDE `_log_rollout_data`, so a dump error at shutdown/checkpoint
could still crash — now swallowed+logged. Net: a relaunch/resume on the live code self-heals the dump
and names the short key.

**Correction + real root cause of the CRASH (runs `…-020140`).** Two earlier theories were wrong:
(1) the short column is NOT a non_tensor key — the live run printed no `mism`, so dropping non_tensor
keys can't help; (2) "self-heal" was insufficient. The actual reason the JOB DIES is a verl bug in
the background-dump executor: `_dump_generations` re-raises a failed background write via `f.result()`
and then SKIPS clearing `self._dump_futures` (the clear line is after the raise) — so the SAME failed
write re-raises on EVERY subsequent step (caught by my `_log_rollout_data` guard, hence the repeated
"rollout dump failed" prints + an 8-deep nested `_log_rollout_data→_dump_generations→f.result()`
traceback) and ultimately propagates UNCAUGHT at shutdown. Fix: override `_dump_generations` to swallow
the surfaced error AND **purge all DONE futures** so a bad write can never re-surface (standalone repro:
verl re-raises 5/5 steps; override → 0 uncaught). The dump is now genuinely non-fatal regardless of WHY
a write fails. Replaced the non_tensor-only check with a full **`dumplens` diagnostic** that prints
EVERY dump column's length (tensor + non_tensor + reward) each step (`DUMP LEN MISMATCH` when one is
off) — the next run will finally NAME the short column for a root fix. The short column itself (why the
write fails) is still unidentified — offline repros are all length-consistent; it's data-dependent on
the cluster.


## 2026-06-13 (h) — k-pack counterfactual-probe budget ratchet (built)

**What.** Implemented the counterfactual-probe ratchet from the TODO, generalized from the
user's "double-rollout" to a **`k`-pack** probe. For a recently-solved problem at budget `B`,
the trainer also rolls out probe packs FORCED to `B−1 … B−k+1`; each pack is its own GRPO
group, and the ratchet pools their correct rollouts to read *true need* directly instead of
the corrupted "the policy always spends its budget" signal (the motivation analysis in the
old TODO / `logs/experiment_stats.md`). Flag-gated under `data.hprl.kpack.enable`, **default
off** → the single-pack downward ratchet runs byte-for-byte as before.

**The rule (k>1, user 2026-06-13).** Gather every correct rollout across all `k` packs; set the
new budget to the smallest `B'` with `≥ require_successes` correct rollouts at `≤ B'` hints —
i.e. the `require_successes`-th smallest pooled correct hint count (`require_successes` default
**2**, the guard against a lone answer-leak/fluke solve; see `hprl-answer-leak-major-step`).
Downward-only, no upward. `k=1` keeps the existing `compute_downward_budget`. A deep probe that
solves frugally pulls the budget down multiple levels in one update.

**Key design pivot — length-preserving SUBSTITUTION, not growth.** Background verification found
this verl build does **not** auto-pad the actor TRAIN path (only the eval path pads): the
per-step row count must stay divisible by `ppo_mini_batch_size` or the dispatch/`_balance_batch`/
`make_iterator` asserts crash (recorded in memory `verl-no-train-path-autopad`). So the expansion
does **not** add rows — it SUBSTITUTES probe packs for an equal number of already-solved,
not-probed rows (`plan_kpack_substitution`), keeping `len(out) == len(in)`. Every divisibility
invariant stock verl already satisfies therefore still holds; no padding, no extra memory, same
step time. (True growth would need `E·n` to be a multiple of `512`, or pad+mask — noted for later.)

**How it threads through verl (no core edits — `verl-changes-flag-gated`).**
- Each dataloader row already gets a fresh `uuid` → its own GRPO group, *before* the rollout
  repeat. So `k` budget-variant rows for one problem naturally form `k` groups; the ratchet
  pools them by `problem_id`. The agent loop re-tokenizes from `raw_prompt` (not the dataset's
  `input_ids`, which is a throwaway `dummy_tensor` here), so a probe variant is fully defined by
  re-rendering its messages + `tools_kwargs` budget — no tokenization in the trainer.
- `HPRLRayPPOTrainer._get_gen_batch` (train-only; guarded on `uid` present, which validation
  lacks) expands gated problems in place before the uid/ repeat, then defers to super.

**Files.**
- `budget_manager.py` — `compute_kpack_budget` (the pooled rule), `plan_kpack_substitution` (pure,
  length-preserving probe/drop planner), `update_group_kpack`, per-problem last-step stats
  (`record_stats`/`get_stats`) for the probe gate, persisted in the state JSON; shared
  `get_create_budget`/`set_create_budget` moved here (verl-free); self-tests extended.
- `hint_prompt.py` — `rerender_messages_for_budget`, the single prompt re-render shared by the
  dataset and the probe expansion (byte-identical at any budget); `hint_dataset` refactored onto it.
- `kpack_expand.py` (new, verl-free) — `render_variant_rows`: deep-copies `select_idxs`'d rows
  (which alias their source) then re-renders prompt + tool budget, stamps a fresh `uid` and a
  unique negative `index` (keeps the rollout-trace counter from merging packs). Source rows untouched.
- `hprl_ray_trainer.py` — `_get_gen_batch` override + `_hprl_expand_kpacks` (classify → plan →
  `select_idxs`/`render_variant_rows`/`concat`); probe summary metrics merged into the step.
- `hint_budget_callback.py` — pools packs by `problem_id`, detects probed problems (>1 distinct
  budget) → k-pack rule with `current_budget = max` over packs, records gate stats, `hprl/kpack_*`
  metrics.
- `config/hprl_trainer.yaml` + `run_hprl_qwen2.5_7b.sh` — `data.hprl.kpack.{enable,k,
  require_successes,gate_min_correct,max_probe_problems}` knobs (+ `HPRL_KPACK_*` env).

**Gate / cost.** Only problems with `≥ gate_min_correct` correct rollouts last step are probed
(probing all-wrong problems is wasted — they're all-wrong at `B−1` too). Drops prefer
already-solved (gated) rows so hard-problem training is preserved; default `max_probe_problems
= ⌊train_batch_size/k⌋` keeps substitution always feasible. Because a problem keeps being probed
as long as its normal pack keeps succeeding, stuck-problem re-probing is largely automatic (the
explicit periodic re-probe and the symmetric `B+1` upward probe from the TODO are NOT built).

**Tests.** Three suites green: `python budget_manager.py --selftest` (rule + planner length-
preservation/gating/dedup/feasibility); `python test_kpack_expansion.py` (verl-free: prompt
re-render, variant construction with **source rows provably untouched**, substitution pooling,
and the callback's k-pack dispatch via a mock batch); and `test_kpack_real_verl.py` run with the
verl env python (`/share5/users/xutao.ma/miniconda3/envs/verl/bin/python`) — drives the ACTUAL
`HPRLRayPPOTrainer._get_gen_batch` against a real `DataProto`, confirming verl's
`select_idxs`/`concat`/`repeat`/pop behave (length preserved, probed problem spans {B,B-1} in
both `extra_info` and `gen_batch`, `batch.repeat(n)` stays aligned with `gen_batch.repeat(n)`, and
the no-`uid` validation guard no-ops). That smoke test surfaced two test-only setup bugs (reading
the popped top-level `tools_kwargs` instead of the surviving `extra_info.tools_kwargs`; missing the
master `data.hprl.enable` in the fake config) — the production paths were correct. Launch wiring
verified: `HPRL_KPACK_*` env → `run_hprl` → `data.hprl.kpack.*` hydra CLI overrides (via
`launch_hprl_cluster.sh` → `ray_cluster_launch.sh`). Still pending: a full live cluster TRAINING
run with `HPRL_KPACK_ENABLE=true` to watch `hprl/kpack_*` + budget-descent curves.


## 2026-06-12 (g) — box-then-call rate as a live training metric

**Why.** The box-then-call pathology (the policy boxes an answer and then emits `<hint_call/>`
in the same turn) was only visible OFFLINE via `logs/plot_tool/hint_call_with_box_analysis.py`
parsing rollout dumps. On run `…v3-20260612-171235` it climbed to ~67% of all hint calls / ~73%
of hint-using rollouts by step ~30 -- worth watching live, not just post-hoc.

**Fix.** Count it at the source. In `hint_agent_loop._handle_generating_state`, at the
`_is_hint_call(text)` branch (where `text` is exactly the emitting turn), bump
`extra_fields["hint_calls_total"]` and, when `\boxed{` is in that turn,
`extra_fields["hint_calls_with_box"]`. Counts every detected call -- served, selector-failed, and
the over-budget terminal one (all pass through this branch before the budget check). Plumbed
through the reward (both return dicts) + colocate pull-through, then aggregated in
`hint_budget_callback`:
  * `hprl/hint_call_with_box_frac` -- per CALL (with_box / total calls)
  * `hprl/hint_call_with_box_rollout_frac` -- per rollout (over ALL rollouts)
  * `hprl/hint_call_with_box_frac_of_hinting` -- per rollout (over hint-USING rollouts)

Matches the offline tool's definitions (`\boxed{` presence in the calling turn, not "is the final
answer"). Verified the keys flow on both reward branches + legacy default via the stubbed
`compute_score` test. Populates from the next run.

**Files touched**
- `hint_agent_loop.py` — count calls + box-in-emitting-turn at `_is_hint_call`; `setdefault` init
- `hint_reward.py`, `hint_reward_manager.py` — pass the two counters through
- `hint_budget_callback.py` — `hprl/hint_call_with_box_{frac,rollout_frac,frac_of_hinting}`

---

## 2026-06-12 (f) — reliable selector-latency logging (verl hint_select timer was dropping to 0.0)

**Finding.** A per-step timing breakdown of run `…v3-20260612-171235` (parsed from the console
`timing_s/*`) showed rollout generation (`gen`) at **64.6%** of each step and the gradient
(`update_actor`) at **24.1%** — and `gen` **tripled** over the run (102s avg over the first 5
steps → 346s over the last 5) as hint usage ramped (137 → ~1315 hinted rollouts/step). But the
hint-selector cost was **invisible**: `timing_s/agent_loop/hint_select/{mean,max,slowest}` logged
a flat **0.0** across all 28 steps despite 27,587 selector calls.

**Root cause (verl core, not ours).** `simple_timer("hint_select", agent_data.metrics)` accumulates
correctly, but `AgentLoopOutput.metrics` is a typed pydantic `AgentLoopMetrics`
(`verl/experimental/agent_loop/agent_loop.py:79`) with only `generate_sequences` / `tool_calls` /
`compute_score` / `num_preempted`. The agent loop passes the raw `agent_data.metrics` dict into
that field (`tool_agent_loop.py:197`); pydantic **drops the undeclared `hint_select` key**, so the
per-sample `model_dump()` (`agent_loop.py:989`) never carries it and `_performance_metrics`'
`metric.get("hint_select", 0.0)` (`:1137`) reads 0.0. The READ side was added to verl core but the
model field never was — an incomplete integration. One-line verl fix would be
`hint_select: float = 0.0` on `AgentLoopMetrics`, but per our no-core-edits rule we instrument on
the override side instead.

**Fix — time the selector on the reliable extra_fields path (no verl edit).**
- `hint_agent_loop.py` — wrap the selector await in `time.perf_counter()` and accumulate
  `extra_fields["hint_select_time"]` (seconds, `+=` across calls) and `["hint_select_calls"]`
  (count). extra_fields DOES survive to `non_tensor_batch` (same path as `hint_call_failed`), unlike
  `agent_data.metrics`. Kept the `simple_timer` too — it starts working for free if verl ever adds
  the field. Also recorded **per-call** `select_latency_s` into each `selector_calls/*.jsonl` dump
  record (off unless `HPRL_SELECTOR_DUMP_DIR` is set).
- `hint_reward.py` / `hint_reward_manager.py` — pass `hint_select_time` / `hint_select_calls`
  through to the result dict (both branches) and the colocate pull-through.
- `hint_budget_callback.py` — new metrics `hprl/hint_select_time_{sum,mean,max,per_call}`
  (`per_call` = total seconds / total calls = the undiluted round-trip cost).

**Analysis artifacts** (in the run dir): `step_timing.{csv,png}` (per-step phase breakdown +
share pie) via the new `step_timing_analysis.py`. Selector time for THIS run stays unrecoverable
(folded into `gen`); the new metrics only populate from the next run.

**Files touched**
- `hint_agent_loop.py` — perf_counter selector timing → extra_fields + per-call dump field
- `hint_reward.py`, `hint_reward_manager.py` — pass the two fields through
- `hint_budget_callback.py` — `hprl/hint_select_time_*` metrics
- new: `step_timing_analysis.py` (per-step timing breakdown tool)

---

## 2026-06-12 (e) — over-budget hint call → floor score, answer ungraded (protocol violation)

**Decision.** An over-budget `<hint_call/>` — the policy emits the sentinel after its
per-problem budget `B_q` is spent — is now treated as a hard **protocol violation**, not a
benign no-op. Previously the rollout just terminated (under `terminate_on_budget_exceeded=true`)
and the reward still graded whatever boxed answer was in `solution_str`, so a model that boxed a
correct answer and then tacked on an illegal hint call still scored a clean solve. The framing
question — grade for *correctness* (the box is right → credit it) vs. *instruction-following*
(an over-budget call is illegal → no credit) — was settled in favor of instruction-following:
asking for help it cannot have ends the rollout at the floor, **answer not graded**.

**Mechanism (flag in the loop → short-circuit in the reward).**
- `hint_agent_loop.py` — the budget-exhausted branch in `_handle_processing_tools_state` now
  *unconditionally* sets `extra_fields["hint_budget_exceeded"]=1` and terminates (the old
  terminate-vs-"nudge to finish" fork is retired — no free chance to finish after an illegal
  call). Initialized via `setdefault(..., 0)` on every rollout for `DataProto.concat` safety.
- `hint_reward.py` — `compute_score` short-circuits on `extra_info["hint_budget_exceeded"]`:
  returns `score = budget_exceeded_reward` (new kwarg, default `None` → `incorrect_reward`),
  `acc=0`, and **skips grading entirely** (no `grade_answer`, no penalty/shaping/call-bonus).
  `applied_hints` / `hint_call_failed` reads hoisted above the branch so both paths share them;
  `hint_budget_exceeded` added to the normal return dict (=0.0) for a consistent schema.
- `hint_reward_manager.py` — colocate-path pull-through so the flag reaches `extra_info` on both
  rollout paths (inline path already merges all of `tool_extra_fields`).

Because floored rollouts carry `acc=0`, the downward ratchet already treats them as failures —
no `budget_manager` change needed.

**Diagnostic.** New `hprl/hint_budget_exceeded` (count) and `hprl/hint_budget_exceeded_frac`
(count / total rollouts) in `hint_budget_callback.py`. Watch the frac against
`hprl/hint_calls_applied` early in a run: it should bite *over-budget* calls only, not chill
legitimate in-budget hint use. Note the penalty lands on the hinted branch, so it pushes the
same direction GRPO already does re: hint suppression.

**Floor value.** Default `incorrect_reward` (−1.0) ties the bottom of the scale but only *equals*
a plain unformatted failure (a formatted wrong answer sits higher at −0.9). To make an
over-budget call *strictly* worse than any honest failure, set
`+custom_reward_function.reward_kwargs.budget_exceeded_reward=-2.0` in the run script (not wired
yet — left at default). Verified all paths with a standalone stubbed-`mathruler` `compute_score`
test (correct-box+violation → −1.0/acc 0; custom floor; normal correct unaffected; numpy 0-d
flag normalizes).

**Files touched**
- `hint_agent_loop.py` — flag + unconditional terminate on over-budget call; `setdefault` init
- `hint_reward.py` — `budget_exceeded_reward` kwarg + floor-score short-circuit; hoisted state
  reads; `hint_budget_exceeded` in both return dicts; docstring
- `hint_reward_manager.py` — colocate pull-through for the flag
- `hint_budget_callback.py` — new `hprl/hint_budget_exceeded{,_frac}` metrics
- `config/hprl_trainer.yaml`, `README.md` — mark `terminate_on_budget_exceeded` retired; document
  the floor-score semantics

---

## 2026-06-12 (d) — effort-gate the hint-call bonus (kill short-CoT-then-hint)

**Motivation (v3 rollout pathology).** Run `…v3-20260612-141816` collapsed to "emit a
~100-char non-attempt, then `<hint_call/>`": the fraction of rollouts calling a hint
climbed **5.8% → 67.8%** over 22 steps (among *failing* rollouts 5.6% → 68.5%),
hints/rollout 0.06 → 1.19, while unaided val acc stayed flat ~0.038 (dipped to 0.015 @
step 10). The `hint_shape_coeff=0.3` effort-shaping penalty was meant to suppress exactly
this and did **not** — `hprl/hint_shape_penalty_mean` was a healthy 0.20 the whole time.

**Root cause — the deterrent and the incentive sat on opposite branches.**
`hint_reward.compute_score` subtracted the shape penalty ONLY in the `if correct:` branch;
the failing branch was `score = base + hint_call_bonus` with no shape term. But acc≈0.15
→ ~85% of rollouts FAIL, and the four reward levels are structurally fixed: correct/no-hint
**+1.10**, correct/hinted 0.40→−0.22 (floored share 0→15%), **wrong/hinted −0.80**,
**wrong/no-hint −0.90**. So a hinted failure ALWAYS beat an unhinted failure by **+0.10**
(the `hint_call_reward`), and that edge sat on 58% of the batch (2384/4096 wrong+hinted @
step 22, mean shape_sum 0.99, `hint_shape_penalty` 0.297 *logged* but 99.5% scored exactly
−0.80 = never subtracted). The penalty's real reach ≈ correct(0.15)×not-floored(~0.46) ≈
**7% of rollouts**, fighting a +0.10 edge on 58% — outnumbered ~8:1 and on the wrong side.
GRPO kept pushing toward the cheapest hinted failure: minimal CoT then a hint. (First run
with `hint_call_reward`, commit `47adae9` — the anti-suppression bonus overshot into
rewarding the cheapest hint use.)

**Fix — effort-gate the bonus on the SAME branch (`hint_reward.py`).** The incorrect-branch
bonus is now `hint_call_bonus = max(0, hint_call_reward − shape_penalty)`, reusing the same
`shape_penalty = coeff·effort_shortfall_sum` the correct branch subtracts; the correct
branch is unchanged. A front-loaded hinted failure (shape_sum≈0.99 → shape_pen 0.30) is
clawed back to **−0.90 = identical to not hinting**; a genuine-struggle hinted failure
(shape_sum≈0) keeps the full +0.10 → −0.80. With `coeff=0.3`, `hint_call_reward=0.1` the
bonus zeroes at shape_sum≥0.333 (measured front-loaders ≈0.99 → fully clawed back). Clamped
at 0 so a shallow hinted failure is never pushed *below* an unhinted one (that would
re-suppress hint use — the very thing the bonus exists to prevent). Invariants preserved:
`correct_floor` −0.75 still > best wrong −0.80; bonus still binary in hint COUNT (no spam
incentive); `hint_shape_coeff≤0` restores the old flat bonus exactly. Verified across all
six outcome cases + the floor by a standalone `compute_score` test.

**Diagnostic.** New `hprl/hint_call_bonus_mean_hinted` in `hint_budget_callback.py` — mean
effective bonus over hinted rollouts; drops toward 0 as the gate claws it back from
front-loaders (the readout for whether the gate is biting on the next run).

**Files touched**
- `hint_reward.py` — effort-gate the incorrect-branch hint-call bonus; docstring/comment
  updates (shape penalty now hits BOTH outcome branches, not correct-only)
- `hint_budget_callback.py` — new `hprl/hint_call_bonus_mean_hinted` metric

---

## 2026-06-12 (c) — budget-0 prompt parity, templated eval sets, standalone re-budget script, train file → 3164

**Motivation (length analysis).** Per-step rollout analysis of run `…v3-20260612-003103`
(`logs/.../length_analysis.{csv,png}`, lengths in Qwen2.5 tokens) split two cohorts:
full response length of **budget-0** rollouts vs. **first-turn** (pre-`<hint_call/>`)
length of hint-call rollouts. Budget-0 length sat flat ~730–800 tok across all of
training; hint-call first turns climbed ~400 → 860 tok (the two cross ~step 60). The
budget-0 cohort's distinct, non-drifting shape traced to a **prompt-style difference**,
not behavior: at budget 0 the prompt dropped the entire tool template (incl. the
"reason step by step … `\boxed{...}`" closer).

**1. Budget-0 now renders the FULL tool template (`hint_prompt.py`).** Removed the
`budget <= 0` early-returns in `render_system`/`render_user`. At budget 0 the system
prompt keeps the tool instruction ("at most **0** time(s)") and the user message ends
with `render_remaining_calls(0)` = "no hint calls remaining, finish on your own." The
budget-0 prompt is now byte-identical to a budget>0 prompt **except the budget digit**,
so the policy sees the same framing/closer with or without hints and the budget-0
length artifact disappears. Both prep AND the dynamic ratchet (`HintBudgetDataset`)
inherit this (shared module). Updated the now-stale "drops the tool instruction"
comment in `prepare_hint_data.py`. Safety: a budget-0 rollout that still emits
`<hint_call/>` hits `hint_agent_loop`'s `len(applied) >= budget` branch → "finish your
solution" nudge — no selector call, no crash.

**2. Eval sets wrapped in the template at budget 0 (`prepare_eval_hint_template.py`,
NEW).** `aime2024.parquet` and `dapo_sample_hard_100.parquet` shipped as bare
single-turn rows (no `agent_name`, no hint pool), so validation ran OUT of the training
prompt distribution. New script re-renders them with the same template at budget 0 →
`dataset/{aime2024,dapo_sample_hard_100}-hint-mt.parquet`. **Budget 0** because these
carry no hint pool to serve (chosen over a positive-budget hollow-call eval;
user-confirmed) — and budget 0 means the agent loop never contacts the selector.
`run_hprl_qwen2.5_7b.sh` `TEST_FILE`/`HARD_TEST_FILE` now point at the `-hint-mt` files;
**originals kept untouched** for the non-HPRL baselines (`run_dapo`/`run_grpo`/
`run_drgrpo`) that still use them. Verified `HintBudgetDataset` re-render is idempotent
(30/30, 100/100) and resolves budget 0 (eval ids absent from the ratchet table; aime
rows have no `problem_id` → baked-0 fallback). `data_source` preserved so verl reports
each set separately.

**3. Budget-setting factored out (`set_hint_budget.py`, NEW).** Splits the budget half
of `prepare_hint_data.upgrade_row` into a standalone tool: input an already-templated
`*-mt.parquet` + a zero-budget id file; recomputes B_q = #major-steps capped at
`--max-budget` (same rule, shared `num_steps`), forces 0 for listed ids, re-renders
system/user from the baked `hprl_system_base`/`hprl_user_base`, and updates
`create_kwargs.{budget,problem}` + `hprl_init_budget`. **In place by default**;
**idempotent** — same `--max-budget` + id set reproduces the parquet byte-for-byte
(verified 3740/3740 on the simplified set). Lets a new easy-bucket curriculum be applied
without regenerating the template or re-joining `hint_full`. ⚠️ pass the same
`--max-budget` prep used (8) or non-zero budgets get silently re-capped.

**4. Applied to `dapo-3164-hint-verl-mt.parquet` + switched TRAIN_FILE.** Ran (3) with
`dataset/unaided_solved_ids.txt` (the same zero set as before): **809/1112** ids matched
this 3164 set → budget 0 (25.6%); resulting dist `{0:809, 2:6, 3:116, 4:1566, 5:645,
6:22}`. Verified 809/809 zero rows have budget-0 + full template, 2355/2355 non-listed
rows keep budget>0 with `hprl_init_budget == create_kwargs.budget`. `run_hprl_qwen2.5_7b
.sh` `TRAIN_FILE` → `dapo-3164-hint-verl-mt.parquet`. (303 listed ids aren't in this set
— expected, it's a subset.)

**Files touched**
- `hint_prompt.py` — budget-0 renders the full template (removed the `<=0` drops)
- `prepare_hint_data.py` — stale budget-0 comment updated
- `prepare_eval_hint_template.py` (NEW) — template eval sets at budget 0
- `set_hint_budget.py` (NEW) — standalone re-budget of a templated mt parquet
- `run_hprl_qwen2.5_7b.sh` — `TRAIN_FILE` → 3164; `TEST_FILE`/`HARD_TEST_FILE` → `-hint-mt`
- `dataset/` — new `aime2024-hint-mt.parquet`, `dapo_sample_hard_100-hint-mt.parquet`;
  `dapo-3164-hint-verl-mt.parquet` re-budgeted in place

---

## 2026-06-12 (b) — budget ratchet: frugal-success primary rule (snap to min-correct)

User spec: *"if the correct rollout has minimum hint calls less than the current
budget, set the next budget to this minimum; else use the current mechanism."* This
promotes the **min-correct** idea (parked in `45d4a7f`, see (c) 2026-06-11) to the
PRIMARY tier and keeps the (N/2)-th-smallest rule as a fallback — resolving the
aggressive-vs-conservative tension by going aggressive *only when a frugal success
exists*.

**New 2-tier rule** (`budget_manager.compute_downward_budget`; m = min hint count
over correct rollouts):
```
m < current_budget  -> new = clamp(m)                  # PRIMARY, UNGATED by C  ("min_frugal")
otherwise           -> (N/2)-th-smallest FALLBACK:
    2C < N          -> unchanged                       # ("unchanged")
    else            -> new = clamp(pivot - decrement)  # ("pivot")
clamp = [min_budget, current_budget]   (strictly downward)
```
- **Ungated by the correct fraction:** a *single* correct rollout that solved under
  budget snaps B_q straight to that count (vs the old `C ≥ N/2` gate). Aggressive by
  design, per the user.
- **The fallback now degenerates** (worth knowing): it is reached only when
  m == current_budget (every success used the FULL budget), so the (N/2)-th-smallest
  pivot == B_q and the branch just squeezes B_q down by `decrement`. Kept the full
  mechanism verbatim — faithful to "use the current mechanism," and robust if a count
  ever exceeds the budget (e.g. the `monotone-down clamp` edge case).

**Metadata / API.** `BudgetUpdate` gains `rule` (`min_frugal`|`pivot`|`unchanged`) and
`min_correct_hint_count`; `as_dict` extended. Fixed a stale dataclass comment
("(N/2)-th-*largest*" → *smallest* — the code always sorted ascending). `hint_budget_
callback` reads only `new_budget`/`changed`/`n_*`, so nothing downstream breaks.
`_selftest` rewritten — **25 checks pass** (frugal: ungated / single-success /
pre-empts-pivot / min_budget-floor / to-zero; fallback: pivot-squeeze / too-few /
no-correct; clamp; manager round-trip + persistence).

**Activation — no new toggle.** It is the *only* downward rule now (replaced in
place), so it is live whenever the ratchet runs: master switch `data.hprl.enable`
(`HPRL_ENABLE`, **default True**). Fires after each step via the `_update_actor` hook
(`hprl_ray_trainer.py`) → `hprl_update_budgets` groups by problem_id →
`compute_downward_budget` → atomic `budget_state.json` → `HintBudgetDataset`
re-renders next epoch. Knobs unchanged: `HPRL_MIN_BUDGET` (clamp floor, BOTH tiers),
`HPRL_DECREMENT` (FALLBACK only now), `HPRL_DEFAULT_BUDGET`. To take effect:
**relaunch** (imports load once; the run uses the live `${SCRIPT_DIR}/budget_manager.py`,
not the archival `hint_rl_src.*` snapshot) and start from a **fresh** per-exp
`budget_state.json` for a clean test (the callback's monotone-down guard never
re-raises an already-lowered B_q). Confirm via the startup line `HPRL ratchet
enabled: budget_state=…` and the `hprl/budget_*` / `frac_ratcheted` wandb scalars.

**Behavioral note / what to watch.** Materially faster, noisier ratcheting — one
lucky low-hint success drops the whole problem's budget (exactly the trade-off (c)
2026-06-11 parked over). Pushes problems into the harder fewer-hint regime sooner,
synergizing with the new hint-call reward (the hint-call-reward entry below + the
suppression memo). Watch `hprl/budget_*` / `frac_ratcheted` for over-fast collapse;
raise `HPRL_MIN_BUDGET` if budgets bottom out too early.

**Files:** `budget_manager.py` (rule body + `_clamp` helper + `BudgetUpdate`
fields/`as_dict` + module header + docstring + `_selftest`). Memory
`hprl-downward-budget-ratchet` updated. **Status:** unit-tested (25/25); not yet run live.

---

## 2026-06-12 — hint-call reward (anti-suppression bonus) + raised correct floor

A reward-side lever against the **GRPO hint-suppression** pathology (the 2026-06-10
(d) entry + the suppression memo: unhinted-correct advantage ~1.10 out-ranks
hinted-correct ~0.70, so hint emission collapses over training even when hints
work). User framing: *if the model is stuck we want it to call a hint, not plow
straight to a wrong answer.*

**New term (`hint_reward.compute_score`).** A wrong answer now scores
`incorrect_reward + b`; a correct answer is unchanged. `b = hint_call_reward`
(default **0.1**) iff the rollout RECEIVED ≥1 applied hint, else 0.
- **Binary, not per-call** — one applied hint earns the full bonus; extra hints add
  nothing, so there is NO incentive to spam `<hint_call/>` (verified: 1 vs 3
  applied hints both → −0.9). The point is purely to keep a positive gradient on
  hint use among the rollouts that **fail anyway**, where it can't trade off against
  solving (the bonus never applies when correct).

**Gate on APPLIED, not on the `<hint_call/>` emission (user-flagged mid-session).**
First cut also counted `hint_call_failed` (a sentinel the selector failed to serve)
as "called a hint"; reverted. A failed call hands the policy the *"no hint
available, continue on your own"* no-op — zero information to course-correct with,
so it behaves like a never-called rollout. Worse, `hint_call_failed` spikes during a
selector **OUTAGE** (the NIC-routing class, 0 hints ever applied), so rewarding the
bare emission would pay out a no-op bonus *exactly when the mechanism is down*,
training the sentinel as a reward-grab decoupled from hint use. So we gate on
`len(applied_hints) >= 1` and keep `hint_call_failed` as the **metric we watch to
detect outages**, not a rewarded signal. Also made `hint_call_failed` path-
independent in `hint_reward_manager` (colocate merge, mirroring `applied_hints`/
`turn_lens`) so that outage metric is reliable on both reward paths.

**Raised the correct-side floor.** The bonus reopened the floor caveat: correct was
floored at `incorrect_reward` (the WORST wrong score), so a heavily-penalized
correct rollout pinned to −1 could sit just *under* a hinted failure (−1 + b). Fix:
`correct_floor = incorrect_reward + format_reward + hint_call_reward` — the **BEST
achievable wrong score** (well-formed AND hinted). A correct rollout can now never
rank below ANY wrong rollout in the GRPO group, even after the full hint + shaping
penalties, so "solve with hints" can't drop under "fail with a hint." Only the
CORRECT branch is floored; the wrong branch (`base + bonus`) is unclamped.

**Knob + logging.** `reward_kwargs.hint_call_reward` (default 0.1, **0 disables**),
env `HINT_CALL_REWARD` in `run_hprl_qwen2.5_7b.sh`. New `reward_extra_info`:
`called_hint` (applied-hint rate over the batch — **the suppression curve to
watch**, should rise/stabilize, not collapse) and `hint_call_bonus` (mean bonus
actually paid).

**Verified** (stubbed grader on `compute_score`, all defaults `incorrect=-1`,
`format=0.1`, `hint_call=0.1` → floor **−0.8**): wrong+applied → −0.9;
wrong+failed-only → **−1.0 (no bonus)** while `hint_call_failed` still logged;
wrong+no-call → −1.0; correct+no-hints → 1.0; correct+hint(default penalty) → 0.28
(no bonus, not floored); **floor exercised** — correct+heavy penalty pins at −0.8 =
best-wrong (invariant `correct ≥ best-wrong` holds with equality at the extreme);
disable (=0) → −1.0.

**Files:** `hint_reward.py` (kwarg, `called_hint`/`hint_call_bonus`, raised
`correct_floor`, header + docstring), `hint_reward_manager.py` (`hint_call_failed`
colocate merge), `run_hprl_qwen2.5_7b.sh` (`HINT_CALL_REWARD` knob + reward_kwarg).
**Status:** unit-tested; not yet run live.

---

## 2026-06-11 (d) — blind-trace bug LIVE-confirmed, instrumented, and FIXED

Closes item #1 of (d) 2026-06-10's "Still to apply". That bug (entry below, cause
#2) was diagnosed offline via replay; this session caught it on a **live** run,
added runtime instrumentation, proved it 285/285, and **applied the loop fix**.

**Triggering case** (user: "is this successful 2-hint rollout the selector's
fault?"). Run `…-v3-20260611-103735`, step 33, problem
`DAPO-Math-17k-04b3e7c6-5c2b-4fe9-8d4c-6a35fd3e53a8` (gt **150**, 5-step pool,
step 5 states the answer). Group of 16; **idx 3630**: acc 1.0, pred 150,
**num_hints 2, score +0.225 — the single highest in the group** (the two honest
5-hint solves scored −1.0). `applied_hints`: call 0 → major step **1**, call 1 →
major step **5** — the selector **jumped 1→5, skipping 2/3/4**, and step 5's hint
is literally `…cos = −√3/2, hence angle B3MA3 = 150 degrees`. The policy had done
only step 1 (coords + circumcircle) then bogus hand-waving (wrong 75° via a fake
"reflection of orthocenter over circumcenter" / arc-subtension argument) — never
located A1/A2/B1/B2 or computed the MA3/MB3 vectors — then transcribed step 5's
formulas verbatim and copied 150. Textbook unearned reveal, and GRPO upweights it.

**Runtime dump (new tooling).** `hint_agent_loop._dump_selector_call`, env-gated on
`HPRL_SELECTOR_DUMP_DIR`, wired through `run_hprl` (`HPRL_DUMP_SELECTOR`, **default
ON while debugging**) into the Ray `runtime_env` — a local-shell export can't reach
workers across the `launch_hprl_cluster`→ray job-spec boundary, hence default-ON.
Per-worker append to `<run dir>/selector_calls/selector_calls.<host>.<pid>.jsonl`;
each record carries `msg_roles`, `n_assistant_in_messages`, `trace`,
`candidate_hints_str`, `selection`, `selector_raw`. `HPRL_DUMP_SELECTOR=0` disables
(writes a lot over a full run — turn off once a step or two is captured).

**Live proof.** Run `…-v3-20260611-143549`: **285/285** dumped calls had
`n_assistant_in_messages == 0` — call_index 0 → `"(The student has not written any
reasoning yet.)"` ×269 (despite a full first turn already written), call_index 1 →
hints-only ×16. One `selector_raw` self-incriminated: *"the student's trace shows
only the two initial hints… there is no evidence of any work."* build_trace is
structurally blind — confirmed at runtime, not just offline replay.

**THE FIX (applied).** `_handle_generating_state`: after the decode, before the
`_is_hint_call` check, append `{"role":"assistant","content":text}` to
`agent_data.messages`. Every assistant turn (incl. the one ending in `<hint_call/>`,
whose progress summary IS the signal) now reaches `build_trace`.
- **Safe** — verified by reading the base loop: `agent_data.messages` feeds ONLY
  `build_trace` after startup. The trained sequence is built from
  `prompt_ids`/`response_mask` (run() finalize, tool_agent_loop.py:177–204), and the
  initial prompt is templated from messages once in `_handle_pending_state` (PENDING,
  before any turn is appended). No token double-count.
- **Effective** — replayed `build_trace` on rollout 3630's real turns: OLD =
  `[hint given] …` (hints-only); NEW leads with the student's turn-1 reasoning
  ("To solve the problem… reflection… symmetry…"), so the prompt's "Guard the final
  step" rule can finally see steps 2–4 are undone.

**Status / next.** Needs a relaunch (143549 predates the fix); after relaunch the
dump should flip to `n_assistant_in_messages > 0`, then disable the dump. Item #1 of
(d)'s list is DONE; still pending there: `SELECTOR_TEMPERATURE`→0.1, the
deterministic last-step guard in `_record_major_step`, and `v4_cite` (the Round 3
entry above) into `utils.selector_prompt`.

**Files:** `hint_agent_loop.py` (trace fix + `_dump_selector_call` + call-site dump),
`run_hprl_qwen2.5_7b.sh` (`HPRL_DUMP_SELECTOR` toggle + `runtime_env` passthrough).

---

## 2026-06-11 (c) — budget ratchet: revised min-correct rule parked; reverted to N/2-th-smallest

**What changed:** the working tree had an unfinished revision of the downward
ratchet in `budget_manager.py`. I committed it for safekeeping, then reverted the
working copy back to the previously-committed rule — so the **active rule is again
the (N/2)-th-smallest pivot**, and the revised rule lives only in a commit.

**Committed `45d4a7f` — the parked revision** (`compute_downward_budget`):
- gate relaxed from `C >= N/2` to `C >= 1` (any correct rollout);
- `m = min(correct hint counts)` (fewest hints any success used);
  - `m <  B_q` → new budget = `m` (cap at the most frugal success);
  - `m == B_q` → new budget = `B_q - decrement` (squeeze by one);
- still strictly downward, clamped to `[min_budget, B_q]`;
- `BudgetUpdate.pivot_hint_count` → `min_correct_hint_count`; self-tests updated/pass.

**Active rule (restored from `f2d3992`, working tree):**
- gate `C >= N/2` (`2*C < N` → unchanged);
- pivot = the **(N/2)-th smallest** correct hint count; new = `pivot - decrement`;
- clamped to `[min_budget, B_q]`. Self-tests pass.

**Why parked, not adopted:** the min-correct rule is more aggressive — a *single*
lucky low-hint success drops the whole problem's budget to that count, vs. the
N/2-th-smallest which requires half the group to be that frugal. Keeping the
conservative N/2 rule active for now; the aggressive variant is recoverable from
`45d4a7f` if we want to A/B it.

**Caveat (git state):** tip commit `45d4a7f` holds the *min-correct* rule but the
working tree holds the *N/2-th-smallest* rule (uncommitted `M`). A stray
`git checkout -- budget_manager.py` / `git stash` would snap the file to the
min-correct version. Did NOT commit the revert as a second commit (left it as a
working-tree change per request).

**Files:** `budget_manager.py` (rule body + `BudgetUpdate` field + `_selftest`).

---

## 2026-06-11 (b) — hint-call parser tightened: own-line + stop required

**Symptom (from v3 rollout audit, run `…dapo-4k-v3-20260611-004305`):** the policy
learns to **spam** `<hint_call/>`. Counting sentinel emissions per step in
`rollouts/*.jsonl` (4096 rollouts/step):

- Early (steps 1–~24): emission is clean & rare — almost every `<hint_call/>` is
  alone on its own line and the turn stops there (step 1: 318 own+stop vs 46
  inline, ~1 emission/record).
- Crossover ~step 33: **inline overtakes own-line** (1,838 own+stop vs 4,250
  inline).
- Late (steps 40–112): inline dominates ~3–5× and occurrences/record hit ~3–4
  (step 37: 2,821 records-with-sentinel but 10,442 occurrences).
- Totals across all 112 steps: own+stop **162,536** / own-but-continued **9,614**
  / inline **484,132** / occ **656,282**.

**Root cause:** the parser was a bare substring match —
`if self.hint_sentinel in text` (`hint_agent_loop.py`). Generation runs to EOS
(no stop string), so ANY inline `<hint_call/>` buried mid-reasoning still tripped
hint injection. That rewards burying the sentinel, and the format drifts away from
the taught "emit it alone on its own line and stop".

**Fix:** detection now keys off the taught format only. New helper
`_is_hint_call(text)` returns true iff, after `rstrip()`, the turn's **last line**
`.strip()`s to exactly `<hint_call/>` — i.e. sentinel on its OWN line AND it's the
last thing emitted before EOS (the model stopped there). Inline / non-terminal /
`own_cont` sentinels now terminate as normal answer turns instead of injecting.
- `\n<hint_call/>\n` → detected. `…text <hint_call/>` → not. `\n<hint_call/>\nmore`
  → not.

**Scope:** detection-only. EOS-only generation unchanged — inline sentinels are now
ignored, not blocked at the sampler. Did NOT add `<hint_call/>` as a stop string
(would contradict the EOS-only design note); flagged as a possible stronger follow-
up. Shaping reward (`hint_shape_*`) untouched — candidate next step is to reward
the own-line+stop format explicitly rather than silently dropping malformed calls.

**Files:** `hint_agent_loop.py` (new `_is_hint_call`, header + docstring comments).
Syntax-checked; not yet run on a live rollout.

---

## 2026-06-11 — selector Round 3: citation-grounded selection (`v4_cite`) — new winner

User idea: to select step N the selector must **quote the trajectory's words**
that solve every earlier candidate step — verbatim excerpts in a new
`completed_steps` JSON array (`{step_id, quote, why}`). Two payoffs: the model
actually looks for evidence before skipping ahead, and the quotes are
**machine-checkable** (substring vs the student-only trace), so the loop can
deterministically clamp picks justified by fabricated/hint-derived "evidence".

Eval on the same jump_last rows as Round 2 (gpt-oss-20b, full trace):

| condition | fail_last T=0.7 (n=240) | T=0.1 (n=120) |
|---|---|---|
| v2_final_gate (prev winner, in prod) | 9.4% | 7.0% |
| **v4_cite prompt** | **6.4%** | **4.2%** |
| **v4_cite + citation enforcement** | **3.0%** | **0.8%** |

Citation audit (`selector/analyze_citations.py`): skip-ahead picks drop to
17/234 resp. 6/120; of their quotes only 60%/44% validate — the rest are
paraphrases presented as quotes (e.g. "Michael's position is x=5t.", nowhere in
the trace) — and the substring check catches all of them. Enforcement = clamp
skip-aheads with missing/fabricated/hint-only quotes to the earliest unproven
step (no extra model calls). Side benefit: structured output cuts parse fails
11.2% → 2.5% (T=0.7), 0 at T=0.1. The single surviving enforced failure at
T=0.1 had all-valid quotes — likely a genuinely-earned label edge case.

**Recommended production stack (supersedes (d)'s list):** blind-trace fix →
`v4_cite` into `utils.selector_prompt` (extra JSON field is loop-compatible) →
citation-enforcement guard in `_record_major_step` (reuse
`analyze_citations.student_only`/`classify_quote`) → `SELECTOR_TEMPERATURE=0.1`.
NOT yet applied to code. Full details: `selector/prompt_improvement_progress.md`
Round 3 + appendix Template E; raw: `selector/prompt_eval/round3_cite*/`.

---

## 2026-06-10 (f) — ADOPT `no_demo_sbs`: demo removed from the hint prompt

User decision on the Round-14 trade-off (entry (e) below): **no-demo wins** —
"no bias in the reasoning style". The Round-13 long demo seeded the "Step k:"
skeleton that live RL amplified to 86% of first turns with ever-shorter
statements; demo-free, the style is the model's own (~5% step-form, genuine
500+-char statements), pre-1st-call CoT doubles (median 341 -> 607 tok), and
Round 14 showed the demo was NOT needed for sentinel format (0 malformed / 720)
nor for fabrication suppression (the IMPORTANT-STOP sentence does that).

**Changes:**
- `hint_prompt.py` — `TOOL_INSTRUCTION`: Example block deleted; closer keeps
  "reason step by step" but drops "in this style"; IMPORTANT-STOP drops
  "(as shown in the example above)". Comment block now explains the no-demo
  rationale + the Round-14 numbers, and warns the closer choice was re-tested
  demo-free (sbs > briefly on BOTH emit and depth — the Round-8 rule does not
  apply to the no-demo stack).
- Byte-verified `render_system(None, 8)` == the tested arm's
  `_system_prompt.txt` (`round14_20260610_230635/no_demo_sbs`).
- **Parquet regenerated**: `dataset/dapo-3740-hint-verl-simplified-mt.parquet`
  (r13 version kept as `*.r13prompt.bak`; same flags: `--max-budget 8`,
  `hint_agent`, no zero-budget ids). Re-verified all 3740 rows: budgets +
  `hprl_system_base`/`hprl_user_base` identical to r13 bake; baked system ==
  `render_system(base,B)` and last user == `render_user(base,B)` row-by-row;
  demo-free wording 3740/3740, stale wording 0.
- **Prompt token distribution**: mean 1032 -> **352**, median 340, p99 559,
  max **1736** (-680 tok/row = the example block). **0 rows over
  `max_prompt_length=2048`** (r13 had 4 left-truncated) — that caveat is gone.
  Side benefit: ~680 fewer prompt tokens per generation turn.

**Expected live profile / what to watch:** offline emit 19.6% sits EXACTLY at
the 20% floor (multi ~2.5%) — watch `num_hints`/emit from step 1; if the live
reward regime decays it below the floor, the queued fallback is a format-only
one-line example or a prose (non-numbered) demo (progress.md Round 14, last
para). "Step k:" first-turn share should fall from ~80-86% toward ~5%, and
pre-hint `reasoning_len` / `hprl/hint_shape_sum_mean_hinted` should improve
from the prompt alone, compounding with the suffix-max effort shaping ((c)).

---

## 2026-06-10 (e) — Round 14: the long demo causes "Step k:" parroting; no-demo arms tested

Live run `…-v3-20260610-220701` (r13 `long_demo_sbs` prompt) confirms the
user-observed pathology: the policy imitates the demo's surface form — RL step
1: 78.3% of first turns use the "Step k:" skeleton (5.4 segs × 262 chars);
step 11: 86.1% × 214 chars and emit 74→97% — **RL amplifies the parroting and
shrinks the per-step statements**.

Offline Round 14 (`hint_call_test/test_no_demo_round14.py`, NEW **EOS-mode**
multi-turn harness mirroring `hint_agent_loop` — no stop=, sentinel substring
detection, full turn kept; required to audit format + fabrication):

| arm | emit | pre-1st-call med tok | "Step k:" form | format near-miss | post-call cont. | fab voice |
|---|---|---|---|---|---|---|
| shipped (control) | 43.3% | 341 | 44.6% | 0 | 7.5% | 1 |
| no_demo_sbs | 19.6% | **607** | **5.4%** | 0 | 3.4% | 0 |
| no_demo_brief | 16.7% | 374 | 5.0% | 0 | 8.5% | 0 |

**Answers to the two monitored worries:** sentinel FORMAT correctness does not
need the demo (0 malformed attempts in 720 rollouts — the instruction sentence
anchors it), and SELF-FABRICATION stays rare without it (3-9% of call turns
continue past the sentinel, median 0 chars; "User (hint):" voice ≤1/240 —
the IMPORTANT-STOP sentence does this work). The demo's real contributions
are emission (43%→~18% without) and the parroted style.

**Decision pending (user):** `no_demo_sbs` kills the imitation and doubles
genuine pre-call CoT but sits exactly at the 20% emission floor. Candidate
next arms if margin is required: format-only one-liner example, prose demo
(no numbered steps), stronger when-to-call paragraph. No code/parquet change
made this session — `hint_prompt.py` still ships r13 `long_demo_sbs`.
Full table + notes: `hint_call_test/progress.md` Round 14; raw:
`hint_call_test/massive_hint_test/round14_20260610_230635/`.

---

## 2026-06-10 (d) — selector answer-leak: root causes + prompt swapped to `v2_final_gate`

Investigated "the hint selector sometimes gives the answer step directly"
(run `…-v3-20260609-171314`, e.g. step 235 triples problem: all 3 successes got
step 5 = "…coordinate sum is 2+251+252=505" verbatim).

**Three stacked causes:**
1. **Answer-bearing pools + verbatim dump (design):** 96.2% of pools carry the
   literal gt in the LAST step's hints; `major_step` mode (`_record_major_step`)
   injects the whole step unfiltered. User verdict: acceptable **iff earned**.
2. **Blind-trace bug (loop):** assistant turns are never appended to
   `agent_data.messages` (neither in `HintAgentLoop._handle_generating_state` nor
   the base loop), so `build_trace` fed the selector a **hints-only trace** — it
   never saw one character of student reasoning (100% step-1 first picks across
   7,101 sampled calls; same-state 2nd picks diverge 2/2/4/5/5 at conf 5). STILL
   UNFIXED in code.
3. **Selector judgment variance (T=0.7):** replaying real failure inputs
   reproduces the last-step pick only ~32% of the time — production failures are
   the sampling tail drawn 16 rollouts × many calls per problem.

**Scale:** unearned final-step reveals (jump_last P(corr)=0.91 + spamwalk_last
P(corr)=0.64) fully account for **5.0% of late "solved" problems (rising from
1.8%)**; on groups solved ONLY unearned, those rollouts get **+0.66 advantage
(98.8% positive)** → GRPO actively reinforces hint-spam/jump on hard problems.
80.5% of all hint calls follow <500 chars of student text.

**Test set + eval (all under `selector/`):** `unearned_final_step_reveals.parquet`
(21,462 offending calls; `trace_as_seen` replays the blind loop, `trace_full` =
intended trace), harness `eval_selector_prompts.py`, log
`prompt_improvement_progress.md` (full prompts in appendix). Live gpt-oss-20b
(sglang DP=2) results on jump_last cases, fail = picked the answer step:

| condition                                   | fail_last      |
|---------------------------------------------|----------------|
| deployed (blind trace, old prompt, T=0.7)   | 32.1%          |
| trace fix only (T=0.7 / T=0.1)              | 22.6% / 16.1%  |
| **v2_final_gate + full trace (0.7 / 0.1)**  | **9.4% / 7.0%** |

v1 (earned-progress rules) is the big lever; v2 adds the asymmetric final-step
guard; v3 (forced checklist) adds nothing. Residual ~7% = stochastic
"credit-by-understanding" + label edge cases → needs the deterministic guard,
not more prompt.

**CHANGE (this session):** `utils.selector_prompt` → `v2_final_gate` (two inserted
sections: "Ground rules for crediting progress", "Guard the final step"); byte
parity with the tested template asserted against `selector/prompt_variants.py`.

**Still to apply (ordered):**
1. blind-trace fix (append decoded assistant turn to `agent_data.messages`) —
   **the prompt's gains assume it**;
2. `SELECTOR_TEMPERATURE` 0.7 → 0.1 (parse fails 11%→4%, small fail_last gain;
   low temp alone does NOT rescue the old prompt: 16.1%);
3. deterministic guard in `_record_major_step`: refuse last-step reveal while
   unrevealed earlier steps exist (+ no-work check) → catches the residual;
4. optional: marked-revealed-steps protocol for the spamwalk loophole
   (`exclude_applied_steps` currently blinds the selector to unexecuted
   revealed steps).

**Files touched:** `utils.py` (prompt + docstring). New analysis artifacts live in
`selector/` and `logs/<run>/` (leak analyzers), not in this package.

---

## 2026-06-10 (c) — effort shaping made ORDER-AWARE (suffix-max)

Replaced the mean-reference shape penalty (entry below, §2) with an order-aware
one. **Why:** the mean reference is order-blind — it can't express "earlier turns
should reason as hard as later turns," which is the actual goal. A short turn was
scored the same whether it came first or last.

**New rule** (user's recursive idea = the suffix maximum). For the rollout's
ordered turn lengths `L_1..L_T`, reference turn `i` against
`M_i = max(L_i..L_T)` (the hardest reasoning at/after turn i):

```
shortfall_i   = relu(M_i - L_i) / M_i          # in [0,1)
shape_sum     = Σ_i shortfall_i                 # one right-to-left pass
```

User's phrasing — "from the front find the longest turn k, score 1..k against
L_k, recurse on k+1" — is exactly this: for j ≤ k=argmax(L_i..L_T), max(L_j..L_T)=L_k.

**Order semantics (the point):**
- reasoning NON-INCREASING (hard early, ease off) → every `M_i=L_i` → **0** (free);
- RISING (shallow early, long late = front-loading) → penalized;
- longest turn(s) + final solve turn contribute 0 (never penalized — they only
  raise the bar for earlier turns); hint-free single-turn rollout → 0.

**Verified** (same multiset, order matters): `[1200,80,80]`→0.0, `[80,1200,80]`→0.93,
`[80,80,1200]`→1.87; `[80,1200,80,600]`→1.8 (turn1 vs 1200, turn3 vs 600).
Decreasing/flat→0; wrong stays −1; correct still floored at −1.

**Changes:** `hint_reward.py` (`effort_shortfall_sum(turn_lens)` now suffix-max,
no longer needs `applied_hints`/`reasoning_len`); call sites + comments in
`hint_reward.py`/`hint_agent_loop.py` updated. Unchanged: correctness-gating, −1
floor, `HINT_SHAPE_COEFF` (0.3), wandb logging (`hint_shape_sum` /
`hprl/hint_shape_sum_mean_hinted`), and `turn_lens` recording (already ordered).
`reasoning_len` on applied-hints still recorded — now redundant for shaping but
kept as an informative per-hint field in the rollout dump.

---

## 2026-06-10 (b) — Round-13 prompt swap: long CoT before the call (`long_demo_sbs`)

The PROMPT-side complement to the effort-shaping penalty below (same
front-loading pathology, §1 of the previous entry): the policy was calling
`<hint_call/>` after a too-short reasoning stub. The reward fix prices the
behaviour; this changes what the prompt teaches in the first place.

**Offline search** (`hint_call_test/test_long_cot_before_call.py`, Rounds
12-13 in `hint_call_test/progress.md`; multi-turn inject loop, 30 hard x 8,
budget 8, Qwen tokenizer):

| arm | emit | multi(>=2) | pre-1st-call tok (mean/med/p90) |
|---|---|---|---|
| shipped (Round-11 prompt) | 62.9% | 22.5% | 217 / 158 / 464 |
| sbs_closer | 31.3% | 4.2% | 391 / 309 / 777 |
| attempt_first | 15.8% | 0.8% | 307 / 250 / 557 |
| long_demo_attempt | 19.2% | 0.4% | 386 / 386 / 531 |
| **long_demo_sbs (ADOPTED)** | **41.3%** | 5.0% | **377 / 350 / 560** |

Two levers, deliberately INVERTING the Round-8 "no step-by-step" rule now that
emission has headroom (62.9% vs the 20% floor):
1. closer "reason briefly" -> "reason **step by step**" (the CoT trigger we
   once removed; alone it doubles pre-call CoT, emit 31%);
2. the demo's first turn rewritten to ~250 tokens of numbered work (wrong
   first attempt + small-case sanity check -> stuck -> call). Imitation
   anchors the length (median~=mean~350) and RECOVERS emission 31%->41%.
"Make a serious attempt first" wording over-suppresses (15.8% < 20% floor) —
not adopted. Round 12 side-result: the demo is still load-bearing (removing
it: emit 18.8%, multi 2.5%) and costs +350 prompt tok, ~0 output tok.

**Changes:**
- `hint_prompt.py` — `TOOL_INSTRUCTION`: new Step-1..12 long-attempt demo
  (literal braces doubled for `.format`), closer -> "reason step by step";
  comment warns against "fixing" it back to "reason briefly".
- `prepare_hint_data.py` — comment only (strip rationale updated: the ONE
  deliberate "step by step" lives in the closer; `strip_cot_trigger` /
  `strip_user_cot_tail` still strip the unconditional copies in the DAPO
  base system / user tail, which stack and over-suppress).
- No code change needed in `hint_dataset.py` / `hint_agent_loop.py` /
  `hint_tool.py` — they share `hint_prompt` renderers; grep confirms no
  stale wording anywhere else.
- **Parquet regenerated**: `dataset/dapo-3740-hint-verl-simplified-mt.parquet`
  (old one kept as `*.r11prompt.bak`; same flags as live: default
  `--max-budget 8`, `agent_name=hint_agent`, NO `--zero-budget-ids` — the
  live parquet had none baked). Verified all 3740 rows: budgets +
  `hprl_system_base`/`hprl_user_base` byte-identical to old; baked system ==
  `render_system(base, B)` and last user == `render_user(base, B)` row-by-row
  (so the ratchet's dynamic re-render stays byte-identical to the bake); new
  wording 3740/3740, old wording 0/3740. Val parquets (`aime2024`,
  `dapo_sample_hard_100`) are unaided — untouched.

**Prompt token distribution** (full chat-template render, Qwen tokenizer,
`add_generation_prompt=True`; NB `apply_chat_template(tokenize=True)` returns
a 2-key dict in this transformers version — render the string and `encode`):

| parquet | min | mean | median | p90 | p99 | max | >2048 |
|---|---|---|---|---|---|---|---|
| old (r11 prompt) | 601 | 700 | 688 | 764 | 907 | 2084 | 1 |
| new (r13 prompt) | 933 | 1032 | 1020 | 1096 | 1239 | 2416 | **4** |

Every row exactly +332 tok (the demo swap). 91% of rows land in (900, 1100];
p99 1239 << `max_prompt_length=2048`, so headroom is fine. **4 rows (0.1%)
now exceed 2048** (`request-27-1` 2416, `request-94-27` 2220, `request-51-25`
2141, `request-82-57` 2098 — giant problem statements; 1 of them was already
over with the old prompt). With `data.truncation='left'` they get LEFT-
truncated, i.e. the head of the system prompt (tool instruction + demo top)
is cut for those 4 — no crash, negligible mass; filter or special-case them
only if they ever matter.

**Expected live effect / what to watch:** pre-hint `reasoning_len` (and
`hprl/hint_shape_sum_mean_hinted`) should rise from the prompt alone; emit
~41% offline (vs 63%) leaves margin over the 20% floor for RL decay;
multi-call drops to ~5% offline — accepted trade. Synergizes with the
effort-shaping penalty (both push "reason more before calling"); if the live
run shows reasoning_len still ~300 chars at step 1, the parquet in the run
config is stale (check the snapshot's TRAIN_FILE).

---

## 2026-06-10 — hint front-loading, effort shaping, ratchet revision

Covers (1) the hint front-loading pathology and the effort-shaping fix, (2) the
revised downward budget ratchet, (3) new wandb metrics, and (4) the cluster-launch
blocker that has so far prevented any of it from running in-cluster.

---

## 1. The problem: the model front-loads hint calls

**Symptom (user report).** When the policy still has hint-call budget left, it
does very short reasoning and immediately calls a hint; it only reasons hard once
the budget is exhausted. We want the opposite — reason hard always, call a hint
only when genuinely stuck. Front-loading wastes hints.

**Evidence** (run `HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260609-171314`, step 100
rollouts, 2048 samples):

- **1840 / 2048** rollouts emit a hint call.
- `num_hints` tracks `hint_budget` almost 1:1 (budget 2 → 2 hints, 3 → 3, …) —
  the model spends essentially its **entire** budget every time, not "when stuck".
- **Median reasoning before the first `<hint_call/>` ≈ 284 chars** (mean 298, min
  84, max 1150) — i.e. almost no genuine attempt.
- Smoking-gun rollout (84 chars, budget 2, used 2 hints, wrong):
  > "I know the product of 11, 21, 31, 41, 51, and 61. I can call for a hint to
  > proceed." → `<hint_call/>`

**Root cause — the reward is timing-blind.** `R = R_acc − Σ penalty`, and the
hint penalty is a fixed importance weight independent of *when* / after how much
effort the hint was called. Conditional on eventually being correct, a hint after
80 chars costs exactly the same as one after 1500 chars of hard reasoning. So the
policy's optimal play is: minimize costly reasoning, call hints ASAP (they raise
P(correct) for free), and only reason hard when forced (budget spent). Nothing in
the reward rewards "try hard first."

---

## 2. The fix: effort-shaping penalty

> **SUPERSEDED by entry 2026-06-10 (c).** The mean-reference formula below was the
> first cut; it's order-blind. The live mechanism now references each turn against
> the SUFFIX MAXIMUM (order-aware). Everything else here — correctness-gating, the
> −1 floor, `HINT_SHAPE_COEFF`, wandb logging — still holds. Kept for history.

Restore the timing signal by penalizing hints called after too-little reasoning,
self-normalized per problem.

**Formula** (per rollout):

```
ref          = mean over this rollout's assistant-turn token lengths (turn_lens)
for each APPLIED hint k:
    shortfall_k   = relu(ref - reasoning_len_k) / ref          # in [0, 1]
shape_sum    = Σ_k shortfall_k                                  # coeff-free signal
shape_penalty = coeff * shape_sum                              # coeff = HINT_SHAPE_COEFF
```

- `reasoning_len_k` = token length of the turn that emitted hint *k* (= reasoning
  since the previous hint). `ref` is the rollout's mean turn length — the long
  final solve turn pulls it up, so shallow pre-hint turns fall far below it.
- **Summed, not averaged**, over calls so spamming shallow hints is monotonically
  worse (averaging would let the policy dilute the penalty by calling more).
- **Mean reference** chosen over max (user preference): more stable; the long
  final turn still creates the signal. Caveat: the mean self-dilutes slightly
  under spam, which the *sum* compensates for.

**Key design decisions:**
- **Correctness-gated.** `shape_penalty` is subtracted only from the *correct*
  reward. A wrong answer stays at exactly `incorrect_reward` (−1); the shaping
  term never farms reward on failures.
- **Floored at −1.** The correct score is `max(base − hint_penalty − shape, incorrect_reward)`.
  Even a fully front-loaded correct rollout never scores below a wrong answer —
  otherwise GRPO would learn to suppress hint use entirely (a known failure mode).
- **Computed always (even when disabled).** `shape_sum` is computed regardless of
  `coeff`, so the front-loading signal is logged even in the control arm
  (`HINT_SHAPE_COEFF=0`).

**Tuning knob:** `HINT_SHAPE_COEFF` (default **0.3**) → `reward_kwargs.hint_shape_coeff`.
Start small; watch rollouts for filler-padding before raising. `0` disables the
penalty (signal still logged).

**Verified locally** (stubbed grader):
- front-load `[80,80,80,1200]` → `shape_sum=2.33`, penalty(0.3)=`0.70`.
- diligent (turns near the mean) → ~`0.10`.
- 3 shallow calls (0.70) < 5 shallow calls (1.05) — spam monotonicity holds.
- wrong answer → score exactly `−1.0` (shaping logged, not applied).
- heavy correct: raw `−1.0625` → **clamped to `−1.0`**; light correct stays `0.087`.
- legacy rollouts without `turn_lens`, hint-free rollouts, `coeff=0` → all `0.0`.

**Files changed:**
- `hint_agent_loop.py` — records `extra_fields["turn_lens"]` (token length of every
  assistant turn) and `reasoning_len` on each applied-hint entry.
- `hint_reward.py` — `effort_shortfall_sum()` (coeff-free) + `effort_shortfall_penalty()`
  (coeff-scaled wrapper); integrated into `compute_score` with the −1 floor; returns
  `hint_shape_sum` + `hint_shape_penalty`.
- `hint_reward_manager.py` — colocate-path fallback so `turn_lens` reaches the reward.
- `run_hprl_qwen2.5_7b.sh` — `HINT_SHAPE_COEFF` (default 0.3) → reward_kwargs.

**Expected effect when it runs:** `num_hints` drops and decouples from budget;
pre-hint `reasoning_len` rises (median ~284 chars should grow); accuracy holds
(if reasoning rises but accuracy doesn't, it's filler-padding — raise the cost of
empty tokens or add an absolute floor).

---

## 3. Revised downward budget ratchet

**Old rule:** if ≥ half the rollouts correct (`C ≥ N/2`), set budget to one below
the `(N/2)`-th-smallest correct hint count; else keep.

**New rule** (per user spec, more aggressive, single best rollout drives it):

```
C == 0 (no correct rollout)   -> keep current budget
else, m = min hints over correct rollouts:
    m <  budget               -> new budget = m       (cap at the best rollout)
    m == budget               -> new budget = budget - 1
clamp to [min_budget, current_budget]                 # strictly downward
```

- **No upward / budget-raising mechanism** — scrubbed the "deferred upward-on-plateau"
  language from docs; clamp guarantees monotone-down.
- `BudgetUpdate.pivot_hint_count` renamed `min_correct_hint_count`.
- **Files:** `budget_manager.py` (rule + self-tests, all passing), `downward_budget_plan.md`,
  `README.md`.
- **Note:** budgets now fall fast and a bit noisily (one lucky low-hint success
  ratchets the whole problem down). Synergizes with effort shaping (fewer hints →
  forced to reason more). Watch `hprl/budget_*`, `hprl/frac_ratcheted`.

---

## 4. New wandb metrics (training-side)

verl only auto-aggregates `reward_extra_info` to wandb on the **validation** path
(where hint metrics are always ~0 — val is single-turn / unaided). So training
signals must be emitted explicitly by `hint_budget_callback` (which already builds
the `hprl/*` scalars). Added:

- **`hprl/active_learning_frac`** — fraction of *problems* (pooled across packs) with
  BOTH a correct and an incorrect rollout (`0 < C < N`). Plus `hprl/num_active_learning`.
  **Caveat under k-pack:** this POOLS a problem's packs, so it counts a problem active
  even when its correct & wrong rollouts live in different packs (one pack all-correct,
  another all-wrong) — a contrast GRPO never sees, since advantages are normalized
  WITHIN each uid (pack). It is therefore an upper bound on the correctness signal GRPO
  trains on. The two views below are the GRPO-honest versions:
- **`hprl/active_learning_frac_packwise`** — fraction of *packs* (uids = GRPO groups)
  whose own rollouts span correct & wrong. Denominator is the pack count
  (`hprl/num_packs`, ≈ `k × n_problems`); plus `hprl/num_active_packs`. This is the
  true rate at which a GRPO group carries a correctness contrast.
- **`hprl/active_learning_frac_anypack`** — fraction of *problems* with AT LEAST ONE
  internally-mixed pack (the real per-problem learnability). Always
  `≤ active_learning_frac`; equals it with k-pack off (one uid per problem). Plus
  `hprl/num_active_learning_anypack`. The gap `active_learning_frac − _anypack` is the
  share of problems whose pooled contrast is a split-pack artifact GRPO can't use.
- **`hprl/hint_shape_sum_mean_hinted`** — coeff-free shortfall sum averaged over
  rollouts that applied a hint. **The front-loading curve to watch** (should fall
  toward 0). Plus `hprl/hint_shape_sum_mean` (all rollouts) and
  `hprl/hint_shape_penalty_mean` (coeff-scaled, what's actually subtracted).

These appear under `hprl/` every training step (requires `HPRL_ENABLE=True`, the
default). Ignore the auto-logged `val-core/.../hint_shape*` — always ~0.

---

## 5. Cluster-launch blocker (INFRA — not code; nothing above has run yet)

Multiple launches on 2026-06-10 all died in cluster bring-up before `main_hprl`
started. None of the code changes above have run in-cluster yet.

**Timeline of symptoms:**
1. **Selector NIC-routing hang** (launch ~11:25): 2 of 4 training nodes
   (`scl-c33-r203-svr03`, `scl-c19c-r43-svr01`) hung on the selector reachability
   probe — couldn't reach the gpt-oss-20b endpoints (`10.20.45.9/.79:30000`) that
   the other two reached fine. **Cleared on relaunch** (next attempt passed 2/2).
   Same class as the `hprl-selector-nic-routing-bug` memory, but a *subset-of-nodes*
   variant even with the correct `10.20.45.x` fabric IP advertised.
2. **`Error retrieving safetensors … Repo id must be in the form …` (selector):**
   **benign red herring.** vLLM probes the HF Hub with the local model path, fails
   validation, retries, falls back to local, and serves fine (proved: `/v1/models
   → 200`, selector check passed). Optional cleanup: set `HF_HUB_OFFLINE=1` +
   `TRANSFORMERS_OFFLINE=1` in the selector serve env (not yet applied).
3. **Master pod repeatedly Terminated / evicted (the real blocker):** master-0
   came up once (Ray head started, `Waiting for 4 nodes to join`, got only **1/4**
   — the others were stuck on symptom #1), then **restarted twice in 3 min,
   hopping nodes** (`svr04` → `svr01`) and `Terminated` at 11:44:10. Every restart
   resets the rendezvous → workers stuck on "master DNS not ready" → no training
   after ~1 hr.

**Root cause:** cluster contention — the 6-pod gang (4 train + 2 selector, 48
GPUs) never stays up together. Likely **preemption/eviction** of the master and/or
**no gang scheduling**, so pieces fall over one at a time and deadlock the
rendezvous.

**Recommended actions (handed to user):**
- `kubectl describe pod …master-0` → read `Last State: Terminated → Reason` and
  `Events` (Preempted / Evicted / OOMKilled / FailedScheduling).
- If preemption/low priority → raise PriorityClass or use a reserved partition;
  retry in a quieter window.
- **Enable gang scheduling** (coscheduling/Volcano) so all 6 pods schedule together
  or not at all — fixes the partial-bringup churn.
- Do NOT set `SELECTOR_REQUIRE_REACHABLE=0` (silent dead-hint trap).

**Status:** code changes staged and unit-tested; will run on the first launch that
actually assembles. No method-side work is blocked on the cluster — only validation.

---

## Files touched this session
- `hint_agent_loop.py` — turn_lens + reasoning_len recording
- `hint_reward.py` — effort_shortfall_sum/penalty, −1 floor, shape metrics
- `hint_reward_manager.py` — turn_lens merge
- `hint_budget_callback.py` — revised-ratchet metrics + active_learning_frac + hint_shape aggregation
- `budget_manager.py` — new downward rule + self-tests
- `run_hprl_qwen2.5_7b.sh` — HINT_SHAPE_COEFF
- `downward_budget_plan.md`, `README.md` — docs

---

## 2026-06-22 — budget-grouped data sampling (generation-step load balancing)

**The problem.** The data is sampled uniformly at random, so each step's batch mixes
problems with very different hint-call budgets B_q — a budget-0 problem (one unaided
turn) sits next to a budget-6 problem (up to six selector rounds, each adding a tool
round-trip + extra generation turns). Async multi-turn rollout ends a step only when
its SLOWEST rollout finishes, so the budget-0 rollouts complete early and their GPUs
idle while the high-budget stragglers run on. The big-budget problems bottleneck the
whole generation step. On the live train file the budgets span 0..6 (baked dist:
{0:802, 2:6, 3:111, 4:1557, 5:641, 6:22}).

**The fix.** A budget-grouped train sampler (`budget_sampler.BudgetGroupedSampler`). At
the START OF EACH EPOCH it orders the epoch's problems by their CURRENT budget B_q and
packs same-budget problems into the same step's batch, so a step's rollouts all run
~the same number of hint rounds and finish together — no straggler, no idle GPUs. The
budget read is the LIVE ratcheted B_q from `budget_state_path` (the same file the
trainer ratchet writes and `HintBudgetDataset` reads), falling back to the parquet's
baked budget for not-yet-ratcheted problems, so the grouping tracks the ratchet as the
run progresses.

**Measured (real `dapo-3139-auto-hint`, batch 128):** mean per-batch budget span
**5.58 → 0.25** (20/24 batches perfectly homogeneous; the few mixed ones are at level
boundaries, e.g. the 6-problem budget-2 level absorbed into one boundary batch).

**Invariants kept identical to the stock sampler.**
- Every emitted batch is EXACTLY `train_batch_size` (PPO mini-batch divisibility holds —
  the verl train path doesn't auto-pad). A random `len % batch_size` remainder is dropped
  each epoch, a DIFFERENT subset every epoch (random-permute → drop the tail → stable-sort
  by budget), so no fixed high-budget problem is ever starved — same expected behavior as
  the stock RandomSampler + drop_last.
- Per-epoch problem multiset otherwise unchanged; only the problem→step assignment and
  intra-batch budget spread change. GRPO groups by uid, not batch membership, so advantages
  are unaffected. Composes with k-pack (packs cluster around the grouped base budget) and
  auto-hint.
- STATEFUL: mirrors torchdata's `_StatefulRandomSamplerIterator` (snapshot + checkpoint the
  torch Generator state + yielded count) → exact mid-epoch resume through StatefulDataLoader
  (verified end-to-end). Train sampler only; validation untouched.

**Wiring (flag-gated override, no verl core edit).** `main_hprl.HPRLTaskRunner.run` rebinds
`main_ppo.create_rl_sampler` to `budget_sampler.wrap_create_rl_sampler(...)` — the same trick
as the existing `RayPPOTrainer` rebind. With `data.hprl.budget_sampling.enable=false` the
wrapper delegates to the stock function (byte-identical). New config block
`data.hprl.budget_sampling.{enable, shuffle_batch_order}`. `shuffle_batch_order=true`
randomizes the order the homogeneous batches run within an epoch (each batch stays
homogeneous; only the step sequence is shuffled), so every step is still a random budget
level (stationary difficulty) rather than a fixed easy→hard ramp.

**Default.** Config default `enable: false` (conservative framework default, like k-pack),
but the run scripts default `HPRL_BUDGET_SAMPLING=true` so `bash run_auto_hint_*.sh` /
`run_hprl_*.sh` use it out of the box. Disable for an apples-to-apples baseline with
`HPRL_BUDGET_SAMPLING=false bash run_...sh`.

**Validation.** `python budget_sampler.py --selftest` (length/divisibility, homogeneity,
epoch-to-epoch variety, remainder rotation, ascending order when shuffle off, exact stateful
resume, live-table override) + a real-parquet integration test through the actual
StatefulDataLoader (homogeneity @ bs=128, the 5.58→0.25 span, exact 3-batch→resume,
num_workers=8). All pass.

### Files touched
- `budget_sampler.py` — NEW: `BudgetGroupedSampler` (stateful) + `wrap_create_rl_sampler` + self-test
- `main_hprl.py` — rebind `create_rl_sampler` to the budget wrapper
- `config/hprl_trainer.yaml` — `data.hprl.budget_sampling` block
- `run_hprl_qwen2.5_7b.sh` — `HPRL_BUDGET_SAMPLING` / `HPRL_BUDGET_SAMPLING_SHUFFLE_ORDER` env + job args
- `run_auto_hint_qwen2.5_7b.sh` — re-affirm the same defaults for the auto-hint wrapper

