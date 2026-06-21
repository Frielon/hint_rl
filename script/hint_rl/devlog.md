# HPRL dev log

Running engineering log. Two parts: **TODO / Planned** (ideas not yet built) directly
below, then the **Done log** (shipped work, newest entry on top — append a new
`## YYYY-MM-DD` section per working session). Move an item from TODO to a dated Done
entry once it lands.

---

# TODO / Planned (not yet done)

- **Citation-enforcement guard** in `hint_agent_loop._record_major_step`: for a pick that skips earlier
  unrevealed candidates, substring-validate each `completed_steps` quote against the student-only trace
  (`analyze_citations.classify_quote` / `student_only`) and clamp the pick to the earliest uncited step.
  Deterministic, no extra model call — turns the `v4_cite` citations (prompt adopted 2026-06-14) into a
  contract. Drops unearned final-step reveal **6.4%→3.0%** @T0.7 / **4.2%→0.8%** @T0.1. Optional companion:
  tighten `utils._as_selection_dict` to a selection-shaped-dict check so the degraded-parse fallback can't
  return a nested `completed_steps` entry as the selection.

---

# Done log

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

