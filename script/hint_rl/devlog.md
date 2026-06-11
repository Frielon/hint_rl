# HPRL dev log

Running engineering log — newest entry on top. Append a new `## YYYY-MM-DD` section
per working session.

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

- **`hprl/active_learning_frac`** — fraction of prompt-groups with BOTH a correct
  and an incorrect rollout (`0 < C < N`); the groups GRPO actually learns a
  correctness contrast from. Plus `hprl/num_active_learning`.
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

