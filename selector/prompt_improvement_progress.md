# Selector prompt improvement — progress log

Goal: stop the **unearned final-step reveal** (selector hands the policy the
answer-bearing last step when the student hasn't earned it), measured on real
failure cases from run `HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260609-171314`.

## Background (what we're fixing)

- ~96% of hint pools carry the literal ground truth in the LAST step's hints;
  `major_step` mode injects the whole step verbatim, so a last-step pick = answer reveal.
- Design verdict: that is fine **if earned** (student did all prior steps, with or
  without hints). The failure is the *unearned* reveal, two kinds:
  `jump_last` (selector skips unrevealed steps; P(rollout correct)=0.91) and
  `spamwalk_last` (hint-collection walk; forced pick, protocol issue not prompt issue).
- Root-cause discovery: the deployed loop never put assistant turns into
  `agent_data.messages`, so `build_trace` fed the selector a **hints-only trace**
  ("blind-trace bug") — the selector guessed progress from hint history at T=0.7.
  See `../script/hint_rl/hint_agent_loop.py:233` and memory note; fix is to append
  the decoded assistant turn in `_handle_generating_state`.

## Setup

- Server: `inference/serve_gpt_oss_20b.sh` — sglang, gpt-oss-20b, DP=2, port 30000,
  reasoning parser separates CoT (`reasoning_content`) from the JSON answer (`content`).
- Test set: `unearned_final_step_reveals.parquet`, `kind == jump_last` only
  (8,938 rows → 3,302 unique selector inputs by `dup_key`; `spamwalk_last` rows are
  forced single-choice picks and cannot distinguish prompts).
- Harness: `eval_selector_prompts.py` (same sampled rows for every variant,
  seed=0; production sampling T=0.7, top_p as configured; parse failures are NOT
  retried, unlike production's 3-attempt retry, so `parse_fail` is a prompt-robustness metric).
- Variants: `prompt_variants.py`. All keep the deployed `<output>` JSON schema, so a
  winner drops into `utils.selector_prompt` unchanged.

| variant | trace | idea |
|---|---|---|
| `as_seen` | hints-only | deployed conditions (blind-trace bug) — replay reference |
| `baseline` | full | deployed prompt + the trace the loop SHOULD have built |
| `v1_earned` | full | + earned-progress rules: hint text is not student work; no new writing → no new progress; don't skip candidates |
| `v2_final_gate` | full | v1 + explicit guard: final step only when every earlier candidate is done in the student's own writing; too-early beats answer-reveal |
| `v3_checklist` | full | v2 + forced per-candidate completed-yes/no checklist in `reasoning_of_major_step` |

Metrics per row (candidate pool = `hints_filtered`, 2–5 steps):
- `fail_last` — picked the answer-bearing last step (**the failure**, lower is better)
- `pick_expected` — picked `expected_step_id` = earliest unrevealed step (reference
  correct pick: by construction the student wrote <500 chars since the last call)
- `other` — non-last, non-expected (avoids the reveal but imperfect anchoring)

Caveats: serving stack differs from training (sglang here vs vLLM in the cluster);
selection-biased test set (every dup_key produced ≥1 observed last-pick in the run,
so absolute rates overstate the average call; comparisons across variants are the point).

## Experiment log

### Smoke (2026-06-10, 8 rows, `as_seen`, T=0.7, tag `smoke`)

Pipeline validated end-to-end (reasoning routed to `reasoning_content`, JSON in
`content`, `parse_output` OK). Two observations:
- Even under deployed (blind) conditions the replay failed only 1/7 scoreable
  rows → the run's observed last-picks are a **tail of the T=0.7 distribution**
  (each problem got many draws across 16 rollouts × steps), not a deterministic
  bug. Variant comparisons need n large enough for ~10–20% base rates.
- Throughput: ~10–12 tok/s per stream, ~1–4k reasoning tokens → 3–7 min/call;
  round sizes chosen accordingly (n=60/variant, concurrency 48, DP=2).

### Round 1 — 60 sampled jump_last cases × 5 variants, T=0.7, top_p 0.95, seed 0 (tag `round1`)

| variant | fail_last | pick_expected | other | parse_fail | notes |
|---|---|---|---|---|---|
| as_seen | **32.1%** | 67.9% | 0% | 4/60 | deployed conditions; the rate to beat (wall 24 min) |
| baseline | 22.6% | 75.5% | 1.9% | 7/60 | trace fix alone: −9.5pts vs as_seen, insufficient by itself |
| v1_earned | 8.9% | 91.1% | 0% | 4/60 | earned-progress rules: −13.7pts vs baseline — the big lever |
| v2_final_gate | 5.5% | 92.7% | 1.8% | 5/60 | + answer-step guard: best so far |
| v3_checklist | 8.6% | 91.4% | 0% | 2/60 | checklist ≈ v1; no gain over v2's gate (but fewest parse fails) |

Round-1 read: with ~55 scoreable rows/variant, v2 (3 fails) vs v3 (5) vs v1 (5) is
within noise of each other, but all are decisively below baseline (12) and as_seen
(18). v2 is the simplest of the leaders → confirm v2 at larger n.

### Round 2 — confirmation (tags `round2_confirm`, `round2_lowtemp`)

| run | variant | n | T | fail_last | pick_expected | parse_fail | notes |
|---|---|---|---|---|---|---|---|
| confirm | v2_final_gate | 240 | 0.7 | **9.4%** | 89.2% | 27/240 | round-1's 5.5% (3/55) was a lucky draw; 9.4% (20/213) is the honest T=0.7 estimate — still ~2.4× better than baseline, ~3.4× better than deployed |
| lowtemp | v2_final_gate | 120 | 0.1 | 7.0% | 89.6% | 5/120 | T=0.1: small gain over 9.4% @0.7; fewer parse fails (4.2% vs 11.2%) |
| lowtemp | baseline | 120 | 0.1 | 16.1% | 81.4% | 2/120 | low temp alone does NOT rescue the deployed prompt |

### Round 3 — citation-grounded selection, `v4_cite` (tags `round3_cite`, `round3_cite_lowtemp`)

Motivation (user, 2026-06-10): instead of asking the selector to *assert* that
earlier steps are done, require it to **cite the trajectory's words** that solve
them — to select step N it must quote, verbatim, the student writing that
carries out every earlier candidate step, in a `completed_steps` JSON array
(`{step_id, quote, why}` per step; `[]` when selecting the earliest candidate).
Two effects: (a) prompts the model to actually look for evidence before
skipping ahead; (b) the quotes are **machine-checkable** — the agent loop can
substring-validate them against the student-only trace and clamp the pick to
the earliest step lacking a valid citation (deterministic enforcement of the
justification, catching fabricated or hint-derived "evidence").

Variant: `v4_cite` = v2_final_gate + "Cite your evidence" workflow section +
`completed_steps` output field (`prompt_variants.py`). Validation + enforcement
simulation: `analyze_citations.py` (quote classes: valid_student / hint_only /
fabricated; enforcement = clamp skip-ahead picks whose earlier candidates lack
a valid_student quote).

| run | n | T | fail_last | + enforcement | pick_expected (→ enforced) | parse_fail | notes |
|---|---|---|---|---|---|---|---|
| round3_cite | 240 | 0.7 | **6.4%** | **3.0%** | 92.7% → 97.0% | 6/240 (2.5%) | same rows as round2_confirm (v2 = 9.4%, parse_fail 11.2%) |
| round3_cite_lowtemp | 120 | 0.1 | **4.2%** | **0.8%** | 95.0% → 99.2% | 0/120 | same rows as round2_lowtemp (v2 = 7.0%) |

Citation audit (`analyze_citations.py`):
- Skip-ahead picks are now rare and mostly unjustified: T=0.7 — 17 skip-aheads,
  7 fully cited+valid; T=0.1 — 6 skip-aheads, 1 fully cited+valid.
- Quote validity (skip-ahead picks only): 60% / 44% `valid_student`; the rest are
  **fabricated paraphrases presented as quotes** (e.g. "Michael's position is
  x=5t." appearing nowhere in the trace) plus a few `hint_only`. **The substring
  check catches every one** — this is exactly the failure mode the citation
  design converts from unobservable to enforceable.
- Enforcement clamped 10/234 (T=0.7) and 5/120 (T=0.1) picks to the earliest
  uncited step. The single surviving enforced failure at T=0.1 was a skip-ahead
  whose quotes all validated — plausibly a genuinely-earned label edge case.
- Side benefit: the structured `completed_steps` field stabilizes the output
  format — parse failures drop 11.2% → 2.5% (T=0.7) and to 0 at T=0.1.

## Findings

- **The deployed failures are stochastic, not deterministic**: replaying the same
  inputs at T=0.7 yields a last-step pick ~32% of the time (`as_seen`, round 1).
  Production saw ≥1 failure per case because each problem is drawn many times
  (16 rollouts × multiple calls).
- **Trace fix alone is insufficient**: full trace + deployed prompt still fails
  22.6% — the prompt happily credits hint content and stated intentions as
  student progress.
- **Earned-progress rules are the big lever** (8.9% fail): explicitly defining
  "completed = the student's own writing carries out the step's substance" and
  "hint text is not student work" fixes most jumps.
- **v1 residual failures (5/56) share one shape**: the selector still credits
  unexecuted work — e.g. "the earlier steps have been *implicitly carried out
  through the hints*", or treats the student's *announcement* of the next task
  ("they need to combine them") as completing the prior step. All residuals are
  conf=5; selector confidence stays uninformative as a gate.
- v2's final-step gate cuts the residual further (5.5%); v3's checklist does NOT
  add on top of v1's rules (8.6%) — explicit "if unsure, go earlier" asymmetry
  beats more structured auditing.
- **v2 residuals (3/55)**: one looks like label ambiguity (the student plausibly
  HAD done the remaining middle steps, so the last step may be earned — the
  `expected_step_id` label assumes no self-done unrevealed work); two are
  "credit-by-understanding" ("effectively completed by their understanding"),
  stochastic non-compliance with rules the prompt already states. Prompt-side
  iteration is at diminishing returns around ~5%; the deterministic loop guard
  (refuse last-step reveal while unrevealed earlier steps exist) would catch all
  of these.
- Parse failures (all variants, 3–12%) are "no <output> block" with finish=stop;
  production's 3-attempt retry in `HintSelector.select` absorbs most of them.

## Conclusion (2026-06-10, updated 2026-06-11 after Round 3)

Winner: **`v4_cite`** (v2's rules + machine-checkable verbatim citations,
user-proposed design) **+ citation enforcement in the loop**. Unearned
final-step reveal rate on real failure cases:

| condition | fail_last (T=0.7 / T=0.1) |
|---|---|
| deployed (blind trace, deployed prompt) | 32.1% / – |
| trace fix only (deployed prompt) | 22.6% / 16.1% |
| v2_final_gate prompt | 9.4% / 7.0% |
| **v4_cite prompt** | **6.4% / 4.2%** |
| **v4_cite + citation enforcement** | **3.0% / 0.8%** |

Recommended production config, in order of importance:
1. **Fix the blind-trace bug** in `hint_agent_loop._handle_generating_state`
   (append the decoded assistant turn to `agent_data.messages`) — all full-trace
   gains assume it.
2. **Adopt the v4_cite prompt** (template in `prompt_variants.py`; extra
   `completed_steps` JSON field is ignored by the current loop, so it is a
   drop-in for `utils.selector_prompt`).
3. **Citation-enforcement guard in `_record_major_step`**: for a pick that skips
   earlier unrevealed steps, substring-validate each `completed_steps` quote
   against the student-only trace (logic in `analyze_citations.py`:
   `student_only`/`classify_quote`); on any missing/fabricated/hint-only quote,
   clamp the pick to the earliest unproven step. Deterministic, no extra model
   calls — turns the justification into a contract.
4. **Lower `SELECTOR_TEMPERATURE` 0.7 → 0.1** — fail_last 3.0% → 0.8% with
   enforcement, and parse failures hit 0 (v4's structured output already cuts
   them 11.2% → 2.5% at T=0.7).

## Status / next steps

- [x] Round 1: 5 variants, n=60, T=0.7 → v2 wins
- [x] Failure analysis of v1/v2 residuals (credit-by-understanding pattern)
- [x] Confirm v2 at n=240 (9.4%) and T=0.1 (7.0%); baseline@0.1 control (16.1%)
- [x] v2 applied to production `utils.selector_prompt` (2026-06-10, devlog (d))
- [x] Round 3: v4_cite (citation-grounded, user design) → 6.4%/4.2%; with
      enforcement 3.0%/0.8% — new winner, supersedes v2 + the blanket
      last-step-refusal guard idea (citation guard is finer-grained)
- [ ] Apply to production: blind-trace fix; v4_cite into `utils.selector_prompt`;
      citation-enforcement guard in `_record_major_step`; temp 0.1
- [ ] Optional: re-test `spamwalk_last` rows under a marked-revealed-steps protocol
      (needs `hints_full` + re-anchoring rules; parquet already carries the columns)

## Appendix — the prompts tested

Six variants were evaluated; they use five distinct templates (`as_seen` and
`baseline` share Template A and differ only in which trace column fills
`{{trace}}`). Templates below are verbatim what was sent to gpt-oss-20b as the
single user message, with `{{problem}}` / `{{trace}}` / `{{hints}}` filled per
test row (`problem`, the variant's trace column, `hints_filtered`). All share
the deployed `<output>` JSON schema, so any of them drops into
`utils.selector_prompt` unchanged. Generated from `prompt_variants.py`.

### Template A — deployed baseline (variants `as_seen` and `baseline`)

Byte-identical to production `script/hint_rl/utils.selector_prompt` (parity asserted in `prompt_variants.py`). Used by:
- `as_seen` — `{{trace}}` = `trace_as_seen` (hints-only trace, replicating the deployed blind-trace bug)
- `baseline` — `{{trace}}` = `trace_full` (student reasoning + injected hints)

````text
You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
````

### Template B — `v1_earned` (`{{trace}}` = `trace_full`)

Template A + the **"Ground rules for crediting progress"** section inserted before "The current step is the earliest step ..." in Workflow step 1.

````text
You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

### Ground rules for crediting progress
- Hints previously given by the tutor appear in the trace. **Hint text is not
  student work.** A step counts as completed only if the *student's own
  writing* carries out its substance after (or independently of) any hint.
  Content that merely echoes a given hint, or a step whose hint was shown but
  that the student never executed, is NOT completed.
- Progress requires new written reasoning. If the student has written little
  or nothing of substance since the last hint was given, then no additional
  step can have been completed since then.
- The candidate list contains only steps whose hints have **not** been given
  yet. Do not skip over an earlier candidate step because you, the tutor, can
  see how it would go -- the student still has to do it.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
````

### Template C — `v2_final_gate` (`{{trace}}` = `trace_full`) — WINNER

Template B + the **"Guard the final step"** section inserted before "## 2. Select a hint within that major step".

````text
You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

### Ground rules for crediting progress
- Hints previously given by the tutor appear in the trace. **Hint text is not
  student work.** A step counts as completed only if the *student's own
  writing* carries out its substance after (or independently of) any hint.
  Content that merely echoes a given hint, or a step whose hint was shown but
  that the student never executed, is NOT completed.
- Progress requires new written reasoning. If the student has written little
  or nothing of substance since the last hint was given, then no additional
  step can have been completed since then.
- The candidate list contains only steps whose hints have **not** been given
  yet. Do not skip over an earlier candidate step because you, the tutor, can
  see how it would go -- the student still has to do it.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

### Guard the final step
The last major step typically states the final answer. Selecting it gives the
answer away, so it is justified **only** when the student's own written work
has correctly completed every earlier candidate step. If you are at all unsure
whether an earlier candidate step is genuinely done, select that earlier step
instead. An unearned answer reveal is a much worse error than a hint that is
one step too early.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
````

### Template D — `v3_checklist` (`{{trace}}` = `trace_full`)

Template C + the forced per-candidate checklist paragraph inserted before "The current step is the earliest step ..." (no gain over Template C in round 1).

````text
You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

### Ground rules for crediting progress
- Hints previously given by the tutor appear in the trace. **Hint text is not
  student work.** A step counts as completed only if the *student's own
  writing* carries out its substance after (or independently of) any hint.
  Content that merely echoes a given hint, or a step whose hint was shown but
  that the student never executed, is NOT completed.
- Progress requires new written reasoning. If the student has written little
  or nothing of substance since the last hint was given, then no additional
  step can have been completed since then.
- The candidate list contains only steps whose hints have **not** been given
  yet. Do not skip over an earlier candidate step because you, the tutor, can
  see how it would go -- the student still has to do it.

Before choosing, go through **every candidate step in order** and give a
verdict: `completed by the student's own work: yes/no`, each with one line of
evidence quoted or paraphrased from the *student's* writing (not from hint
text). The current step is the **first candidate with verdict "no"**. Put this
checklist in `reasoning_of_major_step`.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

### Guard the final step
The last major step typically states the final answer. Selecting it gives the
answer away, so it is justified **only** when the student's own written work
has correctly completed every earlier candidate step. If you are at all unsure
whether an earlier candidate step is genuinely done, select that earlier step
instead. An unearned answer reveal is a much worse error than a hint that is
one step too early.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
````

### Template E — `v4_cite` (`{{trace}}` = `trace_full`) — WINNER after Round 3

Template C + the **"Cite your evidence"** section inserted before "### Guard the
final step", and the `completed_steps` field added to the output JSON (before
`major_step_id`). The cited quotes are machine-checkable; see
`analyze_citations.py` for the validation + enforcement logic.

````text
You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

### Ground rules for crediting progress
- Hints previously given by the tutor appear in the trace. **Hint text is not
  student work.** A step counts as completed only if the *student's own
  writing* carries out its substance after (or independently of) any hint.
  Content that merely echoes a given hint, or a step whose hint was shown but
  that the student never executed, is NOT completed.
- Progress requires new written reasoning. If the student has written little
  or nothing of substance since the last hint was given, then no additional
  step can have been completed since then.
- The candidate list contains only steps whose hints have **not** been given
  yet. Do not skip over an earlier candidate step because you, the tutor, can
  see how it would go -- the student still has to do it.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

### Cite your evidence
For every candidate step you judge completed, you must QUOTE the student's own
words that carry out that step: an exact, verbatim excerpt (roughly 10-200
characters) copied character-for-character from the student's writing in the
trace. Text inside a `[hint given]` block is the tutor's, not the student's --
quoting it does not count. Paraphrases do not count. These citations go into
the `completed_steps` array of your output, one entry per candidate step
listed BEFORE your selected step. If you cannot produce a verbatim student
quote for some earlier candidate step, that step is not completed -- select it
instead. An empty `completed_steps` array means you selected the earliest
remaining candidate step.

### Guard the final step
The last major step typically states the final answer. Selecting it gives the
answer away, so it is justified **only** when the student's own written work
has correctly completed every earlier candidate step. If you are at all unsure
whether an earlier candidate step is genuinely done, select that earlier step
instead. An unearned answer reveal is a much worse error than a hint that is
one step too early.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "completed_steps": [{"step_id": <id of a candidate step earlier than your selection>, "quote": <verbatim excerpt from the student's own writing that carries out this step>, "why": <one line: why this excerpt completes the step>}, ...] (one entry for EVERY candidate step listed before your selected step; [] if you selected the earliest),
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
````
