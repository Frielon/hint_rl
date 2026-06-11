# unearned_final_step_reveals.parquet

Selector test set extracted from run `HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260609-171314`
(extractor: `logs/<run>/` + `/tmp/extract_unearned.py`, 2026-06-10).

One row = one hint call that revealed the pool's FINAL step (which carries the
literal answer for ~96% of pools) **unearned**:

- `kind = jump_last` (8,938 rows): >=1 earlier step was still unrevealed and the
  student wrote <500 chars since the previous call. The selector had 2–5 steps to
  choose from (`n_choices_filtered`) and wrongly picked the last one.
  P(rollout correct) = 0.91.
- `kind = spamwalk_last` (12,524 rows): no skip — all earlier steps had been
  revealed — but the student wrote <500 chars before EVERY call (hint collection
  without work). The filtered pool usually contains only the final step
  (`n_choices_filtered == 1`), so the pick was forced; these rows test protocol
  changes, not selector choice. P(rollout correct) = 0.64.

## Replaying a selector call

Old (run-faithful) protocol — reproduces what the deployed selector saw:

```python
from utils import selector_prompt   # script/hint_rl
prompt = selector_prompt(row.problem, row.trace_as_seen, row.hints_filtered)
```

**Important context:** in this run `build_trace` received `agent_data.messages`,
which never contains assistant turns (neither `HintAgentLoop` nor the base
`ToolAgentLoop._handle_generating_state` appends them). So `trace_as_seen` is
hints-only — the selector NEVER saw the student's reasoning. `trace_full` is the
intended trace (student turns + injected hints, `<hint_call/>` sentinel kept),
reconstructed from the rollout output; use it to evaluate a fixed loop:

```python
prompt = selector_prompt(row.problem, row.trace_full, row.hints_filtered)
```

Pass/fail per row: parse the selection (`utils.parse_output`), take
`hint_selector.step_id_of(selection)`. For `jump_last` rows the pick is a
FAILURE if it equals `last_step_id`; the no-extra-work reference pick is
`expected_step_id` (earliest unrevealed step — by construction the student did
no substantive work since the last call, so credit for skipped steps is
unwarranted). `spamwalk_last` rows with `n_choices_filtered == 1` cannot fail
under the old protocol; use `hints_full` + `trace_full` for redesigned
protocols (e.g. revealed-steps marked instead of excluded).

## Dedup

`trace_as_seen`/`hints_filtered` depend only on (problem, revealed steps, call),
so rows repeat across rollouts and training steps: 5,354 unique `dup_key`
values across 21,462 rows (2,609 unique problems). For old-protocol evals,
`df.drop_duplicates('dup_key')` to weight each distinct selector input once;
`trace_full` differs per rollout, so keep all rows for fixed-loop evals.

## Columns

| column | meaning |
|---|---|
| problem_id, train_step, rollout_index | provenance (rollout = line in `logs/<run>/rollouts/<step>.jsonl`) |
| kind, call_index | failure kind; 0-based index of the offending call |
| ground_truth, acc, hint_budget, num_hints_total | rollout outcome/metadata |
| problem | exact string `HintSelector.select` received (includes the baked initial-budget sentence from data prep) |
| problem_base | clean problem statement (`hprl_user_base`) |
| trace_as_seen | trace as actually fed to the selector (hints-only; `"(The student has not written any reasoning yet.)"` on call 0) |
| trace_full | intended trace: student reasoning + `[hint given]` turns |
| hints_filtered | pool JSON after `exclude_applied_steps` (selector's actual candidate set) |
| hints_full | original full pool JSON |
| revealed_step_ids | steps already revealed before this call (JSON list) |
| picked_step_id | what the run's selector picked (== last_step_id) |
| expected_step_id | earliest unrevealed step (reference pick) |
| last_step_id, n_steps, n_choices_filtered | pool geometry |
| pre_call_text_len | chars the student wrote since the previous call |
| confidence_of_major_step | the run selector's self-reported confidence (93% are 5) |
| injected_hint_body | the step text that was injected (contains the answer) |
| dup_key | hash of (problem_id, call_index, revealed set, kind) for dedup |

Caveats: 2,052 rollouts (~0.4%) were skipped because the output text could not
be segment-aligned (truncated generations); 47 duplicate problem texts in the
source parquet collapse to one meta entry each.
