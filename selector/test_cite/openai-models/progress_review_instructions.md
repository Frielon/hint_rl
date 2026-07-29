# Judge instructions: progress-rephrase review

Input: a chunk file `chunks/chunk_<i>.jsonl` from `build_progress_review.py`.
Each line is one sampled selector response for one problem:

```json
{"problem_id": ..., "model": ..., "step": ..., "selected_hint_id": ...,
 "problem": "<problem statement>",
 "entries": [{"hint_id", "pool_hint", "quote", "why", "progress",
              "quote_located", "trace_window"}, ...]}
```

`pool_hint` is the ORIGINAL hint text from the pool; `progress` is the model's
rephrasing of it as achieved progress; `quote` is the model's verbatim citation
from the student's trace; `trace_window` is the trace context ending at the
located quote (null if the quote couldn't be located).

For EVERY entry of EVERY case, judge the `progress` string:

1. **faithful** — same mathematical idea as `pool_hint`; no solution content
   beyond what the hint already contains; no mathematical error introduced.
2. **achieved phrasing** — stated as progress already made ("You have already
   ..."), not as an instruction still to do.
3. **trace-adapted** — consistent with the student's actual work as evidenced
   by `quote`/`trace_window`; using the student's own notation/variables where
   they differ from the hint is the desired behavior.
4. **supported** — does not assert progress the evidence contradicts or that
   plainly is not in the quote/window (judge only from the given evidence; if
   `quote_located` is false and the quote itself is insufficient, lean on the
   quote text alone and note it).

Verdict per entry:
- `pass`  — all four hold (small stylistic differences are fine).
- `minor` — usable but flawed: e.g. instruction-like phrasing, generic
  restatement not adapted to the trace, slight overstatement.
- `fail`  — wrong math, intent lost or replaced, reveals beyond the hint,
  claims clearly unsupported/contradicted progress, or empty/missing.

Issue tags (list all that apply): `unfaithful`, `leak`, `phrasing`,
`not_adapted`, `unsupported`, `empty`, `other`.

Write one JSON object per line to `verdicts/chunk_<i>.verdicts.jsonl` inside
the check dir (create the `verdicts/` dir if needed), exactly:

```json
{"problem_id": "...", "hint_id": "...", "verdict": "pass|minor|fail",
 "issues": [], "note": "<one line>"}
```

Every entry in the chunk must get exactly one verdict line (match by
problem_id + hint_id; if a case repeats a hint_id, emit one line per
occurrence in file order). Keep notes to one short sentence.
