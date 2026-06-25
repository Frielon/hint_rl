# GPT-OSS-20B Hint-Selection Eval — Progress

_Last updated: 2026-06-22_

Evaluating `gpt-oss-20b` on the Template F hint-selection + citation task, scored
against the Codex (`gpt-5.5`) reference labels in
`test_cite/results/debug/run_20260617_224209`. Two things are measured per row
(problem × reasoning-trace): **hint selection** (does the model pick the same
hint the reference did?) and **citation fidelity** (are the model's
`completed_hints` quotes actually verbatim in the student trace?).

## TL;DR (current, corrected state)

- **Citation is good, not broken.** ~99% of cited quotes are found in the trace,
  ~75–81% are verbatim, ~40% are character-exact, **~1% are fabricated**. The
  "~10% not_found" reported earlier was a **bug** (see below), not model behavior.
- **Hint selection:** the model picks the right *step* ~83–85% of the time
  (`agreement_major_step`); with `x.0`≡`x.1` merged, substep-majority is ~75–79%.
  Strict exact-hint agreement is only ~42–45%, but most of that gap is the
  unavoidable `x.0` (step-guidance) vs `x.1` (first substep) distinction.
- **Trade-off identified:** stricter citation instructions raise verbatim fidelity
  but lower selection accuracy, because they make the model *under-credit* the
  student (88% of selection mismatches now pick an earlier hint than the ref).

## Pipeline & key files (all under `test_cite/gpt_oss_eval/`)

| file | role |
|---|---|
| `run_gpt_oss_eval.sh` | single entry: serve gpt-oss-20b on local vLLM, run the scorer, tear down. Always writes a fresh `results/<label_run>__<model>__<ts>/`. |
| `run_gpt_oss_selection.py` | the scorer/driver. Builds the prompt, samples N=16/row, parses, scores, writes per-row JSON + `_summary.json`. |
| `prompt_template_F.py` | the **editable** prompt (Template F). `prompt_source=local` rebuilds from this every run. |
| `check_citations.py` | standalone verbatim-quote checker → `_citation_check.json`. |
| `reparse_results.py` | **re-parse a run's saved `raw` completions in place** with the current parser (no model re-query). |
| `viewer.py` | local web viewer (default port 8731) to browse run → step → problem with ref + hint set. |
| `../../run_hint_selection_model.py` | shared LaTeX-tolerant parser + OpenAI client. |

Hint pool is **pruned** before prompting: `x.0` step-guidance hints and the
per-hint `type` field are dropped, so the model only sees substep hints
(`prune_hint_pool`). The model therefore never selects `x.0`.

## Runs compared (all: label `run_20260617_224209`, n=16, temp=0.3, top_p=0.95, pruned hints, `prompt_source=local`)

| tag | prompt variant |
|---|---|
| `…__20260622_044924` ("baseline") | new `completed_hints` schema, condensed/terse quote rules. Reparsed with fixed parser. |
| `…__20260622_054527` ("quote-rules") | + detailed **Quote rules** (anti-re-typeset examples, "verify before finishing"). |
| `…__20260622_061914` ("evidence-1st") | + **evidence-first step 4**: find the quote first; a missing quote demotes the hint to the selection. |

(Earlier runs `…_034723` / `…_040238` are obsolete — they predate the prompt-fallback fix and contain mixed old/new-schema rows.)

## Bugs found & fixed (these changed the numbers materially)

1. **Silent old-prompt fallback.** 50/500 label rows store only the rendered
   `prompt` (no structured `problem`/`reasoning_trace`). `rebuild_prompt`
   returned `None` for them and `prompt_for(local)` fell back to the **stored OLD
   prompt** → old schema + un-pruned hints for those rows.
   _Fix:_ `recover_problem_trace` (slice `problem`/`trace` out of the stored
   prompt); `local` mode no longer falls back to the old prompt.

2. **Parser bug `_repair_json_escapes`.** The escape-repair regex corrupted a
   model-correct `\\cdot` (it doubled the 2nd backslash). On those samples the
   parser fell to the field-by-field fallback, which **dropped `completed_hints`
   AND mis-read `hint_id`** (it grabbed the *first completed-hint's* id instead
   of the selected hint). This **understated agreement**.
   _Fix:_ alternation regex that consumes valid escape pairs whole. Recovered
   376 `completed_hints` on the baseline run and corrected ~370 hint_ids
   (`mean_agreement_hint_id` 0.4116 → 0.4362 after reparse).

3. **Trace-join bug (biggest impact on citation).** `check_citations.load_traces`
   read `reasoning_trace` straight from the label → `None`→`""` for the 50
   trace-less rows, so **every quote there scored as `not_found`**, inflating
   not_found ~10×.
   _Fix:_ `_recover_trace` in `load_traces` (same prompt-slicing). Corrected:
   not_found 10.4% → **1.0%**, verbatim 0.73 → **0.81**, reference baseline
   0.92 → **0.998**.

Also: citation stats are now folded into `_summary.json` (`citation` block,
overall + by_step + reference_baseline), scored against the trace the model
actually saw. `check_citations.py` also gained a `word_overlap` diagnostic tier
(**off by default**, `--word-threshold 0`) — judged too weak (false positives).

## Results — citation accuracy (corrected, trace-recovery fix)

| metric | 044924 | 054527 | 061914 |
|---|--:|--:|--:|
| not_found_rate | 0.0129 | 0.0123 | **0.0100** |
| exact_rate | 0.3189 | 0.3637 | **0.4036** |
| verbatim_rate (exact+norm+loose) | 0.7471 | 0.7863 | **0.8078** |
| found_rate (incl. fuzzy) | 0.9871 | 0.9877 | 0.9900 |
| reference baseline found_rate | 0.998 | 0.998 | 0.998 |

Both prompt refinements improved citation **monotonically**.

## Results — hint-selection accuracy

`x.0`≡`x.1` "merged" = treat step-guidance `x.0` and first substep `x.1` as the
same class (the model can't pick `x.0`, so ref `x.0` + model `x.1` is correct).

| run | mean_agree (strict) | mean_agree (merged) | majority (strict) | majority (merged) |
|---|--:|--:|--:|--:|
| 044924 | 0.436 | **0.770** | 0.454 | **0.794** |
| 054527 | 0.429 | 0.762 | 0.440 | 0.778 |
| 061914 | 0.417 | **0.742** | 0.418 | **0.750** |

`agreement_major_step` (right step, any substep) ≈ 0.83–0.85. Selection accuracy
declined monotonically as citation instructions got stricter.

### Mismatch direction (merged, sample-level)

| run | mismatches | beyond ref (over-credit) | before ref (under-credit) |
|---|--:|--:|--:|
| 044924 | 1,819 | 18.4% | 81.6% |
| 054527 | 1,878 | 15.6% | 84.4% |
| 061914 | 2,041 | 11.7% | **88.3%** |

**Dominant error = under-crediting** (model selects an earlier hint than the ref),
and it grows with prompt strictness.

## The citation ↔ selection trade-off

| | merged-majority (selection) | verbatim_rate (citation) |
|---|--:|--:|
| 044924 baseline | **0.794** | 0.747 |
| 054527 quote-rules | 0.778 | 0.786 |
| 061914 evidence-1st | 0.750 | **0.808** |

Same conservatism drives both: the evidence-first gate ("can't quote it → not
achieved → select it") buys ~6pp verbatim at the cost of ~4pp selection. So:
**044924 = best selection, 061914 = best citation, 054527 = middle.**

## Open questions / next steps

- **Pick a config** (or tune): the leading idea is to keep `061914`'s strong
  Quote rules but **soften the "missing-quote → reselect" clause** in step 4 — it
  currently converts "couldn't *copy* a clean quote" into "student didn't *do*
  it," which over-fires and causes the under-crediting. Variants to try:
  (a) drop that clause; (b) allow crediting a clearly-done substep without a
  perfect verbatim span.
- ~38–57 samples/run still drop `completed_hints` to genuine malformed JSON
  (truncation / escapes the repair still misses) — minor.
- Consider whether the reference itself over-credits (the model's "before-ref"
  bias could partly be Codex being lenient).

## How to reproduce

```bash
# new run (serves model, scores, writes results/<...>__<ts>/ incl. citation in _summary.json)
bash run_gpt_oss_eval.sh

# re-score citations standalone (also fine on old runs; recovers traces)
python check_citations.py --results-dir results/<run> --label-run run_20260617_224209

# re-parse an existing run in place after a parser change (no model calls)
python reparse_results.py results/<run>

# browse results
python viewer.py          # http://127.0.0.1:8731
```
