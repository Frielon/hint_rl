# Closed OpenAI-model Hint-Selection + Citation Eval

Evaluates **closed OpenAI models** on the same Template F hint-selection +
citation task as `../gpt_oss_eval/`, scored against the same Codex (`gpt-5.5`)
reference labels (`test_cite/results/debug/run_20260617_224209`, 500 rows,
100/step × 5 steps). Two things measured per row (problem × reasoning-trace):
**hint selection** (does the model pick the same hint the reference did?) and
**citation fidelity** (are the model's `completed_hints` quotes verbatim in the
student trace?).

Default models: `gpt-5-nano`, `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4o-mini`.

## Design — thin wrapper, maximal reuse

`run_openai_selection.py` **reuses `../gpt_oss_eval/run_gpt_oss_selection.py`
wholesale** (import + monkeypatch), so this is apples-to-apples with the
open-weight run:

- **same** editable Template F prompt (`../gpt_oss_eval/prompt_template_F.py`),
- **same** hint-pool pruning (drop `X.0` step-guidance hints + `type`),
- **same** LaTeX-tolerant `<output>` parser,
- **same** selection scoring + citation summary (folded into `_summary.json`).

The only thing swapped is the model-call layer (`drv.one_sample` →
`openai_one_sample`), which speaks `api.openai.com` and adapts per-model quirks:

| model class | detection | params sent |
|---|---|---|
| reasoning (`gpt-5*`, `o1/o3/o4*`) | `is_reasoning_model()` | `max_completion_tokens`, `reasoning_effort` (default `low`); **no** `temperature`/`top_p` (only default allowed) |
| chat (`gpt-4.1-*`, `gpt-4o-*`) | — | `temperature`, `top_p`, `max_tokens` |

`completion_tokens` (from API usage) includes hidden reasoning tokens for
reasoning models, so it stays the clean cross-model cost/length metric.

## Files

| file | role |
|---|---|
| `run_openai_eval.sh` | single entry: loops the model list, one `results/<label>__<model>__<ts>/` per model, then prints the compare table. No server to launch. |
| `run_openai_selection.py` | the driver (thin wrapper over the gpt-oss driver). One model per invocation. |
| `compare_runs.py` | read every run's `_summary.json` → markdown comparison table (selection + citation). |
| **`prompt_template_F.py`** | **editable local single-round prompt** — edit this to revise the prompt these runs use. |
| `env.sh` | `export OPENAI_API_KEY=...` (sourced by the runner if the var is unset). |

## Editing the prompt

The prompt templates are **local editable copies in this folder** — the drivers
load them explicitly and override the shared `gpt_oss_eval` copies, so your edits
here drive the runs (they diverge from the gpt-oss reference once edited, which
is the point):

- single-round: **`prompt_template_F.py`**
- multi-round: **`prompt_template_multiF.py`**

Keep the `{{problem}}` / `{{trace}}` / `{{hints}}` markers and the four output
fields (`hint_id`, `hint`, `completed_hints`, `reasoning_of_hint`) so the
parser/scorer keep working. Each run prints which prompt file it used
(`… (LOCAL editable copy)`) and snapshots it into the result dir's `_scripts/`.
Sanity-check an edit without spending tokens:
`python run_openai_multi_selection.py --model gpt-4o-mini --dry-run --limit 2`.

Output layout mirrors the label run
(`step{N}/<problem_id>/<request_id>.json` + `_summary.json`), so
`../gpt_oss_eval/viewer.py` can browse these runs too.

## How to run

```bash
# all default models, full corpus, n=8/row:
bash run_openai_eval.sh

# subset / bigger sample / smoke test (env-overridable knobs):
MODELS="gpt-4o-mini gpt-4.1-mini" N_SAMPLES=16 bash run_openai_eval.sh
LIMIT=5 bash run_openai_eval.sh            # 5 rows/step smoke
STEPS="1,2" bash run_openai_eval.sh

# one model directly:
source env.sh
python run_openai_selection.py --model gpt-4o-mini -n 8
python run_openai_selection.py --model gpt-5-nano --reasoning-effort low

# compare whatever is in results/ (latest run per model):
python compare_runs.py           # --all for every run; --md compare.md to save
```

**Cost knobs:** full corpus = 500 rows × `N_SAMPLES` calls/model. `gpt-4.1-nano`
/ `gpt-4o-mini` are cheap; `gpt-5-nano` bills reasoning tokens (≈1–2k gen-toks
even at `effort=low`). Use `LIMIT` / `STEPS` / lower `N_SAMPLES` to bound spend.
Runs **resume** by default (skip already-saved rows); `--overwrite` re-queries.

## Multi-round variant (mirrors `../gpt_oss_eval/multi-cite-gpt-eval/`)

Same wrapper idea, against the **multi-round** eval: some hints are pre-marked
`status: "completed"` (given in earlier rounds); the model selects the next hint
among the `pending` ones and cites any pending hints it newly recognizes as
achieved. Reuses `../gpt_oss_eval/multi-cite-gpt-eval/run_multi_selection.py`,
its `prompt_template_multiF.py`, and its static `benchmark.jsonl` (266 rows) —
so runs are apples-to-apples with the gpt-oss reference.

| file | role |
|---|---|
| `run_openai_multi_eval.sh` | entry: loops models → `multi_results/multi__<model>__<ts>/`, then compares. `PARALLEL_MODELS=1` (default) runs all models at once. |
| `run_openai_multi_selection.py` | thin wrapper over `run_multi_selection.py`. |
| **`prompt_template_multiF.py`** | **editable local multi-round prompt** — edit this to revise the prompt these runs use. |
| `compare_multi_runs.py` | multi-round comparison table (`--gpt-oss` adds the reference row). |
| `openai_sampler.py` | shared OpenAI model-call layer (used by both drivers). |

Reported stats match the reference `_summary.json` exactly: `mean_agreement_strict`
/ `mean_agreement_merged`, `majority_{strict,merged}_rate`, `mean_self_consistency`,
`selected_completed_rate` (picks an already-completed hint; want ~0),
`newly_completed_recall` (on the `n_rows_with_gap` rows with achieved-but-unmarked
pending hints), `citation` (verbatim tiers), and the `by_step` breakdown.

```bash
bash run_openai_multi_eval.sh                       # 4 models, 266 rows, n=8
LIMIT=5 bash run_openai_multi_eval.sh               # smoke (5 rows/step)
python compare_multi_runs.py --gpt-oss              # compare vs gpt-oss-20b reference
```

Reference (`gpt-oss-20b`, n=16): agree_merged **0.835**, maj_merged 0.876,
self_cons 0.926, selected_completed 0.0045, newly_recall 0.688,
citation verbatim 0.646 / found 0.943 (1835 quotes).

## Cost reporting

Each run records `prompt_tokens` / `completion_tokens` per call and folds a
`usage` block into `_summary.json` with token totals and an **estimated USD
cost** (also printed at the end of each run and shown in the compare tables).

- Prices live in **`pricing.py`** (`PRICES`, USD per 1M tokens as `(input,
  output)`) — edit them or override at runtime with
  `OPENAI_PRICES_JSON='{"gpt-5-mini":[in,out]}'`. Unknown models show `n/a`.
- Cost is an estimate: `prompt_tokens` at input price (ignores prompt-cache
  discounts → slight over-estimate) + `completion_tokens` at output price (for
  reasoning models this correctly includes the billed hidden reasoning tokens).
- **Back-fill** older runs (no model calls): `python recost.py` reads each run's
  saved tokens, writes a `usage` block, and prints a per-run cost table + total.

Actual cost of the full multi-round run (266 rows × n=8, 4 models): **~$6.98**
(`gpt-4.1-mini` $3.58, `gpt-5-nano` $1.31, `gpt-4o-mini` $1.18, `gpt-4.1-nano`
$0.90) — `gpt-4.1-mini` dominates on output-token price.

## Results — multi-round eval (benchmark.jsonl, 266 rows, n=8)

All numbers vs the `gpt-oss-20b` reference (n=16). `agree_merged` = x.0≡x.1
merged; `sel_compl↓` = fraction picking an already-completed hint (want ~0);
`newly_rec` = recall of achieved-but-unmarked pending hints on the 86 gap rows;
`verbatim`/`found` = citation-quote tiers; `gen_tok` = billed output tokens/call.

| model | agree_merged | maj_merged | self_cons | sel_compl↓ | newly_rec | verbatim | found | gen_tok |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| _gpt-oss-20b (ref, n=16)_ | _0.835_ | _0.876_ | _0.926_ | _0.0045_ | _0.688_ | _0.646_ | _0.943_ | _1260_ |
| **gpt-5-mini** | **0.873** | **0.898** | 0.946 | 0.0000 | **0.869** | **0.927** | 0.983 | 649 |
| gpt-5.4-mini | 0.850 | 0.846 | 0.938 | 0.0005 | 0.769 | 0.816 | 0.953 | 328 |
| gpt-5.4-nano | 0.823 | 0.850 | 0.928 | 0.0005 | 0.641 | 0.755 | **0.995** | 376 |
| gpt-4o-mini | 0.610 | 0.602 | 0.948 | 0.0000 | 0.106 | 0.297 | 0.535 | 120 |
| gpt-4.1-mini | 0.609 | 0.617 | 0.931 | 0.0000 | 0.417 | 0.728 | 0.846 | 243 |
| gpt-5-nano | 0.598 | 0.684 | 0.733 | 0.0000 | 0.593 | 0.117 | 0.354 | 1180 |
| gpt-4.1-nano | 0.553 | 0.571 | 0.862 | 0.0230 | 0.294 | 0.005 | 0.301 | 244 |

- **`gpt-5-mini` beats gpt-oss-20b on every metric** (selection, recall, citation)
  at half the output tokens — best selector tested. Both `gpt-5.4` models are
  close behind and far better than the gpt-4.1/4o batch on citation.
- The gpt-4.1/4o batch is weak on citation (gpt-4.1-nano fabricates: verbatim
  0.005 over 3012 quotes). Caveat: closed models ran n=8 vs the reference's n=16.

## gpt-5.x models — latency & hidden reasoning (from the n=8 run)

All three are **reasoning models** (billed output = hidden reasoning + visible
JSON; the API never returns the reasoning — `reasoning_content` is `None`).
`effort=low`. Latency measured under `row_workers=64` concurrency (incl. queuing).

| model | median lat | mean lat | p90 | billed out | visible | hidden | hidden% |
|---|--:|--:|--:|--:|--:|--:|--:|
| gpt-5.4-mini | 2.7 s | 3.1 s | 4.8 s | 328 | 199 | 129 | 39% |
| gpt-5.4-nano | 2.9 s | 3.7 s | 6.2 s | 376 | 211 | 165 | 44% |
| gpt-5-mini | 7.4 s | 9.6 s | 17.8 s | 649 | 256 | 393 | 61% |

`gpt-5-mini` reasons most (61% of output hidden) → slowest + priciest. Cost math
already correct: billed `completion_tokens` (incl. reasoning) priced as output.

## Projected cost to serve as the RL selector

For run `logs/HPRL-AutoHint-Qwen3-8B-Base-dapo-20260720-235159`:

- **522,932 selector calls** logged in `selector_calls/*.jsonl` (8 worker files,
  verified disjoint — no double-logging).
- **Input:** tiktoken `o200k_base` on a uniform sample of 10,456 calls →
  **mean 5,432 tok/call** (median 4,330; grows ~3.8k→8k over training as traces
  lengthen). Total ≈ **2.84 B input tokens**.
- **Output:** transferred from the eval per model (can't be measured from the run
  — its logged outputs are gpt-oss's): 649 / 328 / 376 tok/call.
- **Prices (assumed):** `gpt-5-mini` & `gpt-5.4-mini` $0.25/$2.00 per 1M in/out;
  `gpt-5.4-nano` $0.05/$0.40. The `gpt-5.4-*` prices are proxied from the gpt-5
  tier — **unverified**; `gpt-5-mini` is best-known.

| model | A: by-call (eval profile) | B: run input + eval out |
|---|--:|--:|
| gpt-5-mini | ≈ $1,098 | ≈ $1,389 |
| gpt-5.4-mini | ≈ $762 | ≈ $1,053 |
| gpt-5.4-nano | ≈ **$162** | ≈ **$221** |

Method B is more accurate (uses the run's real, longer prompts; A uses the
benchmark's shorter 3,204-tok prompts and so under-counts input by ~1.7×). Range
per model: gpt-5-mini **$1.1–1.4k**, gpt-5.4-mini **$0.76–1.05k**, gpt-5.4-nano
**$0.16–0.22k**. Neither applies OpenAI's prompt-cache discount (~5–15% off input
for the shared template prefix). `gpt-5.4-nano` is ~5–6× cheaper and still scores
close to gpt-oss-20b → best price/quality pick for a hosted selector.

## Status

- 2026-07-22: single-round + multi-round wrappers written and validated.
- Full multi-round run (266 rows, n=8) done for **7 models**: gpt-4o-mini,
  gpt-4.1-nano, gpt-4.1-mini, gpt-5-nano (first batch), then gpt-5-mini,
  gpt-5.4-mini, gpt-5.4-nano — all 266/266, 0 API errors. See Results above.
- Added cost reporting (`pricing.py`, `usage` block, `recost.py`) and the RL
  selector-cost projection for the HPRL-AutoHint run (above).
- Fixed a trailing curly-quote in `env.sh` that broke `source env.sh`, and made
  both runners tolerate comma-separated `MODELS` lists.
- **Open:** `gpt-5.4-*` have no verified prices (cost shows `n/a` / uses assumed
  proxy). Fill `pricing.PRICES` when known. Possible next: `effort=minimal` vs
  `low` tradeoff on gpt-5-mini; uncontended single-call latency probe.
