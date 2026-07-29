## OpenAI closed-model MULTI-ROUND eval — hint-selection + citation

_5 run(s); benchmark = benchmark.jsonl_

### Selection, completed-hint handling & cost

| model | n | rows | agree_strict | agree_merged | maj_merged | self_cons | sel_completed↓ | newly_recall | gap_rows | gen_toks | trunc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `gpt-oss-20b` | 16 | 266/266 | 0.605 | 0.835 | 0.876 | 0.926 | 0.0045 | 0.688 | 86 | 1260 | 7 |
| `gpt-4.1-mini` | 8 | 266/266 | 0.434 | 0.609 | 0.617 | 0.931 | 0.0000 | 0.417 | 86 | 243 | 0 |
| `gpt-4.1-nano` | 8 | 266/266 | 0.402 | 0.553 | 0.571 | 0.862 | 0.0230 | 0.294 | 86 | 244 | 0 |
| `gpt-4o-mini` | 8 | 266/266 | 0.433 | 0.610 | 0.602 | 0.948 | 0.0000 | 0.106 | 86 | 120 | 0 |
| `gpt-5-nano` | 8 | 266/266 | 0.428 | 0.598 | 0.684 | 0.733 | 0.0000 | 0.593 | 86 | 1180 | 0 |

### Citation fidelity (model completed_hints quotes vs. the trace)

| model | n_quotes | verbatim_rate | found_rate | not_found_rate | exact | normalized | loose | fuzzy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `gpt-oss-20b` | 1835 | 0.646 | 0.943 | 0.057 | 474 | 58 | 653 | 546 |
| `gpt-4.1-mini` | 843 | 0.728 | 0.846 | 0.154 | 530 | 39 | 45 | 99 |
| `gpt-4.1-nano` | 3012 | 0.005 | 0.301 | 0.699 | 2 | 2 | 10 | 892 |
| `gpt-4o-mini` | 155 | 0.297 | 0.535 | 0.465 | 2 | 1 | 43 | 37 |
| `gpt-5-nano` | 1948 | 0.117 | 0.354 | 0.645 | 105 | 1 | 121 | 463 |

_agree_merged = x.0≡x.1 merged (model can't pick x.0). sel_completed↓ = fraction picking an already-completed hint (lower is better, ~0). newly_recall = of achieved-but-unmarked pending hints in gap rows, fraction re-recognized. verbatim_rate = exact+normalized+loose; found_rate includes fuzzy._
