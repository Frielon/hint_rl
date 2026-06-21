# Checkpoint eval (pass@k math accuracy)

Evaluate the HPRL project checkpoints on the held-out math sets, serving each
checkpoint once on a local vLLM endpoint and rolling every problem out `N` times
(default **128**). The boxed answer is graded with **mathruler** — the exact
grader the training rewards use (`reward/custom_reward.compute_score` for GRPO,
`script/hint_rl/hint_reward.compute_score` for HPRL) — so the numbers are
directly comparable to the wandb val curves and across checkpoints.

## What it runs

| # | label | model | datasets | mode |
|---|---|---|---|---|
| 1 | `baseline` | `model/Qwen2.5-7B-Instruct` (HF) | `aime2024`, `dapo_sample_hard_100` | unaided |
| 2 | `grpo-step200` | `ckpt/GRPO-…-dapo-3139-…/global_step_200` (merged) | `aime2024`, `dapo_sample_hard_100` | unaided |
| 3 | `hprl-v3-step200` | `ckpt/HPRL-…-dapo-4k-v3-…/global_step_200` (merged) | `aime2024-hint-mt`, `dapo_sample_hard_100-hint-mt` | `--hint-mode` |

The GRPO/HPRL checkpoints are verl FSDP shards (`world_size_32`); the launcher
merges them to HF first via `python -m verl.model_merger` (idempotent, cached
under `eval/merged/`). The baseline is already HF.

**HPRL `--hint-mode`.** The `*-hint-mt` sets carry the **budget-0** hint-tool
prompt (the same template training validated on). Faithful to `HintAgentLoop` at
budget 0 + `hint_reward.compute_score`: a sample that **ends** with `<hint_call/>`
alone on its last line is an over-budget protocol violation → scored **acc=0**,
the box is **not** graded (the `hint_budget_exceeded` short-circuit). Every other
sample is graded on its box. The hint-call rate is reported as a behavior
diagnostic, with `acc_box_only` showing accuracy if the penalty were ignored.

## Run (1 node, 8× H100)

```bash
cd eval
./run_eval_ckpts.sh                      # all three, N=128, on all 8 GPUs (DP=8, TP=1)
```

Useful overrides (env vars):

```bash
MODELS="baseline grpo"   ./run_eval_ckpts.sh   # subset (of: baseline grpo hprl)
N=64 CONCURRENCY=256     ./run_eval_ckpts.sh   # fewer rollouts, more in-flight
LIMIT=3                  ./run_eval_ckpts.sh   # smoke test (first 3 problems/set)
TEMPERATURE=0.7          ./run_eval_ckpts.sh   # default 1.0 (matches training rollout)
```

Other knobs: `PORT CONTEXT_LEN MEM_FRAC MAX_NUM_SEQS` (server); `CHUNK TOP_P
TOP_K MAX_TOKENS` (eval); `CONDA_ROOT CONDA_ENV` (env, default `verl`); `DP`
(auto = visible GPU count). Sampling defaults match the training rollout: temp
`1.0`, top_p `1.0`, top_k `-1`, max_tokens `8192` (= `max_response_length`).

The eval client (`eval_ckpts.py`) and server both run in the **`verl`** conda env
(vllm 0.12 + mathruler + pandas + openai + verl), activated conda-free exactly as
`selector/run_eval_h100.sh` does. To eval a single (model, dataset) against an
already-running endpoint, call `eval_ckpts.py` directly (see `--help`).

## Output

`eval/results/run_<ts>/<label>/<dataset>/`:
- **`_summary.json`** — `acc_mean` (avg@N = pass@1), `pass_at_k` curve,
  `format_rate`, `truncation_rate`, timing, per-problem solve rates, and (hint
  mode) `hint_call_rate` / `acc_box_only_mean`.
- **`samples.jsonl`** — one row per sample: `pred`, `acc`, `acc_box_only`,
  `hint_called`, `has_format`, `finish_reason`, and the full `text`.

A combined table across all runs is printed at the end of the launcher.

### Metrics

- **`acc_mean` (avg@N)** — mean accuracy over all `N` rollouts of every problem;
  the unbiased pass@1 estimate and the headline number. For HPRL it is the
  *official* accuracy (over-budget `<hint_call/>` counts as 0).
- **`pass@k`** — unbiased Codex estimator over the `N` samples; `pass@N` is the
  "solved at least once" rate.
- **`hint_call_rate`** (hint mode) — fraction of samples that emitted a terminal
  `<hint_call/>` despite budget 0 (the over-call behavior signal).
- **`acc_box_only_mean`** (hint mode) — box accuracy ignoring the hint-call
  penalty; the gap to `acc_mean` is the accuracy lost to over-budget calls.
