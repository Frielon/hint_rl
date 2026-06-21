#!/usr/bin/env python3
# Copyright 2026
#
# eval_ckpts.py -- math-accuracy eval of ONE checkpoint served on an
# OpenAI-compatible endpoint (vLLM /v1). Mirrors how the GRPO/HPRL training
# validates, but at pass@k scale (each problem rolled out --n times).
#
# It reads a verl-format parquet:
#   * prompt              -> a chat message list (system + user), fed verbatim
#                            to /v1/chat/completions (server applies the chat
#                            template, exactly as the verl rollout does);
#   * reward_model.ground_truth -> the gold answer;
#   * data_source         -> per-set label (aime2024 / math_dapo).
# samples each problem --n times, and grades the boxed answer with mathruler --
# the SAME grader the training rewards use (custom_reward.compute_score /
# hint_reward.compute_score), so the numbers are directly comparable to the
# wandb val curves and across checkpoints.
#
# HPRL hint-template eval (--hint-mode): the eval sets aime2024-hint-mt /
# dapo_sample_hard_100-hint-mt carry the budget-0 hint-tool prompt (the SAME
# template training validated on). Faithful to HintAgentLoop @ budget 0 +
# hint_reward.compute_score: if a sample ENDS with the sentinel <hint_call/>
# alone on its last line, it is an over-budget protocol violation -> acc=0 and
# the box is NOT graded (the hint_budget_exceeded short-circuit). Every other
# sample is graded on its box as usual. The hint-call rate is reported as a
# behavior diagnostic, alongside a box-only accuracy that ignores the penalty.
#
# Output (per model x dataset, under --out-dir):
#   * samples.jsonl  -- one row per sample (pred / acc / finish_reason / hint).
#   * _summary.json  -- avg@n accuracy, pass@k curve, format/truncation rates,
#                       timing, and (hint-mode) hint-call stats.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Optional

import httpx
import pandas as pd
from mathruler.grader import extract_boxed_content, grade_answer
from openai import AsyncOpenAI

# Sentinel returned by extract_boxed_content when nothing is boxed (matches
# hint_reward._NO_BOX).
_NO_BOX = "None"
# The sentinel the HPRL policy emits to request a hint (matches
# hint_agent_loop.HINT_SENTINEL).
HINT_SENTINEL = "<hint_call/>"


# --------------------------------------------------------------------------- #
# grading (identical core to hint_reward / custom_reward)
# --------------------------------------------------------------------------- #
def is_hint_call(text: str) -> bool:
    """True iff the finished turn ends with the sentinel alone on its own line.

    Byte-for-byte the rule HintAgentLoop._is_hint_call uses to recognize an
    over-budget hint call at budget 0: the sentinel is on its OWN line and is the
    LAST thing emitted before EOS. An inline / mid-reasoning sentinel is NOT a
    call.
    """
    stripped = text.rstrip()
    last_nl = stripped.rfind("\n")
    last_line = stripped[last_nl + 1 :] if last_nl != -1 else stripped
    return last_line.strip() == HINT_SENTINEL


def grade_one(text: str, ground_truth: str, hint_mode: bool) -> dict:
    """Grade a single completion the way the training reward would.

    Returns acc/has_format/pred plus, in hint mode, the over-budget flag and a
    diagnostic box-only accuracy (what the box would have scored absent the
    hint-call penalty).
    """
    pred = extract_boxed_content(text)
    has_format = pred != _NO_BOX
    box_correct = bool(has_format and grade_answer(pred, ground_truth))

    hint_called = bool(hint_mode and is_hint_call(text))
    # HPRL @ budget 0: an over-budget <hint_call/> short-circuits to the floor
    # with acc=0 and the box ungraded (hint_reward.compute_score). Otherwise the
    # official acc is just the box correctness.
    acc = 0.0 if hint_called else (1.0 if box_correct else 0.0)
    return {
        "pred": pred,
        "has_format": 1.0 if has_format else 0.0,
        "acc": acc,
        "acc_box_only": 1.0 if box_correct else 0.0,
        "hint_called": 1.0 if hint_called else 0.0,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Codex/HumanEval estimator): P(>=1 correct in k drawn
    without replacement from n samples, c of which are correct)."""
    if k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
async def gen_chunk(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    *,
    model: str,
    messages: list,
    chunk_n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    retries: int,
    seed: Optional[int] = None,
) -> dict:
    """One /v1/chat/completions request asking for ``chunk_n`` samples.

    Returns {"texts": [...], "finish": [...], "completion_tokens": int,
    "error": str|None}. Light exponential-backoff retry; a request that never
    succeeds yields empty texts and an error string (its samples are dropped
    from the denominator and counted in the summary).

    ``seed`` is forwarded to vLLM as a per-request sampling seed. It is REQUIRED
    for independence here: the engine pins a global ``seed=0``, so without a
    distinct per-request seed all samples of a prompt collapse onto a couple of
    reproducible completions (verified). A unique seed per request gives an
    independent — and reproducible — draw.
    """
    extra_body: dict[str, Any] = {"top_k": top_k}
    if seed is not None:
        extra_body["seed"] = seed
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=chunk_n,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                texts = [(c.message.content or "") for c in resp.choices]
                finish = [c.finish_reason for c in resp.choices]
                ctok = int(resp.usage.completion_tokens) if resp.usage else 0
                return {"texts": texts, "finish": finish, "completion_tokens": ctok, "error": None}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            await asyncio.sleep(min(2.0 * (2**attempt), 30.0))
    return {"texts": [], "finish": [], "completion_tokens": 0, "error": last_err}


def load_problems(parquet: str, limit: Optional[int]) -> list[dict]:
    df = pd.read_parquet(parquet)
    if limit:
        df = df.head(limit)
    problems = []
    for i, row in df.iterrows():
        # prompt is an ndarray/list of {role, content} dicts.
        messages = [{"role": m["role"], "content": m["content"]} for m in list(row["prompt"])]
        rm = row["reward_model"]
        gt = rm["ground_truth"] if isinstance(rm, dict) else rm
        extra = row["extra_info"] if "extra_info" in row else None
        pid = None
        if isinstance(extra, dict):
            pid = extra.get("id") or extra.get("problem_id") or extra.get("index")
        problems.append(
            {
                "idx": int(i),
                "problem_id": pid if pid is not None else int(i),
                "data_source": row["data_source"],
                "messages": messages,
                "ground_truth": str(gt),
            }
        )
    return problems


async def run_eval(args) -> dict:
    problems = load_problems(args.dataset, args.limit)
    n_problems = len(problems)
    # Each problem's --n samples are issued as ceil(n/chunk) requests of --chunk
    # samples. chunk=1 (default) => one independent request per sample: this both
    # load-balances perfectly across the DP replicas AND guarantees independent
    # draws (vLLM V1 here returns identical completions for n>1 within a request).
    chunk = max(1, args.chunk)
    base_chunks = [chunk] * (args.n // chunk)
    if args.n % chunk:
        base_chunks.append(args.n % chunk)

    sem = asyncio.Semaphore(args.concurrency)
    # The OpenAI SDK's default httpx pool caps at ~100 connections, which would
    # throttle the semaphore above it; size the pool to the requested concurrency.
    pool = args.concurrency + 16
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
        timeout=args.timeout,
    )
    client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, timeout=args.timeout,
        max_retries=0, http_client=http_client,
    )

    tasks = []
    task_pidx = []  # parallel list: problem idx each request belongs to
    seed_counter = 0  # global, deterministic in iteration order -> reproducible
    for p in problems:
        for cn in base_chunks:
            tasks.append(
                asyncio.create_task(
                    gen_chunk(
                        client,
                        sem,
                        model=args.model,
                        messages=p["messages"],
                        chunk_n=cn,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens,
                        retries=args.retries,
                        seed=args.seed + seed_counter,
                    )
                )
            )
            task_pidx.append(p["idx"])
            seed_counter += 1

    total_tasks = len(tasks)
    print(
        f"[eval] {os.path.basename(args.dataset)}: {n_problems} problems x {args.n} "
        f"samples = {n_problems * args.n} completions across {total_tasks} requests "
        f"(chunk={chunk}, concurrency={args.concurrency}, hint_mode={args.hint_mode})",
        flush=True,
    )

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    wall = time.time() - t0

    by_problem: dict[int, list] = {p["idx"]: [] for p in problems}
    completion_tokens = 0
    errors = 0
    for res, pidx in zip(results, task_pidx):
        if res["error"]:
            errors += 1
        completion_tokens += res["completion_tokens"]
        by_problem[pidx].append(res)

    # ---- grade + aggregate ----
    samples_path = os.path.join(args.out_dir, "samples.jsonl")
    per_problem_stats = []
    all_acc, all_box, all_hint, all_format = [], [], [], []
    n_trunc = 0
    n_samples = 0
    n_problems_any_hint = 0
    pass_k_ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, args.n) if 1 <= k <= args.n})
    pass_k_accum = {k: [] for k in pass_k_ks}

    with open(samples_path, "w") as sf:
        for p in problems:
            pidx = p["idx"]
            gt = p["ground_truth"]
            texts, finishes = [], []
            for res in by_problem[pidx]:
                texts.extend(res["texts"])
                finishes.extend(res["finish"])
            c_correct = 0
            p_accs = []
            p_any_hint = False
            for s_i, (text, fin) in enumerate(zip(texts, finishes)):
                g = grade_one(text, gt, args.hint_mode)
                n_samples += 1
                all_acc.append(g["acc"])
                all_box.append(g["acc_box_only"])
                all_hint.append(g["hint_called"])
                all_format.append(g["has_format"])
                if g["hint_called"] >= 1.0:
                    p_any_hint = True
                if fin == "length":
                    n_trunc += 1
                c_correct += int(g["acc"] >= 1.0)
                p_accs.append(g["acc"])
                rec = {
                    "problem_idx": pidx,
                    "problem_id": p["problem_id"],
                    "data_source": p["data_source"],
                    "sample_idx": s_i,
                    "ground_truth": gt,
                    "pred": g["pred"],
                    "acc": g["acc"],
                    "acc_box_only": g["acc_box_only"],
                    "hint_called": g["hint_called"],
                    "has_format": g["has_format"],
                    "finish_reason": fin,
                }
                if args.save_text:
                    rec["text"] = text[: args.max_store_chars]
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if p_any_hint:
                n_problems_any_hint += 1
            n_got = len(p_accs)
            solve_rate = (sum(p_accs) / n_got) if n_got else 0.0
            per_problem_stats.append(
                {
                    "problem_idx": pidx,
                    "problem_id": p["problem_id"],
                    "data_source": p["data_source"],
                    "n_samples": n_got,
                    "n_correct": c_correct,
                    "solve_rate": solve_rate,
                    "solved_any": 1 if c_correct > 0 else 0,
                }
            )
            for k in pass_k_ks:
                if n_got >= k:
                    pass_k_accum[k].append(pass_at_k(n_got, c_correct, k))

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    summary = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_name": os.path.basename(args.dataset),
        "model": args.model,
        "model_label": args.model_label or args.model,
        "hint_mode": bool(args.hint_mode),
        "n_problems": n_problems,
        "n_requested_per_problem": args.n,
        "n_samples": n_samples,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
        "acc_mean": mean(all_acc),                 # avg@n == pass@1 estimate (official)
        "acc_box_only_mean": mean(all_box),        # ignores the hint-call penalty
        "format_rate": mean(all_format),
        "truncation_rate": (n_trunc / n_samples) if n_samples else 0.0,
        "pass_at_k": {f"pass@{k}": mean(pass_k_accum[k]) for k in pass_k_ks},
        "n_problems_solved_any": sum(s["solved_any"] for s in per_problem_stats),
        "request_errors": errors,
        "total_requests": total_tasks,
        "timing": {
            "wall_s": wall,
            "completion_tokens": completion_tokens,
            "throughput_tok_per_s": (completion_tokens / wall) if wall > 0 else 0.0,
            "mean_completion_tokens": (completion_tokens / n_samples) if n_samples else 0.0,
        },
        "per_problem": per_problem_stats,
    }
    if args.hint_mode:
        summary["hint_call_rate"] = mean(all_hint)
        summary["n_problems_any_hint_call"] = n_problems_any_hint

    summ_path = os.path.join(args.out_dir, "_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console headline ----
    print("=" * 70, flush=True)
    print(f"[eval] {summary['model_label']}  |  {summary['dataset_name']}", flush=True)
    print(
        f"  acc (avg@{args.n})  = {summary['acc_mean']:.4f}   "
        f"format = {summary['format_rate']:.3f}   trunc = {summary['truncation_rate']:.3f}",
        flush=True,
    )
    print("  pass@k: " + "  ".join(f"{k}={v:.3f}" for k, v in summary["pass_at_k"].items()), flush=True)
    if args.hint_mode:
        print(
            f"  [hint] hint_call_rate = {summary['hint_call_rate']:.3f}   "
            f"acc_box_only = {summary['acc_box_only_mean']:.4f}",
            flush=True,
        )
    print(
        f"  wall = {wall:.0f}s   throughput = {summary['timing']['throughput_tok_per_s']:.0f} tok/s   "
        f"errors = {errors}/{total_tasks}",
        flush=True,
    )
    print(f"  summary -> {summ_path}", flush=True)
    print("=" * 70, flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="verl-format parquet to eval on")
    p.add_argument("--model", required=True, help="served model id (vLLM --served-model-name)")
    p.add_argument("--model-label", default=None, help="human label for the summary/console")
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=128, help="samples per problem (rollouts)")
    # IMPORTANT: default 1 (one sample per request). vLLM V1 in this stack returns
    # BYTE-IDENTICAL completions for n>1 within a single request (verified: 4/4
    # identical), which would collapse N rollouts into N/chunk independent draws and
    # bias pass@k. Separate requests sample independently, and prefix-caching makes
    # the shared per-problem prompt nearly free, so chunk=1 costs ~nothing. Only set
    # chunk>1 if you have verified n>1 diversity on your engine.
    p.add_argument("--chunk", type=int, default=1, help="samples per request (keep 1; see code)")
    p.add_argument("--concurrency", type=int, default=192, help="max in-flight requests")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base sampling seed; request i uses seed+i (independent + reproducible draws)",
    )
    p.add_argument(
        "--hint-mode",
        action="store_true",
        help="HPRL hint-template eval: an over-budget <hint_call/> -> acc=0, box ungraded",
    )
    p.add_argument("--limit", type=int, default=None, help="eval only the first N problems (debug)")
    p.add_argument("--save-text", dest="save_text", action="store_true", default=True)
    p.add_argument("--no-save-text", dest="save_text", action="store_false")
    p.add_argument("--max-store-chars", type=int, default=200000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
