#!/usr/bin/env python3
# Copyright 2026
#
# Budget CALIBRATION for auto-hint HPRL.
#
# Runs the auto-hint loop OFFLINE over the whole training set to pick a per-problem
# starting budget B_q, written as a budget-state JSON consumable by HintBudgetDataset /
# BudgetManager (same format as budget_state_hint_wise.json: {"budgets": {pid: B}, ...}).
#
# Procedure (matches auto_hint_agent_loop.AutoHintAgentLoop):
#   * set a fixed probe budget for every problem (default 10);
#   * for each problem, run N rollouts (default 32). Each rollout: the POLICY answers the
#     plain single-turn math prompt; on a WRONG boxed answer the frozen SELECTOR
#     (multi-round Template F) gives the next hint, injected as a user turn, until the
#     answer is correct, the budget is spent, or the pending-hint pool is exhausted;
#   * record each rollout's hints used -- ``len(applied)`` if it solved, else the full
#     ``budget`` (a non-solve "needs at least the whole budget"; a pool-exhausted non-solve
#     is NOT credited its small hint count);
#   * the calibrated B_q = the K-th SMALLEST hints-used over the N rollouts (default K=4) --
#     the budget at which >= K rollouts solved with <= that many hints (clamped to
#     [0, --clamp-max]).
#
# Needs TWO OpenAI-compatible endpoints: the POLICY (the model being calibrated, e.g. the
# base Qwen2.5-7B-Instruct served with vLLM/sglang) and the SELECTOR (frozen gpt-oss-20b,
# via the SELECTOR_* env, same as training). It does NOT import verl / the training stack.
#
# Resumable: per-problem results are cached under <out>.cache/; a re-run skips finished
# problems. Use --limit / --dry-run to test first.
#
# Examples:
#   # dry run: build the first prompts, no model calls
#   python calibrate_budget.py --dry-run --limit 3
#   # real calibration (policy served at :8000, selector via SELECTOR_BASE_URL)
#   python calibrate_budget.py --policy-base-url http://127.0.0.1:8000/v1 \
#       --policy-model Qwen2.5-7B-Instruct --budget 10 --n 32 --rank 4 --workers 128
from __future__ import annotations

import argparse
import json
import asyncio
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

# This module now lives in budget_calibration/; put the hint_rl package dir (its
# parent) on sys.path so the sibling-module imports below still resolve whether run
# directly (python calibrate_budget.py) or via launch_calibration_cluster.sh.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Standalone reuse of the vendored auto-hint components (no verl import).
from hint_selector import build_trace
from selector_multi import build_prompt_multi, pending_hint_ids, render_hints_with_status
from utils import hint_id_of, parse_output

# Boxed-answer grader -- the SAME one the reward + the loop use.
try:
    from mathruler.grader import extract_boxed_content, grade_answer
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"mathruler is required (the grader): {e}")

HERE = Path(__file__).resolve().parent
# budget_calibration/ -> hint_rl(pkg) -> script/ -> hint_rl(project root, holds dataset/)
HINT_RL_HOME = HERE.parent.parent.parent
DS = HINT_RL_HOME / "dataset"


# --------------------------------------------------------------------------- #
# the hint user-turn wording -- byte-identical to AutoHintAgentLoop._format_hint
# (replicated here so this script does not import the verl-dependent agent loop).
# --------------------------------------------------------------------------- #
def format_hint(hint_text: str) -> str:
    body = (hint_text if isinstance(hint_text, str) else "").strip() or "(the selector returned an empty hint)"
    return (
        f"Here is a hint to help you make progress:\n{body}\n\n"
        "Using this hint, continue your reasoning and give your final boxed answer."
    )


def is_correct(messages: list[dict], ground_truth: str) -> bool:
    """Grade the boxed answer over the joined assistant turns (as the loop does)."""
    solution = "\n".join(m["content"] for m in messages if m.get("role") == "assistant")
    pred = extract_boxed_content(solution)
    return pred != "None" and grade_answer(pred, ground_truth)


# --------------------------------------------------------------------------- #
# OpenAI-compatible clients (lazy import so --dry-run needs no openai)
# --------------------------------------------------------------------------- #
class Endpoint:
    """One served model reachable over >=1 OpenAI-compatible base URL (round-robin
    + failover across the URLs, like hint_selector.HintSelector)."""

    def __init__(self, base_urls, model, api_key, temperature, top_p, max_tokens,
                 timeout=600.0, max_retries=3):
        if isinstance(base_urls, str):
            base_urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        self.base_urls = list(base_urls) or ["http://localhost:8000/v1"]
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._clients: dict[str, Any] = {}

    def _client(self, base_url: str):
        c = self._clients.get(base_url)
        if c is None:
            from openai import AsyncOpenAI

            c = AsyncOpenAI(base_url=base_url, api_key=self.api_key, timeout=self.timeout)
            self._clients[base_url] = c
        return c

    async def chat(self, messages: list[dict]) -> Optional[str]:
        """Return the assistant content, or None on failure (after retries/failover).

        CancelledError propagates (re-raised, NOT retried) so an early-stop break can
        abort the in-flight request -- closing the connection makes vLLM abort the
        generation (true zero-overrun early stop)."""
        n = len(self.base_urls)
        start = random.randrange(n)
        for attempt in range(self.max_retries):
            base_url = self.base_urls[(start + attempt) % n]
            try:
                resp = await self._client(base_url).chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- try the next URL / retry
                continue
        return None


async def select_hint(selector: Endpoint, problem: str, trace: str, pool, completed) -> Optional[dict]:
    """One multi-round Template F selector call (async). Returns the parsed dict or None.
    CancelledError propagates so an early-stop break aborts the in-flight selector gen."""
    prompt = build_prompt_multi(problem, trace, render_hints_with_status(pool, completed))
    n = len(selector.base_urls)
    start = random.randrange(n)
    for attempt in range(selector.max_retries):
        base_url = selector.base_urls[(start + attempt) % n]
        try:
            resp = await selector._client(base_url).chat.completions.create(
                model=selector.model, messages=[{"role": "user", "content": prompt}],
                temperature=selector.temperature, top_p=selector.top_p, max_tokens=selector.max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            sel, _err = parse_output(raw)
            if isinstance(sel, dict):
                return sel
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            continue
    return None


# --------------------------------------------------------------------------- #
# one auto-hint rollout -> hints used
# --------------------------------------------------------------------------- #
async def run_rollout(rec: dict, budget: int, max_turns: int, policy: Endpoint, selector: Endpoint,
                      keep_transcript: bool = False) -> dict:
    """Run a single auto-hint rollout for one problem (async). Returns
    {hint_used, correct, n_hints, n_turns, stop} (+ {applied, messages} when
    ``keep_transcript``, for --dump-rollouts).

    ``hint_used`` = len(applied) if the rollout solved, else the full ``budget``.
    Cancelling this coroutine (early-stop break) propagates through the awaits and
    aborts whatever request is in flight."""
    messages = [
        {"role": "system", "content": rec["system"]},
        {"role": "user", "content": rec["user"]},
    ]
    pool, gt, problem = rec["pool"], rec["gt"], rec["problem"]
    applied: list[str] = []      # given hint ids (penalized hints)
    completed: list[str] = []    # given + self-completed hint ids (status rendering)
    correct = False
    stop = "max_turns"
    for _ in range(max_turns):
        text = await policy.chat(messages)
        if text is None:
            stop = "policy_error"
            break
        messages.append({"role": "assistant", "content": text})
        if is_correct(messages, gt):
            correct, stop = True, "correct"
            break
        if len(applied) >= budget:
            stop = "budget"
            break
        if not pending_hint_ids(pool, completed):
            stop = "pool_exhausted"
            break
        sel = await select_hint(selector, problem, build_trace(messages), pool, completed)
        if not isinstance(sel, dict):
            stop = "selector_error"
            break
        hid = hint_id_of(sel)
        ch = sel.get("completed_hints")
        for c in (ch if isinstance(ch, list) else []):
            cid = c.get("hint_id") if isinstance(c, dict) else None
            if cid is not None and str(cid) not in completed:
                completed.append(str(cid))
        if hid is not None and str(hid) not in completed:
            completed.append(str(hid))
        applied.append(str(hid))
        messages.append({"role": "user", "content": format_hint(sel.get("hint"))})
    out = {
        "hint_used": (len(applied) if correct else budget),
        "correct": correct,
        "n_hints": len(applied),
        "n_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "stop": stop,
    }
    if keep_transcript:
        out["applied"] = applied        # given hint ids, in order
        out["messages"] = messages      # full conversation (system, problem, turns, hints)
    return out


def kth_min(values: list[int], k: int) -> int:
    """The k-th SMALLEST value (1-based). If fewer than k values, the largest."""
    s = sorted(values)
    return s[min(k, len(s)) - 1] if s else 0


def rollout_row(pid: str, gt: str, idx: int, r: dict) -> dict:
    """One dumped-rollout record (outcome + full transcript) for the per-problem .jsonl
    'middle file', written incrementally as each rollout lands."""
    return {
        "problem_id": pid,
        "rollout": idx,
        "gt": gt,
        "correct": r["correct"],
        "hint_used": r["hint_used"],
        "n_hints": r["n_hints"],
        "n_turns": r["n_turns"],
        "stop": r["stop"],
        "applied": r.get("applied"),
        "messages": r.get("messages"),
    }


async def calibrate_one_problem(p, args, policy, selector, dump_dir, stop_at):
    """Run a problem's rollouts with up to ``--rollout-workers`` IN FLIGHT, write each to
    its <dump_dir>/<pid>.jsonl ("middle file") as it lands, and BREAK as soon as ``stop_at``
    are correct -- cancelling the still-in-flight rollout tasks, which closes their
    connections so vLLM ABORTS the generations (true zero-overrun early stop). Returns the
    list of rollouts that actually completed (cancelled ones don't count)."""
    pid, gt = p["problem_id"], p["gt"]
    keep = dump_dir is not None
    fh = open(dump_dir / f"{safe(pid)}.jsonl", "w") if keep else None
    sem = asyncio.Semaphore(max(1, args.rollout_workers))
    rolls: list[dict] = []
    n_correct = 0

    async def one():
        async with sem:
            return await run_rollout(p, args.budget, args.max_turns, policy, selector, keep)

    tasks = [asyncio.create_task(one()) for _ in range(args.n)]
    pending = set(tasks)
    try:
        while pending and (not stop_at or n_correct < stop_at):
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    r = t.result()
                except asyncio.CancelledError:
                    continue
                except Exception as e:  # noqa: BLE001 -- a dead rollout = non-solve at budget
                    r = {"hint_used": args.budget, "correct": False, "n_hints": 0, "n_turns": 0, "stop": f"error:{e}"}
                rolls.append(r)
                if fh is not None:
                    fh.write(json.dumps(rollout_row(pid, gt, len(rolls) - 1, r), ensure_ascii=False) + "\n")
                    fh.flush()
                if r.get("correct"):
                    n_correct += 1
    finally:
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)  # let cancellation abort the in-flight gens
        if fh is not None:
            fh.close()
    return rolls


async def run_calibration(args, problems, results, out_path, cache_dir, dump_dir) -> int:
    """Async driver: process ``--problem-workers`` problems concurrently, each running its
    rollouts (up to ``--rollout-workers`` in flight) with early-stop cancellation. As each
    problem finishes, aggregate its budget (kth_min), cache it, and mid-save."""
    policy = Endpoint(args.policy_base_url, args.policy_model, args.policy_api_key,
                      args.policy_temperature, args.policy_top_p, args.policy_max_tokens)
    selector = Endpoint(args.selector_base_urls, args.selector_model, args.selector_api_key,
                        args.selector_temperature, args.selector_top_p, args.selector_max_tokens)

    todo = [p for p in problems if p["problem_id"] not in results]
    print(f"resume     = {len(results)} cached, {len(todo)} to run\n"
          f"out        = {out_path}\ncache      = {cache_dir}\n"
          f"batch      = {args.problem_workers} problems x up to {args.rollout_workers} rollouts in flight "
          f"({args.problem_workers * args.rollout_workers} max concurrent rollouts)\n"
          f"early-stop = at {args.stop_at_correct} correct/problem -> cancel+abort the rest (0=off)\n"
          f"mid-save   = every {args.save_every} finished problems -> {out_path}\n"
          f"dump       = {dump_dir if dump_dir is not None else '(off)'}\n")

    psem = asyncio.Semaphore(max(1, args.problem_workers))
    t0 = time.time()

    async def process(p):
        async with psem:
            rolls = await calibrate_one_problem(p, args, policy, selector, dump_dir, args.stop_at_correct)
        return p["problem_id"], rolls

    tasks = [asyncio.create_task(process(p)) for p in todo]
    try:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), unit="problem",
                         desc="calibrate", dynamic_ncols=True):
            try:
                pid, rolls = await coro
            except Exception as e:  # noqa: BLE001
                print(f"[calibrate] a problem task failed: {e}", flush=True)
                continue
            hint_useds = [r["hint_used"] for r in rolls]
            rec = {
                "problem_id": pid,
                "budget": int(min(args.clamp_max, kth_min(hint_useds, args.rank))),
                "n_solved": sum(1 for r in rolls if r.get("correct")),
                "n_rollouts": len(rolls),          # completed (< args.n when early-stopped)
                "hint_useds": sorted(hint_useds),
                "stops": {s: sum(1 for r in rolls if r.get("stop") == s)
                          for s in sorted({r.get("stop") for r in rolls})},
            }
            results[pid] = rec
            (cache_dir / f"{safe(pid)}.json").write_text(json.dumps(rec))
            if len(results) % max(1, args.save_every) == 0:
                write_payload(build_payload(results, args), out_path)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    payload = build_payload(results, args)
    write_payload(payload, out_path)
    print(f"\ndone in {time.time() - t0:.0f}s -> {out_path}")
    print(f"budget histogram: {payload['meta']['budget_histogram']}")
    print(f"mean problems solved/{args.n}: {payload['meta']['mean_solved_per_problem']}")
    return 0


def build_payload(results: dict, args) -> dict:
    """Aggregate the finished per-problem results into the budget-state JSON payload
    ({"budgets": {pid: B}, "meta": {...}}) -- same format as budget_state_hint_wise.json."""
    import collections

    budgets = {pid: int(results[pid]["budget"]) for pid in results}
    hist = dict(sorted(collections.Counter(budgets.values()).items()))
    # seeded (--resume-from) recs carry no n_solved -> excluded from the solve-rate mean.
    solved = [results[pid]["n_solved"] for pid in results if results[pid].get("n_solved") is not None]
    return {
        "budgets": budgets,
        "meta": {
            "source": "calibrate_budget.py",
            "data": str(args.data),
            "probe_budget": args.budget,
            "n_rollouts": args.n,
            "rank": args.rank,
            "clamp_max": args.clamp_max,
            "max_turns": args.max_turns,
            "policy_model": args.policy_model,
            "selector_model": args.selector_model,
            "n_problems": len(budgets),
            "budget_histogram": {str(k): v for k, v in hist.items()},
            "mean_solved_per_problem": round(sum(solved) / len(solved), 2) if solved else 0.0,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }


def write_payload(payload: dict, path: Path) -> None:
    """Write the budget JSON atomically (tmp + replace) so a concurrent reader (or an
    --aggregate-only run) never sees a half-written file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# load problems
# --------------------------------------------------------------------------- #
def load_problems(path: Path, limit: Optional[int]) -> list[dict]:
    df = pd.read_parquet(path)
    out = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        ck = ((ei.get("tools_kwargs") or {}).get("request_hint", {}) or {}).get("create_kwargs", {}) or {}
        prompt = list(r["prompt"])
        system = next((m["content"] for m in prompt if m.get("role") == "system"), "")
        user = next((m["content"] for m in prompt if m.get("role") == "user"), "")
        pool = ck.get("hints") or ei.get("hint") or ""
        gt = str(ck.get("ground_truth") or (r.get("reward_model") or {}).get("ground_truth") or "")
        pid = ei.get("problem_id")
        if pid is None or not pool or not gt:
            continue
        out.append({"problem_id": str(pid), "system": system, "user": user,
                    "problem": user, "pool": pool, "gt": gt})
        if limit is not None and len(out) >= limit:
            break
    return out


def safe(s: str) -> str:
    return str(s).replace("/", "_").replace(os.sep, "_")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DS / "dapo-3139-auto-hint.parquet"))
    ap.add_argument("--out", default=str(HERE / "budget_state_calibrated.json"))
    ap.add_argument("--budget", type=int, default=10, help="fixed probe budget for every problem")
    ap.add_argument("-n", "--n", type=int, default=32, help="rollouts per problem")
    ap.add_argument("--rank", type=int, default=4, help="take the K-th smallest hints-used as B_q")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="assistant-turn cap per rollout (default budget+2, so all budget hints fit)")
    ap.add_argument("--clamp-max", type=int, default=None,
                    help="ceiling on the calibrated B_q (default = --budget)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (testing)")
    ap.add_argument("--problem-workers", type=int, default=8,
                    help="how many PROBLEMS to process concurrently (the 'batch of problems')")
    ap.add_argument("--rollout-workers", type=int, default=32,
                    help="rollouts IN FLIGHT per problem (each runs up to this many in parallel, then "
                         "early-stops and ABORTS the rest at --stop-at-correct). Total concurrent "
                         "rollouts = problem-workers x rollout-workers.")
    ap.add_argument("--workers", type=int, default=None,
                    help="(deprecated alias) if set, used as --problem-workers")
    ap.add_argument("--save-every", type=int, default=50,
                    help="re-write the aggregated budget JSON every N finished problems (mid-save)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="aggregate the EXISTING per-problem cache into the budget JSON and exit -- "
                         "extracts already-done problems, no model calls")
    ap.add_argument("--cache-dir", default=None,
                    help="per-problem cache dir to read/write (default <out>.cache). Point this at a "
                         "RUNNING job's cache (with a different --out) to extract it without clobbering.")
    ap.add_argument("--resume-from", default=None,
                    help="seed already-done problems from an external budget-state JSON's "
                         "{budgets:{pid:B}} (e.g. budget_state_done_so_far.json): those pids are SKIPPED "
                         "(not re-run) and carried into the output. Complements the per-problem cache; "
                         "cache entries take precedence on overlap.")
    ap.add_argument("--resume-from-rollouts", default=None,
                    help="seed already-done problems by RE-DERIVING their budget from a rollouts dump dir "
                         "(<pid>.jsonl from --dump-rollouts): re-aggregates each problem's rollouts with "
                         "the CURRENT --rank/--clamp-max, skips them, writes sticky cache files. Recovers "
                         "done problems from transcripts when the cache is gone / was cleared.")
    ap.add_argument("--dump-rollouts", default=None,
                    help="dir to dump EVERY rollout's full transcript (one <pid>.jsonl per problem, N "
                         "lines: outcome + the whole conversation incl. injected hints). Only NEWLY-run "
                         "problems are dumped (cached/resumed ones are skipped). Large: ~N*problems rollouts.")
    ap.add_argument("--stop-at-correct", type=int, default=None,
                    help="EARLY-STOP a problem as soon as it has this many CORRECT rollouts (default = "
                         "--rank). 0 = disabled (always run all --n). Big speedup on easy problems; B_q "
                         "then = the largest hint count among those correct solves (more conservative).")
    ap.add_argument("--dry-run", action="store_true", help="build the first prompt only; no model calls")
    # policy endpoint
    ap.add_argument("--policy-base-url", default=os.environ.get("POLICY_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--policy-model", default=os.environ.get("POLICY_MODEL", "Qwen2.5-7B-Instruct"))
    ap.add_argument("--policy-api-key", default=os.environ.get("POLICY_API_KEY", "EMPTY"))
    ap.add_argument("--policy-temperature", type=float, default=float(os.environ.get("POLICY_TEMPERATURE", "1.0")))
    ap.add_argument("--policy-top-p", type=float, default=float(os.environ.get("POLICY_TOP_P", "1.0")))
    ap.add_argument("--policy-max-tokens", type=int, default=int(os.environ.get("POLICY_MAX_TOKENS", "4096")))
    # selector endpoint (same env as training)
    ap.add_argument("--selector-base-urls",
                    default=os.environ.get("SELECTOR_BASE_URLS") or os.environ.get("SELECTOR_BASE_URL", "http://localhost:30000/v1"))
    ap.add_argument("--selector-model", default=os.environ.get("SELECTOR_MODEL", "gpt-oss-20b"))
    ap.add_argument("--selector-api-key", default=os.environ.get("SELECTOR_API_KEY", "EMPTY"))
    ap.add_argument("--selector-temperature", type=float, default=float(os.environ.get("SELECTOR_TEMPERATURE", "0.3")))
    ap.add_argument("--selector-top-p", type=float, default=float(os.environ.get("SELECTOR_TOP_P", "1.0")))
    ap.add_argument("--selector-max-tokens", type=int, default=int(os.environ.get("SELECTOR_MAX_TOKENS", "16000")))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_turns is None:
        args.max_turns = args.budget + 2
    if args.clamp_max is None:
        args.clamp_max = args.budget
    if args.stop_at_correct is None:
        args.stop_at_correct = args.rank
    if args.workers is not None:        # deprecated alias
        args.problem_workers = args.workers

    problems = load_problems(Path(args.data), args.limit)
    print(f"data       = {args.data}  ({len(problems)} problems)")
    print(f"probe      = budget {args.budget}, {args.n} rollouts/problem, B_q = {args.rank}-th smallest hints-used "
          f"(clamp <= {args.clamp_max}), max_turns {args.max_turns}")
    print(f"policy     = {args.policy_base_url}  model={args.policy_model} temp={args.policy_temperature}")
    print(f"selector   = {args.selector_base_urls}  model={args.selector_model} temp={args.selector_temperature}")
    if not problems:
        print("no problems; exiting."); return 0

    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(str(out_path) + ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # resume / aggregate: load finished problems' cached results.
    results: dict[str, dict] = {}
    for p in problems:
        cf = cache_dir / f"{safe(p['problem_id'])}.json"
        if cf.exists():
            try:
                results[p["problem_id"]] = json.loads(cf.read_text())
            except Exception:  # noqa: BLE001
                pass

    # --resume-from: seed already-done problems from an external budget-state JSON's
    # {budgets:{pid:B}} (e.g. budget_state_done_so_far.json). Those pids are skipped (not
    # re-run) and carried into the output; a minimal cache file is written so the resume is
    # sticky for later runs. Cache entries loaded above (full recs) take precedence.
    if args.resume_from:
        try:
            seed = json.loads(Path(args.resume_from).read_text()).get("budgets", {}) or {}
        except Exception as e:  # noqa: BLE001
            print(f"--resume-from: could not read {args.resume_from}: {e}"); return 1
        pid_in_data = {p["problem_id"] for p in problems}
        n_seed = 0
        for pid, b in seed.items():
            pid = str(pid)
            if pid in pid_in_data and pid not in results:
                rec = {"problem_id": pid, "budget": int(b), "n_solved": None,
                       "hint_useds": None, "stops": None, "resumed_from": str(args.resume_from)}
                results[pid] = rec
                (cache_dir / f"{safe(pid)}.json").write_text(json.dumps(rec))  # sticky
                n_seed += 1
        print(f"resume-from = seeded {n_seed} new problems from {args.resume_from} "
              f"({len(seed)} budgets in file)")

    # --resume-from-rollouts: re-derive already-done problems from a rollouts dump dir
    # (<pid>.jsonl from --dump-rollouts). Re-aggregates each problem's rollouts with the
    # CURRENT --rank/--clamp-max (so a full-32 dump yields the same budget the cache would),
    # skips them, and writes sticky cache files. Cache recs loaded above win on overlap.
    if args.resume_from_rollouts:
        rdir = Path(args.resume_from_rollouts)
        safe_to_pid = {safe(p["problem_id"]): p["problem_id"] for p in problems}
        n_seed = 0
        for f in sorted(rdir.glob("*.jsonl")):
            pid = safe_to_pid.get(f.stem)
            if pid is None or pid in results:
                continue
            try:
                rolls = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            except Exception:  # noqa: BLE001
                continue
            if not rolls:
                continue
            hint_useds = [int(r["hint_used"]) for r in rolls if r.get("hint_used") is not None]
            if not hint_useds:
                continue
            rec = {
                "problem_id": pid,
                "budget": int(min(args.clamp_max, kth_min(hint_useds, args.rank))),
                "n_solved": sum(1 for r in rolls if r.get("correct")),
                "n_rollouts": len(rolls),
                "hint_useds": sorted(hint_useds),
                "stops": {s: sum(1 for r in rolls if r.get("stop") == s)
                          for s in sorted({r.get("stop") for r in rolls})},
                "resumed_from_rollouts": str(rdir),
            }
            results[pid] = rec
            (cache_dir / f"{safe(pid)}.json").write_text(json.dumps(rec))
            n_seed += 1
        print(f"resume-from-rollouts = re-derived {n_seed} problems from {rdir}")

    # --aggregate-only: write the budget JSON from whatever is ALREADY cached, then exit
    # (no model calls) -- this is how you extract the already-done problems mid-run.
    if args.aggregate_only:
        if not results:
            print(f"no cached results under {cache_dir}; nothing to aggregate."); return 1
        pl = build_payload(results, args)
        write_payload(pl, out_path)
        print(f"aggregated {len(results)} cached problems -> {out_path}")
        print(f"budget histogram: {pl['meta']['budget_histogram']}  "
              f"mean solved/{args.n}: {pl['meta']['mean_solved_per_problem']}")
        return 0

    if args.dry_run:
        r = problems[0]
        prompt = build_prompt_multi(r["problem"], "(student trace would go here)",
                                    render_hints_with_status(r["pool"], []))
        print(f"\n[dry-run] problem_id={r['problem_id']} gt={r['gt']}")
        print(f"[dry-run] policy messages: system({len(r['system'])} chars) + user({len(r['user'])} chars)")
        print(f"[dry-run] selector prompt: {len(prompt)} chars; first pending hints "
              f"{pending_hint_ids(r['pool'], [])[:5]}")
        print(f"[dry-run] hint user-turn sample:\n{format_hint('(example hint text)')}")
        print("\n[dry-run] OK -- no model calls made.")
        return 0

    dump_dir = Path(args.dump_rollouts) if args.dump_rollouts else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    # The actual rollout run is async (true in-flight cancellation on early-stop).
    return asyncio.run(run_calibration(args, problems, results, out_path, cache_dir, dump_dir))


if __name__ == "__main__":
    raise SystemExit(main())
