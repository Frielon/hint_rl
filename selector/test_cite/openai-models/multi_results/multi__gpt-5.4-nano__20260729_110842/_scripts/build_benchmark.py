#!/usr/bin/env python3
"""Build the multi-round hint-selection benchmark from the Codex label run.

For each label row whose reference selection ``H`` is a hint > 1.1, we simulate
an intermediate state of a multi-round tutoring conversation:

  * the hints positioned before ``H`` are the achieved ones (per the reference
    procedure, ``H`` is the first *un*achieved hint); they are all substeps since
    the pool is pruned of ``x.0`` step-guidance hints;
  * pick a RANDOM cutoff ``C`` among them and mark every hint ``<= C`` as
    ``completed`` (as if given in earlier rounds); the rest are ``pending``;
  * the ground-truth next selection is still ``H`` -- the trace is unchanged, so
    the achieved-but-unmarked hints in ``(C, H)`` must be recognized from the
    trace and ``H`` selected. (Scored with the x.0==x.1 merge downstream.)

Output: one JSON object per line in ``benchmark.jsonl``. Reproducible: the random
cutoff is seeded per-row from (``--seed``, uid).

Usage:
    python build_benchmark.py                       # -> ./benchmark.jsonl
    python build_benchmark.py --steps 2,3 --limit 5
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent              # .../gpt_oss_eval/multi-cite-gpt-eval
GPT_OSS_EVAL = HERE.parent                            # .../gpt_oss_eval
TEST_CITE = GPT_OSS_EVAL.parent                       # .../test_cite
SELECTOR = TEST_CITE.parent                           # .../hint_rl/selector
for p in (str(SELECTOR), str(GPT_OSS_EVAL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# reuse the pruning + trace-recovery from the single-round driver
from run_gpt_oss_selection import prune_hint_pool, recover_problem_trace  # noqa: E402

RESULTS_ROOT = TEST_CITE / "results"


def resolve_label_dir(label_run: str) -> Path:
    p = Path(label_run)
    if p.is_dir():
        return p.resolve()
    for base in (RESULTS_ROOT / "debug", RESULTS_ROOT / "runs", RESULTS_ROOT):
        if (base / label_run).is_dir():
            return (base / label_run).resolve()
    raise SystemExit(f"label run not found: {label_run}")


def pos(hid: Any) -> Optional[tuple[int, int]]:
    if hid is None:
        return None
    maj, _, mn = str(hid).partition(".")
    try:
        return (int(maj), int(mn))
    except ValueError:
        return None


def rng_for(uid: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def problem_trace(rec: dict) -> tuple[Optional[str], Optional[str]]:
    problem, trace = rec.get("problem"), rec.get("reasoning_trace")
    if problem is None or trace is None:
        rp, rt = recover_problem_trace(rec.get("prompt"))
        problem = problem if problem is not None else rp
        trace = trace if trace is not None else rt
    return problem, trace


def ordered_hints(pruned_pool: dict) -> list[tuple[tuple[int, int], str]]:
    out = []
    for st in (pruned_pool or {}).get("steps", []):
        for h in st.get("hints", []):
            p = pos(h.get("hint_id"))
            if p is not None:
                out.append((p, str(h.get("hint_id"))))
    out.sort(key=lambda x: x[0])
    return out


def build_sample(rec: dict, seed: int) -> Optional[dict]:
    ref = rec.get("parsed_answer") or {}
    H = ref.get("hint_id")
    pH = pos(H)
    if pH is None or not (pH > (1, 1)):          # only ref hints strictly > 1.1
        return None
    pool = prune_hint_pool(rec.get("hint_pool"))
    if not isinstance(pool, dict):
        return None
    ordered = ordered_hints(pool)
    candidates = [hid for p, hid in ordered if p < pH]    # achieved hints before H
    if not candidates:
        return None
    problem, trace = problem_trace(rec)
    if problem is None or trace is None:
        return None

    uid = str(rec.get("uid") or rec.get("request_id"))
    C = rng_for(uid, seed).choice(candidates)
    pC = pos(C)
    completed = [hid for p, hid in ordered if p <= pC]
    pending = [hid for p, hid in ordered if p > pC]
    newly = [hid for p, hid in ordered if pC < p < pH]    # achieved but not yet marked

    ref_quotes = {}
    for e in (ref.get("completed_substeps") or ref.get("completed_hints") or []):
        if isinstance(e, dict) and e.get("hint_id") is not None:
            ref_quotes[str(e["hint_id"])] = e.get("quote")

    return {
        "uid": uid,
        "problem_id": rec.get("problem_id"),
        "request_id": rec.get("request_id"),
        "step": rec.get("step"),
        "problem": problem,
        "reasoning_trace": trace,
        "hint_pool": pool,                       # pruned (no x.0, no type)
        "ref_hint_id": str(H),                   # ground-truth next selection
        "cut_hint_id": C,                        # random completed cutoff
        "completed_status": completed,           # marked completed (given earlier)
        "pending": pending,                      # candidates for selection
        "expected_newly_completed": newly,       # achieved-but-unmarked -> model should cite
        "ref_quotes": ref_quotes,                # reference quotes (for analysis)
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-run", default="run_20260617_224209")
    ap.add_argument("--out", default=str(HERE / "benchmark.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default=None, help="comma list of steps to include")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per step")
    args = ap.parse_args()

    label_dir = resolve_label_dir(args.label_run)
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    files = sorted(glob.glob(str(label_dir / "step*" / "*" / "*.json")))

    from collections import Counter
    per_step: Counter = Counter()
    n_seen = n_qualified = 0
    out_samples: list[dict] = []
    cut_dist: Counter = Counter()
    for fp in files:
        try:
            rec = json.loads(Path(fp).read_text())
        except Exception:  # noqa: BLE001
            continue
        step = rec.get("step")
        if steps is not None and step not in steps:
            continue
        if args.limit is not None and per_step[step] >= args.limit:
            continue
        n_seen += 1
        per_step[step] += 1
        s = build_sample(rec, args.seed)
        if s is None:
            continue
        n_qualified += 1
        cut_dist[s["cut_hint_id"]] += 1
        out_samples.append(s)

    out_path = Path(args.out)
    out_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out_samples))

    n_with_newly = sum(1 for s in out_samples if s["expected_newly_completed"])
    print(f"label_dir   : {label_dir}")
    print(f"rows seen   : {n_seen}")
    print(f"benchmark   : {n_qualified} samples (ref hint > 1.1 with >=1 prior hint)  -> {out_path}")
    print(f"  with achieved-but-unmarked hints in (C,H): {n_with_newly}"
          f"  (the rest: cutoff == H's predecessor, pure selection)")
    print(f"  cut_hint (C) distribution: {dict(sorted(cut_dist.items()))}")
    by_step = Counter(s["step"] for s in out_samples)
    print(f"  by step: {dict(sorted(by_step.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
