#!/usr/bin/env python3
"""Post-run checker for the per-completed-hint ``progress`` prompt variant
(prompt_template_multiF_progress.py).

Two jobs:

1. MECHANICAL stats over every saved sample of the given multi_results runs:
   parse rate, how many samples emit completed_hints, and -- the new bit --
   whether every completed_hints entry carries a non-empty ``progress`` string
   (per model). Runs fine on a PARTIAL result dir (files land per row), so it
   doubles as a mid-run schema-compliance probe.

2. REVIEW sampling: for each benchmark problem_id, pick ONE random sample
   (across the given runs) whose selection has >=1 completed_hints entry, and
   emit a self-contained review case per entry: the original pool hint text,
   the model's quote/why/progress, and a fuzzy-located trace window around the
   quote (locator = script/hint_rl/selector_multi.locate_quote_end, the same
   one the training loop uses). These cases are what a human/LLM judge reads to
   decide whether the progress rephrase is faithful + trace-adapted.

Usage:
    python build_progress_review.py --run-dir multi_results/multi__X__ts [--run-dir ...]
        [--out-dir progress_check/<stamp>] [--seed 0] [--stats-only]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "gpt_oss_eval" / "multi-cite-gpt-eval" / "benchmark.jsonl"
HINT_RL_SCRIPT = HERE.parent.parent.parent / "script" / "hint_rl"
sys.path.insert(0, str(HINT_RL_SCRIPT))

from selector_multi import locate_quote_end, pool_hint_texts  # noqa: E402


def load_benchmark() -> dict[str, dict]:
    rows = [json.loads(l) for l in BENCHMARK.read_text().splitlines() if l.strip()]
    return {str(r["request_id"]): r for r in rows}


def iter_records(run_dir: Path):
    for p in sorted(run_dir.glob("step*/*/*.json")):
        try:
            yield json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- row file mid-write during a live run
            continue


def completed_entries(sel) -> list[dict]:
    if not isinstance(sel, dict):
        return []
    cs = sel.get("completed_hints")
    return [e for e in cs if isinstance(e, dict)] if isinstance(cs, list) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--chunks", type=int, default=8, help="review chunk files for judge fan-out")
    ap.add_argument("--window", type=int, default=700, help="trace chars before the quote end")
    args = ap.parse_args()

    bench = load_benchmark()
    stats: dict[str, dict] = {}
    # candidates[problem_id] = list of (model, record, sample)
    candidates: dict[str, list] = defaultdict(list)
    seen_problems: set[str] = set()

    for rd in args.run_dir:
        rd = Path(rd)
        model = rd.name.split("__")[1] if "__" in rd.name else rd.name
        st = stats.setdefault(model, {
            "run_dir": str(rd), "rows": 0, "samples": 0, "api_errors": 0, "parsed": 0,
            "with_completed": 0, "entries": 0, "entries_with_progress": 0,
            "entries_missing_progress": 0, "samples_all_entries_ok": 0,
            "progress_chars": 0,
        })
        for rec in iter_records(rd):
            st["rows"] += 1
            pid = str(rec.get("problem_id"))
            seen_problems.add(pid)
            for s in rec.get("samples") or []:
                if not isinstance(s, dict):
                    continue
                st["samples"] += 1
                if str(s.get("parse_error") or "").startswith("api_error"):
                    st["api_errors"] += 1
                sel = s.get("selection")
                if sel is None:
                    continue
                st["parsed"] += 1
                ents = completed_entries(sel)
                if not ents:
                    continue
                st["with_completed"] += 1
                ok = True
                for e in ents:
                    st["entries"] += 1
                    prog = e.get("progress")
                    if isinstance(prog, str) and prog.strip():
                        st["entries_with_progress"] += 1
                        st["progress_chars"] += len(prog)
                    else:
                        st["entries_missing_progress"] += 1
                        ok = False
                if ok:
                    st["samples_all_entries_ok"] += 1
                candidates[pid].append((model, rec, s))

    for st in stats.values():
        e = st["entries"]
        st["progress_rate"] = round(st["entries_with_progress"] / e, 4) if e else None
        st["mean_progress_chars"] = round(st["progress_chars"] / st["entries_with_progress"], 1) \
            if st["entries_with_progress"] else None
        del st["progress_chars"]

    print(json.dumps(stats, indent=2))

    if args.stats_only:
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else \
        HERE / "progress_check" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    rng = random.Random(args.seed)
    cases, no_candidate = [], []
    for pid in sorted(seen_problems):
        pool = candidates.get(pid)
        if not pool:
            no_candidate.append(pid)
            continue
        model, rec, s = rng.choice(pool)
        row = bench.get(str(rec.get("request_id"))) or {}
        trace = row.get("reasoning_trace") or ""
        hint_texts = pool_hint_texts(row.get("hint_pool") or {})
        entries = []
        for e in completed_entries(s.get("selection")):
            q = e.get("quote") if isinstance(e.get("quote"), str) else ""
            end = locate_quote_end(q, trace) if q else None
            window = trace[max(0, end - args.window):end] if end is not None else None
            entries.append({
                "hint_id": e.get("hint_id"),
                "pool_hint": hint_texts.get(str(e.get("hint_id"))),
                "quote": q,
                "why": e.get("why"),
                "progress": e.get("progress"),
                "quote_located": end is not None,
                "trace_window": window,
            })
        cases.append({
            "problem_id": pid,
            "model": model,
            "step": rec.get("step"),
            "request_id": rec.get("request_id"),
            "sample_index": s.get("index"),
            "already_completed_ids": rec.get("completed_status"),
            "selected_hint_id": s.get("hint_id"),
            "problem": row.get("problem"),
            "entries": entries,
        })

    (out_dir / "review_cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    (out_dir / "no_candidate_problems.json").write_text(json.dumps(no_candidate, indent=2) + "\n")

    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    n = max(1, args.chunks)
    for i in range(n):
        part = cases[i::n]
        if part:
            (chunk_dir / f"chunk_{i}.jsonl").write_text(
                "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in part))

    print(f"\nreview cases : {len(cases)} problems ({sum(len(c['entries']) for c in cases)} entries)"
          f"\nno-candidate : {len(no_candidate)} problems (no sample ever emitted completed_hints)"
          f"\nout          : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
