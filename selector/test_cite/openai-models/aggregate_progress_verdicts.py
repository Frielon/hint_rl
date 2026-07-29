#!/usr/bin/env python3
"""Aggregate judge verdicts for the ``progress``-rephrase review.

Reads <check-dir>/review_cases.jsonl and every <check-dir>/verdicts/*.jsonl
(one verdict object per reviewed completed_hints entry: {problem_id, hint_id,
verdict: pass|minor|fail, issues: [..], note}), joins them, and writes
<check-dir>/report.md with per-model / per-verdict counts and every non-pass
case spelled out. Prints the summary table.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(p: Path):
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("check_dir")
    args = ap.parse_args()
    cd = Path(args.check_dir)

    cases = {c["problem_id"]: c for c in load_jsonl(cd / "review_cases.jsonl")}
    verdicts = []
    for vf in sorted((cd / "verdicts").glob("*.jsonl")):
        verdicts.extend(load_jsonl(vf))

    by_key = {}
    for v in verdicts:
        by_key[(str(v.get("problem_id")), str(v.get("hint_id")))] = v

    expected = [(pid, str(e.get("hint_id"))) for pid, c in cases.items() for e in c["entries"]]
    missing = [k for k in expected if k not in by_key]

    rows = []
    for (pid, hid), v in by_key.items():
        c = cases.get(pid) or {}
        rows.append({
            "problem_id": pid, "hint_id": hid, "model": c.get("model"),
            "verdict": str(v.get("verdict") or "").lower(),
            "issues": v.get("issues") or [], "note": v.get("note") or "",
        })

    n = len(rows)
    vc = Counter(r["verdict"] for r in rows)
    ic = Counter(i for r in rows for i in r["issues"])
    per_model = defaultdict(Counter)
    for r in rows:
        per_model[r["model"]][r["verdict"]] += 1

    lines = ["# progress-rephrase review report", "",
             f"entries judged: {n}   (missing verdicts: {len(missing)})", "",
             "| verdict | count | share |", "|---|---|---|"]
    for k in ("pass", "minor", "fail"):
        lines.append(f"| {k} | {vc.get(k, 0)} | {vc.get(k, 0) / n:.1%} |" if n else f"| {k} | 0 | - |")
    lines += ["", "| model | pass | minor | fail |", "|---|---|---|---|"]
    for m, c in sorted(per_model.items()):
        lines.append(f"| {m} | {c.get('pass', 0)} | {c.get('minor', 0)} | {c.get('fail', 0)} |")
    if ic:
        lines += ["", "issue tags: " + ", ".join(f"{k}={v}" for k, v in ic.most_common())]

    bad = [r for r in rows if r["verdict"] != "pass"]
    if bad:
        lines += ["", "## non-pass cases", ""]
        for r in sorted(bad, key=lambda r: (r["verdict"] != "fail", r["problem_id"])):
            c = cases.get(r["problem_id"]) or {}
            e = next((e for e in c.get("entries", []) if str(e.get("hint_id")) == r["hint_id"]), {})
            lines += [f"### {r['verdict'].upper()}  {r['problem_id']}  hint {r['hint_id']}  ({r['model']})",
                      f"- issues : {', '.join(r['issues']) or '-'}",
                      f"- note   : {r['note']}",
                      f"- hint   : {e.get('pool_hint')}",
                      f"- progress: {e.get('progress')}",
                      f"- quote  : {e.get('quote')}", ""]

    (cd / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    if missing:
        print(f"\nWARN: {len(missing)} entries have no verdict, e.g. {missing[:5]}")
    print(f"\nreport -> {cd / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
