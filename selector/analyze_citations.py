#!/usr/bin/env python
"""Validate v4_cite's `completed_steps` citations and simulate enforcement.

For each eval record (prompt_eval/<tag>/v4_cite.jsonl): re-parse the saved
completion, take the cited quotes, and check each against the trace the
selector actually saw (trace_full of the same sampled row):

  valid_student : normalized quote is a substring of the STUDENT-ONLY text
                  (trace minus `[hint given]` paragraphs)
  hint_only     : found in the full trace but only inside hint text
  fabricated    : not found in the trace at all

Enforcement simulation (what a deterministic loop guard would do): for a pick
that skips ahead, every candidate step listed before the pick must carry a
valid_student citation; otherwise clamp the pick to the EARLIEST candidate
lacking one. Reports fail_last before vs after enforcement.

Usage:
  python analyze_citations.py --tag round3_cite --n 240
  python analyze_citations.py --tag round3_cite_lowtemp --n 120
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/share5/users/xutao.ma/project/hint_rl/script/hint_rl")

from eval_selector_prompts import load_sample  # noqa: E402
from run_hint_selection_model import parse_output  # noqa: E402

HINT_PARA = "[hint given]"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def student_only(trace: str) -> str:
    paras = trace.split("\n\n")
    return "\n\n".join(p for p in paras if not p.lstrip().startswith(HINT_PARA))


def classify_quote(quote: str, trace: str, student_text: str) -> str:
    q = norm(quote)
    if len(q) < 5:
        return "fabricated"  # empty/trivial quotes can't justify anything
    if q in norm(student_text):
        return "valid_student"
    if q in norm(trace):
        return "hint_only"
    return "fabricated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v4_cite")
    args = ap.parse_args()

    rows = load_sample(args.n, args.seed)
    by_key = {r.dup_key: r for _, r in rows.iterrows()}

    path = os.path.join(HERE, "prompt_eval", args.tag, f"{args.variant}.jsonl")
    recs = [json.loads(l) for l in open(path)]

    c = dict(n=0, scoreable=0, fail=0, enf_fail=0, pick_expected=0, enf_expected=0,
             skip_picks=0, skip_fully_cited=0,
             quotes=0, q_valid=0, q_hint=0, q_fab=0,
             rolls_with_fab=0, rolls_missing_cites=0)
    examples_fab = []
    for rec in recs:
        c["n"] += 1
        row = by_key.get(rec["dup_key"])
        if row is None or rec.get("verdict") in ("api_error", "parse_fail"):
            continue
        sel, _ = parse_output(rec.get("content") or "")
        if not isinstance(sel, dict):
            continue
        c["scoreable"] += 1
        trace = row.trace_full
        stud = student_only(trace)
        cand_ids = [str(s.get("step_id"))
                    for s in json.loads(row.hints_filtered).get("steps", [])]
        pick = str(rec.get("picked_step_id"))
        last = str(row.last_step_id)
        expected = str(row.expected_step_id)
        earlier = cand_ids[: cand_ids.index(pick)] if pick in cand_ids else []

        cited = sel.get("completed_steps") or []
        if not isinstance(cited, list):
            cited = []
        verdicts = {}  # step_id -> best verdict among its quotes
        rolls_fab = False
        for entry in cited:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("step_id"))
            v = classify_quote(entry.get("quote") or "", trace, stud)
            c["quotes"] += 1
            c["q_" + {"valid_student": "valid", "hint_only": "hint", "fabricated": "fab"}[v]] += 1
            if v == "fabricated":
                rolls_fab = True
                if len(examples_fab) < 5:
                    examples_fab.append((rec["dup_key"], sid, (entry.get("quote") or "")[:120]))
            order = {"valid_student": 2, "hint_only": 1, "fabricated": 0}
            if order[v] > order.get(verdicts.get(sid, "fabricated"), -1) or sid not in verdicts:
                verdicts[sid] = v
        c["rolls_with_fab"] += rolls_fab

        # enforcement: earliest earlier-candidate WITHOUT a valid_student citation
        enforced = pick
        missing = [sid for sid in earlier if verdicts.get(sid) != "valid_student"]
        if earlier and missing:
            enforced = missing[0]
            c["rolls_missing_cites"] += 1
        if earlier:
            c["skip_picks"] += 1
            if not missing:
                c["skip_fully_cited"] += 1

        c["fail"] += pick == last
        c["enf_fail"] += enforced == last
        c["pick_expected"] += pick == expected
        c["enf_expected"] += enforced == expected

    s = c["scoreable"]
    print(f"=== {args.tag}/{args.variant} (n={c['n']}, scoreable={s}) ===")
    print(f"fail_last:            {c['fail']}/{s} ({c['fail']/max(s,1):.3f})")
    print(f"fail_last ENFORCED:   {c['enf_fail']}/{s} ({c['enf_fail']/max(s,1):.3f})")
    print(f"pick_expected:        {c['pick_expected']/max(s,1):.3f} -> enforced {c['enf_expected']/max(s,1):.3f}")
    print(f"skip-ahead picks:     {c['skip_picks']} (fully cited+valid: {c['skip_fully_cited']})")
    print(f"quotes: {c['quotes']}  valid_student={c['q_valid']} ({c['q_valid']/max(c['quotes'],1):.3f}) "
          f"hint_only={c['q_hint']} fabricated={c['q_fab']}")
    print(f"rollouts with >=1 fabricated quote: {c['rolls_with_fab']}/{s}")
    print(f"rollouts clamped by enforcement:    {c['rolls_missing_cites']}/{s}")
    if examples_fab:
        print("\nexample fabricated quotes:")
        for k, sid, q in examples_fab:
            print(f"  [{k} step {sid}] {q!r}")


if __name__ == "__main__":
    main()
