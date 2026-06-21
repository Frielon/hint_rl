#!/usr/bin/env python
"""Validate v5_substep_cite's `completed_substeps` citations and simulate
enforcement, at SUBSTEP granularity (cf. analyze_citations.py for v4's per-step).

v5 forces a verbatim student quote for EVERY substep_hint (X.1, X.2, ...) that
precedes the selected hint_id -- across all earlier candidate major steps and the
earlier substeps of the selected step. Each quote is classified against the
trace the selector saw:

  valid_student : normalized quote is a substring of the STUDENT-ONLY text
                  (trace minus `[hint given]` paragraphs)
  hint_only     : found in the full trace but only inside hint text
  fabricated    : not found in the trace at all

Enforcement simulation: every substep preceding the selected hint_id must carry a
valid_student citation; otherwise clamp the pick to the MAJOR step of the
earliest such uncited substep. Reports fail_last before vs after enforcement.

Usage:
  python analyze_substep_citations.py --tag round4_substep --n 240
  python analyze_substep_citations.py --tag round4_substep_lowtemp --n 120
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_selector_prompts import load_sample  # noqa: E402
from run_hint_selection_model import parse_output  # noqa: E402
from analyze_citations import norm, student_only, classify_quote  # noqa: E402


def substep_ids(hints_filtered):
    """Ordered list of (major_id_str, hint_id_str) for every substep_hint."""
    out = []
    for s in json.loads(hints_filtered).get("steps", []):
        mid = str(s.get("step_id"))
        for h in s.get("hints", []):
            if h.get("type") == "substep_hint":
                out.append((mid, str(h.get("hint_id"))))
    return out


def _key(hid):
    """(major:int, sub:int) sort key for a hint_id like '3.1'; X.0 sorts first."""
    try:
        a, b = str(hid).split(".")
        return (int(a), int(b))
    except Exception:
        return (10**9, 10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v5_substep_cite")
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
        last = str(row.last_step_id)
        expected = str(row.expected_step_id)
        pick_major = str(rec.get("picked_step_id"))
        pick_hid = str(sel.get("hint_id"))

        subs = substep_ids(row.hints_filtered)             # ordered (major, hid)
        # substeps strictly before the selected hint_id
        pk = _key(pick_hid)
        earlier = [(m, h) for (m, h) in subs if _key(h) < pk]
        earlier_majors = [m for (m, h) in earlier]

        cited = sel.get("completed_substeps") or []
        if not isinstance(cited, list):
            cited = []
        verdicts = {}  # hint_id -> best verdict among its quotes
        rolls_fab = False
        for entry in cited:
            if not isinstance(entry, dict):
                continue
            hid = str(entry.get("hint_id"))
            v = classify_quote(entry.get("quote") or "", trace, stud)
            c["quotes"] += 1
            c["q_" + {"valid_student": "valid", "hint_only": "hint", "fabricated": "fab"}[v]] += 1
            if v == "fabricated":
                rolls_fab = True
                if len(examples_fab) < 5:
                    examples_fab.append((rec["dup_key"], hid, (entry.get("quote") or "")[:120]))
            order = {"valid_student": 2, "hint_only": 1, "fabricated": 0}
            if order[v] > order.get(verdicts.get(hid, "fabricated"), -1) or hid not in verdicts:
                verdicts[hid] = v
        c["rolls_with_fab"] += rolls_fab

        # enforcement: earliest earlier-substep WITHOUT a valid_student citation
        # -> clamp the pick to that substep's MAJOR step.
        enforced = pick_major
        missing = [(m, h) for (m, h) in earlier if verdicts.get(h) != "valid_student"]
        if earlier and missing:
            enforced = missing[0][0]
            c["rolls_missing_cites"] += 1
        if earlier:
            c["skip_picks"] += 1
            if not missing:
                c["skip_fully_cited"] += 1

        c["fail"] += pick_major == last
        c["enf_fail"] += enforced == last
        c["pick_expected"] += pick_major == expected
        c["enf_expected"] += enforced == expected

    s = c["scoreable"]
    print(f"=== {args.tag}/{args.variant} (n={c['n']}, scoreable={s}) ===")
    print(f"fail_last:            {c['fail']}/{s} ({c['fail']/max(s,1):.3f})")
    print(f"fail_last ENFORCED:   {c['enf_fail']}/{s} ({c['enf_fail']/max(s,1):.3f})")
    print(f"pick_expected:        {c['pick_expected']/max(s,1):.3f} -> enforced {c['enf_expected']/max(s,1):.3f}")
    print(f"skip-ahead picks:     {c['skip_picks']} (fully cited+valid: {c['skip_fully_cited']})")
    print(f"substep quotes: {c['quotes']}  valid_student={c['q_valid']} ({c['q_valid']/max(c['quotes'],1):.3f}) "
          f"hint_only={c['q_hint']} fabricated={c['q_fab']}")
    print(f"rollouts with >=1 fabricated quote: {c['rolls_with_fab']}/{s}")
    print(f"rollouts clamped by enforcement:    {c['rolls_missing_cites']}/{s}")
    if examples_fab:
        print("\nexample fabricated quotes:")
        for k, hid, q in examples_fab:
            print(f"  [{k} substep {hid}] {q!r}")


if __name__ == "__main__":
    main()
