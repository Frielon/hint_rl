#!/usr/bin/env python
"""Dump residual failures (or any verdict) from a prompt-eval jsonl for reading.

Usage:
  python inspect_eval_failures.py prompt_eval/round1/v2_final_gate.jsonl            # fail_last rows
  python inspect_eval_failures.py prompt_eval/round1/baseline.jsonl --verdict other --max 5
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("--verdict", default="fail_last")
ap.add_argument("--max", type=int, default=10)
ap.add_argument("--full-content", action="store_true", help="print the whole content, not just the selection reasoning")
args = ap.parse_args()

shown = 0
for line in open(args.jsonl):
    d = json.loads(line)
    if d.get("verdict") != args.verdict:
        continue
    shown += 1
    print("=" * 100)
    print(f"problem_id={d['problem_id']} call={d['call_index']} choices={d['n_choices_filtered']} "
          f"picked={d['picked_step_id']} expected={d['expected_step_id']} last={d['last_step_id']} "
          f"conf={d.get('confidence_of_major_step')} toks={d.get('completion_tokens')}")
    if args.full_content:
        print(d.get("content") or "")
    else:
        print("--- reasoning_of_major_step:")
        print((d.get("reasoning_of_major_step") or "(none)")[:1500])
    if shown >= args.max:
        break
print(f"\n[{shown} '{args.verdict}' rows shown from {args.jsonl}]")
