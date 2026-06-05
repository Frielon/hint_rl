#!/usr/bin/env python3
"""Read step1 (reference_decomposition) files from the debug dir and print a
simplified JSON keeping only major_steps with selected fields.

For each major step keep: step_id, title, purpose, core_conclusion, substeps.
For each substep keep: substep_id, operation.
"""
import argparse
import glob
import json
import os

DEBUG_DIR = "/share5/users/xutao.ma/project/hint_rl/data_pipeline/hint_step_outputs/debug"


def simplify(data):
    major_steps = data.get("result", {}).get("major_steps", [])
    simplified = []
    for step in major_steps:
        simplified.append({
            "step_id": step.get("step_id"),
            "purpose": step.get("title")+': '+step.get("core_conclusion"),
            "hints": [{
                "type": "step_guidence_hint",
                "hint_id": str(step.get("step_id"))+'.0',
                "hint": step.get("purpose"),
            }] + 
            [
                {
                    "type": "substep_hint",
                    "hint_id": sub.get("substep_id"),
                    "hint": sub.get("operation"),
                }
                for sub in step.get("substeps", [])
            ],
        })
    return {"steps": simplified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", default=DEBUG_DIR,
                        help="Root debug directory containing per-problem subdirs")
    parser.add_argument("--problem", default='DAPO-Math-17k-Processed_filtered-request-1-62',
                        help="Only process this problem subdir name (optional)")
    args = parser.parse_args()

    if args.problem:
        pattern = os.path.join(args.debug_dir, args.problem, "*step1*.json")
    else:
        pattern = os.path.join(args.debug_dir, "*", "*step1*.json")

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No step1 files found under {args.debug_dir}")
        return

    for path in files:
        with open(path) as f:
            data = json.load(f)
        problem = os.path.basename(os.path.dirname(path))
        print(f"===== {problem} =====")
        print(json.dumps(simplify(data), indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
