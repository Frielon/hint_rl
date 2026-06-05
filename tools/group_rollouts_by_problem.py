#!/usr/bin/env python3
"""Group rollouts by problem.

Each rollout jsonl file (e.g. logs/<exp_name>/rollouts/<step>.jsonl) contains
many rollouts. The same problem appears multiple times (e.g. 16 rollouts per
problem in a GRPO group) but the lines are interleaved, not consecutive.

A problem is identified by its `input` field (the full prompt). This script adds
two fields to every rollout so they can be grouped:

  - problem_id : a stable 12-hex-char hash of `input` (identical across files /
                 experiments for the same prompt).
  - pid        : a small integer assigned per-experiment (sorted by problem_id),
                 handy for readable grouping/plots within one run.

By default the augmented files are written next to the originals with a
`_grouped` suffix on the subfolder (rollouts -> rollouts_grouped). Use
--inplace to overwrite the originals instead.

Usage:
  python tools/group_rollouts_by_problem.py logs/<exp_name>
  python tools/group_rollouts_by_problem.py logs/<exp_name> --subdirs rollouts val_rollouts
  python tools/group_rollouts_by_problem.py logs/<exp_name> --inplace
"""
import argparse
import hashlib
import json
import os
import sys


def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def iter_jsonl_files(subdir: str):
    for name in sorted(os.listdir(subdir)):
        if name.endswith(".jsonl"):
            yield os.path.join(subdir, name)


def collect_problem_ids(subdir: str):
    """First pass: gather every distinct input -> stable hash across all files."""
    hashes = set()
    for path in iter_jsonl_files(subdir):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                hashes.add(problem_hash(d["input"]))
    # deterministic small integer per problem, sorted by hash for stability
    return {h: i for i, h in enumerate(sorted(hashes))}


def process_subdir(subdir: str, inplace: bool):
    if not os.path.isdir(subdir):
        print(f"  skip (not found): {subdir}")
        return

    pid_map = collect_problem_ids(subdir)
    out_dir = subdir if inplace else subdir.rstrip("/") + "_grouped"
    os.makedirs(out_dir, exist_ok=True)

    n_lines = 0
    for path in iter_jsonl_files(subdir):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid_hash = problem_hash(d["input"])
                d["problem_id"] = pid_hash
                d["pid"] = pid_map[pid_hash]
                rows.append(d)
                n_lines += 1

        out_path = os.path.join(out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(
        f"  {subdir}: {len(pid_map)} distinct problems, {n_lines} rollouts -> {out_dir}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="experiment folder, e.g. logs/<exp_name>")
    ap.add_argument("--subdirs", nargs="+", default=["rollouts", "val_rollouts"],
                    help="rollout subfolders to process (default: rollouts val_rollouts)")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite original files instead of writing *_grouped folders")
    args = ap.parse_args()

    if not os.path.isdir(args.exp_dir):
        sys.exit(f"error: not a directory: {args.exp_dir}")

    print(f"experiment: {args.exp_dir}")
    for sub in args.subdirs:
        process_subdir(os.path.join(args.exp_dir, sub), args.inplace)


if __name__ == "__main__":
    main()
