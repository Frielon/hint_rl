#!/usr/bin/env python3
# Copyright 2026
#
# Extract the problem_ids the policy solves WITHOUT hints, from a run's rollout
# dumps. These "easy" problems are where the HPRL hint penalty is counter-
# productive: a group that contains an unaided-correct rollout (reward 1.1)
# out-ranks every hinted-correct rollout (reward 1.1 - penalty), so GRPO trains
# the <hint_call/> action AWAY globally (see the hint-suppression analysis). The
# fix is the hard-problem curriculum: feed this list to prepare_hint_data.py
# (--zero-budget-ids) so these problems get initial budget 0 (no hints), leaving
# the hint mechanism to act only where the model actually needs it.
#
# A problem counts as "solved unaided" if, across ALL steps it appeared in, it
# has at least --min-unaided-correct rollouts with num_hints==0 AND acc>0.5.
#
# Rollout records carry no problem_id, so we recover it by matching the rollout's
# user-problem text (verbatim after the last "\nuser\n") to the dataset's
# extra_info.tools_kwargs.request_hint.create_kwargs.problem.
#
# Usage:
#   python extract_unaided_solved.py --run-dir <logs/HPRL-...> \
#       [--dataset dataset/...-mt.parquet] [--min-unaided-correct 1] \
#       [--out dataset/unaided_solved_ids.txt]

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_DATASET = os.path.join(HINT_RL_HOME, "dataset", "dapo-3740-hint-verl-simplified-mt.parquet")
DEFAULT_OUT = os.path.join(HINT_RL_HOME, "dataset", "unaided_solved_ids.txt")


def build_problem_index(dataset_path: str) -> dict[str, list[str]]:
    """problem-text -> [problem_id, ...] (a few texts repeat across pids)."""
    df = pd.read_parquet(dataset_path)
    prob2pids: dict[str, list[str]] = defaultdict(list)
    for e in df["extra_info"]:
        prob = e["tools_kwargs"]["request_hint"]["create_kwargs"]["problem"]
        prob2pids[prob.strip()].append(e.get("problem_id"))
    return prob2pids


def extract_problem(inp: str):
    """Recover the user problem statement from a rollout's decoded `input`."""
    i = inp.rfind("\nuser\n")
    if i < 0:
        return None
    seg = inp[i + len("\nuser\n"):]
    j = seg.rfind("\nassistant")
    if j >= 0:
        seg = seg[:j]
    return seg.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="experiment log dir containing rollouts/*.jsonl")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="(-mt) parquet for problem-text -> problem_id")
    ap.add_argument("--min-unaided-correct", type=int, default=1,
                    help="a problem is 'solved unaided' if it has >= this many num_hints==0 & correct rollouts")
    ap.add_argument("--out", default=DEFAULT_OUT, help="write the solved problem_ids here (one per line)")
    args = ap.parse_args()

    prob2pids = build_problem_index(args.dataset)
    all_pids = {pid for pids in prob2pids.values() for pid in pids}
    print(f"dataset: {len(prob2pids)} unique problem texts -> {len(all_pids)} problem_ids")

    files = sorted(glob.glob(os.path.join(args.run_dir, "rollouts", "*.jsonl")),
                   key=lambda f: int(os.path.basename(f)[:-6]))
    if not files:
        raise SystemExit(f"no rollouts/*.jsonl under {args.run_dir}")

    unaided_correct = defaultdict(int)   # pid -> count of unaided-correct rollouts
    seen_pids = set()
    n_rollouts = n_unmatched = 0
    for fn in files:
        for line in open(fn):
            r = json.loads(line)
            n_rollouts += 1
            prob = extract_problem(r.get("input", "") or "")
            pids = prob2pids.get(prob) if prob is not None else None
            if not pids:
                n_unmatched += 1
                continue
            for pid in pids:
                seen_pids.add(pid)
            if int(r.get("num_hints", 0)) == 0 and r.get("acc", 0) > 0.5:
                for pid in pids:
                    unaided_correct[pid] += 1

    solved = sorted(pid for pid, c in unaided_correct.items() if c >= args.min_unaided_correct)

    print(f"scanned {n_rollouts} rollouts across {len(files)} steps ({n_unmatched} unmatched)")
    print(f"coverage: {len(seen_pids)}/{len(all_pids)} problems appeared in rollouts")
    # distribution at a few thresholds, so you can pick min-unaided-correct sensibly
    for thr in (1, 2, 3, 5):
        n = sum(1 for c in unaided_correct.values() if c >= thr)
        print(f"  problems with >= {thr} unaided-correct rollout(s): {n}")
    print(f"=> writing {len(solved)} problem_ids (>= {args.min_unaided_correct} unaided-correct) to {args.out}")

    with open(args.out, "w") as f:
        for pid in solved:
            f.write(f"{pid}\n")


if __name__ == "__main__":
    main()
