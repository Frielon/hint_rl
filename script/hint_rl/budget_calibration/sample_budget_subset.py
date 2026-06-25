#!/usr/bin/env python3
# Copyright 2026
#
# Sample a fixed-size problem subset from an auto-hint parquet, restricted to the
# problems whose budget-state value falls in a [lo, hi] band, and bake that
# budget-state value as each row's INITIAL budget (so the subset starts the
# curriculum at its banded budget).
#
# Selection: a row is eligible iff its extra_info.problem_id is in the budget-state
# file AND lo <= B_q <= hi. We then sample --n of the eligible rows (seeded) and
# re-init each sampled row's budget to its B_q via apply_budget_state's auto-hint
# helper (writes BOTH budget fields the pipeline reads).
#
# Usage:
#   python sample_budget_subset.py \
#       --in  dataset/dapo-3139-auto-hint-hintwise.parquet \
#       --budget-state logs/.../budget_state.json \
#       --lo 3 --hi 5 --n 512 --seed 42 \
#       --out dataset/dapo-512-auto-hint-b3to5.parquet

from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

# This module now lives in budget_calibration/; put the hint_rl package dir (its
# parent) on sys.path so budget_manager (which stays in the package) still imports.
# apply_budget_state moved alongside this file, so it resolves from this dir.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply_budget_state import _set_auto_hint_budget
from budget_manager import load_budget_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source auto-hint parquet")
    ap.add_argument("--budget-state", required=True,
                    help="budget-state JSON ({problem_id: B_q}, BudgetManager.save format)")
    ap.add_argument("--lo", type=int, default=3, help="min budget (inclusive)")
    ap.add_argument("--hi", type=int, default=5, help="max budget (inclusive)")
    ap.add_argument("--n", type=int, default=512, help="number of problems to sample")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (reproducible)")
    ap.add_argument("--out", default=None, help="output parquet (default derived)")
    args = ap.parse_args()

    table = load_budget_table(args.budget_state)  # {problem_id: B_q}
    df = pd.read_parquet(args.inp)

    def _band(row) -> bool:
        pid = (row.get("problem_id") if isinstance(row, dict) else None)
        if pid is None or pid not in table:
            return False
        return args.lo <= int(table[pid]) <= args.hi

    mask = df["extra_info"].map(_band)
    eligible = df[mask].reset_index(drop=True)
    print(f"[sample] eligible (budget in [{args.lo},{args.hi}]): {len(eligible)} / {len(df)}")
    if len(eligible) < args.n:
        raise SystemExit(f"only {len(eligible)} eligible rows < requested n={args.n}")

    sampled = eligible.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    # Bake the banded B_q as each row's init budget (both fields, auto-hint rows).
    dist = Counter()
    new_extra = []
    for _, r in sampled.iterrows():
        ei = r["extra_info"]
        b = int(table[ei["problem_id"]])
        new_extra.append(_set_auto_hint_budget(ei, b))
        dist[b] += 1
    sampled["extra_info"] = new_extra
    print(f"[sample] sampled {len(sampled)} | init-budget dist: {dict(sorted(dist.items()))}")

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.inp))[0]
        out = os.path.join(os.path.dirname(args.inp),
                           f"{stem}-n{args.n}-b{args.lo}to{args.hi}.parquet")
    sampled.to_parquet(out, index=False)
    print(f"[sample] wrote {out}")


if __name__ == "__main__":
    main()
