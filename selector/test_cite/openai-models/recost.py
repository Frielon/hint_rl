#!/usr/bin/env python3
"""Back-fill token usage + estimated cost into a run's ``_summary.json``.

Runs saved before cost reporting was added (or any run at all) already store
``prompt_tokens`` / ``completion_tokens`` per sample, so cost can be computed
after the fact with no model calls. This reads a run dir's per-row JSONs, sums
tokens, prices them via ``pricing.py``, writes a ``usage`` block back into that
run's ``_summary.json``, and prints the cost.

Usage:
    python recost.py                          # every run under results/ + multi_results/
    python recost.py multi_results/<run> ...  # specific run dirs
    python recost.py --no-write               # just print, don't touch _summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pricing

HERE = Path(__file__).resolve().parent
ROOTS = [HERE / "results", HERE / "multi_results"]


def model_of(run_dir: Path) -> str:
    for name in ("_summary.json", "run_config.json"):
        p = run_dir / name
        if p.is_file():
            try:
                m = json.loads(p.read_text()).get("model")
                if m:
                    return m
            except Exception:  # noqa: BLE001
                pass
    # fall back to the dir name: <label>__<model>__<ts>  /  multi__<model>__<ts>
    parts = run_dir.name.split("__")
    return parts[1] if len(parts) >= 3 else run_dir.name


def records_of(run_dir: Path) -> list[dict]:
    recs = []
    for f in glob.glob(str(run_dir / "step*" / "*" / "*.json")):
        try:
            recs.append(json.loads(Path(f).read_text()))
        except Exception:  # noqa: BLE001
            continue
    return recs


def process(run_dir: Path, write: bool) -> dict | None:
    sp = run_dir / "_summary.json"
    recs = records_of(run_dir)
    if not recs:
        return None
    model = model_of(run_dir)
    usage = pricing.usage_and_cost(recs, model)
    if write and sp.is_file():
        try:
            s = json.loads(sp.read_text())
        except Exception:  # noqa: BLE001
            s = {}
        s["usage"] = usage
        sp.write_text(json.dumps(s, indent=2) + "\n")
    return usage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--no-write", action="store_true", help="print only; don't edit _summary.json")
    args = ap.parse_args()

    if args.run_dirs:
        dirs = [d if d.is_absolute() else (HERE / d) for d in args.run_dirs]
    else:
        dirs = []
        for root in ROOTS:
            if root.is_dir():
                dirs += sorted(p for p in root.iterdir() if p.is_dir())

    total = 0.0
    priced = False
    print(f"{'model':<16} {'calls':>7} {'in_tok':>12} {'out_tok':>12} {'total':>12} {'est_cost':>10}")
    print("-" * 74)
    for d in dirs:
        u = process(d, write=not args.no_write)
        if u is None:
            continue
        cost = u["est_cost_usd"]
        if cost is not None:
            total += cost
            priced = True
        cost_s = f"${cost:.4f}" if cost is not None else "n/a"
        print(f"{u['model']:<16} {u['n_calls_with_usage']:>7,} {u['prompt_tokens']:>12,} "
              f"{u['completion_tokens']:>12,} {u['total_tokens']:>12,} {cost_s:>10}")
    print("-" * 74)
    if priced:
        print(f"{'TOTAL':<16} {'':>7} {'':>12} {'':>12} {'':>12} {'$' + format(total, '.4f'):>10}")
    print(f"\nprices: {pricing.PRICES_AS_OF}  (edit pricing.PRICES or set OPENAI_PRICES_JSON)")
    if not args.no_write:
        print("wrote a 'usage' block into each run's _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
