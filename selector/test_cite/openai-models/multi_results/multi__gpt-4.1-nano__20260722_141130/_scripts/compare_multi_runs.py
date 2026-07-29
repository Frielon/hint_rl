#!/usr/bin/env python3
"""Read the ``_summary.json`` of every MULTI-ROUND OpenAI run under
``multi_results/`` and print a markdown comparison table, mirroring the metrics
in ``gpt_oss_eval/multi-cite-gpt-eval/results/.../_summary.json``. Keeps the
latest run per model by default.

Usage:
    python compare_multi_runs.py                    # latest run per model
    python compare_multi_runs.py --all              # every run
    python compare_multi_runs.py multi_results/<run> ...
    python compare_multi_runs.py --gpt-oss          # also show the gpt-oss reference row
    python compare_multi_runs.py --md compare_multi.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "multi_results"
GPT_OSS_REF = (HERE.parent / "gpt_oss_eval" / "multi-cite-gpt-eval" / "results")


def load_runs(dirs: list[Path]) -> list[dict]:
    runs = []
    for d in dirs:
        sp = d / "_summary.json"
        if not sp.is_file():
            continue
        try:
            s = json.loads(sp.read_text())
        except Exception:  # noqa: BLE001
            continue
        cfg = {}
        cp = d / "run_config.json"
        if cp.is_file():
            try:
                cfg = json.loads(cp.read_text())
            except Exception:  # noqa: BLE001
                cfg = {}
        runs.append({"dir": d, "name": d.name, "summary": s, "config": cfg})
    return runs


def latest_per_model(runs: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in runs:
        model = r["summary"].get("model") or r["config"].get("model") or r["name"]
        if model not in best or r["name"] > best[model]["name"]:
            best[model] = r
    return sorted(best.values(), key=lambda r: (r["summary"].get("model") or r["name"]))


def fmt(runs: list[dict]) -> str:
    def val(s, k, d=0.0):
        v = s.get(k)
        return v if v is not None else d

    L = []
    L.append("## OpenAI closed-model MULTI-ROUND eval — hint-selection + citation\n")
    L.append(f"_{len(runs)} run(s); benchmark = "
             f"{Path(runs[0]['config'].get('benchmark', 'benchmark.jsonl')).name if runs else '?'}_\n")

    # --- selection + multi-round-specific metrics ---
    L.append("### Selection, completed-hint handling & cost\n")
    L.append("| model | n | rows | agree_strict | agree_merged | maj_merged | self_cons "
             "| sel_completed↓ | newly_recall | gap_rows | gen_toks | trunc |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in runs:
        s, c = r["summary"], r["config"]
        model = s.get("model") or c.get("model") or r["name"]
        nr = val(s, "newly_completed_recall", None)
        nr_s = f"{nr:.3f}" if isinstance(nr, (int, float)) else "—"
        L.append(
            f"| `{model}` | {c.get('n', '?')} | {val(s,'n_rows_scored',0)}/{val(s,'n_rows',0)} "
            f"| {val(s,'mean_agreement_strict'):.3f} | {val(s,'mean_agreement_merged'):.3f} "
            f"| {val(s,'majority_merged_rate'):.3f} | {val(s,'mean_self_consistency'):.3f} "
            f"| {val(s,'selected_completed_rate'):.4f} | {nr_s} "
            f"| {val(s,'n_rows_with_gap',0)} | {val(s,'mean_completion_tokens'):.0f} "
            f"| {val(s,'total_truncated',0)} |")

    # --- citation ---
    L.append("\n### Citation fidelity (model completed_hints quotes vs. the trace)\n")
    L.append("| model | n_quotes | verbatim_rate | found_rate | not_found_rate "
             "| exact | normalized | loose | fuzzy |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in runs:
        s = r["summary"]
        model = s.get("model") or r["config"].get("model") or r["name"]
        cit = s.get("citation") or {}
        L.append(
            f"| `{model}` | {cit.get('n_quotes', 0)} "
            f"| {cit.get('verbatim_rate', 0):.3f} | {cit.get('found_rate', 0):.3f} "
            f"| {cit.get('not_found_rate', 0):.3f} | {cit.get('exact', 0)} "
            f"| {cit.get('normalized', 0)} | {cit.get('loose', 0)} | {cit.get('fuzzy', 0)} |")

    L.append("\n_agree_merged = x.0≡x.1 merged (model can't pick x.0). "
             "sel_completed↓ = fraction picking an already-completed hint (lower is better, ~0). "
             "newly_recall = of achieved-but-unmarked pending hints in gap rows, fraction re-recognized. "
             "verbatim_rate = exact+normalized+loose; found_rate includes fuzzy._")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every run, not just latest per model")
    ap.add_argument("--gpt-oss", action="store_true",
                    help="also include the gpt-oss reference run(s) from multi-cite-gpt-eval")
    ap.add_argument("--md", default=None, help="also write the table to this file")
    args = ap.parse_args()

    if args.run_dirs:
        dirs = [d if d.is_absolute() else (HERE / d) for d in args.run_dirs]
    elif RESULTS.is_dir():
        dirs = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    else:
        dirs = []

    runs = load_runs(dirs)
    if not args.all and not args.run_dirs:
        runs = latest_per_model(runs)

    if args.gpt_oss and GPT_OSS_REF.is_dir():
        ref_runs = load_runs(sorted(p for p in GPT_OSS_REF.iterdir() if p.is_dir()))
        for rr in ref_runs:
            rr["summary"].setdefault("model", rr["config"].get("model", "gpt-oss-20b"))
        runs = ref_runs + runs

    if not runs:
        print("no runs with _summary.json found under multi_results/ "
              "(run run_openai_multi_eval.sh first).")
        return 0

    table = fmt(runs)
    print(table)
    if args.md:
        Path(args.md).write_text(table)
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
