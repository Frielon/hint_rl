#!/usr/bin/env python3
"""Read the ``_summary.json`` of every OpenAI eval run under ``results/`` and print
a markdown comparison table (selection + citation), mirroring gpt_oss_eval's
progress.md tables. By default it keeps the LATEST run per model.

Usage:
    python compare_runs.py                      # latest run per model in results/
    python compare_runs.py --all                # every run, not just the latest
    python compare_runs.py results/<run> ...    # only these run dirs
    python compare_runs.py --md compare.md      # also write the table to a file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


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
    """Keep only the newest run dir per model (dir names end in __<ts>, so a
    plain name sort is chronological within a model)."""
    best: dict[str, dict] = {}
    for r in runs:
        model = r["summary"].get("model") or r["config"].get("model") or r["name"]
        if model not in best or r["name"] > best[model]["name"]:
            best[model] = r
    return sorted(best.values(), key=lambda r: (r["summary"].get("model") or r["name"]))


def fmt_table(runs: list[dict]) -> str:
    def g(s, *keys, default=0.0):
        cur = s
        for k in keys:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        return cur if cur is not None else default

    lines = []
    lines.append("## OpenAI closed-model eval — Template F hint-selection + citation\n")
    lines.append(f"_{len(runs)} run(s); label = "
                 f"{runs[0]['config'].get('label_run', '?') if runs else '?'}_\n")

    # --- selection + cost ---
    lines.append("### Selection & cost\n")
    lines.append("| model | n | rows | agree_hint | agree_major_step | self_cons | maj_agree | mean_gen_toks | trunc |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in runs:
        s, c = r["summary"], r["config"]
        model = s.get("model") or c.get("model") or r["name"]
        lines.append(
            f"| `{model}` | {c.get('n', '?')} | {s.get('n_rows_scored', 0)}/{s.get('n_rows', 0)} "
            f"| {g(s,'mean_agreement_hint_id'):.3f} | {g(s,'mean_agreement_major_step'):.3f} "
            f"| {g(s,'mean_self_consistency'):.3f} | {g(s,'majority_agree_rate'):.3f} "
            f"| {g(s,'mean_completion_tokens'):.0f} | {s.get('total_truncated', 0)} |")

    # --- citation ---
    lines.append("\n### Citation fidelity (model quotes vs. the trace shown)\n")
    lines.append("| model | n_quotes | exact_rate | verbatim_rate | found_rate | not_found_rate | ref_baseline_found |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in runs:
        s = r["summary"]
        model = s.get("model") or r["config"].get("model") or r["name"]
        mq = g(s, "citation", "model_quotes", default={}) or {}
        rb = g(s, "citation", "reference_baseline", default={}) or {}
        lines.append(
            f"| `{model}` | {mq.get('n_quotes', 0)} "
            f"| {mq.get('exact_rate', 0):.3f} | {mq.get('verbatim_rate', 0):.3f} "
            f"| {mq.get('found_rate', 0):.3f} | {mq.get('not_found_rate', 0):.3f} "
            f"| {rb.get('found_rate', 0):.3f} |")

    # --- token usage & estimated cost ---
    lines.append("\n### Token usage & estimated cost\n")
    lines.append("| model | calls | in_tok | out_tok | total_tok | est_cost |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    total = 0.0
    for r in runs:
        s = r["summary"]
        model = s.get("model") or r["config"].get("model") or r["name"]
        u = s.get("usage") or {}
        cost = u.get("est_cost_usd")
        if isinstance(cost, (int, float)):
            total += cost
        cost_s = f"${cost:.4f}" if isinstance(cost, (int, float)) else "—"
        lines.append(
            f"| `{model}` | {u.get('n_calls_with_usage', 0):,} | {u.get('prompt_tokens', 0):,} "
            f"| {u.get('completion_tokens', 0):,} | {u.get('total_tokens', 0):,} | {cost_s} |")
    lines.append(f"| **total** |  |  |  |  | **${total:.4f}** |")
    lines.append("\n_est_cost estimate: prompt_tokens at input price (ignores cache discount), "
                 "completion_tokens at output price (incl. reasoning). `—` = no `usage` block; "
                 "back-fill with `python recost.py`. Prices in `pricing.py`._")

    lines.append("\n_agree_hint = mean fraction of samples matching the reference hint_id; "
                 "agree_major_step = same major step; maj_agree = modal hint_id == ref. "
                 "verbatim_rate = exact+normalized+loose; found_rate includes fuzzy._")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path,
                    help="specific run dirs (default: all under results/)")
    ap.add_argument("--all", action="store_true",
                    help="show every run, not just the latest per model")
    ap.add_argument("--md", default=None, help="also write the table to this file")
    args = ap.parse_args()

    if args.run_dirs:
        dirs = [d if d.is_absolute() else (HERE / d) for d in args.run_dirs]
    elif RESULTS.is_dir():
        dirs = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    else:
        dirs = []

    runs = load_runs(dirs)
    if not runs:
        print("no runs with _summary.json found under results/ (run run_openai_eval.sh first).")
        return 0
    if not args.all and not args.run_dirs:
        runs = latest_per_model(runs)

    table = fmt_table(runs)
    print(table)
    if args.md:
        Path(args.md).write_text(table)
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
