#!/usr/bin/env python3
"""Plot step-wise eval comparison between methods from ckpt_* result dirs.

Scans result dirs for <method>-step<N>/<dataset>/_summary.json and draws
one panel per dataset with one line per method (avg@k and pass@k rows).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DATASET_ORDER = [
    "aime2024",
    "aime2025",
    "hmmt_nov_2025",
    "dapo_sample_hard_100",
]

DATASET_LABELS = {
    "aime2024": "AIME 2024",
    "aime2025": "AIME 2025",
    "hmmt_nov_2025": "HMMT Nov 2025",
    "dapo_sample_hard_100": "DAPO hard 100",
}

# Categorical slots, fixed order (validated palette).
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

STEP_RE = re.compile(r"^(?P<method>.+)-step(?P<step>\d+)$")


def load_rows(result_dirs: list[Path], methods: list[str]) -> list[dict]:
    rows = []
    for rdir in result_dirs:
        for summary_path in sorted(rdir.glob("*/*/_summary.json")):
            run_name = summary_path.parent.parent.name
            m = STEP_RE.match(run_name)
            if not m or m.group("method") not in methods:
                continue
            dataset = summary_path.parent.name
            with summary_path.open() as f:
                s = json.load(f)
            row = {
                "method": m.group("method"),
                "step": int(m.group("step")),
                "dataset": dataset,
                "acc_mean": float(s["acc_mean"]),
                "truncation_rate": float(s.get("truncation_rate", float("nan"))),
                "mean_completion_tokens": float(
                    s.get("timing", {}).get("mean_completion_tokens", float("nan"))
                ),
                "path": str(summary_path),
            }
            for k, v in s.get("pass_at_k", {}).items():
                row[k] = float(v)
            rows.append(row)
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row}, key=lambda k: (k == "path", k))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset"], r["method"], r["step"])))


def plot(rows: list[dict], methods: list[str], metrics: list[str], out: Path, title: str) -> None:
    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    datasets += sorted({r["dataset"] for r in rows} - set(datasets))

    series: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        for metric in metrics:
            if metric in r:
                series[(metric, r["dataset"], r["method"])].append((r["step"], r[metric]))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
    })
    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows),
        constrained_layout=True, squeeze=False, sharex=True,
    )
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            for mi, method in enumerate(methods):
                pts = sorted(series.get((metric, dataset, method), []))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(
                    xs, [y * 100 for y in ys],
                    color=METHOD_COLORS[mi], linewidth=2,
                    marker="o", markersize=3.5, label=method,
                )
            if i == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset))
            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel(f"{metric} (%)")
            ax.grid(color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=["acc_mean", "pass@8"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--title", default="Eval comparison")
    args = parser.parse_args()

    rows = load_rows(args.results_dirs, args.methods)
    if not rows:
        raise SystemExit("no matching _summary.json found")
    found = sorted({(r["method"], r["step"]) for r in rows})
    print(f"loaded {len(rows)} summaries across {len(found)} checkpoints")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.methods, args.metrics, args.out, args.title)


if __name__ == "__main__":
    main()
