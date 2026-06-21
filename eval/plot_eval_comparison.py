#!/usr/bin/env python3
"""Plot comparison of eval summaries across methods and datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "baseline",
    "grpo-step200",
    "hprl-v3-step250",
    "hprl-v3-0612-step100",
]

METHOD_LABELS = {
    "baseline": "Baseline",
    "grpo-step200": "GRPO step 200",
    "hprl-v3-step250": "HPRL v3 step 250",
    "hprl-v3-0612-step100": "HPRL v3 0612 step 100",
}

DATASET_ORDER = [
    "aime2024",
    "aime2025",
    "dapo_sample_hard_100",
]

DATASET_LABELS = {
    "aime2024": "AIME 2024",
    "aime2025": "AIME 2025",
    "dapo_sample_hard_100": "DAPO hard 100",
}


def normalize_dataset(name: str) -> str:
    return name.removesuffix("-hint-mt")


def run_sort_key(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run_"):
            return part
    return ""


def load_rows(results_dir: Path) -> list[dict]:
    summaries = sorted(results_dir.glob("run_*/*/*/_summary.json"))
    rows_by_key: dict[tuple[str, str], dict] = {}
    for summary_path in summaries:
        dataset_raw = summary_path.parent.name
        method = summary_path.parent.parent.name
        dataset = normalize_dataset(dataset_raw)
        if method not in METHOD_ORDER or dataset not in DATASET_ORDER:
            continue

        with summary_path.open() as f:
            summary = json.load(f)

        row = {
            "run": run_sort_key(summary_path),
            "method": method,
            "dataset": dataset,
            "dataset_raw": dataset_raw,
            "acc_mean": float(summary["acc_mean"]),
            "pass@128": float(summary.get("pass_at_k", {}).get("pass@128", np.nan)),
            "n_problems": int(summary["n_problems"]),
            "n_samples": int(summary["n_samples"]),
            "path": str(summary_path),
        }

        key = (method, dataset)
        if key not in rows_by_key or row["run"] > rows_by_key[key]["run"]:
            rows_by_key[key] = row

    return [rows_by_key[key] for key in sorted(rows_by_key)]


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "acc_mean",
        "pass@128",
        "n_problems",
        "n_samples",
        "run",
        "dataset_raw",
        "path",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(ax, rows: list[dict], metric: str, title: str) -> None:
    x = np.arange(len(DATASET_ORDER))
    width = 0.19
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2) * width
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    values = {
        (row["method"], row["dataset"]): row[metric]
        for row in rows
    }
    for idx, method in enumerate(METHOD_ORDER):
        ys = [values.get((method, dataset), np.nan) * 100 for dataset in DATASET_ORDER]
        bars = ax.bar(
            x + offsets[idx],
            ys,
            width,
            label=METHOD_LABELS[method],
            color=colors[idx],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.bar_label(
            bars,
            labels=["" if np.isnan(y) else f"{y:.1f}" for y in ys],
            padding=2,
            fontsize=8,
            rotation=90,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/eval_comparison.png"))
    parser.add_argument("--csv", type=Path, default=Path("results/eval_comparison.csv"))
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    expected = len(METHOD_ORDER) * len(DATASET_ORDER)
    if len(rows) != expected:
        found = {(row["method"], row["dataset"]) for row in rows}
        missing = [
            (method, dataset)
            for method in METHOD_ORDER
            for dataset in DATASET_ORDER
            if (method, dataset) not in found
        ]
        raise SystemExit(f"Expected {expected} method/dataset rows, found {len(rows)}. Missing: {missing}")

    write_csv(rows, args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    plot_metric(axes[0], rows, "acc_mean", "avg@N / pass@1")
    plot_metric(axes[1], rows, "pass@128", "pass@128")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Evaluation Comparison Across 4 Methods and 3 Datasets", fontsize=14, y=1.12)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
