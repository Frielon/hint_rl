#!/usr/bin/env python
"""
Plot per-problem accuracy improvement (last epoch vs first epoch) for a GRPO run,
classified by the problem's initial (first-epoch) accuracy.

Layout assumed: a rollouts dir containing step files `<step>.jsonl`, each holding
(prompts x group-size) rollouts. Each line has at least the keys `input` and `acc`.
Per-problem accuracy in a step = mean of `acc` over that prompt's samples in the step.
A problem is identified by its prompt text (`input`); its FIRST occurrence across all
steps = first epoch, LAST occurrence = last epoch.

Usage:
  python plot_accuracy_evolution.py <rollouts_dir> [-o OUT.png] [--csv OUT.csv]
                                    [--cache PKL] [--no-cache]

Examples:
  python plot_accuracy_evolution.py /path/to/run/rollouts
  python plot_accuracy_evolution.py /path/to/run/rollouts -o /tmp/evo.png
"""
import argparse, glob, hashlib, json, os, pickle
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# initial-accuracy classes (acc is a multiple of 1/group_size)
CLASSES = [
    ("0.0 (never correct)",  lambda f: f == 0.0),
    ("(0.0, 0.25]",          lambda f: (f > 0.0)  & (f <= 0.25)),
    ("(0.25, 0.5]",          lambda f: (f > 0.25) & (f <= 0.5)),
    ("(0.5, 0.75]",          lambda f: (f > 0.5)  & (f <= 0.75)),
    ("(0.75, 1.0)",          lambda f: (f > 0.75) & (f < 1.0)),
    ("1.0 (always correct)", lambda f: f == 1.0),
]


def build_per_problem(rollouts_dir, cache):
    """Return {prob_hash: {step: mean_acc}} aggregated over all step files."""
    if cache and os.path.exists(cache):
        print(f"[cache] loading {cache}")
        return pickle.load(open(cache, "rb"))
    files = glob.glob(os.path.join(rollouts_dir, "*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl step files found in {rollouts_dir}")

    def step_id(f):
        try:
            return int(os.path.basename(f)[:-6])
        except ValueError:
            return None

    steps = sorted(s for s in (step_id(f) for f in files) if s is not None)
    print(f"[scan] {len(steps)} step files in {rollouts_dir}")
    prob = defaultdict(dict)
    for s in steps:
        agg = defaultdict(lambda: [0.0, 0])
        with open(os.path.join(rollouts_dir, f"{s}.jsonl")) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                h = hashlib.md5(d["input"].encode()).hexdigest()
                a = agg[h]
                a[0] += float(d["acc"]); a[1] += 1
        for h, (sm, c) in agg.items():
            prob[h][s] = sm / c
    prob = dict(prob)
    if cache:
        pickle.dump(prob, open(cache, "wb"))
        print(f"[cache] wrote {cache}")
    return prob


def first_last(prob):
    first, last, fs, ls, keys = [], [], [], [], []
    for h, d in prob.items():
        ss = sorted(d)
        keys.append(h)
        first.append(d[ss[0]]); last.append(d[ss[-1]])
        fs.append(ss[0]); ls.append(ss[-1])
    return (np.array(keys), np.array(first), np.array(last),
            np.array(fs), np.array(ls))


def plot(first, last, out_png, title_extra=""):
    delta = last - first
    N = len(first)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    bins = np.linspace(-1, 1, 33)
    for ax, (name, sel) in zip(axes.ravel(), CLASSES):
        m = sel(first); d = delta[m]; n = int(m.sum())
        ax.hist(d, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="grey", lw=1, ls="--")
        if n:
            ax.axvline(d.mean(), color="#C44E52", lw=1.8, label=f"mean Δ = {d.mean():+.3f}")
            up = int((d > 1e-9).sum()); dn = int((d < -1e-9).sum()); eq = n - up - dn
            ax.text(0.02, 0.97,
                    f"n = {n} ({n/N*100:.1f}%)\n↑ {up/n*100:.0f}%  • {eq/n*100:.0f}%  ↓ {dn/n*100:.0f}%",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(f"initial acc {name}", fontsize=11)
        ax.set_xlabel("improvement  (last − first epoch accuracy)")
        ax.set_ylabel("# problems")
        ax.set_xlim(-1, 1)
    fig.suptitle(
        f"Per-problem accuracy improvement by initial-accuracy class  "
        f"(N={N};  overall mean Δ={delta.mean():+.3f}: {first.mean():.3f}→{last.mean():.3f})"
        + (f"\n{title_extra}" if title_extra else ""),
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", help="path to the rollouts dir (contains <step>.jsonl)")
    ap.add_argument("-o", "--out", help="output PNG (default: <run_dir>/accuracy_evolution_by_initial.png)")
    ap.add_argument("--csv", nargs="?", const="__auto__",
                    help="also write per-problem CSV (default path if flag given without value)")
    ap.add_argument("--cache", help="aggregation pickle cache path (default: derived from rollouts dir)")
    ap.add_argument("--no-cache", action="store_true", help="disable the aggregation cache")
    args = ap.parse_args()

    rollouts = os.path.abspath(args.rollouts.rstrip("/"))
    run_dir = os.path.dirname(rollouts)              # parent of the rollouts dir
    run_tag = os.path.basename(run_dir) or "run"
    out_png = args.out or os.path.join(run_dir, "accuracy_evolution_by_initial.png")
    if args.no_cache:
        cache = None
    else:
        cache = args.cache or os.path.join("/tmp", f"prob_acc_{run_tag}.pkl")

    prob = build_per_problem(rollouts, cache)
    keys, first, last, fs, ls = first_last(prob)
    delta = last - first
    N = len(first)

    print(f"\nproblems: {N}")
    print(f"mean first={first.mean():.4f}  mean last={last.mean():.4f}  mean delta={delta.mean():+.4f}")
    print("\nper-class summary:")
    print(f"  {'class':<24}{'n':>7}{'%':>7}{'mean Δ':>9}{'% improved':>12}")
    for name, sel in CLASSES:
        m = sel(first); d = delta[m]; n = int(m.sum())
        if not n:
            continue
        print(f"  {name:<24}{n:>7}{n/N*100:>6.1f}%{d.mean():>+9.3f}{(d>1e-9).mean()*100:>11.1f}%")

    plot(first, last, out_png, title_extra=run_tag)

    if args.csv:
        csv_path = args.csv if args.csv != "__auto__" else os.path.join(run_dir, "per_problem_acc_first_vs_last.csv")
        import csv
        with open(csv_path, "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["input_md5", "first_step", "last_step",
                        "first_epoch_acc", "last_epoch_acc", "delta"])
            for h, f, l, a, b in zip(keys, fs, ls, first, last):
                w.writerow([h, int(f), int(l), f"{a:.4f}", f"{b:.4f}", f"{b-a:+.4f}"])
        print(f"[csv]  saved -> {csv_path}")


if __name__ == "__main__":
    main()
