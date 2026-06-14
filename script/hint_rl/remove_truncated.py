#!/usr/bin/env python3
# Copyright 2026
#
# Drop rows whose problem statement is cut off mid-sentence from an mt-type
# HPRL parquet (the output of prepare_hint_data.py).
#
# A handful of source rows have the problem statement severed mid-prose -- the
# body cut to "How many of the", or ending "...If the odometer", "...each term
# after the", etc. In rollouts these collapse first-turn reasoning (mean ~49 tok
# for the worst) and trigger a spurious <hint_call/>: the model writes "the
# statement is incomplete ... I need more information" and immediately calls a
# hint. ~27/3164 rows in dapo-3164-hint-verl-mt. These are silent label-quality
# noise, so drop them.
#
# Input : an mt-type parquet, e.g. dataset/dapo-3164-hint-verl-mt.parquet
#   columns: data_source, prompt[list[{role,content}]], ability,
#            reward_model{ground_truth}, extra_info{...}, agent_name, ...
#   The user-turn content carries the appended budget reminder ("You have N hint
#   calls remaining ..."); it is stripped before judging truncation.
#
# Output: the same parquet with truncated rows removed.
#
# Usage:
#   python remove_truncated.py --in <path> --out <path>
#   python remove_truncated.py --in dataset/dapo-3164-hint-verl-mt.parquet \
#       --out dataset/dapo-3164-hint-verl-mt-clean.parquet
#   python remove_truncated.py --in <path>          # in-place dry-run report only
#       --dry-run

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_RL_HOME = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_IN = os.path.join(HINT_RL_HOME, "dataset", "dapo-3164-hint-verl-mt.parquet")
DEFAULT_OUT = os.path.join(HINT_RL_HOME, "dataset", "dapo-3164-hint-verl-mt-clean.parquet")


def user_problem(prompt) -> str:
    """Extract the user problem statement from the chat prompt."""
    for m in prompt:
        if m.get("role") == "user":
            return m.get("content", "")
    # fallback: last message content
    return prompt[-1].get("content", "") if len(prompt) else ""


# --- truncated-problem detection -------------------------------------------
_BUDGET_TAIL_RE = re.compile(r"\n+You have (?:no |\d+ )?hint calls?.*$", re.IGNORECASE | re.DOTALL)
# Trailing figure / URL references: the QUESTION before them can be complete
# (e.g. "...same point as $1993$? \n\nAIME 1993 Problem 9.png"), so strip these
# tails before judging truncation -- a missing figure is a separate concern.
_ASSET_TAIL_RE = re.compile(
    r"(?:\[img\].*?\[/img\]|https?://\S+|\S+\.(?:png|jpe?g|gif)|"
    r"For diagram[^\n]*|AIME[^\n]*\.png)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def problem_is_truncated(text: str) -> bool:
    """True if the problem statement appears cut off mid-sentence.

    Strips the appended budget reminder and any trailing figure/URL references,
    then flags the row if what remains ends in a letter or comma -- i.e. no
    terminal punctuation, display-math close, or CJK full stop, the signature of
    a mid-prose truncation. Empty bodies count. A complete problem keeps its
    real ending here -- including answer-format tails like "..., please provide
    the value of m + n." or "...within \\boxed{}." -- which end in punctuation,
    so they are NOT stripped (stripping them would land on a comma and misfire).
    """
    t = _BUDGET_TAIL_RE.sub("", text or "").strip()
    # strip trailing asset refs (possibly several) so a complete "...?" before
    # a stray ".png" is not misread as truncated.
    for _ in range(3):
        new = _ASSET_TAIL_RE.sub("", t).strip()
        if new == t:
            break
        t = new
    if not t:
        return True
    # complete statements end in real punctuation, a display-math close, a
    # closing brace/paren/bracket (incl. [asy]...[/asy]), $, or a CJK full stop.
    if t[-1] in ".?!}])$" or t[-1] in "。．" or t.endswith("\\]") or t.endswith("\\)"):
        return False
    return bool(re.search(r"[A-Za-z一-鿿,]$", t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN,
                    help="input mt-type parquet")
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT,
                    help="output parquet")
    ap.add_argument("--dry-run", action="store_true",
                    help="report truncated rows but do not write an output parquet")
    args = ap.parse_args()

    out_path = args.out_path

    df = pd.read_parquet(args.in_path)
    print(f"read {len(df)} rows from {args.in_path}")

    mask = df["prompt"].map(lambda p: problem_is_truncated(user_problem(list(p))))
    n_tr = int(mask.sum())
    if n_tr:
        print(f"{'(dry-run) ' if args.dry_run else ''}truncated problem statements: {n_tr}")
        for idx in df.index[mask]:
            pid = (df.at[idx, "extra_info"] or {}).get("problem_id")
            body = _BUDGET_TAIL_RE.sub("", user_problem(list(df.at[idx, "prompt"]))).strip()
            print(f"  - problem_id={pid!r}: ...{body[-60:]!r}")
    else:
        print("no truncated problem statements found")

    if args.dry_run:
        return

    out = df[~mask].reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows ({n_tr} dropped) to {out_path}")


if __name__ == "__main__":
    main()
