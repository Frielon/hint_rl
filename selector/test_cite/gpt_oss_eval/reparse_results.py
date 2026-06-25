#!/usr/bin/env python3
"""Re-parse the saved completions in an existing gpt-oss eval run *in place*.

Every result JSON stores each sample's raw completion (``raw``). After a parser
fix (e.g. the ``_repair_json_escapes`` escape-pair fix that stopped corrupting a
model-correct ``\\cdot``), re-running the parser over those saved raws recovers
fields the old parser dropped -- notably the ``completed_hints`` array whose JSON
the model double-escaped -- WITHOUT re-querying the model.

For each sample we recompute ``selection`` / ``hint_id`` / ``major_step_id`` /
``parse_error`` from the same code paths the live scorer uses, then recompute the
file-level aggregates (``summarize``) and the run's ``_summary.json``
(``corpus_summary``). API-error / empty-raw samples are left untouched.

Usage:
    python reparse_results.py <run_dir>            # rewrite in place
    python reparse_results.py <run_dir> --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # .../test_cite/gpt_oss_eval
SELECTOR = HERE.parent.parent                     # .../hint_rl/selector
sys.path.insert(0, str(SELECTOR))                 # shared parser
sys.path.insert(0, str(HERE))                     # local prompt + driver helpers

from run_hint_selection_model import parse_output, hint_id_of   # noqa: E402
from run_gpt_oss_selection import major_step_of, summarize, corpus_summary  # noqa: E402


def _has_ch(sel) -> bool:
    return isinstance(sel, dict) and "completed_hints" in sel


def reparse_sample(s: dict) -> None:
    """Re-derive selection/hint_id/major_step_id/parse_error from saved raw."""
    raw = s.get("raw")
    if not raw:                       # api-error or --no-save-raw: nothing to redo
        return
    sel, perr = parse_output(raw)
    if perr and s.get("finish_reason") == "length":
        perr = f"truncated (finish_reason=length); {perr}"
    hid = hint_id_of(sel)
    s["hint_id"] = hid
    s["major_step_id"] = major_step_of(sel, hid)
    s["selection"] = sel
    s["parse_error"] = perr


def reparse_file(fp: Path) -> tuple[dict, int]:
    d = json.loads(fp.read_text())
    samples = d.get("samples") or []
    recovered = 0
    for s in samples:
        had = _has_ch(s.get("selection"))
        reparse_sample(s)
        if not had and _has_ch(s.get("selection")):
            recovered += 1
    # recompute aggregates against the reference ids already stored in this file
    d.update(summarize(samples, d.get("ref_hint_id"), d.get("ref_major_step_id")))
    d["samples"] = samples
    return d, recovered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="results/<run> dir to re-parse in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    files = sorted(run.glob("step*/*/*.json"))
    if not files:
        raise SystemExit(f"no step*/*/*.json under {run}")

    records: list[dict] = []
    total_recovered = 0
    files_changed = 0
    for fp in files:
        d, rec = reparse_file(fp)
        total_recovered += rec
        if rec:
            files_changed += 1
        if not args.dry_run:
            fp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        records.append(d)

    summary = corpus_summary(records)
    if not args.dry_run:
        (run / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}run            : {run.name}")
    print(f"{tag}files          : {len(files)}  ({files_changed} with recovered completed_hints)")
    print(f"{tag}samples recovered completed_hints: {total_recovered}")
    print(f"{tag}corpus mean_agreement_hint_id    : {summary['mean_agreement_hint_id']:.4f}")
    print(f"{tag}corpus majority_agree_rate       : {summary['majority_agree_rate']:.4f}")
    if args.dry_run:
        print("(dry-run: no files written)")
    else:
        print(f"wrote {len(files)} files + _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
