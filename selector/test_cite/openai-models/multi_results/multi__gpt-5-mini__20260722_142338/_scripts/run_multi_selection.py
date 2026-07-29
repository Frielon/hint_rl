#!/usr/bin/env python3
"""Multi-round hint-selection eval.

Reads the benchmark from build_benchmark.py (each row: problem + trace + pruned
hint pool with a ``completed_status`` prefix marked as given in earlier rounds),
prompts the served model with the multi-round Template F, draws N samples/row,
and scores:

  * selection vs the reference next-hint ``H`` -- strict and with x.0==x.1 merged
    (the model can only pick substeps, so ref x.0 + model x.1 counts as correct);
  * selected_completed_rate -- how often the model wrongly picks an already
    ``completed`` hint (should be ~0);
  * newly_completed_recall -- on rows with achieved-but-unmarked hints in (C, H),
    how many of those the model re-recognizes in its ``completed_hints``;
  * citation -- are the model's completed_hints quotes verbatim in the trace
    (same tiers as check_citations.py).

Output mirrors the single-round layout: step{N}/<problem_id>/<request_id>.json
plus _summary.json.

Examples:
    python run_multi_selection.py --base-url http://127.0.0.1:30000/v1 -n 16
    python run_multi_selection.py --dry-run --limit 3      # build prompts only, no server
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
GPT_OSS_EVAL = HERE.parent
TEST_CITE = GPT_OSS_EVAL.parent
SELECTOR = TEST_CITE.parent
for p in (str(SELECTOR), str(GPT_OSS_EVAL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_hint_selection_model import parse_output, hint_id_of, make_client, one_sample  # noqa: E402
from check_citations import (  # noqa: E402
    norm as cite_norm, loose as cite_loose, classify as cite_classify, quotes_of as cite_quotes_of,
)
import prompt_template_multiF as PT  # noqa: E402
from prompt_template_multiF import build_prompt, render_hints_with_status  # noqa: E402

_CITE_FOUND = ("exact", "normalized", "loose", "fuzzy")


def canon(hid: Any) -> Optional[str]:
    """Merge x.0 and x.1 into one class (x.0); keep x.2+ distinct."""
    if hid is None:
        return None
    maj, _, mn = str(hid).partition(".")
    try:
        m = int(mn)
    except ValueError:
        return str(hid)
    return f"{maj}.{0 if m <= 1 else m}"


def completed_ids_of(sel: Optional[dict]) -> list[str]:
    if not isinstance(sel, dict):
        return []
    cs = sel.get("completed_hints")
    if cs is None:
        cs = sel.get("completed_substeps")
    return [str(e.get("hint_id")) for e in cs if isinstance(e, dict) and e.get("hint_id") is not None] \
        if isinstance(cs, list) else []


# --------------------------------------------------------------------------- #
def load_benchmark(path: Path, steps, limit) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if steps is not None:
        rows = [r for r in rows if r.get("step") in steps]
    if limit is not None:
        per: Counter = Counter()
        out = []
        for r in rows:
            if per[r.get("step")] >= limit:
                continue
            per[r.get("step")] += 1
            out.append(r)
        rows = out
    return rows


def out_path_for(out_dir: Path, row: dict) -> Path:
    safe = lambda s: str(s).replace("/", "_").replace(os.sep, "_")  # noqa: E731
    return out_dir / f"step{row.get('step')}" / safe(row.get("problem_id")) / f"{safe(row.get('request_id'))}.json"


def summarize(samples: list[dict], row: dict) -> dict[str, Any]:
    ref = row["ref_hint_id"]
    completed = set(row.get("completed_status") or [])
    expected_newly = set(row.get("expected_newly_completed") or [])
    trace = row.get("reasoning_trace") or ""
    tr = {"raw": trace, "norm": cite_norm(trace), "loose": cite_loose(trace)}

    hids = [s["hint_id"] for s in samples if s["hint_id"] is not None]
    n_parsed = len(hids)
    dist = Counter(hids)
    mode_hid = dist.most_common(1)[0][0] if dist else None
    mode_canon = Counter(canon(h) for h in hids).most_common(1)[0][0] if hids else None
    n_trunc = sum(1 for s in samples if s.get("finish_reason") == "length")
    ctoks = [s["completion_tokens"] for s in samples if s.get("completion_tokens") is not None]

    agree_strict = sum(1 for h in hids if str(h) == str(ref)) / n_parsed if n_parsed else 0.0
    agree_merged = sum(1 for h in hids if canon(h) == canon(ref)) / n_parsed if n_parsed else 0.0
    sel_completed = sum(1 for h in hids if str(h) in completed) / n_parsed if n_parsed else 0.0

    # newly-completed recall (only on rows that have a gap)
    recalls = []
    for s in samples:
        if s["hint_id"] is None:
            continue
        got = set(completed_ids_of(s.get("selection")))
        if expected_newly:
            recalls.append(len(got & expected_newly) / len(expected_newly))
    newly_recall = sum(recalls) / len(recalls) if recalls else None

    # citation over the model's completed_hints quotes
    cite = Counter()
    for s in samples:
        for q in cite_quotes_of(s.get("selection")):
            cite[cite_classify(q, tr, 0.85, 0.0)] += 1

    return {
        "n_samples": len(samples),
        "n_parsed": n_parsed,
        "n_truncated": n_trunc,
        "mean_completion_tokens": (sum(ctoks) / len(ctoks)) if ctoks else 0.0,
        "hint_id_distribution": {str(k): v for k, v in dist.items()},
        "mode_hint_id": mode_hid,
        "self_consistency": (dist.most_common(1)[0][1] / n_parsed) if n_parsed else 0.0,
        "agreement_strict": agree_strict,
        "agreement_merged": agree_merged,
        "majority_agrees_strict": (mode_hid is not None and str(mode_hid) == str(ref)),
        "majority_agrees_merged": (mode_canon is not None and mode_canon == canon(ref)),
        "selected_completed_rate": sel_completed,
        "newly_completed_recall": newly_recall,
        "has_gap": bool(expected_newly),
        "citation": dict(cite),
    }


def process_row(row: dict, client, args) -> Optional[dict]:
    out_path = out_path_for(args.out_dir, row)
    if out_path.exists() and not args.overwrite:
        try:
            return json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    hints = render_hints_with_status(row["hint_pool"], row.get("completed_status"))
    prompt = build_prompt(row["problem"], row["reasoning_trace"], hints)

    base = {k: row.get(k) for k in ("uid", "step", "problem_id", "request_id",
                                    "ref_hint_id", "cut_hint_id", "completed_status",
                                    "pending", "expected_newly_completed")}
    base["model"] = args.model

    if args.dry_run:
        base.update({"_prompt_len": len(prompt), "samples": []})
        return base

    samples: list[Optional[dict]] = [None] * args.n
    with ThreadPoolExecutor(max_workers=args.sample_workers) as ex:
        futs = {ex.submit(one_sample, client, args.model, prompt,
                          args.temperature, args.top_p, args.max_tokens): i
                for i in range(args.n)}
        for fut in as_completed(futs):
            i = futs[fut]
            res = fut.result()
            if res.get("error") is not None:
                samples[i] = {"index": i, "hint_id": None, "selection": None,
                              "selected_completed": None, "parse_error": f"api_error: {res['error']}",
                              "finish_reason": None, "completion_tokens": None,
                              "prompt_tokens": None, "latency_s": None, "raw": ""}
                continue
            raw, fin = res["content"], res["finish_reason"]
            sel, perr = parse_output(raw)
            if perr and fin == "length":
                perr = f"truncated (finish_reason=length); {perr}"
            hid = hint_id_of(sel)
            samples[i] = {
                "index": i,
                "hint_id": hid,
                "selection": sel,
                "selected_completed": (str(hid) in set(row.get("completed_status") or [])) if hid is not None else None,
                "parse_error": perr,
                "finish_reason": fin,
                "completion_tokens": res["completion_tokens"],
                "prompt_tokens": res["prompt_tokens"],
                "latency_s": res.get("latency_s"),
                "raw": raw if args.save_raw else "",
            }
    base.update({**summarize(samples, row), "samples": samples})
    return base


def _cite_rate(c: Counter) -> dict[str, Any]:
    tot = sum(c.values()) or 1
    found = sum(c.get(k, 0) for k in _CITE_FOUND)
    return {
        "n_quotes": sum(c.values()),
        "exact": c.get("exact", 0), "normalized": c.get("normalized", 0),
        "loose": c.get("loose", 0), "fuzzy": c.get("fuzzy", 0),
        "not_found": c.get("not_found", 0),
        "verbatim_rate": round((c.get("exact", 0) + c.get("normalized", 0) + c.get("loose", 0)) / tot, 4),
        "found_rate": round(found / tot, 4),
        "not_found_rate": round(c.get("not_found", 0) / tot, 4),
    }


def corpus_summary(records: list[dict]) -> dict[str, Any]:
    scored = [r for r in records if r.get("n_parsed", 0) > 0]
    n = len(scored)
    avg = lambda k: (sum(r[k] for r in scored) / n) if n else 0.0  # noqa: E731
    rate = lambda k: (sum(int(bool(r[k])) for r in scored) / n) if n else 0.0  # noqa: E731
    gap_rows = [r for r in scored if r.get("has_gap") and r.get("newly_completed_recall") is not None]
    cite = Counter()
    for r in records:
        for k, v in (r.get("citation") or {}).items():
            cite[k] += v
    by_step: dict[str, dict] = {}
    for r in scored:
        b = by_step.setdefault(str(r.get("step")), {"n": 0, "agree_merged": 0.0, "maj_merged": 0})
        b["n"] += 1
        b["agree_merged"] += r["agreement_merged"]
        b["maj_merged"] += int(bool(r["majority_agrees_merged"]))
    for b in by_step.values():
        k = b["n"] or 1
        b["mean_agreement_merged"] = round(b.pop("agree_merged") / k, 4)
        b["majority_merged_rate"] = round(b.pop("maj_merged") / k, 4)
    return {
        "n_rows": len(records),
        "n_rows_scored": n,
        "mean_agreement_strict": round(avg("agreement_strict"), 4),
        "mean_agreement_merged": round(avg("agreement_merged"), 4),
        "majority_strict_rate": round(rate("majority_agrees_strict"), 4),
        "majority_merged_rate": round(rate("majority_agrees_merged"), 4),
        "mean_self_consistency": round(avg("self_consistency"), 4),
        "selected_completed_rate": round(avg("selected_completed_rate"), 4),
        "n_rows_with_gap": len(gap_rows),
        "newly_completed_recall": round(sum(r["newly_completed_recall"] for r in gap_rows) / len(gap_rows), 4) if gap_rows else None,
        "mean_completion_tokens": round(avg("mean_completion_tokens"), 1),
        "total_truncated": sum(r.get("n_truncated", 0) for r in records),
        "citation": _cite_rate(cite),
        "by_step": dict(sorted(by_step.items())),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(HERE / "benchmark.jsonl"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no model calls")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME", "gpt-oss-20b"))
    ap.add_argument("-n", "--n", type=int, default=int(os.environ.get("N_SAMPLES", "16")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.3")))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "16384")))
    ap.add_argument("--row-workers", type=int, default=int(os.environ.get("ROW_WORKERS", "32")))
    ap.add_argument("--sample-workers", type=int, default=int(os.environ.get("SAMPLE_WORKERS", "16")))
    ap.add_argument("--save-raw", action="store_true", default=True)
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    rows = load_benchmark(Path(args.benchmark), steps, args.limit)
    if args.out_dir:
        args.out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = HERE / "results" / f"multi__{args.model.replace('/', '_')}__{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"benchmark = {args.benchmark}  ({len(rows)} rows)")
    print(f"out_dir   = {args.out_dir}")
    print(f"endpoint  = {args.base_url}  model={args.model}  n={args.n} temp={args.temperature}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    if not rows:
        print("no rows; exiting.")
        return 0

    (args.out_dir / "run_config.json").write_text(json.dumps(
        {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "timestamp": datetime.now().isoformat(timespec="seconds")}, indent=2))

    client = None if args.dry_run else make_client(args.base_url, args.api_key)
    out_records: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=(1 if args.dry_run else args.row_workers)) as ex:
        futs = {ex.submit(process_row, row, client, args): row for row in rows}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="row", desc="multi-select", dynamic_ncols=True):
            res = fut.result()
            if res is None:
                continue
            op = out_path_for(args.out_dir, futs[fut])
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
            out_records.append(res)

    if args.dry_run:
        plens = [r.get("_prompt_len", 0) for r in out_records]
        print(f"\n[dry-run] built {len(out_records)} prompts; "
              f"mean prompt len {sum(plens)//len(plens) if plens else 0} chars. No model calls.")
        return 0

    summary = corpus_summary(out_records)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ndone in {time.time()-t0:.0f}s -> {args.out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_step"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
