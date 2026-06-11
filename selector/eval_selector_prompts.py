#!/usr/bin/env python3
"""Replay `jump_last` selector calls against a live gpt-oss-20b server with
different prompt variants, and score how often each variant still unearned-
reveals the final (answer-bearing) step.

Test set: unearned_final_step_reveals.parquet (see its README). We use only
``kind == jump_last`` rows -- those are the calls where the selector had a real
choice (2-5 unrevealed steps) and wrongly picked the last one. ``spamwalk_last``
rows are forced picks under the current protocol and cannot distinguish prompts.

Per-row scoring (lower fail_last is better):
  * fail_last      picked step == last_step_id  -> the failure we're fixing
  * pick_expected  picked step == expected_step_id (earliest unrevealed step,
                   the no-extra-work reference: by construction the student
                   wrote <500 chars since the previous call)
  * other          some other (non-last) step -- not the reference pick, but
                   still avoids the answer reveal
  * parse_fail     completion yielded no scoreable selection

Rows are deduped by ``dup_key`` and sampled with a fixed seed, so every variant
sees the SAME inputs (trace_full differs per duplicate row, but each dup_key's
first row is deterministic). Results go to prompt_eval/<tag>/<variant>.jsonl +
summary.json.

Usage (server first: inference/serve_gpt_oss_20b.sh):
    python eval_selector_prompts.py --variants baseline,v1_earned --n 120
    python eval_selector_prompts.py --variants all --n 120 --tag round1
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_hint_selection_model import parse_output, make_client, one_sample  # noqa: E402
from prompt_variants import VARIANTS  # noqa: E402

PARQUET = os.path.join(HERE, "unearned_final_step_reveals.parquet")
OUT_ROOT = os.path.join(HERE, "prompt_eval")


def step_id_of(selection):
    """major_step_id as str, falling back to the integer part of hint_id."""
    if not isinstance(selection, dict):
        return None
    sid = selection.get("major_step_id")
    if sid is not None:
        return str(sid)
    hid = selection.get("hint_id")
    if hid is not None:
        return str(hid).split(".")[0]
    return None


def load_sample(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[df.kind == "jump_last"].drop_duplicates("dup_key")
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def score_row(row, parsed) -> str:
    sid = step_id_of(parsed)
    if sid is None:
        return "parse_fail"
    if sid == str(row.last_step_id):
        return "fail_last"
    if sid == str(row.expected_step_id):
        return "pick_expected"
    return "other"


def run_variant(name, variant, df, client, args, out_dir):
    t0 = time.time()
    records = [None] * len(df)

    def work(i):
        row = df.iloc[i]
        prompt = variant.build(row.problem, getattr(row, variant.trace_field),
                               row.hints_filtered)
        res = one_sample(client, args.model, prompt, args.temperature,
                         args.top_p, args.max_tokens)
        if res.get("error"):
            parsed, perr = None, res["error"]
        else:
            parsed, perr = parse_output(res.get("content") or "")
        return i, row, res, parsed, perr

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i) for i in range(len(df))]
        done = 0
        for fut in as_completed(futs):
            i, row, res, parsed, perr = fut.result()
            verdict = "api_error" if res.get("error") else score_row(row, parsed)
            records[i] = {
                "dup_key": row.dup_key,
                "problem_id": row.problem_id,
                "call_index": int(row.call_index),
                "n_choices_filtered": int(row.n_choices_filtered),
                "expected_step_id": str(row.expected_step_id),
                "last_step_id": str(row.last_step_id),
                "picked_step_id": step_id_of(parsed),
                "verdict": verdict,
                "confidence_of_major_step": (parsed or {}).get("confidence_of_major_step"),
                "finish_reason": res.get("finish_reason"),
                "completion_tokens": res.get("completion_tokens"),
                "latency_s": res.get("latency_s"),
                "parse_error": perr,
                "reasoning_of_major_step": (parsed or {}).get("reasoning_of_major_step"),
                "content": res.get("content"),
            }
            done += 1
            if done % 20 == 0 or done == len(df):
                print(f"  [{name}] {done}/{len(df)}", flush=True)

    with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    counts = {}
    for r in records:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    scoreable = n - counts.get("api_error", 0) - counts.get("parse_fail", 0)
    lat = [r["latency_s"] for r in records if r["latency_s"]]
    toks = [r["completion_tokens"] for r in records if r["completion_tokens"]]
    summary = {
        "variant": name,
        "description": variant.description,
        "trace_field": variant.trace_field,
        "n": n,
        "counts": counts,
        "fail_last_rate": round(counts.get("fail_last", 0) / max(scoreable, 1), 4),
        "pick_expected_rate": round(counts.get("pick_expected", 0) / max(scoreable, 1), 4),
        "scoreable": scoreable,
        "mean_latency_s": round(sum(lat) / max(len(lat), 1), 1),
        "mean_completion_tokens": round(sum(toks) / max(len(toks), 1)),
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"  [{name}] {summary['counts']}  fail_last={summary['fail_last_rate']:.1%} "
          f"expected={summary['pick_expected_rate']:.1%}  ({summary['wall_s']}s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all",
                    help="comma list of variant names, or 'all'")
    ap.add_argument("--n", type=int, default=120, help="rows per variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="round1", help="output dir prompt_eval/<tag>/")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--temperature", type=float, default=0.7)  # prod setting
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    for nm in names:
        if nm not in VARIANTS:
            sys.exit(f"unknown variant {nm!r}; have {list(VARIANTS)}")

    df = load_sample(args.n, args.seed)
    out_dir = os.path.join(OUT_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    client = make_client(args.base_url, "EMPTY")
    print(f"{len(df)} rows (seed={args.seed}), variants={names}, out={out_dir}")

    summaries = []
    for nm in names:
        print(f"== {nm}: {VARIANTS[nm].description}")
        summaries.append(run_variant(nm, VARIANTS[nm], df, client, args, out_dir))
        # merge into any existing summary so variants can be run incrementally
        spath = os.path.join(out_dir, "summary.json")
        prev = {}
        if os.path.exists(spath):
            with open(spath) as f:
                prev = {s["variant"]: s for s in json.load(f)["variants"]}
        for s in summaries:
            prev[s["variant"]] = s
        with open(spath, "w") as f:
            json.dump({"config": {k: v for k, v in vars(args).items()},
                       "variants": list(prev.values())}, f, indent=2)

    print("\n=== summary ===")
    for s in summaries:
        print(f"{s['variant']:15s} fail_last={s['fail_last_rate']:7.1%}  "
              f"expected={s['pick_expected_rate']:7.1%}  scoreable={s['scoreable']}/{s['n']}")


if __name__ == "__main__":
    main()
