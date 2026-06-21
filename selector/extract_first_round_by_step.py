#!/usr/bin/env python3
"""For the v3 run, sample first-round hint calls grouped by the selector's chosen
major step.

A "first-round hint call" is a selector-call record with ``call_index == 0`` (the
student's very first <hint_call/>); because we only log the first turn, every such
record has exactly one assistant turn. We bucket these by the selector's response
``selection.major_step_id`` and, for steps 1..5, reservoir-sample N records each.

For every kept record we store: the problem statement, the student's first-turn
reasoning trace, the selector's response, and the dataset problem_id (joined from
the training parquet via the budget-invariant problem-core text).

Output: one parquet per step under selector/dataset/.
"""
import argparse
import glob
import json
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rollout_viewer as rv  # noqa: E402


def build_problem_map(train_parquet):
    df = pd.read_parquet(train_parquet)
    core2id = {}
    for _, row in df.iterrows():
        prompt = row["prompt"]
        users = [m["content"] for m in prompt if m["role"] == "user"]
        if not users:
            continue
        core = rv._problem_core(users[-1])
        core2id[core] = row["extra_info"]["problem_id"]
    return core2id


def _int_step(ms):
    try:
        return int(ms)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(
        ROOT, "logs", "HPRL-Qwen2.5-7B-Instruct-dapo-4k-v3-20260614-045934"))
    ap.add_argument("--train-parquet", default=os.path.join(
        ROOT, "dataset", "dapo-3139-hint-verl-mt-clean.parquet"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "selector", "dataset"))
    ap.add_argument("--steps", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=100, help="samples per step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading training problem map ...", flush=True)
    core2id = build_problem_map(args.train_parquet)
    print(f"  {len(core2id)} unique problem cores", flush=True)

    shards = sorted(glob.glob(os.path.join(args.run_dir, "selector_calls", "*.jsonl")))
    print(f"streaming {len(shards)} shard(s)", flush=True)

    # reservoir sampling per step
    reservoir = {s: [] for s in steps}
    seen = {s: 0 for s in steps}
    n_ci0 = 0
    miss_pid = 0

    for shard in shards:
        base = os.path.basename(shard)
        n = 0
        with open(shard, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                if d.get("call_index") != 0:
                    continue
                sel = d.get("selection")
                if not isinstance(sel, dict):
                    continue
                step = _int_step(sel.get("major_step_id"))
                if step not in reservoir:
                    continue
                n_ci0 += 1
                seen[step] += 1
                core = rv._problem_core(d.get("problem", ""))
                pid = core2id.get(core)
                if pid is None:
                    miss_pid += 1
                rec = {
                    "problem_id": pid,
                    "problem": d.get("problem", ""),
                    "reasoning_trace": d.get("trace", ""),
                    "selector_response": d.get("selector_raw", ""),
                    "selection": json.dumps(sel, ensure_ascii=False),
                    "major_step_id": step,
                    "request_id": d.get("request_id"),
                    "budget": d.get("budget"),
                    "strategy": d.get("strategy"),
                    "call_index": d.get("call_index"),
                    "n_assistant": d.get("n_assistant_in_messages"),
                    "shard": base,
                }
                res = reservoir[step]
                if len(res) < args.n:
                    res.append(rec)
                else:
                    j = rng.randint(0, seen[step] - 1)
                    if j < args.n:
                        res[j] = rec
        print(f"  [{base}] {n:,} records; "
              f"counts so far: { {s: seen[s] for s in steps} }", flush=True)

    print(f"\ntotal call_index==0 in steps {steps}: {n_ci0:,}; "
          f"problem_id misses: {miss_pid}", flush=True)
    for s in steps:
        res = reservoir[s]
        out = os.path.join(args.out_dir, f"first_round_step{s}.parquet")
        df = pd.DataFrame(res)
        df.to_parquet(out, index=False)
        print(f"step {s}: seen={seen[s]:,} kept={len(res)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
