#!/usr/bin/env python
"""Extract 'unearned final-step reveal' selector calls into a parquet test set.

One row per offending hint call (the call that revealed the pool's FINAL step
unearned). Two kinds:
  jump_last     -- >=1 earlier step still unrevealed and <500 chars of student
                   text since the previous call (selector skipped ahead)
  spamwalk_last -- no skip (all earlier steps already revealed) but the student
                   wrote <500 chars before EVERY call (hint-collection walk)

Each row carries the exact selector inputs to replay the call:
  problem        -- create_kwargs.problem (what HintSelector.select received)
  trace_as_seen  -- what build_trace ACTUALLY produced in this run (hints-only;
                    assistant turns were never appended to agent_data.messages)
  trace_full     -- the INTENDED trace (student reasoning + injected hints,
                    reconstructed from the rollout output)
  hints_filtered -- pool after exclude_applied_steps(applied[:call]) (as seen)
  hints_full     -- original pool JSON
plus labels: picked_step_id (== last_step_id), expected_step_id (earliest
unrevealed step), kind, and rollout metadata.
"""
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "/share5/users/xutao.ma/project/hint_rl/script/hint_rl")
from hint_selector import exclude_applied_steps  # noqa: E402

LOGDIR = sys.argv[1]
PARQUET = sys.argv[2]
OUT = sys.argv[3]
SHORT = 500
PLACEHOLDER = "(The student has not written any reasoning yet.)"
ASSIST = "\nassistant\n"


def norm(s):
    return re.sub(r"\s+", "", s)


# ---------------- dataset meta ----------------
df = pd.read_parquet(PARQUET)
meta = {}
for _, row in df.iterrows():
    ei = row["extra_info"]
    user_base = ei.get("hprl_user_base") or ""
    if not user_base:
        for m in row["prompt"]:
            if m["role"] == "user":
                user_base = m["content"]
    ck = {}
    try:
        ck = ei["tools_kwargs"]["request_hint"]["create_kwargs"]
    except Exception:
        pass
    pool_str = ei.get("hint", "") or ""
    try:
        ids = [str(s.get("step_id")) for s in json.loads(pool_str).get("steps", [])]
    except Exception:
        ids = []
    meta[norm(user_base)] = {
        "problem_id": ei.get("problem_id", ""),
        "problem": ck.get("problem", user_base),
        "problem_base": user_base,
        "gt": str(row["reward_model"].get("ground_truth", "")),
        "pool": pool_str,
        "ids": ids,
    }
print(f"dataset rows: {len(df)}, unique problem keys: {len(meta)}")

files = sorted(glob.glob(os.path.join(LOGDIR, "rollouts", "*.jsonl")),
               key=lambda p: int(os.path.basename(p).split(".")[0]))


def parse_turns(output, n_calls):
    """-> (student_texts, hint_msgs): student text before each call k (0..n_calls-1),
    injected hint message after each call j (0..n_calls-1)."""
    segs = output.split("<hint_call/>")
    student, hint_msgs = [], []
    for k in range(n_calls):
        if k >= len(segs):
            return None, None
        seg = segs[k]
        if k == 0:
            student.append(seg)
        else:
            j = seg.find(ASSIST)
            if j == -1:
                return None, None
            student.append(seg[j + len(ASSIST):])
    for j in range(1, n_calls + 1):
        if j >= len(segs):
            return None, None
        seg = segs[j]
        if not seg.startswith("user\n"):
            return None, None
        e = seg.find(ASSIST)
        hint_msgs.append(seg[5:e] if e != -1 else seg[5:])
    return student, hint_msgs


rows = []
skipped = defaultdict(int)
for fp in files:
    tstep = int(os.path.basename(fp).split(".")[0])
    with open(fp) as f:
        for li, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                skipped["bad_json"] += 1
                continue
            hints = d.get("applied_hints") or []
            if not hints:
                continue
            if d.get("hint_call_failed", 0):
                skipped["had_failed_call"] += 1
                continue
            inp = d.get("input", "")
            u = inp.find("\nuser\n")
            a = inp.rfind("\nassistant\n")
            prob = inp[u + 6:a] if (u != -1 and a != -1) else inp
            prob = re.sub(r"\n*You have .{0,80} for this problem\.?\s*$", "", prob).strip()
            m = meta.get(norm(prob))
            if m is None or not m["ids"]:
                skipped["no_meta"] += 1
                continue
            ids = m["ids"]
            n_steps = len(ids)
            last_id = ids[-1]
            student, hint_msgs = parse_turns(d.get("output", ""), len(hints))
            if student is None:
                skipped["parse_fail"] += 1
                continue
            revealed = set()
            all_short = True
            for k, h in enumerate(hints):
                sid = str(h.get("major_step_id"))
                tlen = len(student[k])
                if tlen >= SHORT:
                    all_short = False
                if sid == last_id:
                    expected = next((s for s in ids if s not in revealed), last_id)
                    jump = expected != last_id and tlen < SHORT
                    spam = expected == last_id and all_short
                    if not (jump or spam):
                        break  # earned: don't extract; final step now revealed
                    kind = "jump_last" if jump else "spamwalk_last"
                    # selector inputs at this call
                    applied_prefix = hints[:k]
                    hints_filtered = exclude_applied_steps(m["pool"], applied_prefix)
                    seen_parts = [f"[hint given] {hint_msgs[j]}" for j in range(k)]
                    trace_as_seen = "\n\n".join(seen_parts).strip() or PLACEHOLDER
                    # intended trace: a0, [hint given] h0, a1, ..., ak (sentinel kept)
                    full_parts = []
                    for j in range(k + 1):
                        full_parts.append(student[j] + "<hint_call/>")
                        if j < k:
                            full_parts.append(f"[hint given] {hint_msgs[j]}")
                    trace_full = "\n\n".join(p.strip() for p in full_parts if p.strip())
                    try:
                        n_choices = len(json.loads(hints_filtered).get("steps", []))
                    except Exception:
                        n_choices = -1
                    dup_key = hashlib.md5(
                        (m["problem_id"] + "|" + str(k) + "|" + ",".join(sorted(revealed)) + "|" + kind).encode()
                    ).hexdigest()[:16]
                    rows.append({
                        "problem_id": m["problem_id"],
                        "train_step": tstep,
                        "rollout_index": li,
                        "kind": kind,
                        "call_index": k,
                        "ground_truth": m["gt"],
                        "acc": float(d.get("acc", 0)),
                        "hint_budget": int(d.get("hint_budget", 0)),
                        "num_hints_total": len(hints),
                        "problem": m["problem"],
                        "problem_base": m["problem_base"],
                        "trace_as_seen": trace_as_seen,
                        "trace_full": trace_full,
                        "hints_filtered": hints_filtered,
                        "hints_full": m["pool"],
                        "revealed_step_ids": json.dumps(sorted(revealed, key=lambda x: ids.index(x) if x in ids else 99)),
                        "picked_step_id": sid,
                        "expected_step_id": expected,
                        "last_step_id": last_id,
                        "n_steps": n_steps,
                        "n_choices_filtered": n_choices,
                        "pre_call_text_len": tlen,
                        "confidence_of_major_step": h.get("confidence_of_major_step"),
                        "injected_hint_body": h.get("hint") or "",
                        "dup_key": dup_key,
                    })
                    break  # at most one final-step reveal per rollout
                revealed.add(sid)

out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out_df.to_parquet(OUT, index=False)
print(f"wrote {len(out_df)} rows -> {OUT}")
print("skipped:", dict(skipped))
print("\nby kind:")
print(out_df["kind"].value_counts().to_string())
print("\nunique dup_keys:", out_df["dup_key"].nunique())
print("unique problems:", out_df["problem_id"].nunique())
print("\njump_last: n_choices_filtered distribution:")
print(out_df[out_df.kind == "jump_last"]["n_choices_filtered"].value_counts().sort_index().to_string())
print("\nacc by kind:")
print(out_df.groupby("kind")["acc"].mean().to_string())
