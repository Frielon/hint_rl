#!/usr/bin/env python3
"""Build a compact offset-index over a run's ``selector_calls/*.jsonl`` dump so the
rollout viewer can show, for each injected hint, the ORIGINAL selector response.

The dump is huge (tens of GB) and carries no step / rollout id, so we do NOT copy
it. We stream each shard once and, for every selector-call record, store only a
small row: a MATCH KEY plus the ``(file, byte-offset, length)`` needed to seek the
full record back out on demand, and a little metadata for the button label. The
viewer opens this sqlite read-only and seeks into the original shards when a hint's
"selector response" button is clicked -- so the 21 GB stays where it is.

Why a content-derived key? The rollout dump has NO ``request_id``, so a selector
record cannot be joined by id. But a record's ``messages`` ends with the EXACT
assistant turn that emitted ``<hint_call/>``, and that turn appears verbatim in the
rollout ``output``. Keying on ``problem_core`` + the student's assistant-turn trace
up to & including that calling turn pins the call (validated end-to-end: the
``problem_core`` recovered from ``selector.problem`` equals the rollout's, and the
trace pins the exact generation). The viewer recomputes the SAME key from a
rollout via ``rollout_viewer.selector_match_key`` -- single source of truth.

One call maps to MANY rollout rows (k-pack substitution shares one generated
trajectory across budget packs), and N GRPO rollouts of a problem can rarely share
an identical short first turn -- so a key maps to a LIST of records; the viewer
shows them all.

Usage:
  python tools/build_selector_index.py --run-dir logs/<EXP> [--jobs N]
    -> writes logs/<EXP>/selector_index.sqlite

Re-run after more rollouts/selector calls land to refresh it (cheap, idempotent;
the output is rebuilt from scratch).
"""
import argparse
import glob
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Single source of truth for _norm / _problem_core / the match key + its version,
# so the index and the viewer can never drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rollout_viewer as rv  # noqa: E402

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)
except Exception:  # pragma: no cover - orjson is present in the inference env
    import json

    def _loads(b):
        return json.loads(b)


# calls schema is shared verbatim by the per-worker shards and the merged DB so
# `INSERT INTO calls SELECT * FROM shard.calls` lines up column-for-column.
CALLS_COLS = (
    "mkey", "file_idx", "offset", "length", "call_index", "request_id",
    "strategy", "budget", "n_assistant", "select_latency_s", "has_selection",
    "selector_err",
)
CREATE_CALLS = (
    "CREATE TABLE calls (\n"
    "  mkey TEXT NOT NULL,\n"
    "  file_idx INTEGER NOT NULL,\n"
    "  offset INTEGER NOT NULL,\n"
    "  length INTEGER NOT NULL,\n"
    "  call_index INTEGER,\n"
    "  request_id TEXT,\n"
    "  strategy TEXT,\n"
    "  budget INTEGER,\n"
    "  n_assistant INTEGER,\n"
    "  select_latency_s REAL,\n"
    "  has_selection INTEGER,\n"
    "  selector_err TEXT\n"
    ")"
)
_INSERT_CALLS = f"INSERT INTO calls ({', '.join(CALLS_COLS)}) VALUES ({', '.join(['?'] * len(CALLS_COLS))})"
BATCH = 5000


def _index_one_file(args):
    """Worker: stream one selector shard, write its rows to a private shard DB.

    Tracks the byte offset of every line so the viewer can ``seek`` the full
    record back out. Returns (file_idx, abspath, shard_db_path, n_rows, n_err).
    """
    file_idx, path, shard_path, limit = args
    abspath = os.path.abspath(path)
    base = os.path.basename(path)
    if os.path.exists(shard_path):
        os.remove(shard_path)
    con = sqlite3.connect(shard_path)
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=OFF")
    con.execute(CREATE_CALLS)
    rows = []
    n = n_err = 0
    offset = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        for line in fh:
            ln = len(line)
            start = offset
            offset += ln
            if not line.strip():
                continue
            try:
                d = _loads(line)
            except Exception:
                n_err += 1
                continue
            msgs = d.get("messages") or []
            asst = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
            problem_core = rv._problem_core(d.get("problem", ""))
            mkey = rv.selector_match_key(problem_core, asst)
            n_asst = sum(1 for a in asst if rv._norm(a))
            sel = d.get("selection")
            rows.append((
                mkey, file_idx, start, ln, d.get("call_index"), d.get("request_id"),
                d.get("strategy"), d.get("budget"), n_asst, d.get("select_latency_s"),
                1 if isinstance(sel, dict) else 0, d.get("selector_err"),
            ))
            n += 1
            if len(rows) >= BATCH:
                con.executemany(_INSERT_CALLS, rows)
                rows.clear()
            if n % 200000 == 0:
                print(f"  [{base}] {n:,} records ({n / max(1e-9, time.time() - t0):.0f}/s)", flush=True)
            if limit and n >= limit:
                break
    if rows:
        con.executemany(_INSERT_CALLS, rows)
    con.commit()
    con.close()
    print(f"  [{base}] done: {n:,} records, {n_err} parse errors, {time.time() - t0:.0f}s", flush=True)
    return file_idx, abspath, shard_path, n, n_err


def build(run_dir, selector_dir, out_path, jobs, limit):
    files = sorted(glob.glob(os.path.join(selector_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl under {selector_dir}")
    print(f"indexing {len(files)} selector shard(s) from {selector_dir}", flush=True)
    jobs = jobs or len(files)
    work = [
        (i, p, f"{out_path}.part{i}", limit)
        for i, p in enumerate(files)
    ]

    t0 = time.time()
    results = []
    if jobs == 1:
        for w in work:
            results.append(_index_one_file(w))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_index_one_file, w) for w in work]
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=lambda r: r[0])

    # merge the shards into the final DB, then build the lookup index.
    if os.path.exists(out_path):
        os.remove(out_path)
    final = sqlite3.connect(out_path)
    final.execute("PRAGMA synchronous=OFF")
    final.execute("PRAGMA journal_mode=OFF")
    final.execute(CREATE_CALLS)
    final.execute("CREATE TABLE files (file_idx INTEGER PRIMARY KEY, path TEXT)")
    final.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    total = total_err = 0
    for file_idx, abspath, shard_path, n, n_err in results:
        final.execute("INSERT INTO files (file_idx, path) VALUES (?, ?)", (file_idx, abspath))
        final.execute("ATTACH DATABASE ? AS shard", (shard_path,))
        final.execute("INSERT INTO calls SELECT * FROM shard.calls")
        final.commit()
        final.execute("DETACH DATABASE shard")
        total += n
        total_err += n_err
    print(f"merged {total:,} rows; building mkey index ...", flush=True)
    final.execute("CREATE INDEX idx_calls_mkey ON calls (mkey)")
    meta = {
        "key_version": str(rv.SELECTOR_KEY_VERSION),
        "run_dir": os.path.abspath(run_dir),
        "selector_dir": os.path.abspath(selector_dir),
        "n_files": str(len(files)),
        "n_records": str(total),
        "n_parse_errors": str(total_err),
        "built_unix": str(int(time.time())),
    }
    final.executemany("INSERT INTO meta (k, v) VALUES (?, ?)", list(meta.items()))
    final.commit()
    final.execute("VACUUM")
    final.commit()
    final.close()
    for _, _, shard_path, _, _ in results:
        try:
            os.remove(shard_path)
        except OSError:
            pass
    size_mb = os.path.getsize(out_path) / 1e6
    print(
        f"wrote {out_path} ({size_mb:.0f} MB, {total:,} records, {total_err} parse errors) "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="experiment log dir (contains selector_calls/)")
    ap.add_argument("--selector-dir", default=None, help="override dir of selector_calls/*.jsonl (default <run-dir>/selector_calls)")
    ap.add_argument("--out", default=None, help="output sqlite path (default <run-dir>/selector_index.sqlite)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers (default: one per shard file)")
    ap.add_argument("--limit", type=int, default=0, help="cap records per file (debug)")
    args = ap.parse_args()
    run_dir = args.run_dir
    selector_dir = args.selector_dir or os.path.join(run_dir, "selector_calls")
    out_path = args.out or os.path.join(run_dir, rv.SELECTOR_INDEX_NAME)
    if not os.path.isdir(selector_dir):
        raise SystemExit(f"not a directory: {selector_dir}")
    build(run_dir, selector_dir, out_path, args.jobs, args.limit)


if __name__ == "__main__":
    main()
