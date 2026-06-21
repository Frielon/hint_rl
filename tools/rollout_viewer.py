#!/usr/bin/env python3
"""Web viewer for rollout logs.

Browse:  experiment  ->  step file  ->  problem  ->  rollouts.

A "problem" is identified by its bare statement — the `input` prompt with the
trailing hint-budget reminder stripped (budget-invariant md5). This groups a
problem's k-pack packs — the same problem rolled out at budgets B, B-1, ... within
one step (see plot_tool/budget_sankey.py) — into ONE entry instead of k near-
duplicates. Within a problem the rollouts are ordered by budget descending (pack B
first, then B-1, ...), each under a budget header. Grouping is on the fly, so this
works on raw `rollouts/` files as well as `*_grouped/` ones; non-kpack runs (one
budget per problem) are unchanged.

Each problem page carries "← prev step / next step →" buttons that jump to the
previous/next step file in which the SAME problem (its budget-invariant key) recurs
-- useful for following one problem across epochs, since adjacent steps share no
problems. This needs every step's problem set, so the run is indexed once into a
`<exp>/.<sub>_problem_steps.json` sidecar (built in the background with live progress,
updated incrementally as new steps land); see `api_problem_steps`.

Multi-turn HPRL rollouts are rendered turn-by-turn: the model emits the
`<hint_call/>` sentinel, and the system injects a curated hint as the next user
turn. The chat template strips `<|im_start|>` / `<|im_end|>`, leaving bare
`system` / `user` / `assistant` role lines, which this viewer parses back into
turns. The new per-rollout HPRL fields (num_hints, hint_budget, hint_penalty,
hint_call_failed, applied_hints) are surfaced in the rollout header.

Selector responses: each injected hint carries a "🔍 selector response" button that
reveals the ORIGINAL frozen-selector call that produced it -- the student trace and
candidate hints the selector saw, its parsed pick (major step, chosen hint, the
reasoning + confidence), and its raw output. This is opt-in: it lights up only when
the run's selector dump has been indexed into `<exp>/selector_index.sqlite` by
`tools/build_selector_index.py` (the dump itself, tens of GB, is never copied --
the button seeks the one record out on demand). The rollout dump has no request_id,
so the link is a content key (problem + assistant-turn trace through the calling
turn); see `selector_match_key`, which the indexer reuses so the keys can't drift.

Run (conda env `inference` has Flask):
  /shared_home/xutao.ma/miniconda3/envs/inference/bin/python \
      tools/rollout_viewer.py --logs-dir logs --port 8000

Then open http://<host>:8000  (use --host 0.0.0.0 for remote access).
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
from collections import Counter, defaultdict

from flask import Flask, abort, jsonify, render_template_string, request

app = Flask(__name__)
LOGS_DIR = os.path.abspath("logs")
ROLLOUT_SUBDIRS = ("rollouts", "rollouts_grouped", "val_rollouts", "val_rollouts_grouped")

# Source datasets: let us map a rollout's wrapped prompt back to its dataset
# problem_id / index / hint. Different runs use different hint datasets (e.g. the
# simplified multi-turn variant vs the original), whose hint format and system
# prompt differ; we load several candidates and pick, per rollout, the one whose
# system base matches. Overridable via --dataset / ROLLOUT_DATASET (comma-sep).
DATASET_PATHS = [p for p in os.environ.get(
    "ROLLOUT_DATASET",
    "dataset/dapo-3740-hint-verl-simplified-mt.parquet,dataset/dapo-3740-hint-verl.parquet",
).split(",") if p.strip()]
_DS = None  # cached list of dataset records, or False if unavailable
_DS_REPORT = []  # human-readable load status, printed at startup
# repo root (…/hint_rl), so relative dataset paths work regardless of launch cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WS = re.compile(r"\s+")
# boilerplate wrapped around the pure problem statement on each side
_DS_SUFFIX = re.compile(r"\s*Let's think step by step and output the final answer within \\boxed\{\}\.?\s*$")
# the rollout's trailing hint-budget reminder, in BOTH its forms: the ">=1 calls"
# form ("You have N hint calls remaining for this problem.") and the budget-0 form
# ("You have no hint calls remaining for this problem, so finish on your own.").
# Stripping it makes the recovered statement budget-invariant, so a problem's k-pack
# packs (budgets B, B-1, ...) share one identity.
_RO_SUFFIX = re.compile(
    r"\s*You have (?:\d+|no) hint calls? remaining for this problem"
    r"(?:, so finish on your own)?\.?\s*$")
_HINT_INSTR = "A hint tool is available"  # start of the HPRL hint-instruction block
_PREFIX_K = 64  # bucket key length for the prefix-match fallback

HINT_CALL = "<hint_call/>"
# bare role lines left after the chat template strips <|im_start|>/<|im_end|>
_LEAD_MARKERS = ("system\n", "user\n", "assistant\n")
_ROLE_MARKERS = (("\nassistant\n", "assistant"), ("\nuser\n", "user"), ("\nsystem\n", "system"))

# ---- selector-call linking ------------------------------------------------- #
# Each injected hint can be traced back to the ORIGINAL frozen-selector call that
# produced it. The rollout dump has no request_id, so we content-join: the key is
# the problem statement plus the student's assistant-turn trace up to & including
# the turn that emitted <hint_call/>. build_selector_index.py precomputes the same
# key over selector_calls/*.jsonl into <exp>/selector_index.sqlite; here we
# recompute it from a rollout's output turns and look the call up. Bump the version
# if the key recipe changes so a stale index is ignored instead of mis-joining.
SELECTOR_KEY_VERSION = 1
SELECTOR_INDEX_NAME = "selector_index.sqlite"


def selector_match_key(problem_core: str, assistant_texts) -> str:
    """Content key joining a rollout's injected hint to its selector-call record.

    ``problem_core`` is the budget-invariant statement (``_problem_core``); the
    selector dump recovers the identical core from its own ``problem`` field.
    ``assistant_texts`` is the ordered list of assistant turns up to & including
    the one that emitted ``<hint_call/>`` -- the selector saw exactly these (its
    ``messages`` ends on that calling turn). Whitespace is normalized per turn so a
    full-sequence decode (rollout output) and a per-turn decode (selector messages)
    hash alike. Empty turns are dropped on both sides for the same reason."""
    h = hashlib.md5()
    h.update(problem_core.encode("utf-8"))
    for a in assistant_texts:
        na = _norm(a)
        if not na:
            continue
        h.update(b"\x1e")
        h.update(na.encode("utf-8"))
    return h.hexdigest()


def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def _problem_core(user_text: str) -> str:
    """Recover the pure problem statement: drop the dataset's trailing
    instruction and the rollout's hint-budget line, then normalize whitespace."""
    return _norm(_DS_SUFFIX.sub("", _RO_SUFFIX.sub("", user_text or "")))


def problem_group_key(rollout_input: str) -> str:
    """Budget-invariant problem id used to group a step file's rollouts.

    A problem's k-pack packs differ ONLY in the trailing hint-budget reminder
    (`You have N hint calls remaining...`), so hashing the raw `input` (as the old
    grouping did) split one problem into k separate list entries. We isolate the
    last user turn, strip the budget line via `_problem_core`, and hash the bare
    statement so all packs of a problem collapse into ONE entry — matching the
    budget-invariant key in plot_tool/budget_sankey.py. Non-kpack runs have a single
    budget per problem, so grouping is unchanged. Within one step file the system
    template is constant, so the question text alone identifies the problem."""
    return problem_hash(_input_problem_core(rollout_input))


def _input_problem_core(rollout_input: str) -> str:
    """Recover a rollout prompt's bare, budget-invariant problem statement -- the
    last user turn with the dataset suffix + hint-budget line stripped. Shared by
    the group key and the selector-call match key so both see the same statement."""
    user = rollout_input.rsplit("\nuser\n", 1)[-1].rsplit("\nassistant", 1)[0]
    return _problem_core(user)


def _system_base(system_text: str, hprl_base) -> str:
    """The clean system base (no hint-instruction block), used to tell which
    dataset a rollout's prompt came from."""
    if hprl_base:
        return _norm(hprl_base)
    return _norm((system_text or "").split(_HINT_INSTR)[0])


def _first_role(prompt, role: str) -> str:
    for m in prompt:
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def _resolve_dataset_path(raw_path: str):
    """Resolve a (possibly relative) dataset path against cwd then the repo root."""
    raw = raw_path.strip()
    for cand in (raw, os.path.join(_REPO_ROOT, raw)):
        cand = os.path.abspath(cand)
        if os.path.isfile(cand):
            return cand
    return None


def _load_datasets():
    """Build (and cache) one lookup record per candidate dataset. Each record is
    {path, sys_base, exact, by_prefix}. Returns None if none are available."""
    global _DS, _DS_REPORT
    if _DS is not None:
        return _DS or None
    _DS_REPORT = []
    try:
        import pandas as pd
    except Exception as e:
        _DS_REPORT.append(f"pandas unavailable ({e}); no dataset problem ids / hints will be shown")
        _DS = False
        return None
    records = []
    for raw_path in DATASET_PATHS:
        path = _resolve_dataset_path(raw_path)
        if not path:
            _DS_REPORT.append(f"NOT FOUND: {raw_path} (looked in cwd and {_REPO_ROOT})")
            continue
        df = pd.read_parquet(path)
        exact = {}
        by_prefix = defaultdict(list)
        sys_base = None
        for _, r in df.iterrows():
            ei = r.get("extra_info") or {}
            if sys_base is None:
                sys_base = _system_base(_first_role(r["prompt"], "system"), ei.get("hprl_system_base"))
            core = _problem_core(_first_role(r["prompt"], "user"))
            idx = ei.get("index")
            info = {
                "ds_index": int(idx) if idx is not None else None,
                "ds_problem_id": ei.get("problem_id"),
                "hint": ei.get("hint"),            # injected hints (dict or list)
                "hint_full": ei.get("hint_full"),  # full reference hints (list), if present
            }
            exact.setdefault(core, info)
            by_prefix[core[:_PREFIX_K]].append((core, info))
        records.append({"path": path, "sys_base": sys_base, "exact": exact, "by_prefix": dict(by_prefix)})
        _DS_REPORT.append(f"loaded {os.path.basename(path)} ({len(df)} problems)")
    if not records:
        _DS_REPORT.append("no datasets loaded; problem ids / hint sets will be hidden")
    _DS = records or False
    return _DS or None


def _lookup_core(rec, core: str):
    if core in rec["exact"]:
        return rec["exact"][core]
    cands = rec["by_prefix"].get(core[:_PREFIX_K], [])
    related = {}
    for c, info in cands:
        if c.startswith(core) or core.startswith(c):
            related[info["ds_problem_id"]] = info
    if len(related) == 1:
        return next(iter(related.values()))
    if len(cands) == 1:
        return cands[0][1]
    return None


def match_dataset_problem(rollout_input: str):
    """Map a rollout's wrapped `input` to its dataset entry.

    The rollout prompt strips the dataset's trailing instruction and may append a
    hint-budget line; we recover the pure problem statement and match it. When
    several datasets contain the same problem (e.g. an original and a simplified
    variant), the dataset whose system base prefixes the rollout's system prompt
    is preferred so the hint/budget shown matches the run that produced it."""
    records = _load_datasets()
    if not records:
        return None
    turns = split_turns(rollout_input, start_role="system")
    sys_txt = _norm(next((t["text"] for t in turns if t["role"] == "system"), ""))
    users = [t["text"] for t in turns if t["role"] == "user"]
    if not users:
        return None
    core = _problem_core(users[-1])
    # try datasets whose system base matches the rollout first
    ordered = sorted(records, key=lambda r: 0 if (r["sys_base"] and sys_txt.startswith(r["sys_base"])) else 1)
    for rec in ordered:
        info = _lookup_core(rec, core)
        if info:
            return info
    return None


def split_turns(text: str, start_role: str = "assistant"):
    """Split a flattened chat string into [{role, text, hint_call}] turns.

    A `<hint_call/>` immediately followed by a `user\\n` marker ends an assistant
    turn (the sentinel is kept visible) and starts the injected-hint user turn.
    Plain `\\nrole\\n` markers switch turns too; a leading bare role line (as in
    the `input` prompt, which begins with `system\\n`) sets the first role.
    """
    if not text:
        return []
    role = start_role
    i = 0
    n = len(text)
    for lead in _LEAD_MARKERS:
        if text.startswith(lead):
            role = lead[:-1]
            i = len(lead)
            break
    turns = []
    buf = i
    while i < n:
        if text.startswith(HINT_CALL, i) and text.startswith("user\n", i + len(HINT_CALL)):
            seg = text[buf:i] + HINT_CALL
            turns.append({"role": role, "text": seg.rstrip("\n"), "hint_call": True})
            i += len(HINT_CALL) + len("user\n")
            buf = i
            role = "user"
            continue
        matched = False
        for marker, next_role in _ROLE_MARKERS:
            if text.startswith(marker, i):
                turns.append({"role": role, "text": text[buf:i], "hint_call": False})
                i += len(marker)
                buf = i
                role = next_role
                matched = True
                break
        if matched:
            continue
        i += 1
    tail = text[buf:]
    turns.append({"role": role, "text": tail, "hint_call": HINT_CALL in tail})
    return [t for t in turns if t["text"].strip()]


def safe_join(base: str, *parts: str) -> str:
    """Join and ensure the result stays within base (no path traversal)."""
    path = os.path.abspath(os.path.join(base, *parts))
    if path != base and not path.startswith(base + os.sep):
        abort(400, "invalid path")
    return path


def read_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _num(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _step_sort_key(filename: str):
    """Numeric-step ordering for step files ('100.jsonl' < '101.jsonl'), with a
    lexical fallback for non-numeric stems. Shared by the step list and the
    problem-step index so 'next/prev step' agrees with the dropdown order."""
    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    return (0, int(stem)) if stem.isdigit() else (1, stem)


# ---------------------------------------------------------------- selector index
# Per-injected-hint link to the original selector call, via the sqlite offset index
# build_selector_index.py writes to <exp>/selector_index.sqlite. The DB is tiny
# (key + file/offset/length + light metadata); the heavy record is seeked out of
# the original selector_calls/*.jsonl on demand, so the 21 GB dump is not copied.

_SEL_INDEX_OK = {}  # exp -> bool: index present AND key-version compatible


def _open_selector_db(exp: str):
    """Open a run's selector index read-only, or None if absent/incompatible.

    A fresh connection per call keeps this thread-safe under Flask's threaded
    server; buttons are clicked rarely so the open cost is irrelevant. A
    key-version mismatch (stale index vs a changed key recipe) is treated as
    absent so we never mis-join."""
    path = safe_join(LOGS_DIR, exp, SELECTOR_INDEX_NAME)
    if not os.path.isfile(path):
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        con.row_factory = sqlite3.Row
        ver = con.execute("SELECT v FROM meta WHERE k='key_version'").fetchone()
    except sqlite3.Error:
        return None
    if not ver or str(ver["v"]) != str(SELECTOR_KEY_VERSION):
        con.close()
        return None
    return con


def _selector_index_ok(exp: str) -> bool:
    if exp not in _SEL_INDEX_OK:
        con = _open_selector_db(exp)
        _SEL_INDEX_OK[exp] = con is not None
        if con is not None:
            con.close()
    return _SEL_INDEX_OK[exp]


def _read_selector_record(path: str, offset: int, length: int):
    """Seek the full selector-call record back out of its original shard."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(length)
        return json.loads(raw)
    except (OSError, ValueError):
        return None


def annotate_selector_keys(problem_core: str, turns):
    """Tag each genuinely injected-hint (user) turn with its selector_match_key.

    Walk a rollout's output turns accumulating assistant turns; the key over the
    assistant trace SO FAR -- through the turn that emitted ``<hint_call/>`` -- is
    the join into the selector index. A key is set ONLY on a user turn that
    immediately follows a ``hint_call`` assistant turn: that is exactly a real
    injected hint (a served hint, or the "no hint available" message after a
    selector failure -- both produce a dumped selector call). A user turn split out
    of the model's OWN text by a stray literal ``user\\n`` (an early-training
    artifact, ``hint_call`` False) is not a real call and gets no button."""
    acc = []
    prev_hint_call = False
    for t in turns:
        role = t.get("role")
        if role == "assistant":
            acc.append(t.get("text", ""))
            prev_hint_call = bool(t.get("hint_call"))
        elif role == "user":
            if prev_hint_call:
                t["sel_key"] = selector_match_key(problem_core, acc)
            prev_hint_call = False
        else:
            prev_hint_call = False
    return turns


# ---------------------------------------------------------------- API

@app.get("/api/exps")
def api_exps():
    exps = []
    for name in sorted(os.listdir(LOGS_DIR)):
        d = os.path.join(LOGS_DIR, name)
        if not os.path.isdir(d):
            continue
        subs = [s for s in ROLLOUT_SUBDIRS if os.path.isdir(os.path.join(d, s))]
        if subs:
            exps.append({"name": name, "subdirs": subs})
    return jsonify(exps)


@app.get("/api/steps")
def api_steps():
    exp = request.args["exp"]
    sub = request.args["sub"]
    subdir = safe_join(LOGS_DIR, exp, sub)
    if not os.path.isdir(subdir):
        abort(404)
    files = [f for f in os.listdir(subdir) if f.endswith(".jsonl")]
    return jsonify(sorted(files, key=_step_sort_key))  # numeric step order


@app.get("/api/problems")
def api_problems():
    exp = request.args["exp"]
    sub = request.args["sub"]
    step = request.args["step"]
    path = safe_join(LOGS_DIR, exp, sub, step)
    if not os.path.isfile(path):
        abort(404)
    groups = {}
    order = []
    for d in read_jsonl(path):
        h = problem_group_key(d["input"])  # budget-invariant: merges k-pack packs
        if h not in groups:
            ds = match_dataset_problem(d["input"]) or {}
            groups[h] = {
                "problem_id": h,
                "pid": d.get("pid"),
                "ds_index": ds.get("ds_index"),
                "ds_problem_id": ds.get("ds_problem_id"),
                "gts": d.get("gts"),
                "count": 0,
                "n_correct": 0,
                "sum_hints": 0.0,
                "n_hint_failed": 0,
                "pack_budgets": Counter(),
                "preview": d["input"][-300:],
            }
            order.append(h)
        g = groups[h]
        g["count"] += 1
        if _num(d, "acc") > 0:
            g["n_correct"] += 1
        g["sum_hints"] += _num(d, "num_hints")
        if _num(d, "hint_call_failed") > 0:
            g["n_hint_failed"] += 1
        g["pack_budgets"][int(_num(d, "hint_budget"))] += 1
    out = []
    for h in order:
        g = groups[h]
        g["avg_hints"] = round(g["sum_hints"] / g["count"], 2) if g["count"] else 0.0
        g.pop("sum_hints", None)
        # distinct budgets this problem ran at this step, B first. >1 == it was split
        # into k-pack probe packs (cf. budget_sankey / hint_budget_callback).
        g["pack_budgets"] = sorted(g["pack_budgets"], reverse=True)
        out.append(g)
    return jsonify(out)


@app.get("/api/rollouts")
def api_rollouts():
    exp = request.args["exp"]
    sub = request.args["sub"]
    step = request.args["step"]
    pid = request.args["problem_id"]
    path = safe_join(LOGS_DIR, exp, sub, step)
    if not os.path.isfile(path):
        abort(404)
    matches = [d for d in read_jsonl(path) if problem_group_key(d["input"]) == pid]
    problem_input = input_turns = ds_match = None
    pack_budgets = []
    problem_core = ""
    if matches:
        # canonical prompt = the highest-budget pack (the assigned budget B); the forced
        # probe packs B-1,... frame the same problem with a lower budget reminder.
        canon = max(matches, key=lambda d: _num(d, "hint_budget"))
        problem_input = canon["input"]
        input_turns = split_turns(canon["input"], start_role="system")
        ds_match = match_dataset_problem(canon["input"])
        # budget-invariant statement -> the selector-call match key (same core the
        # selector dump recovers from its own `problem`). One per problem group.
        problem_core = _input_problem_core(canon["input"])
        # order rollouts B first, then B-1, ... (stable sort keeps file order within a budget)
        matches.sort(key=lambda d: -_num(d, "hint_budget"))
        pack_budgets = sorted({int(_num(d, "hint_budget")) for d in matches}, reverse=True)
    out = []
    for d in matches:
        turns = split_turns(d.get("output", ""), start_role="assistant")
        annotate_selector_keys(problem_core, turns)  # tag injected-hint turns
        out.append({
            "turns": turns,
            "score": d.get("score"),
            "acc": d.get("acc"),
            "reward": d.get("reward"),
            "pred": d.get("pred"),
            "gts": d.get("gts"),
            "has_format": d.get("has_format"),
            "num_hints": d.get("num_hints"),
            "hint_budget": d.get("hint_budget"),
            "hint_penalty": d.get("hint_penalty"),
            "hint_call_failed": d.get("hint_call_failed"),
            "applied_hints": d.get("applied_hints"),
        })
    def _parse_json(raw):
        if raw is None or isinstance(raw, (dict, list)):
            return raw
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    ds_info = None
    if ds_match:
        ds_info = {
            "ds_index": ds_match.get("ds_index"),
            "ds_problem_id": ds_match.get("ds_problem_id"),
            "hint": _parse_json(ds_match.get("hint")),
            "hint_full": _parse_json(ds_match.get("hint_full")),
        }
    return jsonify({
        "input": problem_input,
        "input_turns": input_turns,
        "dataset": ds_info,
        "pack_budgets": pack_budgets,  # distinct budgets, B first (>1 == k-packed)
        "has_selector_index": _selector_index_ok(exp),  # show "selector response" buttons?
        "rollouts": out,
    })


@app.get("/api/selector_call")
def api_selector_call():
    """Resolve an injected hint's match key to its original selector call(s).

    Looks the key up in the run's selector index, then seeks each matching record
    out of the original selector_calls/*.jsonl. Usually one match; k-pack /
    duplicate-trajectory GRPO rollouts can yield a few, all returned (capped)."""
    exp = request.args["exp"]
    key = request.args["key"]
    con = _open_selector_db(exp)
    if con is None:
        return jsonify({"calls": [], "total": 0, "error": "no selector index for this run"})
    try:
        rows = con.execute(
            "SELECT file_idx, offset, length FROM calls WHERE mkey=? ORDER BY n_assistant, call_index",
            (key,),
        ).fetchall()
        files = {r["file_idx"]: r["path"] for r in con.execute("SELECT file_idx, path FROM files")}
    finally:
        con.close()
    MAX = 12
    calls = []
    for r in rows[:MAX]:
        path = files.get(r["file_idx"])
        rec = _read_selector_record(path, r["offset"], r["length"]) if path else None
        if rec is None:
            continue
        sel = rec.get("selection") if isinstance(rec.get("selection"), dict) else None
        calls.append({
            "request_id": rec.get("request_id"),
            "call_index": rec.get("call_index"),
            "strategy": rec.get("strategy"),
            "budget": rec.get("budget"),
            "select_latency_s": rec.get("select_latency_s"),
            "selector_err": rec.get("selector_err"),
            "applied_step_ids_before": rec.get("applied_step_ids_before"),
            "n_assistant_in_messages": rec.get("n_assistant_in_messages"),
            "problem": rec.get("problem"),
            "trace": rec.get("trace"),
            "candidate_hints_str": rec.get("candidate_hints_str"),
            "selection": sel,
            "selector_raw": rec.get("selector_raw"),
        })
    return jsonify({"calls": calls, "total": len(rows), "truncated": len(rows) > len(calls)})


# ---------------------------------------------------------------- problem-step index
# "Jump to the next/previous step where this problem appears." Adjacent training
# steps share NO problems (each step samples a fresh batch); a problem recurs only
# when the dataset wraps, ~once per epoch, so finding its other steps means knowing
# every step file's problem set. We build that once per (exp, sub): a map
# {problem_group_key -> [step files]}, persisted to a sidecar in the exp dir and
# updated incrementally as new step files land. The first build streams every step
# file (tens of seconds for a long run), so it runs in a background thread with live
# progress; navigation buttons poll until it is ready. Bump PROB_INDEX_VERSION if the
# problem key recipe changes so a stale sidecar is rebuilt instead of mis-joining.

PROB_INDEX_VERSION = 1
PROB_STEPS_SUFFIX = "_problem_steps.json"  # sidecar name: .<sub>_problem_steps.json

_PROB_IDX = {}            # (exp, sub) -> build-state dict (see api_problem_steps)
_PROB_IDX_LOCK = threading.Lock()  # guards check-and-start so we build each key once


def _prob_sidecar_path(exp: str, sub: str) -> str:
    return safe_join(LOGS_DIR, exp, "." + sub + PROB_STEPS_SUFFIX)


def _file_sig(path: str):
    """Cheap change-signature for a step file: [size, mtime]. Step files are written
    once per step, so this distinguishes a new/rewritten file from an unchanged one."""
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime)]


def _current_step_sigs(subdir: str):
    sigs = {}
    for f in os.listdir(subdir):
        if f.endswith(".jsonl"):
            try:
                sigs[f] = _file_sig(os.path.join(subdir, f))
            except OSError:
                pass
    return sigs


def _pids_in_file(path: str):
    """Stream a step file and return the set of budget-invariant problem keys in it
    (the same key /api/problems and /api/rollouts use). Streaming keeps memory flat;
    a partial last line in a step still being written is skipped, not fatal."""
    pids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            inp = d.get("input")
            if inp:
                pids.add(problem_group_key(inp))
    return pids


def _load_prob_sidecar(path: str):
    """Return (index, covered_sigs) from the sidecar, or ({}, {}) if absent or from
    an incompatible key version (treated as absent so we rebuild, never mis-join)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    if data.get("version") != PROB_INDEX_VERSION:
        return {}, {}
    return data.get("index") or {}, data.get("files") or {}


def _save_prob_sidecar(path: str, index, files):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"version": PROB_INDEX_VERSION, "files": files, "index": index}, f)
        os.replace(tmp, path)
    except OSError:
        pass  # read-only logs dir: keep the in-memory index, just don't persist


def _build_prob_index(exp: str, sub: str, entry: dict):
    """Background worker: bring (exp, sub)'s problem->steps index up to date.

    Reuses the sidecar and scans only step files not already covered; a previously
    covered file whose signature changed (rare: a rewritten step) forces a full
    rebuild so stale entries can't linger. Updates entry['progress'] as it goes."""
    try:
        subdir = safe_join(LOGS_DIR, exp, sub)
        sidecar = _prob_sidecar_path(exp, sub)
        index, covered = _load_prob_sidecar(sidecar)
        cur = _current_step_sigs(subdir)
        if any(n not in cur or cur[n] != sig for n, sig in covered.items()):
            index, covered = {}, {}  # a covered file vanished/changed -> rebuild clean
        todo = sorted((n for n in cur if n not in covered), key=_step_sort_key)
        entry["progress"] = [0, len(todo)]
        for i, name in enumerate(todo):
            for p in _pids_in_file(os.path.join(subdir, name)):
                index.setdefault(p, []).append(name)
            covered[name] = cur[name]
            entry["progress"] = [i + 1, len(todo)]
        for p, lst in index.items():
            index[p] = sorted(set(lst), key=_step_sort_key)
        entry["index"] = index
        entry["files"] = covered
        entry["state"] = "ready"
        _save_prob_sidecar(sidecar, index, covered)
    except Exception as e:  # never leave a wedged thread; next request retries
        entry["error"] = repr(e)
        entry["state"] = "error"


@app.get("/api/problem_steps")
def api_problem_steps():
    """All step files in this run that contain the given problem, in step order.

    The client picks the prev/next neighbour of the step it is viewing. Returns
    {status:'indexing', done, total} while the (cached, incremental) index builds,
    {status:'ready', steps:[...]} once done, or {status:'error', error}. A cheap
    signature probe each call re-triggers an incremental build when new step files
    appear, so a run still training stays current."""
    exp = request.args["exp"]
    sub = request.args["sub"]
    pid = request.args["problem_id"]
    subdir = safe_join(LOGS_DIR, exp, sub)
    if not os.path.isdir(subdir):
        abort(404)
    cur = _current_step_sigs(subdir)
    key = (exp, sub)
    with _PROB_IDX_LOCK:
        entry = _PROB_IDX.get(key)
        need_build = (
            entry is None
            or entry["state"] == "error"
            or (entry["state"] == "ready" and entry["files"] != cur)  # new/changed steps
        )
        if need_build:
            entry = {"state": "building", "progress": [0, 0],
                     "index": {}, "files": {}, "error": None}
            _PROB_IDX[key] = entry
            threading.Thread(target=_build_prob_index, args=(exp, sub, entry),
                             daemon=True).start()
        state = entry["state"]
        progress = list(entry["progress"])
        steps = entry["index"].get(pid, []) if state == "ready" else []
    if state == "building":
        done, total = progress
        return jsonify({"status": "indexing", "done": done, "total": total})
    if state == "error":
        return jsonify({"status": "error", "error": entry["error"]})
    return jsonify({"status": "ready", "steps": steps})


# ---------------------------------------------------------------- UI

PAGE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Rollout Viewer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         color: #1c2330; background: #f4f6fa; }
  header { background: #1f2a44; color: #fff; padding: 10px 16px; font-weight: 600; }
  .bars { display: flex; gap: 10px; padding: 10px 16px; flex-wrap: wrap;
          background: #fff; border-bottom: 1px solid #e0e4ea; }
  .ctrl { display: flex; flex-direction: column; font-size: 12px; gap: 3px; }
  .ctrl label { color: #66708a; }
  select { padding: 6px 8px; min-width: 220px; border: 1px solid #c7cdda;
           border-radius: 6px; background: #fff; font-size: 13px; }
  .layout { display: flex; height: calc(100vh - 110px); }
  #problems { width: 320px; overflow-y: auto; border-right: 1px solid #e0e4ea;
              background: #fff; }
  .pitem { padding: 8px 12px; border-bottom: 1px solid #eef0f4; cursor: pointer;
           font-size: 12px; }
  .pitem:hover { background: #eef3ff; }
  .pitem.active { background: #dde7ff; }
  .pitem.all-ok { border-left: 4px solid #1e8e3e; background: #f1fbf3; }
  .pitem.all-ok:hover { background: #e3f7e8; }
  .pitem.all-fail { border-left: 4px solid #c5221f; background: #fdf3f3; }
  .pitem.all-fail:hover { background: #fbe8e8; }
  .pitem.all-ok.active, .pitem.all-fail.active { background: #dde7ff; }
  .pitem .meta { color: #66708a; margin-top: 2px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 10px;
           font-size: 11px; font-weight: 600; }
  .ok { background: #d8f5dd; color: #137333; }
  .bad { background: #fde0e0; color: #b3261e; }
  .hintb { background: #fff0d6; color: #9a6700; }
  .failb { background: #f3e0ff; color: #7a3ea8; }
  .dsb { background: #e3ecff; color: #2747a8; font-family: ui-monospace, monospace; }
  .packb { background: #e3ecff; color: #2747a8; font-family: ui-monospace, monospace; }
  #main { flex: 1; overflow-y: auto; padding: 16px; }
  .dsid { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .dsid-pid { font-family: ui-monospace, monospace; font-size: 12px; color: #66708a; }
  .prompt { margin-bottom: 16px; }
  .roll { background: #fff; border: 1px solid #e0e4ea; border-radius: 8px;
          margin-bottom: 12px; }
  .roll-head { display: flex; gap: 14px; align-items: center; padding: 8px 12px;
               border-bottom: 1px solid #eef0f4; font-size: 12px; flex-wrap: wrap; }
  .roll-head .k { color: #66708a; }
  .roll-body { padding: 12px; }
  details.pack { margin: 16px 0 0; }
  details.pack > summary.packhdr { cursor: pointer; margin: 0 0 8px; padding: 6px 12px;
             font-size: 12px; font-weight: 700; color: #2747a8; background: #e9f0ff;
             border-left: 4px solid #2747a8; border-radius: 4px; }
  details.pack > summary.packhdr:hover { background: #dde7ff; }
  .hints-meta { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 12px 0; }
  .hchip { font-size: 11px; background: #fff6e6; border: 1px solid #f0d9a8;
           color: #8a5a00; border-radius: 6px; padding: 2px 7px; }
  .turn { border-radius: 8px; margin-bottom: 8px; overflow: hidden;
          border: 1px solid #e0e4ea; }
  .turn-label { font-size: 10px; font-weight: 700; text-transform: uppercase;
                letter-spacing: .05em; padding: 3px 10px; }
  .turn-body { padding: 10px 12px; white-space: pre-wrap;
               font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.5; }
  .turn-asst { border-color: #d4dcf0; }
  .turn-asst .turn-label { background: #eef2fc; color: #3b5bbf; }
  .turn-hint { border-color: #f0d9a8; }
  .turn-hint .turn-label { background: #fff0d6; color: #9a6700; }
  .turn-hint .turn-body { background: #fffaf0; }
  .turn-sys { border-color: #dde1e8; }
  .turn-sys .turn-label { background: #eef0f4; color: #66708a; }
  .turn-sys .turn-body { background: #fafbfc; color: #4a5468; }
  .hc { background: #ffe0e0; color: #b3261e; font-weight: 700;
        border-radius: 4px; padding: 0 3px; }
  details.hintset { margin-bottom: 16px; border: 1px solid #f0d9a8;
                    border-radius: 8px; background: #fffaf0; }
  details.hintset > summary { cursor: pointer; padding: 8px 12px; font-size: 13px;
        font-weight: 600; color: #8a5a00; }
  .hintset-body { padding: 4px 12px 12px; }
  .hstep { border-top: 1px solid #f0e2c2; padding: 8px 0; }
  .hstep-head { font-size: 12px; font-weight: 700; color: #5a4300; margin-bottom: 3px; }
  .hstep-field { font-size: 12px; margin: 2px 0; line-height: 1.4; }
  .hstep-field .k { color: #9a6700; font-weight: 600; }
  .hsub { font-size: 11px; margin: 2px 0 2px 14px; color: #4a5468;
          font-family: ui-monospace, monospace; line-height: 1.45; }
  .htype { display: inline-block; font-size: 9px; font-weight: 700; border-radius: 6px;
           padding: 0 5px; background: #e3ecff; color: #2747a8; text-transform: uppercase; }
  .diff { display: inline-block; font-size: 10px; font-weight: 700; border-radius: 8px;
          padding: 0 6px; margin-left: 6px; text-transform: uppercase; }
  .diff-easy { background: #d8f5dd; color: #137333; }
  .diff-moderate { background: #fff0d6; color: #9a6700; }
  .diff-hard { background: #fde0e0; color: #b3261e; }
  .hint { color: #66708a; padding: 40px; text-align: center; }
  h3 { margin: 0 0 8px; font-size: 13px; color: #66708a; text-transform: uppercase;
       letter-spacing: .04em; }
  .mono { font-family: ui-monospace, monospace; }
  /* jump to the prev/next step where this same problem appears */
  .probnav { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
             margin-bottom: 14px; padding: 8px 12px; background: #eef2fc;
             border: 1px solid #d4dcf0; border-radius: 8px; }
  .navbtn { font-size: 12px; font-weight: 600; cursor: pointer; border: 1px solid #9fb2e6;
            background: #fff; color: #2747a8; border-radius: 6px; padding: 4px 12px; }
  .navbtn:hover:not([disabled]) { background: #dde7ff; }
  .navbtn[disabled] { color: #aab2c4; border-color: #dde2ee; background: #f6f8fc;
                      cursor: default; }
  .navmsg { font-size: 12px; color: #5a6478; }
  .navmsg b { color: #2747a8; }
  /* selector-response button + panel, on each injected-hint turn */
  .selbtn { float: right; font-size: 10px; font-weight: 600; text-transform: none;
            letter-spacing: 0; cursor: pointer; border: 1px solid #c7b07a;
            background: #fff7e8; color: #8a5a00; border-radius: 6px; padding: 1px 8px; }
  .selbtn:hover { background: #ffeec9; }
  .selbtn.on { background: #8a5a00; color: #fff; border-color: #8a5a00; }
  .selpanel { display: none; border-top: 1px dashed #e0c98a; background: #fffdf8;
              padding: 10px 12px; }
  .selnote { color: #8a5a00; font-size: 11px; margin-bottom: 6px; }
  .selcall { border: 1px solid #ead9b0; border-radius: 8px; background: #fff; margin-bottom: 10px; }
  .selcall-head { padding: 6px 10px; border-bottom: 1px solid #f0e2c2; font-size: 11px;
                  color: #6a5a3a; background: #fbf7ee; border-radius: 8px 8px 0 0;
                  display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .selcall-head .k { color: #9a6700; font-weight: 600; }
  .selcall-body { padding: 8px 10px; }
  .selrow { margin: 6px 0; font-size: 12px; line-height: 1.45; }
  .selrow .k { color: #2747a8; font-weight: 700; display: block; font-size: 10px;
               text-transform: uppercase; letter-spacing: .03em; margin-bottom: 1px; }
  .selrow .v { white-space: pre-wrap; }
  .selrow.hl .v { background: #fff6e6; border-left: 3px solid #e0b450; padding: 4px 8px;
                  border-radius: 0 4px 4px 0; }
  .selcomplete { margin: 3px 0 3px 10px; font-size: 11px; color: #4a5468; line-height: 1.4; }
  .selcomplete .sid { font-weight: 700; color: #137333; }
  .seldetails { margin-top: 8px; }
  .seldetails > summary { cursor: pointer; font-size: 11px; color: #2747a8; font-weight: 600; }
  .selpre { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 11px;
            background: #f6f8fc; border: 1px solid #e0e4ea; border-radius: 6px;
            padding: 8px; margin-top: 4px; max-height: 360px; overflow: auto; }
</style>
</head>
<body>
<header>Rollout Viewer</header>
<div class="bars">
  <div class="ctrl"><label>Experiment</label><select id="exp"></select></div>
  <div class="ctrl"><label>Split</label><select id="sub"></select></div>
  <div class="ctrl"><label>Step file</label><select id="step"></select></div>
</div>
<div class="layout">
  <div id="problems"><div class="hint">select a step file</div></div>
  <div id="main"><div class="hint">select a problem</div></div>
</div>
<script>
const $ = id => document.getElementById(id);
let state = {};
let selSeq = 0;  // unique id per selector-response panel
let navSeq = 0;  // generation token: ignore stale problem-step poll results

async function getJSON(url) { const r = await fetch(url); return r.json(); }

async function loadExps() {
  const exps = await getJSON('/api/exps');
  state.exps = exps;
  $('exp').innerHTML = exps.map(e => `<option value="${e.name}">${e.name}</option>`).join('');
  if (exps.length) onExpChange();
}

function onExpChange() {
  const e = state.exps.find(x => x.name === $('exp').value);
  $('sub').innerHTML = e.subdirs.map(s => `<option>${s}</option>`).join('');
  onSubChange();
}

async function onSubChange() {
  const exp = $('exp').value, sub = $('sub').value;
  const steps = await getJSON(`/api/steps?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}`);
  $('step').innerHTML = steps.map(s => `<option>${s}</option>`).join('');
  onStepChange();
}

async function onStepChange(selectPid) {
  const exp = $('exp').value, sub = $('sub').value, step = $('step').value;
  if (!step) return;
  const probs = await getJSON(`/api/problems?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}&step=${encodeURIComponent(step)}`);
  $('main').innerHTML = '<div class="hint">select a problem</div>';
  $('problems').innerHTML = probs.map((p, i) => {
    const rate = p.count ? Math.round(100 * p.n_correct / p.count) : 0;
    const cls = rate >= 50 ? 'ok' : 'bad';
    const allOk = p.count > 0 && p.n_correct === p.count;
    const allFail = p.count > 0 && p.n_correct === 0;
    const itemCls = allOk ? 'pitem all-ok' : allFail ? 'pitem all-fail' : 'pitem';
    const label = (p.pid !== null && p.pid !== undefined) ? ('#' + p.pid) : p.problem_id;
    const dsBadge = (p.ds_index != null)
      ? ` &nbsp; <span class="badge dsb" title="${esc(p.ds_problem_id||'')}">DAPO #${esc(p.ds_index)}</span>` : '';
    const hintBadge = (p.avg_hints != null)
      ? ` &nbsp; <span class="badge hintb">⌀${p.avg_hints} hints</span>` : '';
    const failBadge = (p.n_hint_failed > 0)
      ? ` &nbsp; <span class="badge failb">${p.n_hint_failed} fail</span>` : '';
    const packBadge = (p.pack_budgets && p.pack_budgets.length > 1)
      ? ` &nbsp; <span class="badge packb" title="k-pack budgets (B first)">${p.pack_budgets.length} packs B=${esc(p.pack_budgets.join(','))}</span>` : '';
    return `<div class="${itemCls}" data-pid="${p.problem_id}">
       <div><b>${label}</b> &nbsp; <span class="badge ${cls}">${p.n_correct}/${p.count}</span>${dsBadge}${hintBadge}${failBadge}${packBadge}
            &nbsp; <span class="meta">gt=${esc(p.gts)}</span></div>
       <div class="meta">${esc(p.preview||'')}</div>
     </div>`;
  }).join('');
  [...document.querySelectorAll('.pitem')].forEach(el => {
    el.onclick = () => {
      document.querySelectorAll('.pitem').forEach(x => x.classList.remove('active'));
      el.classList.add('active');
      loadRollouts(el.dataset.pid);
    };
  });
  // arrived here by jumping to another step for the same problem -> reopen it
  if (selectPid) {
    const el = document.querySelector(`.pitem[data-pid="${selectPid}"]`);
    if (el) { el.click(); el.scrollIntoView({ block: 'center' }); }
    else $('main').innerHTML = `<div class="hint">this problem isn't in ${esc(step)}</div>`;
  }
}

function esc(s){ return (s==null?'':String(s)).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }

function fmtBody(text){
  return esc(text).replace(/&lt;hint_call\/&gt;/g, '<span class="hc">&lt;hint_call/&gt;</span>');
}

function diffBadge(d){
  if (!d) return '';
  const k = String(d).toLowerCase();
  const cls = (k==='easy'||k==='moderate'||k==='hard') ? 'diff-'+k : 'diff-moderate';
  return `<span class="diff ${cls}">${esc(d)}</span>`;
}

function field(k,v){ return v ? `<div class="hstep-field"><span class="k">${k}:</span> ${esc(v)}</div>` : ''; }

// simplified-mt format: {steps:[{step_id, purpose, hints:[{type, hint_id, hint}]}]}
function renderHintDict(h){
  return (h.steps||[]).map(s => {
    const hints = (s.hints||[]).map(hh => {
      const ty = String(hh.type||'').replace(/_hint$/,'').replace(/_/g,' ');
      return `<div class="hsub"><b>${esc(hh.hint_id)}</b>`
           + (ty?` <span class="htype">${esc(ty)}</span>`:'')
           + ` ${esc(hh.hint)}</div>`;
    }).join('');
    return `<div class="hstep">
      <div class="hstep-head">Step ${esc(s.step_id)}</div>
      ${field('purpose', s.purpose)}
      ${hints}
    </div>`;
  }).join('');
}

// original format: [{step_id, title, purpose, core_conclusion, difficulty, substeps:[...]}]
function renderHintList(arr){
  return arr.map(s => {
    const subs = Array.isArray(s.substeps) ? s.substeps.map(ss =>
      `<div class="hsub"><b>${esc(ss.substep_id)}</b> ${esc(ss.operation)} ${diffBadge(ss.difficulty)}</div>`
    ).join('') : '';
    return `<div class="hstep">
      <div class="hstep-head">Step ${esc(s.step_id)} — ${esc(s.title)} ${diffBadge(s.difficulty)}</div>
      ${field('purpose', s.purpose)}
      ${field('conclusion', s.core_conclusion)}
      ${subs}
      ${field('variants', s.acceptable_variants)}
    </div>`;
  }).join('');
}

function renderHintSet(ds){
  if (!ds) return '';
  const h = ds.hint;
  let body = '', n = 0;
  if (h && Array.isArray(h.steps)) { body = renderHintDict(h); n = h.steps.length; }
  else if (Array.isArray(h) && h.length) { body = renderHintList(h); n = h.length; }
  else if (Array.isArray(ds.hint_full) && ds.hint_full.length) { body = renderHintList(ds.hint_full); n = ds.hint_full.length; }
  if (!body) return '';
  // when both an injected-hint set and a full reference set exist, offer the latter too
  let extra = '';
  if (h && Array.isArray(h.steps) && Array.isArray(ds.hint_full) && ds.hint_full.length) {
    extra = `<details class="hintset" style="margin-top:8px">
      <summary>Full reference hints — ${ds.hint_full.length} step(s)</summary>
      <div class="hintset-body">${renderHintList(ds.hint_full)}</div>
    </details>`;
  }
  return `<details class="hintset">
    <summary>Hint set — ${n} step(s)</summary>
    <div class="hintset-body">${body}</div>
  </details>${extra}`;
}

function renderTurns(turns){
  if (!turns || !turns.length) return '<div class="hint">no turns</div>';
  return turns.map(t => {
    const cls = t.role === 'user' ? 'turn-hint'
              : t.role === 'system' ? 'turn-sys' : 'turn-asst';
    const label = t.role === 'user' ? 'injected hint' : t.role;
    // an injected-hint turn that we can trace to its selector call gets a button
    let btn = '', panel = '';
    if (t.role === 'user' && t.sel_key && state.hasSelIndex) {
      const pid = 'sel' + (selSeq++);
      btn = `<button class="selbtn" data-key="${esc(t.sel_key)}" data-panel="${pid}">🔍 selector response</button>`;
      panel = `<div class="selpanel" id="${pid}"></div>`;
    }
    return `<div class="turn ${cls}">
      <div class="turn-label">${label}${btn}</div>
      <div class="turn-body">${fmtBody(t.text)}</div>
      ${panel}
    </div>`;
  }).join('');
}

// ---- selector-call rendering (the original frozen-selector response) ----
function selRow(k, v, hl){
  if (v == null || v === '') return '';
  return `<div class="selrow${hl?' hl':''}"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`;
}
function prettyMaybeJSON(s){
  if (s == null) return '';
  try { return JSON.stringify(JSON.parse(s), null, 2); } catch(e) { return String(s); }
}
function renderSelectorCall(c){
  const s = c.selection || {};
  const conf = (s.confidence_of_major_step != null) ? s.confidence_of_major_step
             : (s.confidence_of_hint != null) ? s.confidence_of_hint : null;
  const failBadge = c.selector_err ? `<span class="badge bad">selector error</span>`
                  : (!c.selection ? `<span class="badge bad">no selection parsed</span>` : '');
  const meta = [
    (c.call_index!=null?`<span><span class="k">call</span> #${esc(c.call_index)}</span>`:''),
    (c.strategy?`<span><span class="k">strategy</span> ${esc(c.strategy)}</span>`:''),
    (c.budget!=null?`<span><span class="k">budget</span> ${esc(c.budget)}</span>`:''),
    (c.select_latency_s!=null?`<span><span class="k">latency</span> ${esc(c.select_latency_s)}s</span>`:''),
    (c.n_assistant_in_messages!=null?`<span><span class="k">trace turns</span> ${esc(c.n_assistant_in_messages)}</span>`:''),
    (c.request_id?`<span><span class="k">req</span> <span class="mono">${esc(c.request_id)}</span></span>`:''),
    failBadge,
  ].filter(Boolean).join('');
  const completed = Array.isArray(s.completed_steps) && s.completed_steps.length
    ? `<div class="selrow"><span class="k">completed steps (per selector)</span>` + s.completed_steps.map(cs =>
        `<div class="selcomplete"><span class="sid">step ${esc(cs.step_id)}</span>`
        + (cs.quote?` — “${esc(cs.quote)}”`:'') + (cs.why?`<br>${esc(cs.why)}`:'') + `</div>`).join('') + `</div>`
    : '';
  const body = [
    completed,
    selRow('major step', (s.major_step_id!=null? s.major_step_id : '') + (conf!=null?`  ·  confidence ${conf}`:''), true),
    selRow('why this step', s.reasoning_of_major_step),
    selRow('hint id', s.hint_id),
    selRow('why this hint', s.reasoning_of_hint),
    selRow('chosen hint', s.hint, true),
    c.selector_err ? selRow('error', c.selector_err) : '',
  ].filter(Boolean).join('');
  const details = [
    `<details class="seldetails"><summary>raw selector output</summary><div class="selpre">${esc(c.selector_raw||'(empty)')}</div></details>`,
    `<details class="seldetails"><summary>candidate hints shown to selector</summary><div class="selpre">${esc(prettyMaybeJSON(c.candidate_hints_str))}</div></details>`,
    `<details class="seldetails"><summary>student trace seen by selector</summary><div class="selpre">${esc(c.trace||'(empty)')}</div></details>`,
    `<details class="seldetails"><summary>problem</summary><div class="selpre">${esc(c.problem||'(empty)')}</div></details>`,
  ].join('');
  return `<div class="selcall">
    <div class="selcall-head">${meta}</div>
    <div class="selcall-body">${body}${details}</div>
  </div>`;
}
function renderSelectorCalls(data){
  if (data.error) return `<div class="selnote">${esc(data.error)} — build it with tools/build_selector_index.py</div>`;
  if (!data.calls || !data.calls.length) return `<div class="selnote">no matching selector call found in the index</div>`;
  let note = '';
  if (data.truncated) note = `<div class="selnote">showing ${data.calls.length} of ${data.total} matching selector calls</div>`;
  else if (data.calls.length > 1) note = `<div class="selnote">${data.calls.length} matching selector calls (shared/duplicate trajectories)</div>`;
  return note + data.calls.map(renderSelectorCall).join('');
}
async function toggleSelector(btn){
  const panel = document.getElementById(btn.dataset.panel);
  if (!panel) return;
  if (panel.dataset.open === '1') {  // collapse
    panel.style.display = 'none'; panel.dataset.open = '0'; btn.classList.remove('on');
    return;
  }
  if (!panel.dataset.loaded) {
    panel.innerHTML = '<div class="selnote">loading selector call…</div>';
    panel.style.display = 'block';
    const exp = $('exp').value;
    try {
      const data = await getJSON(`/api/selector_call?exp=${encodeURIComponent(exp)}&key=${encodeURIComponent(btn.dataset.key)}`);
      panel.innerHTML = renderSelectorCalls(data);
    } catch(e) {
      panel.innerHTML = `<div class="selnote">failed to load selector call</div>`;
    }
    panel.dataset.loaded = '1';
  }
  panel.style.display = 'block'; panel.dataset.open = '1'; btn.classList.add('on');
}

// ---- jump to the prev/next step where this same problem appears ----
function stepNum(f){ const s = String(f).replace(/\.jsonl$/, ''); return /^\d+$/.test(s) ? parseInt(s,10) : null; }

function gotoStep(stepFile, pid){
  if ($('step').value !== stepFile) $('step').value = stepFile;
  onStepChange(pid);  // rebuild that step's problem list, then reopen this problem
}

// fill #probnav from the index: prev/next step neighbours of the current step
function renderProblemNav(bar, pid, curStepFile, steps){
  const cur = stepNum(curStepFile);
  let prev = null, next = null;  // nearest earlier / later step containing this problem
  if (cur != null) {
    for (const f of steps) {
      const n = stepNum(f);
      if (n == null || n === cur) continue;
      if (n < cur && (prev == null || n > stepNum(prev))) prev = f;
      if (n > cur && (next == null || n < stepNum(next))) next = f;
    }
  }
  const total = steps.length;
  const pos = steps.indexOf(curStepFile);
  const occ = pos >= 0
      ? `occurrence <b>${pos+1}</b> of <b>${total}</b>`
      : `appears in <b>${total}</b> step${total===1?'':'s'}`;
  const prevBtn = prev
      ? `<button class="navbtn" data-step="${esc(prev)}">← step ${esc(stepNum(prev))}</button>`
      : `<button class="navbtn" disabled title="no earlier step has this problem">← prev step</button>`;
  const nextBtn = next
      ? `<button class="navbtn" data-step="${esc(next)}">step ${esc(stepNum(next))} →</button>`
      : `<button class="navbtn" disabled title="no later step has this problem">next step →</button>`;
  bar.innerHTML = `${prevBtn}<span class="navmsg">same problem · ${occ}</span>${nextBtn}`;
  bar.querySelectorAll('.navbtn[data-step]').forEach(b => { b.onclick = () => gotoStep(b.dataset.step, pid); });
}

async function loadProblemNav(pid, token){
  const bar = $('probnav');
  if (!bar) return;
  const exp = $('exp').value, sub = $('sub').value, step = $('step').value;
  let data;
  try {
    data = await getJSON(`/api/problem_steps?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}&problem_id=${encodeURIComponent(pid)}`);
  } catch(e) { if (token === navSeq) bar.innerHTML = `<span class="navmsg">couldn't load the step index</span>`; return; }
  if (token !== navSeq) return;  // user moved to another problem; drop this result
  if (data.status === 'indexing') {
    const pct = data.total ? Math.round(100 * data.done / data.total) : 0;
    bar.innerHTML = `<span class="navmsg">indexing this run to locate the problem's other steps… <b>${pct}%</b> (${data.done}/${data.total})</span>`;
    setTimeout(() => { if (token === navSeq) loadProblemNav(pid, token); }, 1000);  // poll the build
    return;
  }
  if (data.status === 'error') { bar.innerHTML = `<span class="navmsg">step index error: ${esc(data.error||'')}</span>`; return; }
  renderProblemNav(bar, pid, step, data.steps || []);
}

async function loadRollouts(pid) {
  const exp = $('exp').value, sub = $('sub').value, step = $('step').value;
  const navToken = ++navSeq;  // invalidates any in-flight nav poll from a prior problem
  $('main').innerHTML = '<div class="hint">loading…</div>';
  const data = await getJSON(`/api/rollouts?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}&step=${encodeURIComponent(step)}&problem_id=${encodeURIComponent(pid)}`);
  state.hasSelIndex = !!data.has_selector_index;  // gate the per-hint selector buttons
  const promptTurns = data.input_turns && data.input_turns.length
      ? renderTurns(data.input_turns)
      : `<div class="turn turn-sys"><div class="turn-body">${fmtBody(data.input)}</div></div>`;
  let html = '<div id="probnav" class="probnav"></div>';  // prev/next-step nav, filled async
  if (data.dataset && data.dataset.ds_index != null) {
    html += `<div class="dsid">
       <span class="badge dsb">DAPO #${esc(data.dataset.ds_index)}</span>
       <span class="dsid-pid">${esc(data.dataset.ds_problem_id||'')}</span>
     </div>`;
  }
  html += `<h3>Prompt</h3><div class="prompt">${promptTurns}</div>`;
  html += renderHintSet(data.dataset);
  const packs = data.pack_budgets || [];
  const multiPack = packs.length > 1;            // k-pack: this problem ran at >1 budget
  const maxB = packs.length ? packs[0] : null;   // assigned budget B (packs sorted desc)
  // per-pack tallies for the foldable header: rollout count + accuracy
  const packCount = {}, packCorrect = {};
  data.rollouts.forEach(r => {
    const b = r.hint_budget;
    packCount[b] = (packCount[b]||0) + 1;
    if (parseFloat(r.acc) > 0) packCorrect[b] = (packCorrect[b]||0) + 1;
  });
  html += `<h3>${data.rollouts.length} rollouts`
        + (multiPack ? ` &nbsp;·&nbsp; ${packs.length} k-packs, ordered B=${esc(packs.join(', '))}` : '')
        + `</h3>`;
  let lastB;
  data.rollouts.forEach((r, i) => {
    if (multiPack && r.hint_budget !== lastB) {  // new budget group -> B first, then B-1, ...
      if (lastB !== undefined) html += `</details>`;   // close the previous pack
      lastB = r.hint_budget;
      const b = r.hint_budget, delta = maxB - b;
      const tag = delta === 0 ? 'B — assigned' : `B−${delta} — counterfactual probe`;
      const nc = packCorrect[b] || 0, n = packCount[b], pct = n ? Math.round(100*nc/n) : 0;
      // foldable pack (open by default) -> fold one to jump to the next; accuracy on the bar
      html += `<details class="pack" open><summary class="packhdr">`
            + `budget ${esc(b)} &nbsp;·&nbsp; ${tag} &nbsp;·&nbsp; ${n} rollouts`
            + ` &nbsp;·&nbsp; <span class="badge ${pct>=50?'ok':'bad'}">${nc}/${n} correct · ${pct}%</span>`
            + `</summary>`;
    }
    const ok = parseFloat(r.acc) > 0;
    const hintsTxt = (r.num_hints != null)
      ? `${esc(r.num_hints)}${r.hint_budget != null ? '/' + esc(r.hint_budget) : ''}` : null;
    const head = [
      `<b>#${i}</b>`,
      `<span><span class="badge ${ok?'ok':'bad'}">${ok?'correct':'wrong'}</span></span>`,
      `<span><span class="k">pred</span> ${esc(r.pred)}</span>`,
      `<span><span class="k">gt</span> ${esc(r.gts)}</span>`,
      `<span><span class="k">score</span> ${esc(r.score)}</span>`,
      r.reward!=null ? `<span><span class="k">reward</span> ${esc(r.reward)}</span>` : '',
      hintsTxt!=null ? `<span><span class="k">hints</span> <span class="badge hintb">${hintsTxt}</span></span>` : '',
      r.hint_penalty!=null ? `<span><span class="k">hint pen</span> ${esc(r.hint_penalty)}</span>` : '',
      parseFloat(r.hint_call_failed)>0 ? `<span><span class="badge failb">hint call failed</span></span>` : '',
      `<span><span class="k">format</span> ${esc(r.has_format)}</span>`,
    ].filter(Boolean).join('\n        ');
    let hintsMeta = '';
    if (r.applied_hints && r.applied_hints.length) {
      hintsMeta = '<div class="hints-meta">' + r.applied_hints.map(h =>
        `<span class="hchip">#${esc(h.call_index)} · step ${esc(h.major_step_id)} · conf ${esc(h.confidence_of_major_step)} · [${esc((h.hint_ids||[]).join(', '))}]</span>`
      ).join('') + '</div>';
    }
    html += `<div class="roll">
      <div class="roll-head">
        ${head}
      </div>
      ${hintsMeta}
      <div class="roll-body">${renderTurns(r.turns)}</div>
    </div>`;
  });
  if (multiPack && lastB !== undefined) html += `</details>`;  // close the last pack
  $('main').innerHTML = html;
  document.querySelectorAll('.selbtn').forEach(b => { b.onclick = () => toggleSelector(b); });
  loadProblemNav(pid, navToken);  // find the prev/next step where this problem recurs
}

$('exp').onchange = onExpChange;
$('sub').onchange = onSubChange;
$('step').onchange = () => onStepChange();  // no arg: a manual step switch selects no problem
loadExps();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs", help="directory containing experiment folders")
    ap.add_argument("--dataset", default=",".join(DATASET_PATHS),
                    help="comma-separated source parquet(s) for mapping prompts back to "
                         "dataset problem ids / hints (the one whose system base matches "
                         "each rollout is used)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    LOGS_DIR = os.path.abspath(args.logs_dir)
    DATASET_PATHS = [p for p in args.dataset.split(",") if p.strip()]
    print(f"serving logs from {LOGS_DIR} at http://{args.host}:{args.port}")
    _load_datasets()
    for line in _DS_REPORT:
        print(f"  dataset: {line}")
    # Report which runs have a selector index (enables the per-hint "selector
    # response" buttons); for the rest, point at the builder.
    try:
        exps = [n for n in sorted(os.listdir(LOGS_DIR)) if os.path.isdir(os.path.join(LOGS_DIR, n))]
        indexed = [n for n in exps if os.path.isfile(os.path.join(LOGS_DIR, n, SELECTOR_INDEX_NAME))]
        if indexed:
            print(f"  selector index: {len(indexed)} run(s) -> per-hint selector buttons enabled: {', '.join(indexed)}")
        if len(indexed) < len(exps):
            print("  selector index: build for a run with "
                  "`python tools/build_selector_index.py --run-dir logs/<EXP>`")
    except OSError:
        pass
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
