#!/usr/bin/env python3
"""Web viewer for rollout logs.

Browse:  experiment  ->  step file  ->  problem  ->  rollouts.

A "problem" is identified by the `input` prompt (stable md5 of `input`, matching
tools/group_rollouts_by_problem.py). Rollouts of the same problem are grouped on
the fly, so this works on raw `rollouts/` files as well as `*_grouped/` ones.

Multi-turn HPRL rollouts are rendered turn-by-turn: the model emits the
`<hint_call/>` sentinel, and the system injects a curated hint as the next user
turn. The chat template strips `<|im_start|>` / `<|im_end|>`, leaving bare
`system` / `user` / `assistant` role lines, which this viewer parses back into
turns. The new per-rollout HPRL fields (num_hints, hint_budget, hint_penalty,
hint_call_failed, applied_hints) are surfaced in the rollout header.

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
from collections import defaultdict

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
_RO_SUFFIX = re.compile(r"\s*You have \d+ hint calls? remaining for this problem\.?\s*$")
_HINT_INSTR = "A hint tool is available"  # start of the HPRL hint-instruction block
_PREFIX_K = 64  # bucket key length for the prefix-match fallback

HINT_CALL = "<hint_call/>"
# bare role lines left after the chat template strips <|im_start|>/<|im_end|>
_LEAD_MARKERS = ("system\n", "user\n", "assistant\n")
_ROLE_MARKERS = (("\nassistant\n", "assistant"), ("\nuser\n", "user"), ("\nsystem\n", "system"))


def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def _problem_core(user_text: str) -> str:
    """Recover the pure problem statement: drop the dataset's trailing
    instruction and the rollout's hint-budget line, then normalize whitespace."""
    return _norm(_DS_SUFFIX.sub("", _RO_SUFFIX.sub("", user_text or "")))


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
    # sort numerically by step number when possible
    def key(f):
        stem = f[:-6]
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return jsonify(sorted(files, key=key))


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
        h = d.get("problem_id") or problem_hash(d["input"])
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
    out = []
    for h in order:
        g = groups[h]
        g["avg_hints"] = round(g["sum_hints"] / g["count"], 2) if g["count"] else 0.0
        g.pop("sum_hints", None)
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
    out = []
    problem_input = None
    input_turns = None
    ds_match = None
    for d in read_jsonl(path):
        h = d.get("problem_id") or problem_hash(d["input"])
        if h != pid:
            continue
        if problem_input is None:
            problem_input = d["input"]
            input_turns = split_turns(d["input"], start_role="system")
            ds_match = match_dataset_problem(d["input"])
        out.append({
            "turns": split_turns(d.get("output", ""), start_role="assistant"),
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
        "rollouts": out,
    })


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

async function onStepChange() {
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
    return `<div class="${itemCls}" data-pid="${p.problem_id}">
       <div><b>${label}</b> &nbsp; <span class="badge ${cls}">${p.n_correct}/${p.count}</span>${dsBadge}${hintBadge}${failBadge}
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
    return `<div class="turn ${cls}">
      <div class="turn-label">${label}</div>
      <div class="turn-body">${fmtBody(t.text)}</div>
    </div>`;
  }).join('');
}

async function loadRollouts(pid) {
  const exp = $('exp').value, sub = $('sub').value, step = $('step').value;
  $('main').innerHTML = '<div class="hint">loading…</div>';
  const data = await getJSON(`/api/rollouts?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}&step=${encodeURIComponent(step)}&problem_id=${encodeURIComponent(pid)}`);
  const promptTurns = data.input_turns && data.input_turns.length
      ? renderTurns(data.input_turns)
      : `<div class="turn turn-sys"><div class="turn-body">${fmtBody(data.input)}</div></div>`;
  let html = '';
  if (data.dataset && data.dataset.ds_index != null) {
    html += `<div class="dsid">
       <span class="badge dsb">DAPO #${esc(data.dataset.ds_index)}</span>
       <span class="dsid-pid">${esc(data.dataset.ds_problem_id||'')}</span>
     </div>`;
  }
  html += `<h3>Prompt</h3><div class="prompt">${promptTurns}</div>`;
  html += renderHintSet(data.dataset);
  html += `<h3>${data.rollouts.length} rollouts</h3>`;
  data.rollouts.forEach((r, i) => {
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
  $('main').innerHTML = html;
}

$('exp').onchange = onExpChange;
$('sub').onchange = onSubChange;
$('step').onchange = onStepChange;
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
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
