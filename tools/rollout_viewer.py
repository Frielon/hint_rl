#!/usr/bin/env python3
"""Web viewer for rollout logs.

Browse:  experiment  ->  step file  ->  problem  ->  rollouts.

A "problem" is identified by the `input` prompt (stable md5 of `input`, matching
tools/group_rollouts_by_problem.py). Rollouts of the same problem are grouped on
the fly, so this works on raw `rollouts/` files as well as `*_grouped/` ones.

Run (conda env `inference` has Flask):
  /shared_home/xutao.ma/miniconda3/envs/inference/bin/python \
      tools/rollout_viewer.py --logs-dir logs --port 8000

Then open http://<host>:8000  (use --host 0.0.0.0 for remote access).
"""
import argparse
import hashlib
import json
import os

from flask import Flask, abort, jsonify, render_template_string, request

app = Flask(__name__)
LOGS_DIR = os.path.abspath("logs")
ROLLOUT_SUBDIRS = ("rollouts", "rollouts_grouped", "val_rollouts", "val_rollouts_grouped")


def problem_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


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
            groups[h] = {
                "problem_id": h,
                "pid": d.get("pid"),
                "gts": d.get("gts"),
                "count": 0,
                "n_correct": 0,
                "preview": d["input"][-300:],
            }
            order.append(h)
        g = groups[h]
        g["count"] += 1
        try:
            if float(d.get("acc", 0)) > 0:
                g["n_correct"] += 1
        except (TypeError, ValueError):
            pass
    return jsonify([groups[h] for h in order])


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
    for d in read_jsonl(path):
        h = d.get("problem_id") or problem_hash(d["input"])
        if h != pid:
            continue
        if problem_input is None:
            problem_input = d["input"]
        out.append({
            "output": d.get("output", ""),
            "score": d.get("score"),
            "acc": d.get("acc"),
            "reward": d.get("reward"),
            "pred": d.get("pred"),
            "gts": d.get("gts"),
            "has_format": d.get("has_format"),
        })
    return jsonify({"input": problem_input, "rollouts": out})


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
  .pitem .meta { color: #66708a; margin-top: 2px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 10px;
           font-size: 11px; font-weight: 600; }
  .ok { background: #d8f5dd; color: #137333; }
  .bad { background: #fde0e0; color: #b3261e; }
  #main { flex: 1; overflow-y: auto; padding: 16px; }
  .prompt { background: #fff; border: 1px solid #e0e4ea; border-radius: 8px;
            padding: 12px; white-space: pre-wrap; font-family: ui-monospace, monospace;
            font-size: 12px; max-height: 240px; overflow-y: auto; margin-bottom: 16px; }
  .roll { background: #fff; border: 1px solid #e0e4ea; border-radius: 8px;
          margin-bottom: 12px; }
  .roll-head { display: flex; gap: 14px; align-items: center; padding: 8px 12px;
               border-bottom: 1px solid #eef0f4; font-size: 12px; flex-wrap: wrap; }
  .roll-head .k { color: #66708a; }
  .roll-body { padding: 12px; white-space: pre-wrap; font-family: ui-monospace, monospace;
               font-size: 12px; line-height: 1.5; }
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
    const label = (p.pid !== null && p.pid !== undefined) ? ('#' + p.pid) : p.problem_id;
    return `<div class="pitem" data-pid="${p.problem_id}">
       <div><b>${label}</b> &nbsp; <span class="badge ${cls}">${p.n_correct}/${p.count}</span>
            &nbsp; <span class="meta">gt=${p.gts ?? ''}</span></div>
       <div class="meta">${(p.preview||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</div>
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

async function loadRollouts(pid) {
  const exp = $('exp').value, sub = $('sub').value, step = $('step').value;
  $('main').innerHTML = '<div class="hint">loading…</div>';
  const data = await getJSON(`/api/rollouts?exp=${encodeURIComponent(exp)}&sub=${encodeURIComponent(sub)}&step=${encodeURIComponent(step)}&problem_id=${encodeURIComponent(pid)}`);
  let html = `<h3>Prompt</h3><div class="prompt">${esc(data.input)}</div>`;
  html += `<h3>${data.rollouts.length} rollouts</h3>`;
  data.rollouts.forEach((r, i) => {
    const ok = parseFloat(r.acc) > 0;
    html += `<div class="roll">
      <div class="roll-head">
        <b>#${i}</b>
        <span><span class="badge ${ok?'ok':'bad'}">${ok?'correct':'wrong'}</span></span>
        <span><span class="k">pred</span> ${esc(r.pred)}</span>
        <span><span class="k">gt</span> ${esc(r.gts)}</span>
        <span><span class="k">score</span> ${esc(r.score)}</span>
        ${r.reward!=null?`<span><span class="k">reward</span> ${esc(r.reward)}</span>`:''}
        <span><span class="k">format</span> ${esc(r.has_format)}</span>
      </div>
      <div class="roll-body">${esc(r.output)}</div>
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    LOGS_DIR = os.path.abspath(args.logs_dir)
    print(f"serving logs from {LOGS_DIR} at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
