#!/usr/bin/env python3
"""Local web viewer for the gpt-oss hint-selection eval results.

Browse ``results/<run>/step{N}/<problem_id>/<request_id>.json`` interactively:
pick a run, a step, then a problem. Each problem page joins the eval result with
its Codex *label source* (``test_cite/results/debug/<label_run>/...`` at the same
relative path) so you see, side by side:

  * the math problem + the student's reasoning trace,
  * the full HINT SET (hint_pool) with the model's vote count overlaid on each
    hint and the reference hint highlighted,
  * the REFERENCE selection (Codex's gold answer),
  * the model's N samples with per-sample match badges, confidences, reasoning
    and raw completion.

No third-party deps -- pure stdlib ``http.server``. Math is rendered with
MathJax from a CDN (needs internet in the *browser*, not on this host).

Usage:
    python viewer.py                     # serve on http://127.0.0.1:8731
    python viewer.py --port 8123
    python viewer.py --host 0.0.0.0      # expose on the network
    # remote box? forward a port instead:  ssh -L 8731:localhost:8731 <host>

Then open the printed URL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent            # .../test_cite/gpt_oss_eval
TEST_CITE = HERE.parent                            # .../test_cite

# Overridable via CLI; populated in main().
RESULTS_ROOT = HERE / "results"
LABEL_ROOT = TEST_CITE / "results"

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")

# --------------------------------------------------------------------------- #
# small fs helpers (with an mtime-keyed JSON cache so re-browsing is instant)
# --------------------------------------------------------------------------- #
_cache: dict[str, tuple[float, object]] = {}


def load_json(path: Path | str):
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = str(p)
    hit = _cache.get(key)
    if hit and hit[0] == st.st_mtime:
        return hit[1]
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    _cache[key] = (st.st_mtime, data)
    return data


def safe_component(s: str) -> bool:
    return bool(s) and bool(SAFE.match(s)) and ".." not in s


def within(child: Path, parent: Path) -> bool:
    try:
        child = Path(child).resolve()
        parent = Path(parent).resolve()
        return child == parent or parent in child.parents
    except Exception:  # noqa: BLE001
        return False


def run_dir(run: str) -> Path:
    if not safe_component(run):
        raise ValueError("bad run")
    p = (RESULTS_ROOT / run).resolve()
    if not within(p, RESULTS_ROOT) or not p.is_dir():
        raise ValueError("run not found")
    return p


def list_runs() -> list[str]:
    if not RESULTS_ROOT.is_dir():
        return []
    out = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and (d / "run_config.json").exists() or (d.is_dir() and any(
                (d / f"step{i}").is_dir() for i in range(1, 12))):
            out.append(d.name)
    return out


def label_dir_of(run: str) -> Path | None:
    """Resolve the Codex label source dir that this eval run was scored against."""
    cfg = load_json(run_dir(run) / "run_config.json") or {}
    name = cfg.get("label_run") or run.split("__")[0]
    cands: list[Path] = []
    ld = cfg.get("label_dir")
    if ld:
        cands.append(Path(ld))                       # may be a container path -> filtered
    if safe_component(str(name)):
        cands += [LABEL_ROOT / "debug" / name, LABEL_ROOT / "runs" / name, LABEL_ROOT / name]
    for c in cands:
        try:
            if Path(c).is_dir():
                return Path(c).resolve()
        except Exception:  # noqa: BLE001
            continue
    return None


def steps_of(run: str) -> list[int]:
    rd = run_dir(run)
    out = []
    for d in rd.iterdir():
        m = re.fullmatch(r"step(\d+)", d.name)
        if d.is_dir() and m:
            out.append(int(m.group(1)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# API payload builders
# --------------------------------------------------------------------------- #
def api_runs() -> dict:
    runs = []
    for name in list_runs():
        cfg = load_json(run_dir(name) / "run_config.json") or {}
        summ = load_json(run_dir(name) / "_summary.json") or {}
        runs.append({
            "name": name,
            "model": cfg.get("model"),
            "label_run": cfg.get("label_run") or name.split("__")[0],
            "timestamp": cfg.get("timestamp"),
            "n_rows": summ.get("n_rows"),
            "mean_agreement_hint_id": summ.get("mean_agreement_hint_id"),
            "majority_agree_rate": summ.get("majority_agree_rate"),
        })
    return {"runs": runs}


def api_run(run: str) -> dict:
    rd = run_dir(run)
    cfg = load_json(rd / "run_config.json") or {}
    summ = load_json(rd / "_summary.json") or {}
    cit = load_json(rd / "_citation_check.json")
    ldir = label_dir_of(run)
    steps = []
    for s in steps_of(run):
        sd = rd / f"step{s}"
        n_prob = 0
        n_req = 0
        for pd in os.scandir(sd):
            if pd.is_dir():
                n_prob += 1
                n_req += sum(1 for f in os.scandir(pd.path) if f.name.endswith(".json"))
        steps.append({"step": s, "n_problems": n_prob, "n_requests": n_req,
                      "summary": (summ.get("by_step") or {}).get(str(s))})
    return {
        "name": run,
        "config": cfg,
        "summary": summ,
        "citation": cit,
        "label_dir": str(ldir) if ldir else None,
        "label_found": ldir is not None,
        "steps": steps,
    }


def _req_meta(rec: dict, fname: str) -> dict:
    return {
        "request_id": rec.get("request_id") or fname,
        "agreement_hint_id": rec.get("agreement_hint_id"),
        "agreement_major_step": rec.get("agreement_major_step"),
        "majority_agrees_hint_id": rec.get("majority_agrees_hint_id"),
        "self_consistency": rec.get("self_consistency"),
        "mode_hint_id": rec.get("mode_hint_id"),
        "ref_hint_id": rec.get("ref_hint_id"),
        "ref_major_step_id": rec.get("ref_major_step_id"),
        "n_parsed": rec.get("n_parsed"),
        "n_samples": rec.get("n_samples"),
        "n_truncated": rec.get("n_truncated"),
        "mean_completion_tokens": rec.get("mean_completion_tokens"),
    }


def api_problems(run: str, step: int) -> dict:
    sd = run_dir(run) / f"step{step}"
    if not sd.is_dir():
        return {"problems": []}
    problems = []
    for pd in sorted(sd.iterdir()):
        if not pd.is_dir():
            continue
        reqs = []
        for f in sorted(pd.glob("*.json")):
            reqs.append(_req_meta(load_json(f) or {}, f.stem))
        ag = [r["agreement_hint_id"] for r in reqs if isinstance(r["agreement_hint_id"], (int, float))]
        problems.append({
            "problem_id": pd.name,
            "n_requests": len(reqs),
            "n_majority_agree": sum(1 for r in reqs if r["majority_agrees_hint_id"]),
            "mean_agreement_hint_id": (sum(ag) / len(ag)) if ag else None,
            "n_truncated": sum((r["n_truncated"] or 0) for r in reqs),
            "requests": reqs,
        })
    return {"problems": problems, "step": step}


def api_problem(run: str, step: int, pid: str) -> dict:
    if not safe_component(pid):
        raise ValueError("bad problem id")
    pd = (run_dir(run) / f"step{step}" / pid).resolve()
    if not within(pd, run_dir(run)) or not pd.is_dir():
        raise ValueError("problem not found")
    ldir = label_dir_of(run)
    reqs = []
    for f in sorted(pd.glob("*.json")):
        ev = load_json(f) or {}
        label = None
        if ldir:
            sp = (ldir / f"step{step}" / pid / f.name)
            if within(sp, ldir):
                src = load_json(sp)
                if src:
                    label = {
                        "problem": src.get("problem"),
                        "reasoning_trace": src.get("reasoning_trace"),
                        "hint_pool": src.get("hint_pool"),
                        "parsed_answer": src.get("parsed_answer"),
                        "prompt": src.get("prompt"),
                        "budget": src.get("budget"),
                        "strategy": src.get("strategy"),
                        "call_index": src.get("call_index"),
                        "ref_model": src.get("model"),
                        "reasoning_effort": src.get("reasoning_effort"),
                        "n_words": src.get("n_words"),
                    }
        reqs.append({"eval": ev, "label": label})
    return {"problem_id": pid, "step": step, "label_found": ldir is not None, "requests": reqs}


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "EvalViewer/1.0"

    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        g = lambda k: (q.get(k, [None])[0])  # noqa: E731
        try:
            if path == "/" or path == "/index.html":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/runs":
                self._json(api_runs())
            elif path == "/api/run":
                self._json(api_run(g("run")))
            elif path == "/api/problems":
                self._json(api_problems(g("run"), int(g("step"))))
            elif path == "/api/problem":
                self._json(api_problem(g("run"), int(g("step")), g("pid")))
            else:
                self._json({"error": "not found"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    do_HEAD = do_GET


# --------------------------------------------------------------------------- #
# Front-end (single page, embedded). Raw string: backslashes are literal so the
# MathJax delimiters and any JS escapes survive verbatim.
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hint-selection eval viewer</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  startup: { typeset: false }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  :root{
    --bg:#f6f7f9; --panel:#fff; --ink:#1c2430; --muted:#6b7785; --line:#e3e7ec;
    --accent:#2563eb; --good:#1a7f37; --goodbg:#e8f5ec; --bad:#c0392b; --badbg:#fdecea;
    --warnbg:#fff6e5; --warn:#a86700; --gold:#b8860b; --goldbg:#fbf3da;
    --chip:#eef2f7; --code:#f3f5f8;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink);background:var(--bg);display:flex;height:100vh;overflow:hidden}
  a{color:var(--accent);text-decoration:none}

  /* sidebar */
  #side{width:360px;min-width:360px;background:var(--panel);border-right:1px solid var(--line);
        display:flex;flex-direction:column;height:100%}
  #side .top{padding:12px 14px;border-bottom:1px solid var(--line)}
  #side h1{font-size:14px;margin:0 0 8px;letter-spacing:.2px}
  select,input,button{font:inherit;color:inherit}
  select{width:100%;padding:7px 8px;border:1px solid var(--line);border-radius:8px;background:#fff}
  .steps{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .steptab{flex:1 1 auto;min-width:56px;text-align:center;padding:6px 4px;border:1px solid var(--line);
           border-radius:8px;cursor:pointer;background:#fff;font-size:12px}
  .steptab .pct{display:block;font-size:11px;color:var(--muted)}
  .steptab.on{border-color:var(--accent);background:#eef4ff;color:var(--accent)}
  .filters{display:flex;gap:6px;margin-top:10px;align-items:center}
  .filters input[type=text]{flex:1;padding:6px 8px;border:1px solid var(--line);border-radius:8px}
  .filters select{width:auto;padding:6px 6px}
  .cklabel{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);margin-top:8px;cursor:pointer}

  #plist{overflow:auto;flex:1;padding:6px}
  .pitem{padding:8px 10px;border:1px solid var(--line);border-left:4px solid var(--line);
         border-radius:8px;margin:5px 2px;cursor:pointer;background:#fff}
  .pitem:hover{background:#fafbfc}
  .pitem.on{outline:2px solid var(--accent);outline-offset:-2px}
  .pitem .pid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;word-break:break-all;color:var(--ink)}
  .pitem .meta{display:flex;gap:8px;margin-top:4px;font-size:11.5px;color:var(--muted);align-items:center}
  .pitem.agree{border-left-color:var(--good)}
  .pitem.disagree{border-left-color:var(--bad)}
  .pitem.mixed{border-left-color:var(--warn)}

  /* main */
  #main{flex:1;overflow:auto;padding:18px 22px;height:100%}
  .muted{color:var(--muted)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:0 0 16px}
  .card h2{margin:0 0 10px;font-size:15px}
  .card h3{margin:14px 0 6px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:1100px){.grid2{grid-template-columns:1fr}}

  .badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;
         background:var(--chip);color:var(--ink);white-space:nowrap}
  .b-good{background:var(--goodbg);color:var(--good)}
  .b-bad{background:var(--badbg);color:var(--bad)}
  .b-warn{background:var(--warnbg);color:var(--warn)}
  .b-gold{background:var(--goldbg);color:var(--gold)}
  .b-accent{background:#e7efff;color:var(--accent)}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .kv{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:12.5px;margin-bottom:4px}
  .prose{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
  pre.raw{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:10px;
          overflow:auto;white-space:pre-wrap;word-wrap:break-word;font-size:12px;margin:6px 0 0}
  details{margin-top:6px}
  summary{cursor:pointer;color:var(--accent);font-size:12.5px}

  /* hint pool */
  .step-block{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0;background:#fff}
  .step-block.refstep{border-color:var(--gold);background:var(--goldbg)}
  .step-purpose{font-size:12.5px;color:var(--muted);margin:2px 0 8px}
  .hint{display:grid;grid-template-columns:54px 1fr auto;gap:8px;align-items:start;padding:6px 0;border-top:1px dashed var(--line)}
  .hint:first-of-type{border-top:none}
  .hint .hid{font-weight:700;font-family:ui-monospace,Menlo,monospace}
  .hint.refhint .hid{color:var(--gold)}
  .hint .txt{}
  .hint .type{font-size:10.5px;color:var(--muted);display:block;margin-top:2px}
  .votebar{height:7px;background:var(--accent);border-radius:4px;min-width:2px}
  .votewrap{width:120px;text-align:right}
  .votewrap .vt{font-size:11px;color:var(--muted)}

  /* samples */
  .dist{display:flex;flex-direction:column;gap:4px;margin:6px 0}
  .distrow{display:grid;grid-template-columns:60px 1fr 64px;gap:8px;align-items:center;font-size:12px}
  .distrow .bar{height:14px;border-radius:4px;background:#cbd5e1}
  .distrow.ref .bar{background:var(--gold)}
  .distrow.mode .bar{background:var(--accent)}
  .sample{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0}
  .sample.match{border-left:4px solid var(--good)}
  .sample.nomatch{border-left:4px solid var(--bad)}
  .sample .shead{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px}
  .conf{font-size:11px;color:var(--muted)}
  .placeholder{color:var(--muted);text-align:center;margin-top:18%;font-size:15px}
  .reqtabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
  .reqtab{padding:5px 10px;border:1px solid var(--line);border-radius:8px;cursor:pointer;background:#fff;font-size:12px}
  .reqtab.on{border-color:var(--accent);background:#eef4ff;color:var(--accent)}
  table.steptbl{border-collapse:collapse;font-size:12.5px;width:100%}
  table.steptbl th,table.steptbl td{border:1px solid var(--line);padding:4px 8px;text-align:right}
  table.steptbl th:first-child,table.steptbl td:first-child{text-align:left}
</style>
</head>
<body>
<div id="side">
  <div class="top">
    <h1>Hint-selection eval viewer</h1>
    <select id="runsel"></select>
    <div class="steps" id="steps"></div>
    <div class="filters">
      <input id="filter" type="text" placeholder="filter problem id...">
      <select id="sort">
        <option value="id">sort: id</option>
        <option value="agree_asc">agreement ↑</option>
        <option value="agree_desc">agreement ↓</option>
      </select>
    </div>
    <label class="cklabel"><input type="checkbox" id="disonly"> disagreements only</label>
  </div>
  <div id="plist"></div>
</div>
<div id="main"><div class="placeholder">Loading…</div></div>

<script>
const $ = s => document.querySelector(s);
const api = async (p) => (await fetch(p)).json();
const esc = s => (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const num = (x,d=2) => (typeof x==='number' && isFinite(x)) ? x.toFixed(d) : '–';
const pct = x => (typeof x==='number' && isFinite(x)) ? (x*100).toFixed(0)+'%' : '–';

const state = { run:null, runData:null, step:null, problems:[], pid:null, detail:null, reqIdx:0 };

function typeset(node){ if(window.MathJax && MathJax.typesetPromise){ MathJax.typesetPromise([node]).catch(()=>{}); } }

/* ---------------- routing via hash #run/step/pid ---------------- */
function writeHash(){
  const p = [state.run, state.step, state.pid].filter(x=>x!=null).map(encodeURIComponent).join('/');
  const h = '#'+p;
  if(location.hash !== h) history.replaceState(null,'',h);
}
function readHash(){
  const raw = decodeURIComponent(location.hash.replace(/^#/,''));
  const [run,step,pid] = raw.split('/');
  return { run: run||null, step: step?parseInt(step):null, pid: pid||null };
}

/* ---------------- init ---------------- */
async function init(){
  const {runs} = await api('/api/runs');
  const sel = $('#runsel');
  sel.innerHTML = runs.map(r=>{
    const tag = [r.model, r.mean_agreement_hint_id!=null?('agree '+pct(r.mean_agreement_hint_id)):null]
                .filter(Boolean).join(' · ');
    return `<option value="${esc(r.name)}">${esc(r.name)}${tag?'  —  '+esc(tag):''}</option>`;
  }).join('');
  sel.onchange = ()=>selectRun(sel.value, null, null);
  $('#filter').oninput = renderProblems;
  $('#sort').onchange = renderProblems;
  $('#disonly').onchange = renderProblems;
  window.addEventListener('hashchange', onHash);

  const h = readHash();
  const first = runs[0] && runs[0].name;
  await selectRun(h.run && runs.some(r=>r.name===h.run) ? h.run : first, h.step, h.pid);
}

function onHash(){
  const h = readHash();
  if(h.run && h.run!==state.run){ selectRun(h.run, h.step, h.pid); return; }
  if(h.step && h.step!==state.step){ selectStep(h.step, h.pid); return; }
  if(h.pid && h.pid!==state.pid){ openProblem(h.pid); }
}

/* ---------------- run + step ---------------- */
async function selectRun(run, step, pid){
  if(!run){ $('#main').innerHTML='<div class="placeholder">No runs found under results/.</div>'; return; }
  state.run = run; state.pid=null; state.detail=null;
  $('#runsel').value = run;
  state.runData = await api('/api/run?run='+encodeURIComponent(run));
  renderSteps();
  const steps = state.runData.steps.map(s=>s.step);
  const chosen = (step && steps.includes(step)) ? step : steps[0];
  await selectStep(chosen, pid);
  renderRunOverview();   // landing view in main when no problem chosen
}

function renderSteps(){
  const box = $('#steps');
  box.innerHTML = state.runData.steps.map(s=>{
    const sm = s.summary||{};
    const rate = sm.majority_agree_rate;
    return `<div class="steptab" data-step="${s.step}">step ${s.step}
      <span class="pct">${s.n_problems}p · ${rate!=null?pct(rate):'–'}</span></div>`;
  }).join('');
  box.querySelectorAll('.steptab').forEach(t=>{
    t.onclick = ()=>selectStep(parseInt(t.dataset.step), null);
  });
  highlightStep();
}
function highlightStep(){
  $('#steps').querySelectorAll('.steptab').forEach(t=>{
    t.classList.toggle('on', parseInt(t.dataset.step)===state.step);
  });
}

async function selectStep(step, pid){
  state.step = step; highlightStep();
  const data = await api(`/api/problems?run=${encodeURIComponent(state.run)}&step=${step}`);
  state.problems = data.problems;
  renderProblems();
  if(pid){ openProblem(pid); }
  else { state.pid=null; renderRunOverview(); }
  writeHash();
}

/* ---------------- problem list ---------------- */
function problemClass(p){
  if(!p.n_requests) return '';
  if(p.n_majority_agree===p.n_requests) return 'agree';
  if(p.n_majority_agree===0) return 'disagree';
  return 'mixed';
}
function renderProblems(){
  const f = $('#filter').value.trim().toLowerCase();
  const sort = $('#sort').value;
  const disonly = $('#disonly').checked;
  let items = state.problems.slice();
  if(f) items = items.filter(p=>p.problem_id.toLowerCase().includes(f));
  if(disonly) items = items.filter(p=>p.n_majority_agree < p.n_requests);
  if(sort==='agree_asc') items.sort((a,b)=>(a.mean_agreement_hint_id??9)-(b.mean_agreement_hint_id??9));
  else if(sort==='agree_desc') items.sort((a,b)=>(b.mean_agreement_hint_id??-1)-(a.mean_agreement_hint_id??-1));
  else items.sort((a,b)=>a.problem_id.localeCompare(b.problem_id));

  $('#plist').innerHTML = items.map(p=>{
    const reqtag = p.n_requests>1 ? `<span class="badge">${p.n_majority_agree}/${p.n_requests}✓</span>`
                                  : `<span class="badge ${p.n_majority_agree?'b-good':'b-bad'}">${p.n_majority_agree?'✓':'✗'} maj</span>`;
    const trunc = p.n_truncated ? `<span class="badge b-warn">${p.n_truncated} trunc</span>`:'';
    return `<div class="pitem ${problemClass(p)} ${p.problem_id===state.pid?'on':''}" data-pid="${esc(p.problem_id)}">
      <div class="pid">${esc(p.problem_id)}</div>
      <div class="meta">${reqtag}<span>agree ${pct(p.mean_agreement_hint_id)}</span>${trunc}</div>
    </div>`;
  }).join('') || '<div class="muted" style="padding:14px">no problems match.</div>';
  $('#plist').querySelectorAll('.pitem').forEach(it=>{
    it.onclick = ()=>openProblem(it.dataset.pid);
  });
}

/* ---------------- run overview (landing) ---------------- */
function renderRunOverview(){
  if(state.pid) return;
  const d = state.runData, s = d.summary||{}, c = d.config||{};
  const by = s.by_step||{};
  const rows = Object.keys(by).sort().map(k=>{
    const b=by[k];
    return `<tr><td>step ${k}</td><td>${b.n_rows??'–'}</td><td>${pct(b.mean_agreement_hint_id)}</td>
      <td>${pct(b.mean_agreement_major_step)}</td><td>${pct(b.majority_agree_rate)}</td></tr>`;
  }).join('');
  let cit = '';
  if(d.citation){
    const q = d.citation.model_quotes||{};
    const ref = d.citation.reference_quotes_baseline||{};
    cit = `<div class="card"><h2>Citation check (run-level)</h2>
      <div class="kv"><span>samples w/ quotes: <b>${d.citation.n_samples_with_quotes}/${d.citation.n_samples}</b></span></div>
      <h3>model quotes</h3>
      <div class="kv"><span>n=${q.n_quotes}</span><span>verbatim ${pct(q.verbatim_rate)}</span>
        <span>found ${pct(q.found_rate)}</span><span>not found ${pct(q.not_found_rate)}</span></div>
      <h3>reference baseline</h3>
      <div class="kv"><span>n=${ref.n_quotes}</span><span>verbatim ${pct(ref.verbatim_rate)}</span>
        <span>found ${pct(ref.found_rate)}</span></div></div>`;
  }
  $('#main').innerHTML = `
    <div class="card">
      <h2>${esc(d.name)}</h2>
      <div class="kv">
        <span>model <b>${esc(c.model)}</b></span>
        <span>label run <b class="mono">${esc(d.config?.label_run||'')}</b></span>
        <span>n/row <b>${esc(c.n)}</b></span><span>temp <b>${esc(c.temperature)}</b></span>
        <span>top_p <b>${esc(c.top_p)}</b></span><span>prompt <b>${esc(c.prompt_source)}</b></span>
      </div>
      <div class="kv">
        <span>rows scored <b>${s.n_rows_scored??'–'}/${s.n_rows??'–'}</b></span>
        <span>agreement(hint) <b>${pct(s.mean_agreement_hint_id)}</b></span>
        <span>agreement(step) <b>${pct(s.mean_agreement_major_step)}</b></span>
        <span>majority agree <b>${pct(s.majority_agree_rate)}</b></span>
        <span>self-consistency <b>${pct(s.mean_self_consistency)}</b></span>
        <span>mean tokens <b>${num(s.mean_completion_tokens,0)}</b></span>
        <span>truncated <b>${s.total_truncated??'–'}</b></span>
      </div>
      ${d.label_found?'':'<div class="badge b-warn">label source not found — hint set &amp; problem text unavailable</div>'}
      <h3>by step</h3>
      <table class="steptbl"><thead><tr><th>step</th><th>rows</th><th>agree(hint)</th>
        <th>agree(step)</th><th>majority</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    ${cit}
    <div class="muted" style="margin-left:4px">← pick a problem on the left to inspect the hint set, the reference, and the model's samples.</div>`;
}

/* ---------------- problem detail ---------------- */
async function openProblem(pid){
  state.pid = pid; state.reqIdx = 0;
  renderProblems();
  $('#main').innerHTML = '<div class="placeholder">Loading problem…</div>';
  state.detail = await api(`/api/problem?run=${encodeURIComponent(state.run)}&step=${state.step}&pid=${encodeURIComponent(pid)}`);
  renderDetail();
  writeHash();
  $('#main').scrollTop = 0;
}

function renderDetail(){
  const d = state.detail;
  if(!d || !d.requests || !d.requests.length){
    $('#main').innerHTML = '<div class="placeholder">No data for this problem.</div>'; return;
  }
  const tabs = d.requests.length>1 ? `<div class="reqtabs">${d.requests.map((r,i)=>{
      const ev=r.eval||{};
      const ok = ev.majority_agrees_hint_id;
      return `<div class="reqtab ${i===state.reqIdx?'on':''}" data-i="${i}">req ${i+1}
        <span class="badge ${ok?'b-good':'b-bad'}">${ok?'✓':'✗'}</span></div>`;
    }).join('')}</div>` : '';
  $('#main').innerHTML = tabs + `<div id="reqbody"></div>`;
  if(d.requests.length>1){
    $('#main').querySelectorAll('.reqtab').forEach(t=>{
      t.onclick = ()=>{ state.reqIdx = parseInt(t.dataset.i);
        $('#main').querySelectorAll('.reqtab').forEach(x=>x.classList.toggle('on', x===t));
        renderRequest(); };
    });
  }
  renderRequest();
}

function selectionHTML(sel){
  if(!sel) return '<span class="muted">— none —</span>';
  const cs = (sel.completed_substeps&&sel.completed_substeps.length)
      ? sel.completed_substeps.map(esc).join(', ') : '∅';
  return `
    <div class="kv">
      <span>major step <b>${esc(sel.major_step_id)}</b> <span class="conf">(conf ${esc(sel.confidence_of_major_step)})</span></span>
      <span>hint <b class="mono">${esc(sel.hint_id)}</b> <span class="conf">(conf ${esc(sel.confidence_of_hint)})</span></span>
      <span>completed substeps: <b>${cs}</b></span>
    </div>
    <div class="prose" style="margin:6px 0"><b>hint:</b> ${esc(sel.hint)}</div>
    <details><summary>reasoning</summary>
      <div class="prose" style="margin-top:6px"><b>major step:</b> ${esc(sel.reasoning_of_major_step)}</div>
      <div class="prose" style="margin-top:6px"><b>hint:</b> ${esc(sel.reasoning_of_hint)}</div>
    </details>`;
}

function hintPoolHTML(pool, refHid, dist, modeHid){
  if(!pool || !pool.steps){ return '<div class="muted">hint set unavailable (no label source).</div>'; }
  const refStep = refHid!=null ? String(refHid).split('.')[0] : null;
  const maxv = Math.max(1, ...Object.values(dist||{}));
  const seen = new Set();
  let html = pool.steps.map(st=>{
    const isRefStep = String(st.step_id)===refStep;
    const hints = (st.hints||[]).map(h=>{
      const isRef = String(h.hint_id)===String(refHid);
      const v = (dist&&dist[h.hint_id])||0; seen.add(String(h.hint_id));
      const isMode = String(h.hint_id)===String(modeHid);
      const w = Math.round(110*v/maxv);
      const tags = [isRef?'<span class="badge b-gold">REF</span>':'',
                    isMode&&v?'<span class="badge b-accent">model maj</span>':''].join(' ');
      return `<div class="hint ${isRef?'refhint':''}">
        <div class="hid">${esc(h.hint_id)}</div>
        <div class="txt"><span class="prose">${esc(h.hint)}</span><span class="type">${esc(h.type)} ${tags}</span></div>
        <div class="votewrap"><div class="votebar" style="width:${w}px;margin-left:auto;background:${isRef?'var(--gold)':'var(--accent)'}"></div>
          <span class="vt">${v} vote${v===1?'':'s'}</span></div>
      </div>`;
    }).join('');
    return `<div class="step-block ${isRefStep?'refstep':''}">
      <div><b>step ${esc(st.step_id)}</b> ${isRefStep?'<span class="badge b-gold">ref step</span>':''}</div>
      <div class="step-purpose prose">${esc(st.purpose)}</div>${hints}</div>`;
  }).join('');
  // votes for hint_ids not present in the pool (off-pool / hallucinated)
  const off = Object.keys(dist||{}).filter(k=>!seen.has(String(k)));
  if(off.length){
    html += `<div class="step-block"><b>off-pool selections</b>${off.map(k=>
      `<div class="hint"><div class="hid">${esc(k)}</div><div class="txt muted">not in hint pool</div>
       <div class="votewrap"><span class="vt">${dist[k]} vote(s)</span></div></div>`).join('')}</div>`;
  }
  return html;
}

function distHTML(dist, refHid, modeHid, nParsed){
  const entries = Object.entries(dist||{}).sort((a,b)=>b[1]-a[1]);
  if(!entries.length) return '<div class="muted">no parsed samples.</div>';
  const max = Math.max(...entries.map(e=>e[1]));
  return `<div class="dist">${entries.map(([hid,c])=>{
    const cls = String(hid)===String(refHid)?'ref':(String(hid)===String(modeHid)?'mode':'');
    return `<div class="distrow ${cls}"><span class="mono">${esc(hid)}</span>
      <span class="bar" style="width:${Math.round(100*c/max)}%"></span>
      <span>${c}/${nParsed} ${String(hid)===String(refHid)?'<span class="badge b-gold">ref</span>':''}</span></div>`;
  }).join('')}</div>`;
}

function sampleHTML(s, refHid, refStep){
  const hidMatch = s.hint_id!=null && String(s.hint_id)===String(refHid);
  const stepMatch = s.major_step_id!=null && String(s.major_step_id)===String(refStep);
  const trunc = s.finish_reason==='length';
  const sel = s.selection;
  const badges = [
    `<span class="badge ${hidMatch?'b-good':'b-bad'}">hint ${esc(s.hint_id)} ${hidMatch?'✓':'✗'}</span>`,
    `<span class="badge ${stepMatch?'b-good':'b-bad'}">step ${esc(s.major_step_id)} ${stepMatch?'✓':'✗'}</span>`,
    trunc?'<span class="badge b-warn">truncated</span>':'',
    s.parse_error?`<span class="badge b-bad">parse err</span>`:'',
    `<span class="conf">${esc(s.completion_tokens)} tok</span>`,
    (sel?`<span class="conf">conf ${esc(sel.confidence_of_hint)}/${esc(sel.confidence_of_major_step)}</span>`:'')
  ].join(' ');
  const body = sel ? `
      <div class="prose" style="margin:4px 0"><b>hint:</b> ${esc(sel.hint)}</div>
      <details><summary>reasoning &amp; raw</summary>
        <div class="prose" style="margin-top:6px"><b>major step:</b> ${esc(sel.reasoning_of_major_step)}</div>
        <div class="prose" style="margin-top:6px"><b>hint:</b> ${esc(sel.reasoning_of_hint)}</div>
        ${s.raw?`<pre class="raw">${esc(s.raw)}</pre>`:''}
      </details>`
    : `<div class="muted">${esc(s.parse_error||'no parsed selection')}</div>
       ${s.raw?`<details><summary>raw</summary><pre class="raw">${esc(s.raw)}</pre></details>`:''}`;
  return `<div class="sample ${hidMatch?'match':'nomatch'}">
    <div class="shead"><b>#${s.index}</b> ${badges}</div>${body}</div>`;
}

function renderRequest(){
  const r = state.detail.requests[state.reqIdx];
  const ev = r.eval||{}, lab = r.label||{};
  const hasLabel = !!r.label;
  const ref = ev.ref_selection || lab.parsed_answer || null;
  const refHid = ev.ref_hint_id!=null ? ev.ref_hint_id : (ref&&ref.hint_id);
  const refStep = ev.ref_major_step_id!=null ? ev.ref_major_step_id : (ref&&ref.major_step_id);
  const dist = ev.hint_id_distribution||{};
  const samples = ev.samples||[];

  const head = `<div class="card">
    <h2 class="mono" style="word-break:break-all">${esc(state.detail.problem_id)}</h2>
    <div class="kv">
      <span>step <b>${esc(ev.step)}</b></span>
      <span>request <b class="mono">${esc(ev.request_id)}</b></span>
      <span>model <b>${esc(ev.model)}</b></span>
      ${lab.strategy?`<span>strategy <b>${esc(lab.strategy)}</b></span>`:''}
      ${lab.budget!=null?`<span>budget <b>${esc(lab.budget)}</b></span>`:''}
      ${lab.ref_model?`<span>ref model <b>${esc(lab.ref_model)}</b> (${esc(lab.reasoning_effort)})</span>`:''}
    </div>
    <div class="kv">
      <span class="badge ${ev.majority_agrees_hint_id?'b-good':'b-bad'}">${ev.majority_agrees_hint_id?'majority ✓ ref':'majority ✗ ref'}</span>
      <span>agreement(hint) <b>${pct(ev.agreement_hint_id)}</b></span>
      <span>agreement(step) <b>${pct(ev.agreement_major_step)}</b></span>
      <span>self-consistency <b>${pct(ev.self_consistency)}</b></span>
      <span>parsed <b>${ev.n_parsed}/${ev.n_samples}</b></span>
      <span>truncated <b>${ev.n_truncated}</b></span>
      <span>mean tokens <b>${num(ev.mean_completion_tokens,0)}</b></span>
    </div></div>`;

  const traceTxt = lab.reasoning_trace
      ? esc(lab.reasoning_trace)
      : `<span class="muted">${hasLabel?'(empty — student has not started this trace)':'unavailable'}</span>`;
  const problemCard = `<div class="card">
    <h2>Problem</h2>
    ${hasLabel?'':'<div class="badge b-warn">label source not found — problem text, trace &amp; hint set unavailable</div>'}
    <div class="prose">${lab.problem?esc(lab.problem):'<span class="muted">—</span>'}</div>
    <details><summary>student reasoning trace${lab.n_words?` (${lab.n_words} words)`:''}</summary>
      <div class="prose" style="margin-top:8px">${traceTxt}</div>
    </details>
    ${lab.prompt?`<details><summary>exact prompt sent to model</summary><pre class="raw">${esc(lab.prompt)}</pre></details>`:''}
  </div>`;

  const refCard = `<div class="card">
    <h2>Reference selection <span class="badge b-gold">gold</span></h2>
    <div class="kv"><span>hint <b class="mono">${esc(refHid)}</b></span><span>major step <b>${esc(refStep)}</b></span></div>
    ${selectionHTML(ref)}
  </div>`;

  const poolCard = `<div class="card">
    <h2>Hint set <span class="muted" style="font-weight:400">(model votes overlaid)</span></h2>
    ${hintPoolHTML(lab.hint_pool, refHid, dist, ev.mode_hint_id)}
  </div>`;

  const distCard = `<div class="card">
    <h2>Model hint distribution</h2>
    ${distHTML(dist, refHid, ev.mode_hint_id, ev.n_parsed)}
  </div>`;

  const samplesCard = `<div class="card">
    <h2>Samples <span class="muted" style="font-weight:400">(${samples.length})</span></h2>
    ${samples.map(s=>sampleHTML(s, refHid, refStep)).join('') || '<div class="muted">none</div>'}
  </div>`;

  const body = head
    + problemCard
    + `<div class="grid2">${poolCard}${refCard}</div>`
    + distCard
    + samplesCard;
  $('#reqbody').innerHTML = body;
  typeset($('#reqbody'));
}

init();
</script>
</body>
</html>
"""


def main() -> int:
    global RESULTS_ROOT, LABEL_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 to expose)")
    ap.add_argument("--port", type=int, default=8731, help="bind port (default 8731)")
    ap.add_argument("--results", default=str(RESULTS_ROOT),
                    help="results dir to browse (default: ./results)")
    ap.add_argument("--label-root", default=str(LABEL_ROOT),
                    help="root holding the Codex label runs (default: ../results)")
    args = ap.parse_args()

    RESULTS_ROOT = Path(args.results).resolve()
    LABEL_ROOT = Path(args.label_root).resolve()

    runs = list_runs()
    print(f"results : {RESULTS_ROOT}")
    print(f"labels  : {LABEL_ROOT}")
    print(f"runs    : {len(runs)} found" + (f"  ({', '.join(runs)})" if runs else ""))
    if not runs:
        print("WARNING: no runs found under the results dir.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = args.host if args.host not in ("0.0.0.0", "::") else "localhost"
    print(f"\n  →  http://{shown}:{args.port}\n")
    if args.host in ("0.0.0.0", "::"):
        print("  (bound on all interfaces; or SSH-forward:  ssh -L "
              f"{args.port}:localhost:{args.port} <this-host>)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
