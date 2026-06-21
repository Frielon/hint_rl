#!/usr/bin/env python3
"""Interactive viewer for Template F labeling results.

Serves a small web page with three pickers:
  - RUN     -- which run to inspect (results/{runs,debug}/run_<timestamp>)
  - STEP    -- which step within the run (step1 .. step5)
  - PROBLEM -- which labeled call within that step (+ prev/next buttons)

The detail page shows, for the selected call:
  - the Template F answer vs the original selector answer (with match badges)
  - the Template F completed-substep citations (and the original's citations)
  - metrics: word count, latency, parse/error status
  - the INPUT reasoning trace (with the cited quotes highlighted)
  - the problem statement
  - the HINT SET (candidate pool), with the Template F pick and original pick marked

Point --dir at a results dir (default: results next to this script). It discovers
every run under results/runs/ and results/debug/.

No third-party dependencies -- uses only the Python standard library.

Usage:
    python3 view_results.py [--dir DIR] [--host HOST] [--port PORT]
"""
import argparse
import json
import os
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP_RE = re.compile(r"^step(\d+)$")


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def discover_runs(root):
    """Return ordered [(run_id, abs_path)]; run_id is '<mode>/<run_name>'.

    A run is an immediate child dir of results/runs/ or results/debug/ that holds
    >=1 stepN/ subdir. Newest (by name) first within each mode.
    """
    runs = []
    for mode in ("runs", "debug"):
        base = os.path.join(root, mode)
        if not os.path.isdir(base):
            continue
        subs = []
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if os.path.isdir(d) and list_steps(d):
                subs.append((f"{mode}/{name}", d))
        subs.sort(key=lambda kv: natural_key(kv[0]), reverse=True)
        runs.extend(subs)
    # fallback: --dir itself is a single run dir (holds stepN/ directly)
    if not runs and list_steps(root):
        runs.append((os.path.basename(os.path.normpath(root)), root))
    return runs


def list_steps(run_path):
    """Sorted list of integer step ids present as stepN/ dirs under run_path."""
    out = []
    try:
        for name in os.listdir(run_path):
            m = STEP_RE.match(name)
            if m and os.path.isdir(os.path.join(run_path, name)):
                out.append(int(m.group(1)))
    except OSError:
        pass
    return sorted(out)


def list_results(run_path, step):
    """Return [(problem_id, request_id)] for stepN/<problem_id>/<request_id>.json."""
    step_dir = os.path.join(run_path, f"step{step}")
    out = []
    if not os.path.isdir(step_dir):
        return out
    for pid in os.listdir(step_dir):
        pdir = os.path.join(step_dir, pid)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.endswith(".json"):
                out.append((pid, fn[:-5]))
    out.sort(key=lambda pr: (natural_key(pr[0]), pr[1]))
    return out


def problem_entries(run_path, step):
    """[(key, label)] for the step; label disambiguated when a pid repeats."""
    pairs = list_results(run_path, step)
    pid_counts = Counter(pid for pid, _ in pairs)
    entries = []
    for pid, rid in pairs:
        label = pid if pid_counts[pid] == 1 else f"{pid}  [{rid[:8]}]"
        entries.append((f"{pid}/{rid}", label))
    return entries


def load_result(run_path, step, key):
    """Load the record for stepN/<key>.json, guarding against path traversal."""
    if "/" not in key:
        raise ValueError("bad key")
    pid, rid = key.rsplit("/", 1)
    step_dir = os.path.realpath(os.path.join(run_path, f"step{step}"))
    path = os.path.realpath(os.path.join(step_dir, pid, rid + ".json"))
    if not path.startswith(step_dir + os.sep) or not os.path.isfile(path):
        raise FileNotFoundError(key)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Template F Results</title>
<style>
  :root { --bg:#0f1117; --panel:#1a1d27; --panel2:#232734; --fg:#e6e9ef;
          --muted:#9aa3b2; --accent:#6ea8fe; --good:#3fb950; --bad:#f85149;
          --warn:#d29922; --border:#2d3340; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--fg); line-height:1.5; }
  header { position:sticky; top:0; z-index:10; background:var(--panel);
           border-bottom:1px solid var(--border); padding:12px 20px;
           display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; white-space:nowrap; }
  label.sel { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }
  select { background:var(--panel2); color:var(--fg); border:1px solid var(--border);
           border-radius:6px; padding:7px 10px; font-size:14px; max-width:46vw; }
  button { background:var(--panel2); color:var(--fg); border:1px solid var(--border);
           border-radius:6px; padding:7px 12px; font-size:14px; cursor:pointer; }
  button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  #count { color:var(--muted); font-size:13px; white-space:nowrap; }
  main { padding:20px; max-width:1100px; margin:0 auto; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:16px 18px; margin-bottom:16px; }
  .card h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase;
             letter-spacing:.05em; color:var(--muted); }
  .stats { display:flex; flex-wrap:wrap; gap:10px; }
  .stat { background:var(--panel2); border:1px solid var(--border); border-radius:8px;
          padding:8px 12px; min-width:110px; }
  .stat .v { font-size:20px; font-weight:600; }
  .stat .k { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .prompt, .reason, .hinttext { white-space:pre-wrap; word-wrap:break-word; }
  .prompt { background:var(--panel2); border-radius:8px; padding:12px;
            font-size:13px; max-height:420px; overflow:auto; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
          font-weight:600; border:1px solid var(--border); }
  .pill.good { color:var(--good); border-color:var(--good); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  .pill.accent { color:var(--accent); border-color:var(--accent); }
  .kv { margin:6px 0; }
  .kv b { color:var(--muted); font-weight:600; }
  .reason { font-size:13px; color:#cdd3de; }
  .hinttext { background:var(--panel2); border-left:3px solid var(--accent);
              padding:8px 12px; border-radius:0 6px 6px 0; margin-top:6px; font-size:13px; }
  .cite-item { margin:10px 0; }
  .poolstep { border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin-bottom:10px; }
  .poolstep.selstep { border-color:var(--good); }
  .hint-row { padding:7px 10px; border-radius:6px; margin-top:8px; border:1px solid transparent; }
  .hint-row.sel { background:rgba(63,185,80,.10); border-color:var(--good); }
  .hint-row.orig { background:rgba(110,168,254,.08); border-color:var(--accent); }
  mark.cite { background:rgba(210,153,34,.35); color:#ffe9b0; border-radius:3px; padding:0 1px; }
  .small { font-size:12px; color:var(--muted); }
  details summary { cursor:pointer; color:var(--accent); font-size:13px; }
  .loading { color:var(--muted); padding:40px; text-align:center; }
</style>
</head>
<body>
<header>
  <h1>&#129504; Template F Results</h1>
  <label class="sel">run <select id="runpicker"></select></label>
  <label class="sel">step <select id="steppicker"></select></label>
  <button id="prev" title="Previous">&#8592;</button>
  <select id="picker"></select>
  <button id="next" title="Next">&#8594;</button>
  <span id="count"></span>
</header>
<main id="content"><div class="loading">Loading&hellip;</div></main>

<script>
const esc = s => (s===null||s===undefined) ? "" :
  String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const fmt = (x,d=3) => (typeof x==="number") ? x.toFixed(d) : esc(x);
let RUNS=[], STEPS=[], PROBLEMS=[], CUR={run:null, step:null, key:null};

function parseHash(){
  const parts = decodeURIComponent(location.hash.slice(1)).split("||");
  return {run:parts[0]||null, step:parts[1]||null, key:parts[2]||null};
}
function setHash(){ location.hash = encodeURIComponent([CUR.run, CUR.step||"", CUR.key||""].join("||")); }

async function getJSON(u){ const r=await fetch(u); if(!r.ok) throw new Error(r.status); return r.json(); }

async function init(){
  RUNS = (await getJSON("api/runs")).runs;
  const rsel=document.getElementById("runpicker");
  rsel.innerHTML = RUNS.map(r=>`<option value="${esc(r.id)}">${esc(r.label)}</option>`).join("");
  const h=parseHash();
  let i = RUNS.findIndex(x=>x.id===h.run); if(i<0) i=0;
  if(!RUNS.length){ document.getElementById("content").innerHTML='<div class="loading">No runs found.</div>'; return; }
  rsel.value=RUNS[i].id;
  await loadRun(RUNS[i].id, h.step, h.key);
}

async function loadRun(runId, wantStep, wantKey){
  CUR.run=runId; document.getElementById("runpicker").value=runId;
  STEPS = (await getJSON("api/steps?run="+encodeURIComponent(runId))).steps;
  const ssel=document.getElementById("steppicker");
  ssel.innerHTML = STEPS.map(s=>`<option value="${esc(s.id)}">step ${esc(s.id)} (${esc(s.n)})</option>`).join("");
  if(!STEPS.length){ document.getElementById("content").innerHTML='<div class="loading">No steps in this run.</div>'; return; }
  let s = STEPS.findIndex(x=>String(x.id)===String(wantStep)); if(s<0) s=0;
  ssel.value=STEPS[s].id;
  await loadStep(STEPS[s].id, wantKey);
}

async function loadStep(stepId, wantKey){
  CUR.step=stepId; document.getElementById("steppicker").value=stepId;
  PROBLEMS = (await getJSON("api/problems?run="+encodeURIComponent(CUR.run)+"&step="+encodeURIComponent(stepId))).problems;
  const sel=document.getElementById("picker");
  sel.innerHTML = PROBLEMS.map((p,i)=>`<option value="${i}">${esc(p.label)}</option>`).join("");
  document.getElementById("count").textContent = PROBLEMS.length+" problems";
  if(!PROBLEMS.length){ document.getElementById("content").innerHTML='<div class="loading">No results in this step.</div>'; return; }
  let idx = wantKey ? PROBLEMS.findIndex(p=>p.key===wantKey) : 0; if(idx<0) idx=0;
  setIdx(idx);
}

function setIdx(i){
  i=Math.max(0, Math.min(PROBLEMS.length-1, i));
  document.getElementById("picker").value=i;
  loadResult(i);
}

async function loadResult(i){
  const p=PROBLEMS[i]; CUR.key=p.key; setHash();
  document.getElementById("prev").disabled=(i<=0);
  document.getElementById("next").disabled=(i>=PROBLEMS.length-1);
  const c=document.getElementById("content");
  c.innerHTML='<div class="loading">Loading&hellip;</div>';
  try{
    const d = await getJSON("api/result?run="+encodeURIComponent(CUR.run)+"&step="+encodeURIComponent(CUR.step)+"&key="+encodeURIComponent(p.key));
    render(d);
  }catch(e){ c.innerHTML='<div class="loading">Error loading result ('+esc(e)+')</div>'; }
}

function selBlock(sel){
  if(!sel) return '<div class="small">(no parsed selection)</div>';
  let h="";
  if(sel.major_step_id!==undefined)
    h+=`<div class="kv"><b>major step:</b> ${esc(sel.major_step_id)} <span class="small">(conf ${esc(sel.confidence_of_major_step)})</span></div>`;
  if(sel.reasoning_of_major_step) h+=`<div class="reason">${esc(sel.reasoning_of_major_step)}</div>`;
  if(sel.hint_id!==undefined)
    h+=`<div class="kv" style="margin-top:8px"><b>hint id:</b> <span class="pill accent">${esc(sel.hint_id)}</span> <span class="small">(conf ${esc(sel.confidence_of_hint)})</span></div>`;
  if(sel.reasoning_of_hint) h+=`<div class="reason">${esc(sel.reasoning_of_hint)}</div>`;
  if(sel.hint) h+=`<div class="hinttext">${esc(sel.hint)}</div>`;
  return h;
}

function citeList(arr){
  if(!arr || !arr.length) return '<div class="small">(none)</div>';
  return arr.map(c=>{
    const id = (c.hint_id!==undefined)?c.hint_id:c.step_id;
    return `<div class="cite-item"><span class="pill accent">${esc(id)}</span>
      <div class="hinttext">${esc(c.quote)}</div>
      <div class="small">${esc(c.why)}</div></div>`;
  }).join("");
}

function highlightTrace(trace, cites){
  let E = esc(trace);
  (cites||[]).forEach(c=>{
    const q = c && c.quote; if(!q) return;
    const eq = esc(q);
    if(eq && E.indexOf(eq)>=0) E = E.split(eq).join('<mark class="cite">'+eq+'</mark>');
  });
  return E;
}

function hintPool(pool, selHint, origHint, selStep){
  if(!pool || !pool.steps) return '<div class="small">(hint pool not recorded -- re-run to capture)</div>';
  return pool.steps.map(st=>{
    const isSel = String(st.step_id)===String(selStep);
    const hints = (st.hints||[]).map(hn=>{
      const sel=String(hn.hint_id)===String(selHint), orig=String(hn.hint_id)===String(origHint);
      let cls="hint-row"+(sel?" sel":(orig?" orig":"")), tags="";
      if(sel) tags+=' <span class="pill good">Template F</span>';
      if(orig) tags+=' <span class="pill accent">original</span>';
      return `<div class="${cls}"><span class="pill accent">${esc(hn.hint_id)}</span> <span class="small">${esc(hn.type)}</span>${tags}
        <div class="hinttext">${esc(hn.hint)}</div></div>`;
    }).join("");
    return `<div class="poolstep ${isSel?'selstep':''}"><div class="kv"><b>Step ${esc(st.step_id)}</b>${isSel?' <span class="pill good">selected step</span>':''}: ${esc(st.purpose)}</div>${hints}</div>`;
  }).join("");
}

function render(d){
  const pf=d.parsed_answer, og=d.original_answer;
  const selHint=pf?pf.hint_id:null, origHint=og?og.hint_id:null, selStep=pf?pf.major_step_id:null;
  const hintMatch = pf&&og&&String(pf.hint_id)===String(og.hint_id);
  const stepMatch = pf&&og&&String(pf.major_step_id)===String(og.major_step_id);

  let h = `<div class="card"><h2>${esc(d.problem_id)}</h2>
    <div class="small">run: ${esc(CUR.run)} &middot; step ${esc(d.step)} &middot; req ${esc(d.request_id)} &middot; ${esc(d.model)}/${esc(d.reasoning_effort)} &middot; ${esc(d.timestamp)}</div>`;
  if(d.error) h += `<div class="kv"><span class="pill bad">error</span> <span class="small">${esc(d.error)}</span></div>`;
  h += `</div>`;

  h += `<div class="card"><h2>Selection</h2><div class="stats">
    <div class="stat"><div class="v">${pf?esc(pf.hint_id):'&mdash;'}</div><div class="k">Template F hint</div></div>
    <div class="stat"><div class="v">${og?esc(og.hint_id):'&mdash;'}</div><div class="k">original hint</div></div>
    <div class="stat"><div class="v">${pf&&og?(hintMatch?'yes':'no'):'&mdash;'}</div><div class="k">hint match</div></div>
    <div class="stat"><div class="v">${pf&&og?(stepMatch?'yes':'no'):'&mdash;'}</div><div class="k">step match</div></div>
    <div class="stat"><div class="v">${esc(d.n_words)}</div><div class="k">words</div></div>
    <div class="stat"><div class="v">${fmt(d.latency_s,1)}</div><div class="k">latency s</div></div>
    </div></div>`;

  h += `<div class="card"><h2>Template F answer</h2>${selBlock(pf)}
    <h2 style="margin-top:14px">Completed substeps (citations)</h2>${citeList(pf&&pf.completed_substeps)}</div>`;

  h += `<div class="card"><h2>Original answer</h2>${selBlock(og)}
    <h2 style="margin-top:14px">Cited (original)</h2>${citeList(og&&(og.completed_substeps||og.completed_steps))}</div>`;

  h += `<div class="card"><h2>Input reasoning trace</h2>` +
    (d.reasoning_trace ? `<div class="prompt">${highlightTrace(d.reasoning_trace, pf&&pf.completed_substeps)}</div>`
                       : `<div class="small">(trace not recorded -- re-run to capture)</div>`) + `</div>`;

  if(d.problem) h += `<div class="card"><h2>Problem statement</h2><div class="prompt">${esc(d.problem)}</div></div>`;

  h += `<div class="card"><h2>Hint set (candidate pool)</h2>${hintPool(d.hint_pool, selHint, origHint, selStep)}</div>`;

  if(d.raw_answer)
    h += `<div class="card"><h2>Raw model output</h2><details><summary>show</summary><div class="prompt" style="margin-top:8px">${esc(d.raw_answer)}</div></details></div>`;
  if(d.prompt)
    h += `<div class="card"><h2>Full prompt sent</h2><details><summary>show</summary><div class="prompt" style="margin-top:8px">${esc(d.prompt)}</div></details></div>`;

  document.getElementById("content").innerHTML = h;
}

document.getElementById("runpicker").addEventListener("change", e=>loadRun(e.target.value, null, null));
document.getElementById("steppicker").addEventListener("change", e=>loadStep(e.target.value, null));
document.getElementById("picker").addEventListener("change", e=>setIdx(parseInt(e.target.value)));
document.getElementById("prev").addEventListener("click", ()=>setIdx(parseInt(document.getElementById("picker").value)-1));
document.getElementById("next").addEventListener("click", ()=>setIdx(parseInt(document.getElementById("picker").value)+1));
window.addEventListener("keydown", e=>{
  if(e.target.tagName==="SELECT") return;
  if(e.key==="ArrowLeft")  setIdx(parseInt(document.getElementById("picker").value)-1);
  if(e.key==="ArrowRight") setIdx(parseInt(document.getElementById("picker").value)+1);
});
init();
</script>
</body>
</html>
"""


def make_handler(root):
    def runs_map():
        runs = discover_runs(root)
        return runs, {rid: path for rid, path in runs}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            if path == "/":
                self._send(200, PAGE, "text/html")
                return

            runs, by_id = runs_map()
            if path == "/api/runs":
                self._send(200, json.dumps({"runs": [{"id": rid, "label": rid} for rid, _ in runs]}))
                return

            run_path = by_id.get(qs.get("run", [""])[0])
            if run_path is None:
                self._send(404, json.dumps({"error": "unknown run"}))
                return

            if path == "/api/steps":
                steps = [{"id": s, "n": len(list_results(run_path, s))} for s in list_steps(run_path)]
                self._send(200, json.dumps({"steps": steps}))
            elif path == "/api/problems":
                step = qs.get("step", [""])[0]
                entries = problem_entries(run_path, step)
                self._send(200, json.dumps({"problems": [{"key": k, "label": l} for k, l in entries]}))
            elif path == "/api/result":
                step = qs.get("step", [""])[0]
                key = qs.get("key", [""])[0]
                try:
                    self._send(200, json.dumps(load_result(run_path, step, key)))
                except (FileNotFoundError, ValueError):
                    self._send(404, json.dumps({"error": "unknown result"}))
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(e)}))
            else:
                self._send(404, json.dumps({"error": "not found"}))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="results dir holding runs/ and debug/ (default: ./results)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        raise SystemExit(f"directory not found: {root}")
    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"no runs (dirs with stepN/) found under {root}")

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    shown = args.host if args.host not in ("0.0.0.0", "") else "localhost"
    print(f"Discovered {len(runs)} run(s) under {root}:")
    for rid, path in runs:
        steps = list_steps(path)
        n = sum(len(list_results(path, s)) for s in steps)
        print(f"  - {rid:40s} steps={steps} ({n} results)")
    print(f"Open  http://{shown}:{args.port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
