#!/usr/bin/env python3
"""Interactive viewer for selector results (any run).

Serves a small web page with two dropdowns:
  - a RUN picker  -- choose which eval run to inspect (e.g. run_20260608_200550,
    or the flat baseline that lives directly under the model dir)
  - a PROBLEM picker (plus prev/next buttons) -- choose which problem to view

For each problem it shows:
  - the math problem prompt
  - the codex (reference) hint selection
  - the hint-id distribution / self-consistency / agreement stats
  - every individual model sample with its selection and reasoning

Point --dir at the model directory (default: selector_gpt-oss-20b). The viewer
discovers every run beneath it: the directory itself if it holds problem dirs
directly, plus each immediate subdir (e.g. run_<timestamp>/) that does.

No third-party dependencies -- uses only the Python standard library.

Usage:
    python3 view_selector.py [--dir DIR] [--host HOST] [--port PORT]
"""
import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selector_gpt-oss-20b")
ROOT_RUN_ID = "(root)"  # sentinel id for problems living directly under --dir


def natural_key(name):
    """Sort 'request-10-2' / 'run_2026...' names numerically rather than lexically."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def is_problem_dir(d):
    """A problem dir is any directory containing a samples.json file."""
    return os.path.isfile(os.path.join(d, "samples.json"))


def has_problem_children(d):
    try:
        for name in os.listdir(d):
            if is_problem_dir(os.path.join(d, name)):
                return True
    except OSError:
        pass
    return False


def discover_runs(root):
    """Return an ordered list of (run_id, abs_path).

    A "run" is any directory that directly contains >=1 problem dir. We include
    `root` itself (if it holds problem dirs directly -- the flat baseline) plus
    each immediate subdir that holds problem dirs (e.g. run_<timestamp>/).
    """
    runs = []
    if has_problem_children(root):
        runs.append((ROOT_RUN_ID, root))
    subs = []
    for name in sorted(os.listdir(root), key=natural_key):
        d = os.path.join(root, name)
        if os.path.isdir(d) and not is_problem_dir(d) and has_problem_children(d):
            subs.append((name, d))
    # newest runs first among the timestamped subdirs
    subs.sort(key=lambda kv: natural_key(kv[0]), reverse=True)
    runs.extend(subs)
    return runs


def run_label(run_id, root):
    if run_id == ROOT_RUN_ID:
        return f"{os.path.basename(os.path.normpath(root))} (flat)"
    return run_id


def list_problems(run_path):
    out = []
    for name in os.listdir(run_path):
        if is_problem_dir(os.path.join(run_path, name)):
            out.append(name)
    out.sort(key=natural_key)
    return out


def load_problem(run_path, pid):
    with open(os.path.join(run_path, pid, "samples.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# HTML / JS front-end (served at "/")
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Selector Viewer</title>
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
           border-radius:6px; padding:7px 10px; font-size:14px; }
  #picker { min-width:320px; max-width:50vw; }
  #runpicker { min-width:220px; }
  button { background:var(--panel2); color:var(--fg); border:1px solid var(--border);
           border-radius:6px; padding:7px 12px; font-size:14px; cursor:pointer; }
  button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  #count { color:var(--muted); font-size:13px; white-space:nowrap; }
  main { padding:20px; max-width:1100px; margin:0 auto; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:16px 18px; margin-bottom:16px; }
  .card h2 { margin:0 0 10px; font-size:14px; text-transform:uppercase;
             letter-spacing:.05em; color:var(--muted); }
  .stats { display:flex; flex-wrap:wrap; gap:10px; }
  .stat { background:var(--panel2); border:1px solid var(--border); border-radius:8px;
          padding:8px 12px; min-width:120px; }
  .stat .v { font-size:20px; font-weight:600; }
  .stat .k { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .prompt, .reason, .hinttext { white-space:pre-wrap; word-wrap:break-word; }
  .prompt { background:var(--panel2); border-radius:8px; padding:12px;
            font-size:13px; max-height:340px; overflow:auto; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
          font-weight:600; border:1px solid var(--border); }
  .pill.good { color:var(--good); border-color:var(--good); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  .pill.accent { color:var(--accent); border-color:var(--accent); }
  .bar-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
  .bar-label { width:54px; font-size:13px; color:var(--muted); text-align:right; }
  .bar-track { flex:1; background:var(--panel2); border-radius:5px; overflow:hidden; height:22px; }
  .bar-fill { height:100%; background:var(--accent); display:flex; align-items:center;
              padding-left:8px; font-size:12px; color:#0b1020; font-weight:600; white-space:nowrap; }
  .kv { margin:6px 0; }
  .kv b { color:var(--muted); font-weight:600; }
  .reason { font-size:13px; color:#cdd3de; }
  .hinttext { background:var(--panel2); border-left:3px solid var(--accent);
              padding:8px 12px; border-radius:0 6px 6px 0; margin-top:6px; font-size:13px; }
  .sample { border:1px solid var(--border); border-radius:8px; margin-bottom:10px; overflow:hidden; }
  .sample.match { border-left:4px solid var(--good); }
  .sample.nomatch { border-left:4px solid var(--bad); }
  .sample.err { border-left:4px solid var(--warn); }
  .sample-head { display:flex; gap:10px; align-items:center; padding:10px 14px;
                 cursor:pointer; background:var(--panel2); flex-wrap:wrap; }
  .sample-head .idx { font-weight:600; }
  .sample-head .meta { margin-left:auto; color:var(--muted); font-size:12px; }
  .sample-body { padding:12px 14px; display:none; border-top:1px solid var(--border); }
  .sample.open .sample-body { display:block; }
  .small { font-size:12px; color:var(--muted); }
  details summary { cursor:pointer; color:var(--accent); font-size:13px; margin-top:8px; }
  .loading { color:var(--muted); padding:40px; text-align:center; }
</style>
</head>
<body>
<header>
  <h1>&#129504; Selector Viewer</h1>
  <label class="sel">run <select id="runpicker"></select></label>
  <button id="prev" title="Previous">&#8592; Prev</button>
  <select id="picker"></select>
  <button id="next" title="Next">Next &#8594;</button>
  <span id="count"></span>
</header>
<main id="content"><div class="loading">Loading&hellip;</div></main>

<script>
const esc = s => (s===null||s===undefined) ? "" :
  String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const fmt = (x, d=3) => (typeof x === "number") ? x.toFixed(d) : esc(x);
let RUNS = [];        // [{id, label}]
let PROBLEMS = [];
let CUR_RUN = null;

function parseHash() {
  // hash form: #<run>||<problem>
  const raw = decodeURIComponent(location.hash.slice(1));
  const i = raw.indexOf("||");
  if (i < 0) return {run: raw || null, pid: null};
  return {run: raw.slice(0, i) || null, pid: raw.slice(i + 2) || null};
}
function setHash(run, pid) {
  location.hash = encodeURIComponent(run + "||" + (pid || ""));
}

async function init() {
  const r = await fetch("api/runs");
  RUNS = (await r.json()).runs;
  const rsel = document.getElementById("runpicker");
  rsel.innerHTML = RUNS.map(run=>`<option value="${esc(run.id)}">${esc(run.label)}</option>`).join("");
  const h = parseHash();
  let runIdx = RUNS.findIndex(x => x.id === h.run);
  if (runIdx < 0) runIdx = 0;
  rsel.value = RUNS[runIdx].id;
  await loadRun(RUNS[runIdx].id, h.pid);
}

async function loadRun(runId, wantPid) {
  CUR_RUN = runId;
  document.getElementById("runpicker").value = runId;
  const r = await fetch("api/problems?run=" + encodeURIComponent(runId));
  PROBLEMS = (await r.json()).problems;
  const sel = document.getElementById("picker");
  sel.innerHTML = PROBLEMS.map((p,i)=>`<option value="${i}">${esc(p)}</option>`).join("");
  document.getElementById("count").textContent = PROBLEMS.length + " problems";
  if (!PROBLEMS.length) {
    document.getElementById("content").innerHTML = '<div class="loading">No problems in this run.</div>';
    return;
  }
  let idx = wantPid ? PROBLEMS.indexOf(wantPid) : 0;
  if (idx < 0) idx = 0;
  sel.value = idx;
  loadProblem(idx);
}

function setIdx(i) {
  i = Math.max(0, Math.min(PROBLEMS.length-1, i));
  document.getElementById("picker").value = i;
  loadProblem(i);
}

async function loadProblem(i) {
  const pid = PROBLEMS[i];
  setHash(CUR_RUN, pid);
  document.getElementById("prev").disabled = (i<=0);
  document.getElementById("next").disabled = (i>=PROBLEMS.length-1);
  const c = document.getElementById("content");
  c.innerHTML = '<div class="loading">Loading '+esc(pid)+'&hellip;</div>';
  const r = await fetch("api/problem?run="+encodeURIComponent(CUR_RUN)+"&id="+encodeURIComponent(pid));
  if (!r.ok) { c.innerHTML = '<div class="loading">Error loading '+esc(pid)+'</div>'; return; }
  render(await r.json());
}

function distBars(dist, mode) {
  const total = Object.values(dist).reduce((a,b)=>a+b,0) || 1;
  return Object.entries(dist).sort((a,b)=>parseFloat(a[0])-parseFloat(b[0])).map(([k,v])=>{
    const pct = 100*v/total;
    const star = (k===String(mode)) ? " &#9733;" : "";
    return `<div class="bar-row"><div class="bar-label">hint ${esc(k)}${star}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%">${v} (${pct.toFixed(0)}%)</div></div></div>`;
  }).join("");
}

function selBlock(sel) {
  if (!sel) return '<div class="small">(no parsed selection)</div>';
  let h = "";
  if (sel.major_step_id !== undefined)
    h += `<div class="kv"><b>major step:</b> ${esc(sel.major_step_id)} `+
         `<span class="small">(conf ${esc(sel.confidence_of_major_step)})</span></div>`;
  if (sel.reasoning_of_major_step)
    h += `<div class="reason">${esc(sel.reasoning_of_major_step)}</div>`;
  if (sel.hint_id !== undefined)
    h += `<div class="kv" style="margin-top:8px"><b>hint id:</b> `+
         `<span class="pill accent">${esc(sel.hint_id)}</span> `+
         `<span class="small">(conf ${esc(sel.confidence_of_hint)})</span></div>`;
  if (sel.reasoning_of_hint)
    h += `<div class="reason">${esc(sel.reasoning_of_hint)}</div>`;
  if (sel.hint)
    h += `<div class="hinttext">${esc(sel.hint)}</div>`;
  return h;
}

function render(d) {
  const codexHint = String(d.codex_hint_id);
  const codexStep = d.codex_major_step_id;
  const stats = [
    ["self consistency", fmt(d.self_consistency)],
    ["agreement w/ codex", fmt(d.agreement_with_codex)],
    ["majority agrees", d.majority_agrees_with_codex ? "yes" : "no"],
    ["mode hint", esc(d.mode_hint_id)],
    ["codex hint", esc(codexHint)],
    ["codex step", esc(codexStep)],
    ["parsed", esc(d.n_parsed)+"/"+esc(d.n_samples)],
    ["truncated", esc(d.n_truncated)],
    ["mean tokens", fmt(d.mean_completion_tokens,0)],
  ];

  let h = `<div class="card"><h2>${esc(d.problem_id)}</h2>
    <div class="small">model: ${esc(d.model)} &middot; run: ${esc(CUR_RUN)}</div></div>`;

  h += `<div class="card"><h2>Statistics</h2><div class="stats">` +
       stats.map(([k,v])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("") +
       `</div></div>`;

  h += `<div class="card"><h2>Hint-id distribution (across samples)</h2>` +
       distBars(d.hint_id_distribution||{}, d.mode_hint_id) + `</div>`;

  h += `<div class="card"><h2>Codex reference selection</h2>` + selBlock(d.codex_selection) + `</div>`;

  const prompt = (d.meta && d.meta.prompt) ? d.meta.prompt : "(no prompt)";
  h += `<div class="card"><h2>Problem prompt</h2><div class="prompt">${esc(prompt)}</div></div>`;

  // samples -- compare each sample's STEP and HINT against codex
  let srows = "";
  (d.samples||[]).forEach(s => {
    let cls = "err", badge = '<span class="pill">parse error</span>';
    if (!s.parse_error && s.selection) {
      const stepMatch = String(s.major_step_id) === String(codexStep);
      const hintMatch = String(s.hint_id) === codexHint;
      if (hintMatch) { cls = "match"; badge = '<span class="pill good">matches codex</span>'; }
      else if (stepMatch) { cls = "nomatch"; badge = '<span class="pill bad">step ok, hint '+esc(s.hint_id)+'</span>'; }
      else { cls = "nomatch"; badge = '<span class="pill bad">step '+esc(s.major_step_id)+' (codex '+esc(codexStep)+')</span>'; }
    }
    const reasoning = s.reasoning_content
      ? `<details><summary>reasoning trace (${esc(s.reasoning_content.length)} chars)</summary>`+
        `<div class="reason" style="margin-top:6px">${esc(s.reasoning_content)}</div></details>` : "";
    srows += `<div class="sample ${cls}">
      <div class="sample-head" onclick="this.parentNode.classList.toggle('open')">
        <span class="idx">#${esc(s.index)}</span> ${badge}
        <span class="meta">${esc(s.finish_reason)} &middot; ${esc(s.completion_tokens)} tok &middot; ${fmt(s.latency_s,1)}s</span>
      </div>
      <div class="sample-body">${selBlock(s.selection)}${reasoning}</div>
    </div>`;
  });
  h += `<div class="card"><h2>Samples (${(d.samples||[]).length}) &mdash; click to expand</h2>${srows}</div>`;

  document.getElementById("content").innerHTML = h;
}

document.getElementById("runpicker").addEventListener("change", e => loadRun(e.target.value, null));
document.getElementById("picker").addEventListener("change", e => setIdx(parseInt(e.target.value)));
document.getElementById("prev").addEventListener("click", () => setIdx(parseInt(document.getElementById("picker").value)-1));
document.getElementById("next").addEventListener("click", () => setIdx(parseInt(document.getElementById("picker").value)+1));
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  setIdx(parseInt(document.getElementById("picker").value)-1);
  if (e.key === "ArrowRight") setIdx(parseInt(document.getElementById("picker").value)+1);
});
init();
</script>
</body>
</html>
"""


def make_handler(root):
    def runs_map():
        # rediscover on each request so newly-finished runs show up on reload
        runs = discover_runs(root)
        return runs, {rid: path for rid, path in runs}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter logs
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
                self._send(200, json.dumps({
                    "runs": [{"id": rid, "label": run_label(rid, root)} for rid, _ in runs]
                }))
                return

            run_id = qs.get("run", [ROOT_RUN_ID])[0]
            run_path = by_id.get(run_id)
            if run_path is None:
                self._send(404, json.dumps({"error": "unknown run"}))
                return

            if path == "/api/problems":
                self._send(200, json.dumps({"problems": list_problems(run_path)}))
            elif path == "/api/problem":
                pid = qs.get("id", [""])[0]
                if not is_problem_dir(os.path.join(run_path, pid)):
                    self._send(404, json.dumps({"error": "unknown problem"}))
                    return
                try:
                    self._send(200, json.dumps(load_problem(run_path, pid)))
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(e)}))
            else:
                self._send(404, json.dumps({"error": "not found"}))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help="model/results dir holding one or more runs (default: selector_gpt-oss-20b)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        raise SystemExit(f"directory not found: {root}")
    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"no runs (dirs with problem/samples.json) found under {root}")

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    shown = args.host if args.host not in ("0.0.0.0", "") else "localhost"
    print(f"Discovered {len(runs)} run(s) under {root}:")
    for rid, path in runs:
        n = len(list_problems(path))
        print(f"  - {run_label(rid, root):40s} {n:4d} problems")
    print(f"Open  http://{shown}:{args.port}/   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
