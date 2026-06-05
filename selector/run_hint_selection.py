#!/usr/bin/env python3
"""Run codex-based hint selection over the massive hint-asking reasoning traces.

For each problem under massive_hint_test, take the *first* rollout that actually
asked for a hint (the first one with ``emitted == True``, ordered by index), feed
its reasoning trace plus the simplified candidate hint set (from the step1
reference_decomposition in the debug dir) into the selector prompt, and run that
prompt through Codex.

The Codex JSON output is saved under ``this_dir/<problem_id>/hint_selection.json``
together with metainfo: the problem statement, the candidate hint set and its
source path, and the full hint-selection prompt.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seletor_prompt import selector_prompt  # noqa: E402
from simplify_step1 import simplify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TRACES_DIR = "/share5/users/xutao.ma/project/hint_rl/data_pipeline/hint_call_test/massive_hint_test"
DEBUG_DIR = "/share5/users/xutao.ma/project/hint_rl/data_pipeline/hint_step_outputs/debug"
OUT_DIR = os.path.join(HERE, "selector_codex")

# ---- codex parameters (mirrors tools/ask_codex.py) ----
MODEL = "gpt-5.5"
REASONING_EFFORT = "high"
CODEX_BIN = "codex"
TIMEOUT_SECONDS = 600
SANDBOX = "read-only"
# -------------------------------------------------------

OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


def call_codex(prompt: str) -> str:
    """Run a one-shot Codex exec session and return its stdout."""
    import subprocess

    cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", "--sandbox", SANDBOX]
    if MODEL:
        cmd += ["--model", MODEL]
    if REASONING_EFFORT:
        cmd += ["--config", f"model_reasoning_effort='{REASONING_EFFORT}'"]
    cmd.append("-")
    proc = subprocess.run(
        cmd, input=prompt, text=True, capture_output=True, timeout=TIMEOUT_SECONDS
    )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr[-2000:]}")
    return proc.stdout


def parse_output(raw: str):
    """Pull the JSON object out of the <output>...</output> block, if possible."""
    m = OUTPUT_RE.search(raw)
    block = m.group(1) if m else raw
    try:
        return json.loads(block), None
    except json.JSONDecodeError as e:
        return None, str(e)


def find_first_hint_rollout(traces_path: str):
    """Return (trace, rollout_index, metainfo, rollout_file) for the first hint-asking
    rollout, or None if the problem has no rollout that emitted a hint call."""
    files = sorted(glob.glob(os.path.join(traces_path, "*.json")))
    if not files:
        return None
    rollout_file = files[-1]  # latest run for this problem
    with open(rollout_file) as f:
        data = json.load(f)
    metainfo = data.get("metainfo", {})
    rollouts = sorted(data.get("rollouts", []), key=lambda r: r.get("index", 0))
    for r in rollouts:
        if r.get("emitted"):
            text = r.get("text", "")
            n = r.get("preamble_chars")
            trace = text[:n] if isinstance(n, int) else text
            return trace.strip(), r.get("index"), metainfo, rollout_file
    return None


def load_hint_set(problem_id: str):
    """Return (simplified_hints, step1_path) or (None, None) if no step1 file."""
    pattern = os.path.join(DEBUG_DIR, problem_id, "*step1*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    step1_path = files[-1]
    with open(step1_path) as f:
        data = json.load(f)
    return simplify(data), step1_path


def process_problem(problem_id: str, overwrite: bool):
    out_problem_dir = os.path.join(OUT_DIR, problem_id)
    existing = glob.glob(os.path.join(out_problem_dir, "hint_selection*.json"))
    if existing and not overwrite:
        return problem_id, "skip (exists)"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(out_problem_dir, f"hint_selection_{ts}.json")

    traces_path = os.path.join(TRACES_DIR, problem_id)
    found = find_first_hint_rollout(traces_path)
    if found is None:
        return problem_id, "skip (no hint-asking rollout)"
    trace, rollout_index, metainfo, rollout_file = found

    hint_set, step1_path = load_hint_set(problem_id)
    if hint_set is None:
        return problem_id, "skip (no step1 hint set)"

    problem_statement = metainfo.get("statement", "")
    hints_str = json.dumps(hint_set, indent=2, ensure_ascii=False)
    prompt = selector_prompt(problem_statement, trace, hints_str)

    try:
        raw = call_codex(prompt)
    except Exception as e:  # noqa: BLE001
        return problem_id, f"ERROR codex: {e}"

    selection, parse_err = parse_output(raw)

    record = {
        "problem_id": problem_id,
        "selection": selection,
        "raw_codex_output": raw,
        "parse_error": parse_err,
        "meta": {
            "problem": problem_statement,
            "answer": metainfo.get("answer"),
            "difficulty": metainfo.get("difficulty"),
            "trace": trace,
            "rollout_index": rollout_index,
            "rollout_file": rollout_file,
            "hint_set": hint_set,
            "hint_set_path": step1_path,
            "hint_selection_prompt": prompt,
            "codex": {
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "sandbox": SANDBOX,
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    os.makedirs(out_problem_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    status = "ok" if selection is not None else f"ok (parse_error: {parse_err})"
    return problem_id, status


def main():
    global TRACES_DIR, DEBUG_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default=TRACES_DIR)
    parser.add_argument("--debug-dir", default=DEBUG_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--problem", default=None,
                        help="Process only this problem id (for testing)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many problems")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of concurrent codex calls")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run problems whose output already exists")
    args = parser.parse_args()

    TRACES_DIR = args.traces_dir
    DEBUG_DIR = args.debug_dir
    OUT_DIR = args.out_dir

    if args.problem:
        problems = [args.problem]
    else:
        problems = sorted(
            d for d in os.listdir(TRACES_DIR)
            if os.path.isdir(os.path.join(TRACES_DIR, d))
        )
    if args.limit:
        problems = problems[:args.limit]

    print(f"Processing {len(problems)} problems with {args.workers} worker(s)...")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_problem, p, args.overwrite): p for p in problems
        }
        for fut in as_completed(futures):
            problem_id, status = fut.result()
            done += 1
            print(f"[{done}/{len(problems)}] {problem_id}: {status}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
