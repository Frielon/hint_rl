#!/usr/bin/env python3
"""Test a local (sglang-served) model on the hint-selection task and compare its
hint_id choices against the Codex reference in ``selector_codex/``.

Design: for an apples-to-apples comparison we *reuse the exact prompt* that Codex
was given. Every file under ``selector_codex/<problem_id>/hint_selection*.json``
already stores that prompt at ``meta.hint_selection_prompt`` and Codex's own
choice at ``selection.hint_id``. For each such problem we send the identical
prompt to the model ``--n`` times (default 16) with temperature > 0, parse the
``<output>`` JSON, and record:

  * the model's hint_id distribution over the N samples
  * self-consistency  = (count of the modal hint_id) / (parsed samples)
  * agreement_with_codex = fraction of parsed samples matching Codex's hint_id
  * majority_agrees_with_codex = (modal hint_id == Codex hint_id)

Results are written to ``selector_<model>/<problem_id>/samples.json`` and an
aggregate ``selector_<model>/_summary.json``.

The model is reached through the OpenAI-compatible endpoint sglang exposes
(default http://localhost:30000/v1). Launch the server first, e.g.::

    ./inference/serve_qwen3.5_27b.sh

Usage::

    python run_hint_selection_model.py --model Qwen3.5-27B --n 16
    python run_hint_selection_model.py --problem DAPO-...-100-5 --n 16   # single
    python run_hint_selection_model.py --compare-only --model Qwen3.5-27B # re-stat
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from seletor_prompt import selector_prompt  # noqa: E402

CODEX_DIR = os.path.join(HERE, "selector_codex")

OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _extract_last_json_object(text: str):
    """Find the last balanced {...} block in text and json.loads it, or None."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            c = text[start]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    obj = _loads_lenient(text[start:end + 1])
                    if obj is not None:
                        return obj
                    break
        end = text.rfind("}", 0, end)
    return None


def _repair_json_escapes(s: str) -> str:
    r"""Double any backslash that isn't a valid JSON escape introducer.

    Local math models routinely emit LaTeX inside JSON string values, e.g.
    "side \(6\sqrt{2}\)". Sequences like \( or \s are invalid JSON escapes and
    make json.loads raise "Invalid \escape". Doubling stray backslashes turns
    them into literal backslashes so the JSON parses (LaTeX preserved verbatim).

    The alternation consumes a *valid* escape pair (\\, \", \n, ...) whole and
    leaves it untouched, only doubling genuinely-stray backslashes. A bare
    `\(?!...)` would otherwise land on the SECOND backslash of a model-correct
    `\\cdot` and corrupt it into `\\\cdot`, breaking the parse."""
    return re.sub(r'\\(["\\/bfnrtu])|\\',
                  lambda m: m.group(0) if m.group(1) else r'\\\\', s)


def _loads_lenient(block: str):
    """json.loads, retrying with backslash-escape repair. Returns obj or None."""
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json_escapes(block))
        except json.JSONDecodeError:
            return None


# Expected keys of the selection object, by value type.
_INT_KEYS = ("major_step_id", "confidence_of_major_step", "confidence_of_hint")
_STR_KEYS = ("reasoning_of_major_step", "reasoning_of_hint", "hint")
# Match a JSON string body tolerantly: any run of non-quote/non-backslash chars
# or backslash-anything (so \" stays inside and invalid LaTeX escapes like \( or
# \frac are still consumed). Stops at the first unescaped closing quote.
_STR_BODY = r'((?:[^"\\]|\\.)*)'


def _unescape_loose(s: str) -> str:
    """Best-effort decode of an extracted JSON string body. Tries a real JSON
    decode (with escape repair); on failure falls back to minimal unescaping so
    the LaTeX-bearing text is preserved verbatim."""
    v = _loads_lenient('"' + s + '"')
    if isinstance(v, str):
        return v
    return (s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
             .replace("\\r", "\r"))


def _hard_parse(text: str):
    """Field-by-field regex extraction of the selection object, tolerant of any
    JSON the standard parser rejects (the common case: unescaped LaTeX
    backslashes in the reasoning/hint strings). Returns a dict (possibly
    partial) or None if nothing useful is found.

    This recovers the fields we actually score on -- ``hint_id`` and
    ``major_step_id`` -- plus the human-readable text, without requiring the
    whole blob to be valid JSON."""
    if not text:
        return None
    obj = {}
    for k in _INT_KEYS:
        m = re.search(r'"%s"\s*:\s*(-?\d+)' % k, text)
        if m:
            obj[k] = int(m.group(1))
    # hint_id may be quoted ("1.0") or bare (1.0); keep it as a string either way
    m = re.search(r'"hint_id"\s*:\s*"?(\d+(?:\.\d+)?)"?', text)
    if m:
        obj["hint_id"] = m.group(1)
    for k in _STR_KEYS:
        m = re.search(r'"%s"\s*:\s*"%s"' % (k, _STR_BODY), text, re.DOTALL)
        if m:
            obj[k] = _unescape_loose(m.group(1))
    # only count it as a recovery if we got something we can score on
    if obj.get("hint_id") is not None or obj.get("major_step_id") is not None:
        return obj
    return None


def parse_output(raw: str):
    """Pull the JSON object out of <output>...</output>, then fall back to a
    brace-matched object and a fenced ```json block. Tolerates invalid LaTeX
    backslash escapes. Returns (obj, error)."""
    m = OUTPUT_RE.search(raw)
    if m:
        block = m.group(1).strip()
        # strip a ```json fence if the model added one inside the tags
        block = re.sub(r"^```(?:json)?\s*|\s*```$", "", block.strip())
        obj = _loads_lenient(block)
        if obj is not None:
            return obj, None
        obj = _extract_last_json_object(block)
        if obj is not None:
            return obj, None
        err = "json decode failed (after escape repair)"
    else:
        err = "no <output> block"
    obj = _extract_last_json_object(raw)
    if obj is not None:
        return obj, None
    # hard fallback: field-by-field regex extraction that tolerates JSON the
    # parser rejects (e.g. unescaped LaTeX backslashes). Prefer the <output>
    # block when present, else scan the whole completion.
    obj = _hard_parse(m.group(1) if m else raw) or _hard_parse(raw)
    if obj is not None:
        return obj, None
    return None, err


def hint_id_of(selection):
    if not isinstance(selection, dict):
        return None
    hid = selection.get("hint_id")
    return str(hid) if hid is not None else None


# --------------------------------------------------------------------------- #
# model call
# --------------------------------------------------------------------------- #
def make_client(base_url: str, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def one_sample(client, model, prompt, temperature, top_p, max_tokens, retries=3):
    """Return a dict for a single completion (light retry), or {"error": ...}.

    Captures both the answer ``content`` and, when the server uses a reasoning
    parser (e.g. gpt-oss), the separate ``reasoning_content`` chain-of-thought,
    plus the authoritative ``completion_tokens`` from the API usage field. The
    latter counts ALL generated tokens (CoT + answer) and is the clean
    cross-model length metric. finish_reason == "length" means max_tokens cut
    the model off before it finished."""
    last_err = None
    for attempt in range(retries):
        try:
            t_start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            latency = time.time() - t_start
            choice = resp.choices[0]
            msg = choice.message
            usage = resp.usage
            return {
                "content": msg.content or "",
                "reasoning_content": getattr(msg, "reasoning_content", None),
                "finish_reason": choice.finish_reason,
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "latency_s": latency,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    return {"error": last_err}


# --------------------------------------------------------------------------- #
# per-problem
# --------------------------------------------------------------------------- #
def load_codex_record(problem_dir):
    files = sorted(glob.glob(os.path.join(problem_dir, "hint_selection*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def summarize_samples(samples, codex_hint_id):
    parsed = [s["hint_id"] for s in samples if s["hint_id"] is not None]
    dist = Counter(parsed)
    mode_hint_id, mode_count = (dist.most_common(1)[0] if dist else (None, 0))
    n_parsed = len(parsed)
    n_truncated = sum(1 for s in samples if s.get("finish_reason") == "length")
    ctoks = [s["completion_tokens"] for s in samples
             if s.get("completion_tokens") is not None]
    return {
        "n_samples": len(samples),
        "n_parsed": n_parsed,
        "n_truncated": n_truncated,
        "mean_completion_tokens": (sum(ctoks) / len(ctoks)) if ctoks else 0.0,
        "hint_id_distribution": dict(dist),
        "mode_hint_id": mode_hint_id,
        "self_consistency": (mode_count / n_parsed) if n_parsed else 0.0,
        "agreement_with_codex": (
            sum(1 for h in parsed if h == codex_hint_id) / n_parsed
        ) if n_parsed else 0.0,
        "majority_agrees_with_codex": (
            mode_hint_id is not None and mode_hint_id == codex_hint_id
        ),
    }


def process_problem(problem_id, args, client):
    problem_dir = os.path.join(args.codex_dir, problem_id)
    codex_rec = load_codex_record(problem_dir)
    if codex_rec is None:
        return problem_id, "skip (no codex record)", None

    # Rebuild the prompt from the stored components so we use the *current*
    # selector_prompt() template (seletor_prompt.py), not the prompt that was
    # baked into the codex record. Fall back to the stored prompt only if the
    # components are missing (older records).
    meta = codex_rec.get("meta", {}) or {}
    problem_statement = meta.get("problem")
    trace = meta.get("trace")
    hint_set = meta.get("hint_set")
    if problem_statement is not None and trace is not None and hint_set is not None:
        hints_str = json.dumps(hint_set, indent=2, ensure_ascii=False)
        prompt = selector_prompt(problem_statement, trace, hints_str)
    else:
        prompt = meta.get("hint_selection_prompt")
    if not prompt:
        return problem_id, "skip (no prompt: missing components and no stored prompt)", None
    codex_sel = codex_rec.get("selection") or {}
    codex_hint_id = hint_id_of(codex_sel)

    out_problem_dir = os.path.join(args.out_dir, problem_id)
    out_path = os.path.join(out_problem_dir, "samples.json")

    if args.compare_only:
        if not os.path.exists(out_path):
            return problem_id, "skip (no samples to compare)", None
        rec = json.load(open(out_path))
        # re-parse each sample's saved raw content with the current (fixed)
        # parser so escape-repair etc. is applied without re-querying the model
        for s in rec["samples"]:
            if s.get("raw"):
                sel, perr = parse_output(s["raw"])
                s["selection"] = sel
                s["hint_id"] = hint_id_of(sel)
                s["major_step_id"] = (sel or {}).get("major_step_id") if sel else None
                if perr and s.get("finish_reason") == "length":
                    perr = f"truncated (finish_reason=length); {perr}"
                s["parse_error"] = perr
        stats = summarize_samples(rec["samples"], codex_hint_id)
        rec.update(stats)
        rec["codex_hint_id"] = codex_hint_id
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        return problem_id, "recomputed", rec

    if os.path.exists(out_path) and not args.overwrite:
        rec = json.load(open(out_path))
        return problem_id, "skip (exists)", rec

    # draw N samples concurrently within this problem
    samples = [None] * args.n
    with ThreadPoolExecutor(max_workers=args.sample_workers) as ex:
        futs = {
            ex.submit(one_sample, client, args.model, prompt,
                      args.temperature, args.top_p, args.max_tokens): i
            for i in range(args.n)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            res = fut.result()
            if res.get("error") is not None:
                samples[i] = {"index": i, "hint_id": None, "major_step_id": None,
                              "selection": None,
                              "parse_error": f"api_error: {res['error']}",
                              "finish_reason": None, "completion_tokens": None,
                              "reasoning_content": None, "latency_s": None,
                              "raw": ""}
                continue
            raw = res["content"]
            finish_reason = res["finish_reason"]
            sel, perr = parse_output(raw)
            if perr and finish_reason == "length":
                perr = f"truncated (finish_reason=length); {perr}"
            samples[i] = {
                "index": i,
                "hint_id": hint_id_of(sel),
                "major_step_id": (sel or {}).get("major_step_id") if sel else None,
                "selection": sel,
                "parse_error": perr,
                "finish_reason": finish_reason,
                "completion_tokens": res["completion_tokens"],
                "prompt_tokens": res["prompt_tokens"],
                "latency_s": res.get("latency_s"),
                "reasoning_content": res["reasoning_content"],
                "raw": raw,
            }

    stats = summarize_samples(samples, codex_hint_id)
    rec = {
        "problem_id": problem_id,
        "model": args.model,
        "codex_hint_id": codex_hint_id,
        "codex_major_step_id": codex_sel.get("major_step_id"),
        "codex_selection": codex_sel,
        **stats,
        "samples": samples,
        "meta": {
            "prompt": prompt,
            "base_url": args.base_url,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    os.makedirs(out_problem_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    status = (f"ok  self={stats['self_consistency']:.2f} "
              f"agree={stats['agreement_with_codex']:.2f} "
              f"maj={'Y' if stats['majority_agrees_with_codex'] else 'n'} "
              f"({stats['n_parsed']}/{stats['n_samples']} parsed, "
              f"{stats['n_truncated']} truncated, "
              f"{stats['mean_completion_tokens']:.0f} gen-toks)")
    return problem_id, status, rec


# --------------------------------------------------------------------------- #
# progress logging
# --------------------------------------------------------------------------- #
class ProgressLogger:
    """Writes a timestamped, live-tailable progress log to a file AND stdout.

    The file is line-buffered and flushed after every write so it can be
    followed with `tail -f` while the eval runs."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", buffering=1)  # line-buffered

    def log(self, message, stdout=True):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if stdout:
            print(message, flush=True)

    def close(self):
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def compute_timing(recs, wall_seconds):
    """Aggregate inference-time metrics for RL cost estimation.

    The key number for RL projection is effective aggregate throughput
    (completion tokens / wall-clock) measured with the server saturated, which
    is exactly the regime an RL rollout phase runs in."""
    lat = [s["latency_s"] for r in recs for s in r["samples"]
           if s.get("latency_s") is not None]
    ctoks = [s["completion_tokens"] for r in recs for s in r["samples"]
             if s.get("completion_tokens") is not None]
    n_samples = sum(r["n_samples"] for r in recs)
    total_tokens = sum(ctoks)
    lat_sorted = sorted(lat)
    median_lat = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    return {
        "wall_seconds": round(wall_seconds, 1),
        "total_samples": n_samples,
        "total_completion_tokens": total_tokens,
        "throughput_tokens_per_sec": round(total_tokens / wall_seconds, 1) if wall_seconds else 0.0,
        "throughput_samples_per_sec": round(n_samples / wall_seconds, 3) if wall_seconds else 0.0,
        "mean_call_latency_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "median_call_latency_s": round(median_lat, 2),
        "mean_completion_tokens_per_call": round(total_tokens / len(ctoks), 1) if ctoks else 0.0,
    }


def write_summary(records, args, plog=None, wall_seconds=None):
    recs = [r for r in records if r is not None and "self_consistency" in r]
    n = len(recs)
    if n == 0:
        print("No records to summarize.")
        return
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    timing = compute_timing(recs, wall_seconds) if wall_seconds else None
    summary = {
        "model": args.model,
        "n_problems": n,
        "n": args.n,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_tokens": args.max_tokens},
        "mean_self_consistency": mean([r["self_consistency"] for r in recs]),
        "mean_agreement_with_codex": mean([r["agreement_with_codex"] for r in recs]),
        "majority_agree_rate": mean(
            [1.0 if r["majority_agrees_with_codex"] else 0.0 for r in recs]),
        "mean_parse_rate": mean([r["n_parsed"] / r["n_samples"] for r in recs]),
        "mean_truncation_rate": mean(
            [r.get("n_truncated", 0) / r["n_samples"] for r in recs]),
        "mean_completion_tokens": mean(
            [r.get("mean_completion_tokens", 0.0) for r in recs]),
        "timing": timing,
        "per_problem": sorted(
            [{"problem_id": r["problem_id"],
              "codex_hint_id": r["codex_hint_id"],
              "mode_hint_id": r["mode_hint_id"],
              "self_consistency": round(r["self_consistency"], 3),
              "agreement_with_codex": round(r["agreement_with_codex"], 3),
              "majority_agrees_with_codex": r["majority_agrees_with_codex"]}
             for r in recs],
            key=lambda x: x["problem_id"]),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    emit = plog.log if plog else (lambda m, **k: print(m, flush=True))
    emit("=" * 60, stdout=False) if plog else print("\n" + "=" * 60)
    emit(f"SUMMARY  model={args.model}  problems={n}  n={args.n}")
    emit(f"  mean self-consistency      : {summary['mean_self_consistency']:.3f}")
    emit(f"  mean agreement w/ codex    : {summary['mean_agreement_with_codex']:.3f}")
    emit(f"  majority == codex rate     : {summary['majority_agree_rate']:.3f}")
    emit(f"  mean parse rate            : {summary['mean_parse_rate']:.3f}")
    emit(f"  mean truncation rate       : {summary['mean_truncation_rate']:.3f}")
    emit(f"  mean completion tokens     : {summary['mean_completion_tokens']:.0f}")
    if timing:
        emit(f"  -- inference timing (server saturated) --")
        emit(f"  wall time                  : {timing['wall_seconds']:.0f}s")
        emit(f"  total completion tokens    : {timing['total_completion_tokens']}")
        emit(f"  throughput                 : {timing['throughput_tokens_per_sec']:.0f} tok/s"
             f"  ({timing['throughput_samples_per_sec']:.2f} calls/s)")
        emit(f"  mean/median call latency   : {timing['mean_call_latency_s']:.2f}s"
             f" / {timing['median_call_latency_s']:.2f}s")
    emit(f"  summary -> {os.path.join(args.out_dir, '_summary.json')}")
    emit("=" * 60, stdout=False) if plog else print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen3.5-27B",
                        help="served-model-name on the sglang endpoint")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--codex-dir", default=CODEX_DIR,
                        help="reference Codex outputs to read prompts/hint_ids from")
    parser.add_argument("--out-dir", default=None,
                        help="base output dir (default: selector_<model>). Each "
                             "sampling run writes into a timestamped subdir "
                             "<out-dir>/run_<YYYYmmdd_HHMMSS>/. For --compare-only, "
                             "pass the specific run subdir to recompute in place.")
    parser.add_argument("--n", type=int, default=16,
                        help="samples per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="this is a heavy reasoning model; long traces need a "
                             "big budget or <output> gets truncated. Needs server "
                             "--context-length >= prompt + this.")
    parser.add_argument("--problem", default=None, help="only this problem id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="problems processed concurrently")
    parser.add_argument("--sample-workers", type=int, default=8,
                        help="concurrent samples within one problem")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-only", action="store_true",
                        help="recompute stats from existing samples, no model calls")
    parser.add_argument("--no-run-subdir", action="store_true",
                        help="use --out-dir as-is instead of creating a "
                             "run_<timestamp> subdir under it; for callers that "
                             "manage the run directory themselves (e.g. the launch script)")
    parser.add_argument("--progress-log", default=None,
                        help="progress log file (default: <out-dir>/progress.log). "
                             "Line-buffered and tailable while the eval runs.")
    args = parser.parse_args()

    if args.out_dir is None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
        args.out_dir = os.path.join(HERE, f"selector_{safe}")
    # Each sampling run lands in its own timestamped subdir, so successive runs
    # never clobber each other and --overwrite is unnecessary. compare-only
    # recomputes in place on an existing run dir, so it is left as given.
    if not args.compare_only and not args.no_run_subdir:
        run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(args.out_dir, run_stamp)
    if args.progress_log is None:
        args.progress_log = os.path.join(args.out_dir, "progress.log")

    if args.problem:
        problems = [args.problem]
    else:
        problems = sorted(
            d for d in os.listdir(args.codex_dir)
            if os.path.isdir(os.path.join(args.codex_dir, d))
        )
    if args.limit:
        problems = problems[:args.limit]

    client = None if args.compare_only else make_client(args.base_url, args.api_key)

    os.makedirs(args.out_dir, exist_ok=True)
    plog = ProgressLogger(args.progress_log)
    total = len(problems)
    mode = "compare-only" if args.compare_only else "sampling"
    plog.log(f"START {mode}: {total} problems  model={args.model}  n={args.n}  "
             f"base_url={args.base_url}  out={args.out_dir}")

    records, done = [], 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_problem, p, args, client): p for p in problems}
            for fut in as_completed(futs):
                problem_id, status, rec = fut.result()
                done += 1
                records.append(rec)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                plog.log(f"[{done}/{total}] {problem_id}: {status}  "
                         f"| {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
        wall = time.time() - t0
        write_summary(records, args, plog=plog,
                      wall_seconds=(None if args.compare_only else wall))
        plog.log(f"DONE: {done}/{total} problems in {wall:.0f}s")
    finally:
        plog.close()


if __name__ == "__main__":
    main()
