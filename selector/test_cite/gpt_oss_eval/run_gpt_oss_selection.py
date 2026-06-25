#!/usr/bin/env python3
"""Test gpt-oss-20b on the Template F hint-selection task and compare its choices
against the Codex reference labels under ``test_cite/results/.../run_<ts>/``.

Apples-to-apples: every labeled result JSON already stores the *exact* Template F
prompt Codex was shown (``prompt``) plus Codex's parsed answer (``parsed_answer``,
with ``hint_id`` / ``major_step_id`` / ``completed_substeps``). For each row we
send that identical prompt to the served gpt-oss model ``--n`` times at
temperature > 0, parse the ``<output>`` JSON with the shared LaTeX-tolerant
parser, and score against the Codex reference.

If a row has no stored ``prompt`` (e.g. a non-debug run) we rebuild it from
``problem`` + ``reasoning_trace`` + ``hint_pool`` via Template F's ``build_prompt``,
so this works on debug and non-debug label runs alike.

Per row we record:
  * the model's hint_id distribution over the N samples
  * self_consistency          = (count of modal hint_id) / (parsed samples)
  * agreement_hint_id         = fraction of parsed samples whose hint_id == ref
  * agreement_major_step      = fraction whose major_step_id == ref
  * majority_agrees_hint_id   = (modal hint_id == ref hint_id)
  * truncation count + mean completion tokens (CoT + answer, from API usage)

Output layout mirrors the label run (step{N}/<problem_id>/<request_id>.json) plus
a top-level ``_summary.json`` with corpus-level aggregates.

All paths are resolved relative to this file, so the tree can live at any mount
(``/share5/...`` on the dev box, ``/xutao/...`` in a job container, ...).

Examples:
    # served by run_gpt_oss_eval.sh on the same node:
    python run_gpt_oss_selection.py --base-url http://127.0.0.1:30000/v1 -n 16
    python run_gpt_oss_selection.py --label-run run_20260617_224209 --steps 1,2 -n 8
    python run_gpt_oss_selection.py --limit 5            # first 5 rows per step (smoke)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

HERE = Path(__file__).resolve().parent           # .../test_cite/gpt_oss_eval
TEST_CITE = HERE.parent                           # .../test_cite
SELECTOR = TEST_CITE.parent                       # .../hint_rl/selector
sys.path.insert(0, str(SELECTOR))                 # run_hint_selection_model (parser + client)
sys.path.insert(0, str(HERE))                     # LOCAL prompt_template_F (edit it here!)

# Shared, battle-tested LaTeX-tolerant parser + OpenAI client helpers.
from run_hint_selection_model import (  # noqa: E402
    parse_output, hint_id_of, make_client, one_sample,
)
# The EDITABLE Template F prompt lives next to this file (test_cite/gpt_oss_eval/
# prompt_template_F.py). HERE is first on sys.path so this local copy wins over
# the one in test_cite/. Edit prompt_template_F.TEMPLATE_F to revise the prompt;
# with --prompt-source local (default) those edits drive every model call.
import prompt_template_F                           # noqa: E402
from prompt_template_F import build_prompt         # noqa: E402
# citation-scoring primitives (same dir) -- reused so _summary.json carries the
# verbatim-quote stats without a separate check_citations.py pass.
from check_citations import (  # noqa: E402
    norm as _cite_norm, loose as _cite_loose,
    classify as _cite_classify, quotes_of as _cite_quotes_of,
)

RESULTS_ROOT = TEST_CITE / "results"              # debug/ and runs/ live here


# --------------------------------------------------------------------------- #
# locating the labeled run + its result files
# --------------------------------------------------------------------------- #
def resolve_label_dir(label_run: str) -> Path:
    """Resolve ``--label-run`` to a concrete run dir.

    Accepts an absolute/relative path, or a bare ``run_<ts>`` name that we look
    up under results/debug/ then results/runs/.
    """
    p = Path(label_run)
    if p.is_dir():
        return p.resolve()
    for base in (RESULTS_ROOT / "debug", RESULTS_ROOT / "runs", RESULTS_ROOT):
        cand = base / label_run
        if cand.is_dir():
            return cand.resolve()
    raise SystemExit(f"FATAL: label run not found: {label_run} "
                     f"(looked under {RESULTS_ROOT}/debug and /runs)")


def discover_records(label_dir: Path, steps: Optional[set[int]],
                     limit: Optional[int]) -> list[dict[str, Any]]:
    """Load every per-call label JSON under step*/<problem_id>/<request_id>.json."""
    out: list[dict[str, Any]] = []
    per_step: Counter = Counter()
    for path in sorted(glob.glob(str(label_dir / "step*" / "*" / "*.json"))):
        try:
            rec = json.loads(Path(path).read_text())
        except Exception:  # noqa: BLE001
            continue
        step = rec.get("step")
        if steps is not None and step not in steps:
            continue
        if limit is not None and per_step[step] >= limit:
            continue
        per_step[step] += 1
        rec["_src_path"] = path
        out.append(rec)
    return out


def prune_hint_pool(pool: Any) -> Any:
    """Drop step-guidance (``X.0``) hints and the per-hint ``type`` field.

    Leaves only substep hints, each shown as ``{"hint_id", "hint"}``. Accepts the
    Template F hint-pool dict (``{"steps": [{"hints": [...]}, ...]}``); a JSON
    string is parsed first, and anything unrecognized is returned unchanged.
    """
    if isinstance(pool, str):
        try:
            pool = json.loads(pool)
        except Exception:  # noqa: BLE001
            return pool
    if not isinstance(pool, dict):
        return pool
    steps = []
    for st in pool.get("steps", []):
        hints = [{k: v for k, v in h.items() if k != "type"}
                 for h in st.get("hints", [])
                 if h.get("type") != "step_guidence_hint"
                 and not str(h.get("hint_id", "")).endswith(".0")]
        steps.append({**st, "hints": hints})
    return {**pool, "steps": steps}


_M_PROBLEM = "## Math problem"
_M_TRACE = "## Student's current reasoning trace"
_M_HINTS = "## Candidate hints"


def recover_problem_trace(prompt: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Pull ``(problem, trace)`` out of a previously rendered Template F prompt.

    Some label rows store only the rendered ``prompt`` (no structured
    ``problem`` / ``reasoning_trace`` fields). Every Template F prompt lays the
    inputs out under the same section headers, so we slice between them. Without
    this, such rows can't be rebuilt and would silently fall back to the stored
    OLD prompt (old output schema, un-pruned hints). Returns ``(None, None)`` if
    the markers aren't found.
    """
    if not isinstance(prompt, str):
        return None, None
    i_p, i_t, i_h = prompt.find(_M_PROBLEM), prompt.find(_M_TRACE), prompt.find(_M_HINTS)
    if not (0 <= i_p < i_t < i_h):
        return None, None
    problem = prompt[i_p + len(_M_PROBLEM):i_t].strip("\n").strip()
    trace = prompt[i_t + len(_M_TRACE):i_h].strip("\n").strip()
    return (problem or None), (trace or None)


def rebuild_prompt(rec: dict[str, Any]) -> Optional[str]:
    """Build the prompt from the LOCAL editable Template F + this row's inputs.

    The hint pool is pruned first (``prune_hint_pool``): the model is shown only
    substep hints, with the ``type`` field and every ``X.0`` step-guidance hint
    removed. If the row lacks structured ``problem`` / ``reasoning_trace`` (some
    label rows store only the rendered prompt), they are recovered from the
    stored prompt so the LOCAL template still drives the run.
    """
    problem, trace, pool = rec.get("problem"), rec.get("reasoning_trace"), rec.get("hint_pool")
    if pool is None:
        return None
    if problem is None or trace is None:
        rp, rt = recover_problem_trace(rec.get("prompt"))
        problem = problem if problem is not None else rp
        trace = trace if trace is not None else rt
    if problem is None or trace is None:
        return None
    pool = prune_hint_pool(pool)
    hints_str = pool if isinstance(pool, str) else json.dumps(pool, indent=2, ensure_ascii=False)
    return build_prompt(problem, trace, hints_str)


def trace_for(rec: dict[str, Any]) -> str:
    """The student trace actually shown to the model for this row (matching
    rebuild_prompt): the structured ``reasoning_trace`` field, else recovered
    from the stored prompt. Used to score the cited quotes against the right
    trace -- including the rows whose label lacks a structured trace."""
    trace = rec.get("reasoning_trace")
    if trace is None:
        _, trace = recover_problem_trace(rec.get("prompt"))
    return trace or ""


def prompt_for(rec: dict[str, Any], source: str) -> Optional[str]:
    """Resolve the prompt for a row.

    source="local"  -> rebuild from the editable prompt_template_F.py here, so
                       your prompt edits drive the run (default). Does NOT fall
                       back to the stored prompt: the stored prompt is the OLD
                       template, so falling back would silently mix old-schema
                       rows into the run. Rows that can't be rebuilt return None
                       (recorded as a visible "no prompt" error instead).
    source="stored" -> reuse the exact prompt Codex was shown (apples-to-apples,
                       ignores any local prompt edits). Falls back to a rebuild.
    """
    stored = rec.get("prompt")
    stored = stored if isinstance(stored, str) and stored.strip() else None
    if source == "stored":
        return stored or rebuild_prompt(rec)
    return rebuild_prompt(rec)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def major_step_of(sel: Optional[dict], hint_id: Optional[str]) -> Optional[int]:
    """Major-step id for a parsed selection.

    Template F's newer output schema no longer emits a separate ``major_step_id``;
    the selected ``hint_id`` ("X.Y") already encodes the major step as its integer
    part. Use an explicit ``major_step_id`` when one is present (e.g. an older
    Codex reference), else derive it from the hint_id.
    """
    if isinstance(sel, dict) and isinstance(sel.get("major_step_id"), int):
        return sel["major_step_id"]
    if hint_id is None:
        return None
    try:
        return int(str(hint_id).split(".")[0])
    except (ValueError, IndexError):
        return None


def summarize(samples: list[dict], ref_hint_id, ref_major_step) -> dict[str, Any]:
    hids = [s["hint_id"] for s in samples if s["hint_id"] is not None]
    msids = [s["major_step_id"] for s in samples if s.get("major_step_id") is not None]
    dist = Counter(hids)
    mode_hid, mode_n = (dist.most_common(1)[0] if dist else (None, 0))
    n_parsed = len(hids)
    n_trunc = sum(1 for s in samples if s.get("finish_reason") == "length")
    ctoks = [s["completion_tokens"] for s in samples if s.get("completion_tokens") is not None]
    return {
        "n_samples": len(samples),
        "n_parsed": n_parsed,
        "n_truncated": n_trunc,
        "mean_completion_tokens": (sum(ctoks) / len(ctoks)) if ctoks else 0.0,
        "hint_id_distribution": {str(k): v for k, v in dist.items()},
        "mode_hint_id": mode_hid,
        "self_consistency": (mode_n / n_parsed) if n_parsed else 0.0,
        "agreement_hint_id": (
            sum(1 for h in hids if h == ref_hint_id) / n_parsed) if n_parsed else 0.0,
        "agreement_major_step": (
            sum(1 for m in msids if m == ref_major_step) / len(msids)) if msids else 0.0,
        "majority_agrees_hint_id": (mode_hid is not None and mode_hid == ref_hint_id),
    }


def snapshot_scripts(out_dir: Path) -> None:
    """Copy every script this run depends on into out_dir/_scripts/, so a result
    dir is fully self-describing (the prompt template + driver + entry script +
    the shared parser/client module that produced these numbers)."""
    dst = out_dir / "_scripts"
    dst.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),                       # this driver
        HERE / "run_gpt_oss_eval.sh",                   # the entry bash script
        Path(prompt_template_F.__file__).resolve(),     # the prompt actually used
        SELECTOR / "run_hint_selection_model.py",       # shared parser + OpenAI client
        HERE / "check_citations.py",                    # citation scorer (folded into summary)
    ]
    manifest: list[str] = []
    for src in sources:
        try:
            (dst / src.name).write_text(src.read_text())
            manifest.append(f"{src.name}  <-  {src}")
        except Exception as exc:  # noqa: BLE001
            manifest.append(f"{src.name}  <-  {src}  (FAILED: {exc})")
    (dst / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


def out_path_for(out_dir: Path, rec: dict[str, Any]) -> Path:
    """Mirror the label layout: step{N}/<problem_id>/<request_id>.json."""
    step = rec.get("step")
    pid = str(rec.get("problem_id"))
    rid = str(rec.get("request_id"))
    safe = lambda s: s.replace("/", "_").replace(os.sep, "_")  # noqa: E731
    return out_dir / f"step{step}" / safe(pid) / f"{safe(rid)}.json"


def process_record(rec: dict[str, Any], client, args) -> Optional[dict[str, Any]]:
    out_path = out_path_for(args.out_dir, rec)
    if out_path.exists() and not args.overwrite:
        try:
            return json.loads(out_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    prompt = prompt_for(rec, args.prompt_source)
    ref = rec.get("parsed_answer") or {}
    ref_hint_id = hint_id_of(ref)
    ref_major_step = major_step_of(ref, ref_hint_id)

    base = {
        "uid": rec.get("uid"),
        "step": rec.get("step"),
        "problem_id": rec.get("problem_id"),
        "request_id": rec.get("request_id"),
        "model": args.model,
        "ref_hint_id": ref_hint_id,
        "ref_major_step_id": ref_major_step,
        "ref_selection": ref or None,
    }

    if not prompt:
        base.update({"error": "no prompt (missing stored prompt and components)",
                     "samples": [], **summarize([], ref_hint_id, ref_major_step)})
        return base

    # draw N samples concurrently within this row
    samples: list[Optional[dict]] = [None] * args.n
    with ThreadPoolExecutor(max_workers=args.sample_workers) as ex:
        futs = {ex.submit(one_sample, client, args.model, prompt,
                          args.temperature, args.top_p, args.max_tokens): i
                for i in range(args.n)}
        for fut in as_completed(futs):
            i = futs[fut]
            res = fut.result()
            if res.get("error") is not None:
                samples[i] = {"index": i, "hint_id": None, "major_step_id": None,
                              "selection": None, "parse_error": f"api_error: {res['error']}",
                              "finish_reason": None, "completion_tokens": None,
                              "prompt_tokens": None, "latency_s": None, "raw": ""}
                continue
            raw = res["content"]
            fin = res["finish_reason"]
            sel, perr = parse_output(raw)
            if perr and fin == "length":
                perr = f"truncated (finish_reason=length); {perr}"
            hid = hint_id_of(sel)
            samples[i] = {
                "index": i,
                "hint_id": hid,
                "major_step_id": major_step_of(sel, hid),
                "selection": sel,
                "parse_error": perr,
                "finish_reason": fin,
                "completion_tokens": res["completion_tokens"],
                "prompt_tokens": res["prompt_tokens"],
                "latency_s": res.get("latency_s"),
                "raw": raw if args.save_raw else "",
            }
    base.update({**summarize(samples, ref_hint_id, ref_major_step), "samples": samples})
    return base


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-run", default="run_20260617_224209",
                    help="labeled run dir or bare run_<ts> name under results/debug|runs "
                         "(default: run_20260617_224209)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <here>/results/<label_run>__<model>)")
    ap.add_argument("--steps", default=None,
                    help="comma list of steps to score (default: all present)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per step (smoke test). Default: all.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-query rows already saved in out-dir (default: resume)")
    ap.add_argument("--prompt-source", choices=["local", "stored"],
                    default=os.environ.get("PROMPT_SOURCE", "local"),
                    help="local: rebuild from the editable prompt_template_F.py here "
                         "(default, so prompt edits take effect); stored: reuse the exact "
                         "prompt Codex saw (apples-to-apples, ignores local edits)")
    # model / endpoint
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME", "gpt-oss-20b"))
    ap.add_argument("-n", "--n", type=int, default=int(os.environ.get("N_SAMPLES", "16")),
                    help="samples per row")
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.7")))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "16384")))
    # concurrency
    ap.add_argument("--row-workers", type=int, default=int(os.environ.get("ROW_WORKERS", "16")),
                    help="rows processed concurrently")
    ap.add_argument("--sample-workers", type=int, default=int(os.environ.get("SAMPLE_WORKERS", "16")),
                    help="samples drawn concurrently within a row")
    ap.add_argument("--save-raw", action="store_true", default=True,
                    help="store each sample's raw completion (default on; --no-save-raw off)")
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    return ap.parse_args()


def corpus_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r.get("n_parsed", 0) > 0]
    n = len(scored)
    avg = lambda k: (sum(r[k] for r in scored) / n) if n else 0.0  # noqa: E731
    by_step: dict[str, dict] = {}
    for r in records:
        s = str(r.get("step"))
        b = by_step.setdefault(s, {"n_rows": 0, "n_scored": 0,
                                   "sum_agree_hint": 0.0, "sum_agree_major": 0.0,
                                   "n_majority_agree": 0})
        b["n_rows"] += 1
        if r.get("n_parsed", 0) > 0:
            b["n_scored"] += 1
            b["sum_agree_hint"] += r["agreement_hint_id"]
            b["sum_agree_major"] += r["agreement_major_step"]
            b["n_majority_agree"] += int(bool(r["majority_agrees_hint_id"]))
    for b in by_step.values():
        k = b["n_scored"] or 1
        b["mean_agreement_hint_id"] = b.pop("sum_agree_hint") / k
        b["mean_agreement_major_step"] = b.pop("sum_agree_major") / k
        b["majority_agree_rate"] = b["n_majority_agree"] / k
    return {
        "n_rows": len(records),
        "n_rows_scored": n,
        "mean_agreement_hint_id": avg("agreement_hint_id"),
        "mean_agreement_major_step": avg("agreement_major_step"),
        "mean_self_consistency": avg("self_consistency"),
        "majority_agree_rate": (sum(int(bool(r["majority_agrees_hint_id"])) for r in scored) / n) if n else 0.0,
        "mean_completion_tokens": avg("mean_completion_tokens"),
        "total_truncated": sum(r.get("n_truncated", 0) for r in records),
        "by_step": dict(sorted(by_step.items())),
    }


# --------------------------------------------------------------------------- #
# citation stats (verbatim-quote check) folded into the corpus summary
# --------------------------------------------------------------------------- #
_CITE_FOUND = ("exact", "normalized", "loose", "fuzzy")
_CITE_FUZZY = 0.85


def _cite_rate(c: Counter) -> dict[str, Any]:
    tot = sum(c.values()) or 1
    found = sum(c[k] for k in _CITE_FOUND)
    return {
        "n_quotes": sum(c.values()),
        "exact": c["exact"], "normalized": c["normalized"], "loose": c["loose"],
        "fuzzy": c["fuzzy"], "not_found": c["not_found"], "empty": c["empty"],
        "exact_rate": round(c["exact"] / tot, 4),
        "verbatim_rate": round((c["exact"] + c["normalized"] + c["loose"]) / tot, 4),
        "found_rate": round(found / tot, 4),
        "not_found_rate": round(c["not_found"] / tot, 4),
    }


def corpus_citation(records: list[dict[str, Any]],
                    rid_to_trace: dict[str, str]) -> dict[str, Any]:
    """Verbatim-quote check over every sample's cited quotes, folded into the
    summary. Each quote is classified against the trace the model actually saw
    (``rid_to_trace``, recovered for rows whose label lacks a structured trace),
    mirroring check_citations.py's tiers (word_overlap disabled)."""
    overall: Counter = Counter()
    by_step: dict[str, Counter] = {}
    ref_total = ref_found = 0
    n_swq = n_all_found = 0
    tr_cache: dict[str, dict] = {}
    for r in records:
        rid = str(r.get("request_id"))
        trace = rid_to_trace.get(rid, "")
        tr = tr_cache.get(rid)
        if tr is None:
            tr = {"raw": trace, "norm": _cite_norm(trace), "loose": _cite_loose(trace)}
            tr_cache[rid] = tr
        step = str(r.get("step"))
        bs = by_step.setdefault(step, Counter())
        for q in _cite_quotes_of(r.get("ref_selection")):
            ref_total += 1
            if _cite_classify(q, tr, _CITE_FUZZY, 0.0) in _CITE_FOUND:
                ref_found += 1
        for s in r.get("samples", []):
            qs = _cite_quotes_of(s.get("selection"))
            if not qs:
                continue
            n_swq += 1
            n_found = 0
            for q in qs:
                cls = _cite_classify(q, tr, _CITE_FUZZY, 0.0)
                overall[cls] += 1
                bs[cls] += 1
                if cls in _CITE_FOUND:
                    n_found += 1
            if n_found == len(qs):
                n_all_found += 1
    return {
        "fuzzy_threshold": _CITE_FUZZY,
        "n_samples_with_quotes": n_swq,
        "n_samples_all_found": n_all_found,
        "all_quotes_found_rate": round(n_all_found / (n_swq or 1), 4),
        "model_quotes": _cite_rate(overall),
        "reference_baseline": {
            "n_quotes": ref_total,
            "found_rate": round(ref_found / (ref_total or 1), 4),
        },
        "by_step": {k: _cite_rate(v) for k, v in sorted(by_step.items())},
    }


def main() -> int:
    args = parse_args()
    label_dir = resolve_label_dir(args.label_run)
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None

    if args.out_dir:
        args.out_dir = Path(args.out_dir).resolve()
    else:
        tag = args.model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")    # current run timestamp
        args.out_dir = HERE / "results" / f"{label_dir.name}__{tag}__{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records_in = discover_records(label_dir, steps, args.limit)
    print(f"label_dir = {label_dir}")
    print(f"out_dir   = {args.out_dir}")
    print(f"endpoint  = {args.base_url}   model={args.model}")
    print(f"prompt    = {args.prompt_source}  ({prompt_template_F.__file__})")
    print(f"rows      = {len(records_in)}  n={args.n} temp={args.temperature} "
          f"max_tokens={args.max_tokens} row_workers={args.row_workers}")
    if not records_in:
        print("no rows to score; exiting.")
        return 0

    # snapshot every related script used for this run, for provenance / diffing.
    snapshot_scripts(args.out_dir)

    (args.out_dir / "run_config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "label_dir": str(label_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    client = make_client(args.base_url, args.api_key)
    out_records: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.row_workers) as ex:
        futs = {ex.submit(process_record, rec, client, args): rec for rec in records_in}
        bar = tqdm(as_completed(futs), total=len(futs), unit="row",
                   desc="gpt-oss selection", dynamic_ncols=True, smoothing=0.1)
        for fut in bar:
            rec = futs[fut]
            res = fut.result()
            if res is None:
                continue
            op = out_path_for(args.out_dir, rec)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
            out_records.append(res)
            scored = [r for r in out_records if r.get("n_parsed", 0) > 0]
            if scored:
                bar.set_postfix(
                    agree=f"{sum(r['agreement_hint_id'] for r in scored)/len(scored):.2f}",
                    refresh=False)
        bar.close()

    summary = corpus_summary(out_records)
    rid_to_trace = {str(rec.get("request_id")): trace_for(rec) for rec in records_in}
    summary["citation"] = corpus_citation(out_records, rid_to_trace)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ndone in {time.time()-t0:.0f}s  ->  {args.out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_step"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
