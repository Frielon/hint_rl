#!/usr/bin/env python3
"""Check whether the model's cited substep quotes are actually present (verbatim)
in the student's reasoning trace.

Template F requires every entry in ``completed_substeps`` to carry a ``quote``
that is an *exact, verbatim excerpt copied character-for-character* from the
student's writing in the trace. This tool joins each model result back to its
trace (by ``request_id``, from the label run) and tests every cited quote:

  * exact      -- the quote is a substring of the trace, character-for-character
  * normalized -- substring after collapsing all runs of whitespace to one space
                  (so a quote that only differs in spacing/newlines still counts)
  * loose      -- substring after notation-normalization: unicode math mapped to
                  LaTeX command names (⌈->lceil, √->sqrt, ≤->leq, π->pi, ...),
                  then everything but [a-z0-9] dropped. Catches the dominant
                  failure where the model rewrote the trace's LaTeX as unicode or
                  ASCII ("\\lceil\\sqrt{n}\\rceil" vs "⌈√n⌉" vs "ceil sqrt(n)").
  * fuzzy      -- not contained even loosely, but >= --fuzzy-threshold of the
                  quote's (loose-normalized) characters appear, in order, in the
                  trace. Catches paraphrases that keep most of the wording.
  * word_overlap -- (DIAGNOSTIC, NOT counted as found) not matched above, but
                  >= --word-threshold of the quote's distinct WORDS appear
                  somewhere in the trace, in any order. Flags rejects that merely
                  reuse the trace's vocabulary -- often the model recomposing a
                  line from in-trace tokens, not a verbatim excerpt. Weak
                  evidence, so it is its own bucket and stays OUT of found_rate.
  * not_found  -- none of the above -> paraphrased beyond recognition or fabricated

The traces here are first-round calls with no ``[hint given]`` blocks, so the
whole trace is the student's own writing: found-in-trace == found-in-student.

It reports quote-level and sample-level pass rates (overall + per step), and as a
sanity baseline runs the same check on the Codex reference quotes (``ref_selection``),
which should pass at ~100%.

Usage:
    python check_citations.py            # defaults to run_20260617_224209__gpt-oss-20b
    python check_citations.py --results-dir results/<run> --label-run run_<ts>
    python check_citations.py --show-failures 20   # print example bad quotes
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent           # .../test_cite/gpt_oss_eval
TEST_CITE = HERE.parent                           # .../test_cite
RESULTS_ROOT = TEST_CITE / "results"              # label runs (debug/ , runs/)

_WS = re.compile(r"\s+")

# unicode math -> the LaTeX command NAME the trace uses (no backslash), so a
# unicode-rewritten quote and the trace's "\cmd{...}" collapse to the same token
# once we drop non-alphanumerics below.
_UNI = {
    "≤": "leq", "≥": "geq", "≠": "neq", "≈": "approx", "≡": "equiv",
    "×": "times", "÷": "div", "·": "cdot", "±": "pm", "∓": "mp",
    "√": "sqrt", "∛": "sqrt", "π": "pi", "∞": "infty", "∑": "sum",
    "∏": "prod", "∫": "int", "∈": "in", "∉": "notin", "⊆": "subseteq",
    "⊂": "subset", "∪": "cup", "∩": "cap", "∅": "emptyset", "∂": "partial",
    "∇": "nabla", "∀": "forall", "∃": "exists", "→": "to", "←": "gets",
    "↔": "leftrightarrow", "⇒": "implies", "⇔": "iff", "∘": "circ",
    "⌈": "lceil", "⌉": "rceil", "⌊": "lfloor", "⌋": "rfloor",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "θ": "theta", "λ": "lambda", "μ": "mu", "σ": "sigma", "φ": "phi",
    "ω": "omega", "Δ": "delta", "Σ": "sum", "Π": "prod", "Ω": "omega",
    "°": "circ", "′": "prime", "″": "prime", "…": "dots", "⋯": "cdots",
    "⋮": "vdots", "⋱": "ddots",
}
_UNI_RE = re.compile("|".join(map(re.escape, _UNI)))
_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def loose(s: str) -> str:
    """Notation-insensitive canonical form: unicode math -> LaTeX names, lower,
    drop everything but [a-z0-9]. "\\lceil\\sqrt{n}\\rceil", "⌈√n⌉" and
    "ceil sqrt(n)" all reduce to comparable alnum tokens."""
    s = _UNI_RE.sub(lambda m: f" {_UNI[m.group()]} ", s)
    return _ALNUM.sub("", s.lower())


def fuzzy_coverage(q_loose: str, t_loose: str) -> float:
    """Fraction of the quote's loose chars that appear, in order, in the trace.
    1.0 when q_loose is a substring of t_loose; lower for partial paraphrases."""
    if not q_loose:
        return 0.0
    sm = difflib.SequenceMatcher(None, q_loose, t_loose, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(q_loose)


def words(s: str) -> list[str]:
    """Word tokens for the order-independent overlap tier: unicode math -> LaTeX
    names, lowercased, split on non-alphanumerics, 1-char tokens dropped."""
    s = _UNI_RE.sub(lambda m: f" {_UNI[m.group()]} ", s)
    return [w for w in _ALNUM.split(s.lower()) if len(w) >= 2]


def word_coverage(quote: str, trace_words: set) -> float:
    """Fraction of the quote's DISTINCT words present anywhere in the trace, in
    any order. A weak, order-blind signal: high values include the model
    recomposing a line from in-trace tokens, so it is reported separately and
    never folded into found_rate."""
    qw = set(words(quote))
    return (len(qw & trace_words) / len(qw)) if qw else 0.0


_TRACE_HDR = "## Student's current reasoning trace"
_HINTS_HDR = "## Candidate hints"


def _recover_trace(prompt: Optional[str]) -> str:
    """Slice the trace out of a rendered Template F prompt, for label rows that
    store only the rendered ``prompt`` (no structured ``reasoning_trace``).
    Without this, such rows are scored against an EMPTY trace and every cited
    quote is spuriously 'not_found' -- which massively inflates not_found."""
    if not isinstance(prompt, str):
        return ""
    it, ih = prompt.find(_TRACE_HDR), prompt.find(_HINTS_HDR)
    if 0 <= it < ih:
        return prompt[it + len(_TRACE_HDR):ih].strip("\n").strip()
    return ""


def load_traces(label_dir: Path) -> dict[str, str]:
    """request_id -> the trace the model was actually shown. Uses the structured
    ``reasoning_trace`` when present, else recovers it from the rendered
    ``prompt`` (some label rows store only the latter)."""
    out: dict[str, str] = {}
    for p in glob.glob(str(label_dir / "step*" / "*" / "*.json")):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = r.get("request_id")
        if rid is None:
            continue
        trace = r.get("reasoning_trace")
        if not trace:
            trace = _recover_trace(r.get("prompt"))
        out[str(rid)] = trace or ""
    return out


def resolve_label_dir(label_run: str) -> Path:
    p = Path(label_run)
    if p.is_dir():
        return p.resolve()
    for base in (RESULTS_ROOT / "debug", RESULTS_ROOT / "runs", RESULTS_ROOT):
        if (base / label_run).is_dir():
            return (base / label_run).resolve()
    raise SystemExit(f"label run not found: {label_run}")


def quotes_of(selection: Optional[dict]) -> list[str]:
    """Every cited quote in a parsed selection: the completed-hints quotes,
    plus a top-level 'quote' if the parse collapsed to a single substep entry.

    Accepts both the newer ``completed_hints`` key and the older
    ``completed_substeps`` (Codex reference labels), so old and new runs score."""
    if not isinstance(selection, dict):
        return []
    qs: list[str] = []
    cs = selection.get("completed_hints")
    if cs is None:
        cs = selection.get("completed_substeps")
    if isinstance(cs, list):
        for c in cs:
            if isinstance(c, dict) and isinstance(c.get("quote"), str):
                qs.append(c["quote"])
    elif isinstance(selection.get("quote"), str):   # malformed single-entry parse
        qs.append(selection["quote"])
    return qs


def classify(quote: str, tr: dict, fuzzy_threshold: float, word_threshold: float = 0.0) -> str:
    """tr is the cached trace forms: {'raw','norm','loose','words'}. First tier wins.
    word_overlap is a last-resort diagnostic tier (order-blind word match), tried
    only after fuzzy fails; pass word_threshold=0 to disable it."""
    q = quote.strip()
    if not q:
        return "empty"
    if q in tr["raw"]:
        return "exact"
    qn = norm(q)
    if qn and qn in tr["norm"]:
        return "normalized"
    ql = loose(q)
    if ql and ql in tr["loose"]:
        return "loose"
    if ql and fuzzy_threshold > 0 and fuzzy_coverage(ql, tr["loose"]) >= fuzzy_threshold:
        return "fuzzy"
    if word_threshold > 0 and word_coverage(q, tr["words"]) >= word_threshold:
        return "word_overlap"
    return "not_found"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/run_20260617_224209__gpt-oss-20b",
                    help="model result dir (with step*/.../*.json), abs or relative to here")
    ap.add_argument("--label-run", default="run_20260617_224209",
                    help="label run holding the traces (bare run_<ts> or a path)")
    ap.add_argument("--show-failures", type=int, default=0,
                    help="print this many example not_found quotes")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.85,
                    help="min in-order loose-char coverage to count a quote as 'fuzzy' "
                         "(0 disables the fuzzy tier). Default 0.85")
    ap.add_argument("--word-threshold", type=float, default=0.0,
                    help="min fraction of the quote's distinct WORDS present in the "
                         "trace (any order) to label an otherwise-not_found quote "
                         "'word_overlap'. Reported separately; NOT counted as found. "
                         "Default 0 (disabled); set e.g. 0.6 to enable the diagnostic.")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (HERE / results_dir).resolve()
    label_dir = resolve_label_dir(args.label_run)
    traces = load_traces(label_dir)
    trace_forms: dict[str, dict] = {}   # rid -> {'raw','norm','loose'}

    files = sorted(glob.glob(str(results_dir / "step*" / "*" / "*.json")))
    if not files:
        raise SystemExit(f"no result files under {results_dir}")

    # counters
    qcls = Counter()                 # model quote classifications
    ref_qcls = Counter()            # codex reference quote classifications
    by_step_q = {}                  # step -> Counter of model quote classes
    sample_all_exact = Counter()    # samples where every quote is exact (incl. found/total)
    n_samples = n_samples_with_quotes = 0
    sample_frac_sum = 0.0           # mean fraction of quotes found (any tier), over samples-with-quotes
    failures: list[dict] = []
    missing_trace = 0
    FOUND = ("exact", "normalized", "loose", "fuzzy")

    for fp in files:
        rec = json.loads(Path(fp).read_text())
        rid = str(rec.get("request_id"))
        trace = traces.get(rid)
        if trace is None:
            missing_trace += 1
            continue
        tr = trace_forms.get(rid)
        if tr is None:
            tr = {"raw": trace, "norm": norm(trace), "loose": loose(trace),
                  "words": set(words(trace))}
            trace_forms[rid] = tr
        step = rec.get("step")
        bs = by_step_q.setdefault(step, Counter())

        # reference (codex) quotes -- sanity baseline
        for q in quotes_of(rec.get("ref_selection")):
            ref_qcls[classify(q, tr, args.fuzzy_threshold, args.word_threshold)] += 1

        # model samples
        for s in rec.get("samples", []):
            sel = s.get("selection")
            qs = quotes_of(sel)
            n_samples += 1
            if not qs:
                continue
            n_samples_with_quotes += 1
            n_found = 0
            for q in qs:
                cls = classify(q, tr, args.fuzzy_threshold, args.word_threshold)
                qcls[cls] += 1
                bs[cls] += 1
                if cls in FOUND:
                    n_found += 1
                elif cls == "not_found" and len(failures) < args.show_failures:
                    failures.append({"request_id": rid, "step": step,
                                     "quote": q, "hint_id": s.get("hint_id")})
            sample_frac_sum += n_found / len(qs)
            if n_found == len(qs):
                sample_all_exact["all_found"] += 1
            sample_all_exact["total"] += 1

    def rate(c: Counter) -> dict:
        tot = sum(c.values()) or 1
        found = sum(c[k] for k in FOUND)
        return {
            "n_quotes": sum(c.values()),
            "exact": c["exact"], "normalized": c["normalized"],
            "loose": c["loose"], "fuzzy": c["fuzzy"],
            "word_overlap": c["word_overlap"],
            "not_found": c["not_found"], "empty": c["empty"],
            "exact_rate": round(c["exact"] / tot, 4),
            # verbatim-or-notation-rewritten (exact+normalized+loose):
            "verbatim_rate": round((c["exact"] + c["normalized"] + c["loose"]) / tot, 4),
            # any tier incl. fuzzy paraphrase (word_overlap excluded -- weak signal):
            "found_rate": round(found / tot, 4),
            # diagnostic only: rejects that reuse the trace's vocabulary out of order
            "word_overlap_rate": round(c["word_overlap"] / tot, 4),
            "not_found_rate": round(c["not_found"] / tot, 4),
        }

    n_swq = n_samples_with_quotes or 1
    summary = {
        "results_dir": str(results_dir),
        "label_run": str(label_dir),
        "n_result_files": len(files),
        "missing_trace": missing_trace,
        "n_samples": n_samples,
        "n_samples_with_quotes": n_samples_with_quotes,
        "model_quotes": rate(qcls),
        "sample_level": {
            "mean_found_fraction": round(sample_frac_sum / n_swq, 4),
            "all_quotes_found_rate": round(
                sample_all_exact["all_found"] / (sample_all_exact["total"] or 1), 4),
            "n_samples_all_found": sample_all_exact["all_found"],
            "n_samples_scored": sample_all_exact["total"],
        },
        "reference_quotes_baseline": rate(ref_qcls),
        "by_step": {str(k): rate(v) for k, v in sorted(by_step_q.items(), key=lambda x: str(x[0]))},
    }

    print(json.dumps(summary, indent=2))
    if failures:
        print("\n=== example not_found (paraphrased/fabricated) quotes ===")
        for f in failures:
            print(f"[step{f['step']} hint {f['hint_id']}] {f['quote'][:120]!r}")

    out = args.out or (results_dir / "_citation_check.json")
    Path(out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
