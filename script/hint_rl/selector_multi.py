# Copyright 2026
#
# Multi-round (Template-F-multi) selector prompt + a fuzzy quote LOCATOR for the
# AUTO-HINT rollout (auto_hint_agent_loop).
#
# SELF-CONTAINED: everything here is a LOCAL copy (stdlib-only deps) -- there is NO
# runtime import from anywhere outside ``script/hint_rl``. The content was copied
# once (not imported) from ``selector/test_cite/gpt_oss_eval/multi-cite-gpt-eval``;
# that path is named below only to record where the copy came from, exactly like
# utils.py vendors the offline selector_prompt. Two things live here:
#
#   1. The multi-round Template F prompt. Unlike utils.selector_prompt (the
#      single-pick v4_cite prompt), this renders the WHOLE hint pool with a
#      per-hint ``status`` (completed | pending): hints already given/verified in
#      earlier rounds are ``completed`` and never re-selected; the selector picks
#      the next ``pending`` hint and ALSO reports, in ``completed_hints``, any
#      pending hint the student newly achieved THIS round -- each with a verbatim
#      ``quote`` from the student's trace. The auto-hint loop uses those quotes to
#      decide the verified-prefix gradient boundary (see locate_quote_end).
#      Verbatim copy of prompt_template_multiF.TEMPLATE_MULTI_F + render/build.
#
#   2. ``locate_quote_end`` -- where in a piece of student text a cited quote ends
#      (a CHARACTER offset), via the same notation-insensitive ``loose`` /
#      ``fuzzy_coverage`` machinery check_citations.py scores citations with. The
#      auto-hint loop maps that offset to a token index to set the per-turn
#      "promote only up to the last verified sentence" boundary.

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# Multi-round Template F prompt (verbatim from prompt_template_multiF.py)
# --------------------------------------------------------------------------- #
TEMPLATE_MULTI_F = r"""You are an expert math tutor helping a student over multiple rounds. In earlier rounds some hints were already given to the student and verified as achieved; those are marked `status: "completed"`. Your task is to choose the first hint that the student has not yet achieved, considering only the hints still marked `status: "pending"`, based on the student's reasoning trace.

# Inputs

## Math problem
<problem>
{{problem}}
</problem>

## Student's current reasoning trace
<Student_reasoning_trace>
{{trace}}
</Student_reasoning_trace>

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `hint_id`, the `hint` text, and a `status`:
  - `status: "completed"`: this hint was already given and achieved in an earlier round.
  - `status: "pending"`: this hint is still open — only pending hints can be selected.

{{hints}}

# Core rule:

Select the earliest `pending` substep hint, in order, whose mathematical idea is **not clearly achieved** in the student’s trace. Treat every `completed` hint as already done: never select a `completed` hint and never include it in `completed_hints`.

# Definitions:

* A hint is **achieved** only if the student’s trace clearly demonstrates the underlying mathematical idea of that hint.
* The student does not need to use the same wording as the hint.
* A hint is **not achieved** if the idea is missing, only partially present, contradicted, used incorrectly, or based on an unjustified guess.
* Do not skip an earlier unachieved `pending` hint even if the student has achieved later hints.
* A `pending` hint may already be achieved in the student’s trace (the student carried it out on their own). Recognize these, mark them completed this round (with a quote), and keep scanning the remaining pending hints.
* The hints already marked `status: "completed"` were handled in earlier rounds: do not re-verify them, do not select them, and do not include them in `completed_hints`.
* For every hint that you mark as completed, you must provide an exact quote from the student’s trace proving completion.
* The quote must be copied character-for-character from the student’s trace.
* Preserve LaTeX markup exactly.
* Do not convert symbols to Unicode.
* Do not paraphrase the quote.
* The quote must be findable by exact string search in the student’s trace.
* If you cannot find an exact supporting quote, do not mark that hint as completed.

# Procedure:

1. Scan the `pending` hints in order:

   * First by `step_id`.
   * Then by the order of hints within that step.
   * Skip every `completed` hint — it was already handled in an earlier round.

2. For each `pending` hint, decide whether the student achieved it:

   * If achieved, add it to `completed_hints` with exact quote evidence according to the following step 4.
   * If not achieved, stop immediately and select that pending hint.

3. Rephrase the selected hint only if necessary:

   * Keep the mathematical intent of the original hint unchanged.
   * Make it clear and helpful for the student.
   * Do not reveal the full solution unless the original hint already does.
   * If no rephrasing is needed, use the original hint text.

4. Write completed hints:

   * For each `pending` hint before the selected hint that the student achieved this round, include an entry in `completed_hints` with:
     - `hint_id`: the hint's id.
     - `quote`: the exact quote from the student's trace that demonstrates achievement.
     - `why`: a one-line explanation of why this quote demonstrates that the student achieved the hint.
   * Do not include any `completed`-status hint in `completed_hints` (those were already given in earlier rounds).
   * For example, if the selected hint is "3.2" and hints "1.1" and "1.2" are already `status: "completed"`, then `completed_hints` contains the `pending` hints among "2.1", "2.2", and "3.1" that the student has achieved this round, but no `completed`-status hints and nothing from "3.2" or later.

# Output format
Return a single JSON object wrapped in <output> tags. The output must contains four fields: `hint_id` is the id of the selected pending hint (e.g., "2.1" or "2.2"), `hint` is its text, rephrased per step 3 if necessary, `completed_hints` is a list of the pending hints that were achieved this round, and `reasoning_of_hint` is an explanation of why this hint was selected.

<output>
{
"completed_hints": [
{
"hint_id": "<id of a pending substep_hint achieved this round, before the selected hint_id, e.g. "2.1">",
"quote": "<a string copied character-for-character from the student's trace — LaTeX markup preserved exactly, no Unicode conversion, no paraphrase; must be findable by exact search in the trace>",
"why": "<one line, in your own words, explaining why this excerpt carries out this substep>"
} ## One entry for EACH pending hint achieved this round before the selected hint; use [] if none
],
"hint_id": "<hint_id of the selected pending hint>",
"reasoning_of_hint": "<why this hint was selected: explain why this is the first unachieved pending substep>",
"hint": "<the selected hint text, rephrased according to step 3 if necessary>"
}
</output>
"""


def render_hints_with_status(pool: Any, completed: Iterable[str]) -> str:
    """Render the FULL hint pool for the {{hints}} slot, tagging each hint with
    status completed|pending. Each hint is shown as {hint_id, hint, status}.

    Unlike the ``exclude_applied_*`` helpers (which DROP already-surfaced hints),
    this keeps every hint and only flips its status -- the multi-round prompt is
    built to reason over completed + pending together. ``completed`` is the set of
    hint ids already given/verified in earlier rounds. On any parse problem the
    pool string is returned unchanged. (Verbatim from prompt_template_multiF.)
    """
    done = {str(h) for h in (completed or [])}
    if isinstance(pool, str):
        try:
            pool = json.loads(pool)
        except Exception:  # noqa: BLE001
            return pool
    steps = []
    for st in (pool or {}).get("steps", []):
        hints = [{
            "hint_id": h.get("hint_id"),
            "hint": h.get("hint"),
            "status": "completed" if str(h.get("hint_id")) in done else "pending",
        } for h in st.get("hints", [])]
        steps.append({"step_id": st.get("step_id"), "purpose": st.get("purpose"), "hints": hints})
    return json.dumps({"steps": steps}, indent=2, ensure_ascii=False)


def build_prompt_multi(problem: str, trace: str, hints_rendered: str) -> str:
    """Fill the multi-round Template F placeholders. ``hints_rendered`` must be the
    output of ``render_hints_with_status`` (the status-tagged pool)."""
    return (TEMPLATE_MULTI_F
            .replace("{{problem}}", problem)
            .replace("{{trace}}", trace)
            .replace("{{hints}}", hints_rendered))


def pending_hint_ids(pool: Any, completed: Iterable[str]) -> list[str]:
    """The hint ids still ``pending`` (in pool order) given ``completed``.

    Used by the auto-hint loop to detect pool exhaustion (no pending hint left to
    give) WITHOUT a selector round-trip. Returns [] on a parse problem (treated as
    exhausted by the caller's len()==0 check, which is the safe "stop" default)."""
    done = {str(h) for h in (completed or [])}
    if isinstance(pool, str):
        try:
            pool = json.loads(pool)
        except Exception:  # noqa: BLE001
            return []
    out: list[str] = []
    for st in (pool or {}).get("steps", []):
        for h in st.get("hints", []):
            hid = h.get("hint_id")
            if hid is not None and str(hid) not in done:
                out.append(str(hid))
    return out


def prune_hint_pool(pool: Any) -> Any:
    """Drop step-guidance (``X.0``) hints and the per-hint ``type`` field, leaving
    only substep hints (each ``{"hint_id", "hint"}``).

    Mirrors ``run_gpt_oss_selection.prune_hint_pool`` -- the pruning the offline
    selector eval (``multi-cite-gpt-eval``) applied at benchmark-build time -- so the
    auto-hint rollout can present the selector the SAME x.0-free pools it was scored
    on (eval/train parity). Accepts the Template F pool dict (``{"steps": [{"hints":
    [...]}, ...]}``) or its JSON string; anything unrecognized is returned unchanged.
    A pruned dict is returned (the downstream render/pending helpers accept a dict).
    """
    if isinstance(pool, str):
        try:
            pool = json.loads(pool)
        except Exception:  # noqa: BLE001 -- malformed pool: pass it through unfiltered
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


# --------------------------------------------------------------------------- #
# Fuzzy quote locator (notation-insensitive, from check_citations.py)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
# unicode math -> the LaTeX command NAME the trace uses (no backslash), so a
# unicode-rewritten quote and the trace's "\cmd{...}" collapse to the same token
# once non-alphanumerics are dropped. (Verbatim from check_citations._UNI.)
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


def loose(s: str) -> str:
    """Notation-insensitive canonical form: unicode math -> LaTeX names, lower,
    drop everything but [a-z0-9]. (Verbatim from check_citations.loose.)"""
    s = _UNI_RE.sub(lambda m: f" {_UNI[m.group()]} ", s)
    return _ALNUM.sub("", s.lower())


def fuzzy_coverage(q_loose: str, t_loose: str) -> float:
    """Fraction of the quote's loose chars that appear, in order, in the trace
    (1.0 == loose substring). (Verbatim from check_citations.fuzzy_coverage.)"""
    if not q_loose:
        return 0.0
    sm = difflib.SequenceMatcher(None, q_loose, t_loose, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(q_loose)


def locate_quote_end(quote: str, text: str, fuzzy_threshold: float = 0.8) -> Optional[int]:
    """Character offset in ``text`` where the best match of ``quote`` ENDS, or None.

    The auto-hint loop needs not just whether a cited quote is present (what
    check_citations.classify reports) but WHERE it ends, to bound the
    verified-prefix gradient mask. The match is decided notation-insensitively
    (the loose/fuzzy_coverage tiers, robust to LaTeX-vs-unicode rewrites); the END
    offset is then read off a raw (lowercased) ``difflib`` alignment so it indexes
    the ORIGINAL ``text``.

      * exact substring         -> end = find + len(quote)
      * whitespace-normalized   -> end via a collapsed-space alignment (handles a
                                   quote that only differs in spacing/newlines)
      * loose/fuzzy >= threshold -> end of the FIRST raw occurrence (anchored on
                                   the earliest longest common run)

    Returns None when coverage is below ``fuzzy_threshold`` (no trustworthy match)
    -- the caller then treats the turn as having nothing verified.
    """
    q = (quote or "").strip()
    if not q or not text:
        return None

    # 1) exact substring (the common case for a well-behaved verbatim quote).
    i = text.find(q)
    if i != -1:
        return i + len(q)

    # 2) whitespace-only difference: align on lowercased raw; accept if the quote's
    #    collapsed form is a substring of the text's collapsed form.
    if _WS.sub(" ", q).strip().lower() in _WS.sub(" ", text).strip().lower():
        end = _raw_match_end(q, text)
        if end is not None:
            return end

    # 3) notation-insensitive (loose) coverage gate, then a raw alignment for the
    #    offset. loose() handles the dominant LaTeX<->unicode<->ascii rewrite.
    ql, tl = loose(q), loose(text)
    if ql and fuzzy_coverage(ql, tl) >= fuzzy_threshold:
        end = _raw_match_end(q, text)
        if end is not None:
            return end
    return None


def _raw_match_end(quote: str, text: str) -> Optional[int]:
    """End offset (in ``text``) of the FIRST occurrence of ``quote``. None if
    nothing aligns.

    Anchored on the longest common run between the lowercased quote and text --
    ``difflib.find_longest_match`` returns the run that starts EARLIEST in ``text``
    (ties broken to the earliest position), so when the quote matches several
    sentences the end lands in the FIRST matched sentence. The end is that run's
    text position extended by the quote chars trailing the run, so a quote whose
    tail paraphrases slightly still bounds at roughly its true end rather than
    being dragged forward by a stray late-matching token (the failure mode of the
    old ``max``-over-all-blocks behaviour)."""
    ql, tl = quote.lower(), text.lower()
    m = difflib.SequenceMatcher(None, ql, tl, autojunk=False).find_longest_match(
        0, len(ql), 0, len(tl))
    if m.size == 0:
        return None
    return min(len(text), m.b + (len(quote) - m.a))
