"""Local copies of the offline selector prompt + tolerant <output> parser.

These were previously imported from ``${HINT_RL_HOME}/selector`` (``seletor_prompt.py``
and ``run_hint_selection_model.py``) via a sys.path hack. They are vendored here so
that everything the HPRL rollout needs lives inside ``script/hint_rl`` and the package
has no cross-folder import dependency.

The <output> parser is kept verbatim. The prompt template is the ``v4_cite``
variant (selector/prompt_variants.py): the v2_final_gate template ("Ground rules for
crediting progress" + "Guard the final step") plus a "Cite your evidence" section and
a ``completed_steps`` output field requiring a verbatim student-quote citation for
each earlier candidate step. This cut unearned final-step (answer-bearing) reveals
32.1% -> 6.4% @T=0.7 / 4.2% @T=0.1 on replayed failure cases (and to 3.0% / 0.8% with
the optional citation-enforcement guard in the loop) -- see
selector/prompt_improvement_progress.md. NOTE: those gains assume the selector sees
the student's reasoning; they require the blind-trace fix in hint_agent_loop
(assistant turns appended to agent_data.messages). The ``completed_steps`` field is
currently ignored by the loop (a drop-in), pending the enforcement guard in
_record_major_step.
"""
from __future__ import annotations

import json
import re

# --------------------------------------------------------------------------- #
# Selector prompt (from selector/seletor_prompt.py)
# --------------------------------------------------------------------------- #
def selector_prompt(problem: str, trace: str, hints: str) -> str:
    prompt = r"""You are an expert math tutor. Your task is to select the single most relevant hint to help a student make progress toward the solution of a math problem, based on where they currently are in their reasoning.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step. Each hint has a `type`, a `hint_id`, and the `hint` text:
  - `type: "step_guidence_hint"` (hint_id `X.0`): a high-level nudge toward *what* this step requires, without revealing the substeps.
  - `type: "substep_hint"` (hint_id `X.1`, `X.2`, ...): a finer hint walking through *how* to carry out the step, in order.

{{hints}}

# Workflow

## 1. Identify the major step the student is currently working on
Go through the major steps in order and, for each one, **verify whether the student has actually completed it correctly** — do not take the student's claims at face value:
- Independently check the student's setup, calculations, and stated conclusions for that step. Re-derive or sanity-check them yourself rather than trusting what the student asserts.
- A step counts as completed **only if** the student has correctly carried out its substance. A step is **not** completed if the student's work for it is wrong, rests on a false premise, sets the problem up incorrectly, or has merely been *named or recognized* without being executed. A confident but incorrect conclusion means the step is **not** complete.
- Conversely, do not rewind to an earlier step over a trivial, purely mechanical omission (e.g., an unstated arithmetic sum) when the student has clearly done that step's substantive reasoning.

### Ground rules for crediting progress
- Hints previously given by the tutor appear in the trace. **Hint text is not
  student work.** A step counts as completed only if the *student's own
  writing* carries out its substance after (or independently of) any hint.
  Content that merely echoes a given hint, or a step whose hint was shown but
  that the student never executed, is NOT completed.
- Progress requires new written reasoning. If the student has written little
  or nothing of substance since the last hint was given, then no additional
  step can have been completed since then.
- The candidate list contains only steps whose hints have **not** been given
  yet. Do not skip over an earlier candidate step because you, the tutor, can
  see how it would go -- the student still has to do it.

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

### Cite your evidence
For every candidate step you judge completed, you must QUOTE the student's own
words that carry out that step: an exact, verbatim excerpt (roughly 10-200
characters) copied character-for-character from the student's writing in the
trace. Text inside a `[hint given]` block is the tutor's, not the student's --
quoting it does not count. Paraphrases do not count. These citations go into
the `completed_steps` array of your output, one entry per candidate step
listed BEFORE your selected step. If you cannot produce a verbatim student
quote for some earlier candidate step, that step is not completed -- select it
instead. An empty `completed_steps` array means you selected the earliest
remaining candidate step.

### Guard the final step
The last major step typically states the final answer. Selecting it gives the
answer away, so it is justified **only** when the student's own written work
has correctly completed every earlier candidate step. If you are at all unsure
whether an earlier candidate step is genuinely done, select that earlier step
instead. An unearned answer reveal is a much worse error than a hint that is
one step too early.

## 2. Select a hint within that major step
- If the student has not yet recognized *what* this step requires, select the `step_guidence_hint` (the `X.0` hint).
- If the student understands what the step requires but is stuck on execution, select the earliest `substep_hint` they have not yet completed.

## 3. Rephrase the hint to fit the student's work, only if needed
- Match the student's terminology and notation. For example, if the hint refers to "the cell (50,50)" but the student writes "the center square c", rewrite the hint to refer to "the center square c".
- Do not add new information, reveal later steps, or give away more than the original hint. Preserve its level of disclosure.

## Confidence scale
Rate each confidence judgment as an integer from 1 to 5:
- 5: Certain. The trace gives direct, unambiguous evidence for this choice.
- 4: Confident. Strong evidence, with only minor or unlikely alternatives.
- 3: Moderate. The choice is the best fit, but the trace is partial or admits other plausible readings.
- 2: Low. The trace is sparse or ambiguous; this is a tentative guess among several candidates.
- 1: Very low. Little to no evidence; essentially a guess.

# Output format
Return a single JSON object wrapped in <output> tags. `hint_id` is the id of the selected hint (e.g., "2.0" or "2.1"). `hint` is its text, rephrased per step 3 if necessary. Confidence values are integers from 1 to 5, as defined above.

<output>
{
  "completed_steps": [{"step_id": <id of a candidate step earlier than your selection>, "quote": <verbatim excerpt from the student's own writing that carries out this step>, "why": <one line: why this excerpt completes the step>}, ...] (one entry for EVERY candidate step listed before your selected step; [] if you selected the earliest),
  "major_step_id": <step_id>,
  "reasoning_of_major_step": <why this is the earliest step not yet completed correctly: cite the trace, and state how you verified whether each prior step was actually done correctly (note any error that makes an earlier step incomplete)>,
  "confidence_of_major_step": <1-5>,
  "hint_id": <hint_id of the selected hint>,
  "reasoning_of_hint": <why this hint: either why the step-guidance hint, or why this is the earliest incomplete substep>,
  "confidence_of_hint": <1-5>,
  "hint": <the selected hint text, rephrased per step 3 if necessary>
}
</output>
"""
    prompt = (prompt
              .replace("{{problem}}", problem)
              .replace("{{trace}}", trace)
              .replace("{{hints}}", hints))

    return prompt


# --------------------------------------------------------------------------- #
# Tolerant <output> JSON parser (from selector/run_hint_selection_model.py)
# --------------------------------------------------------------------------- #
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


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
    """Double any backslash that isn't a valid JSON escape introducer.

    Local math models routinely emit LaTeX inside JSON string values, e.g.
    "side \\(6\\sqrt{2}\\)". Sequences like \\( or \\s are invalid JSON escapes
    and make json.loads raise "Invalid \\escape". Doubling stray backslashes
    turns them into literal backslashes so the JSON parses (and the LaTeX is
    preserved verbatim in the decoded string).

    Already-valid ``\\\\`` pairs are consumed atomically. Selectors mix escape
    styles in one blob (``\\(`` next to ``\\\\frac``); a per-character lookahead
    doubled the second backslash of ``\\\\le`` into ``\\\\\\le``, so the repaired
    JSON stayed invalid and the hint text was lost downstream (~5% of applied
    hints in the 20260723 dolci run arrived empty this way)."""
    return re.sub(r'(\\\\)|\\(?!["\\/bfnrtu])', lambda m: m.group(1) or r'\\', s)


# Single-escaped LaTeX commands that shadow VALID JSON escapes: "\frac" is a
# well-formed JSON string that silently decodes to formfeed+"rac" (likewise
# \boxed, \theta, \times, \right, ...). 3.6% of this run's applied hints
# carried such control-char corruption. A backslash before [bfrt]+letter or a
# \u not followed by 4 hex digits is always LaTeX in this domain (corpus scan
# of 21,922 hint bodies found zero intended backspace/formfeed/tab-then-letter
# uses), so double it BEFORE parsing. \n stays a newline: "...\nThen..." prose
# dominates; the few \n-prefixed LaTeX commands are restored post-decode.
_LATEX_SHADOW_RE = re.compile(r'(\\\\)|\\(?=[bfrt][A-Za-z]|u(?![0-9a-fA-F]{4}))')
# Post-decode: a newline immediately followed by one of these suffixes can only
# be a mangled \n-command (\neq -> newline+"eq", \nmid, \not -> newline+"ot",
# ...); prose after an intended newline ("\nequation (3)", "\nother") fails the
# word boundary. Longest-first where prefixes overlap. Delivered fields only.
_NL_CMD_RE = re.compile(
    r'\n(?=(?:eq|abla|otin|ot|mid|cong|sim|leq|geq|le|ge|subseteq|supseteq|'
    r'prec|succ|parallel|vdash)\b)')


def _normalize_latex_escapes(s: str) -> str:
    """Pre-parse pass doubling LaTeX-shadow escapes (see _LATEX_SHADOW_RE);
    valid ``\\\\`` pairs are consumed atomically, so it is idempotent and
    composes with _repair_json_escapes."""
    return _LATEX_SHADOW_RE.sub(lambda m: m.group(1) or r'\\', s)


def _restore_nl_commands(s: str) -> str:
    """Undo newline-decoded \\n-commands (``a \\neq b`` -> newline+"eq b")."""
    return _NL_CMD_RE.sub(lambda m: "\\n", s)


def _loads_lenient(block: str):
    """json.loads, retrying with backslash-escape repair. Returns obj or None.

    ``strict=False`` additionally tolerates raw control characters inside
    strings -- selectors quote multi-line student text verbatim."""
    block = _normalize_latex_escapes(block)
    try:
        return json.loads(block, strict=False)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json_escapes(block), strict=False)
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
    the LaTeX-bearing text is preserved verbatim (``\\\\`` pairs consumed first
    so ``\\\\neq`` doesn't lose its backslash to the ``\\n`` rule)."""
    v = _loads_lenient('"' + s + '"')
    if isinstance(v, str):
        return v
    return (s.replace("\\\\", "\x00").replace('\\"', '"').replace("\\n", "\n")
             .replace("\\t", "\t").replace("\\r", "\r").replace("\x00", "\\"))


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
    # hint_id may be quoted ("1.0") or bare (1.0); keep it as a string either
    # way. Citation entries inside completed_hints carry their own "hint_id"
    # (immediately followed by their "quote"/"why" key), so classify those out
    # instead of merely deprioritizing them: on a response that never emitted a
    # top-level id (truncation, or the model omitted it), attaching a cited
    # already-achieved id makes the runtime back-fill and inject the WRONG
    # hint. Among the true candidates, prefer the match nearest the top-level
    # "hint"/"reasoning_of_hint" keys.
    ids = list(re.finditer(r'"hint_id"\s*:\s*"?(\d+(?:\.\d+)?)"?', text))
    noncite = [i for i in ids if not _CITE_NEXT_RE.match(text, i.end())]
    if noncite:
        anchor = re.search(r'"(?:hint|reasoning_of_hint)"\s*:\s*"', text)
        best = (min(noncite, key=lambda i: abs(i.start() - anchor.start()))
                if anchor else noncite[-1])
        obj["hint_id"] = best.group(1)
    for k in _STR_KEYS:
        m = re.search(r'"%s"\s*:\s*"%s"' % (k, _STR_BODY), text, re.DOTALL)
        if m:
            obj[k] = _unescape_loose(m.group(1))
    # only count it as a recovery if we got something we can act on: an id to
    # back-fill from the pool, or hint text (the runtime remaps an id-less
    # text onto the offered pool by exact/fuzzy match).
    if (obj.get("hint_id") is not None or obj.get("major_step_id") is not None
            or str(obj.get("hint") or "").strip()):
        return obj
    return None


# A citation's own "hint_id" is immediately followed by its "quote"/"why" key
# ({hint_id, quote, why} schema echo); a top-level id is followed by selection
# keys. Used to classify id occurrences during regex salvage.
_CITE_NEXT_RE = re.compile(r'\s*,?\s*"(?:quote|why)"\s*:')


def _is_citation_item(d: dict) -> bool:
    """A ``completed_hints`` entry ({hint_id, quote, why}) masquerading as the
    selection. When the full blob is unparseable, the balanced-brace fallback
    can salvage one of these nested objects instead; it carries a plausible
    hint_id, so without this guard it becomes an applied hint with empty text
    (and the cited id gets charged + dropped from future pools)."""
    return (("quote" in d or "why" in d)
            and not any(k in d for k in ("hint", "completed_hints",
                                         "reasoning_of_hint", "major_step_id")))


_SELECTION_KEYS = ("hint_id", "hint", "major_step_id", "completed_hints",
                   "reasoning_of_hint", "reasoning_of_major_step",
                   "confidence_of_hint", "confidence_of_major_step")


def _as_selection_dict(obj):
    """Coerce a parsed selection to a dict, or None.

    The selector occasionally emits a JSON array of candidate selections instead
    of the single object the prompt asks for; take the first dict element (its
    top candidate) rather than failing the call. Anything else (string, number,
    list without dicts, a lone citation item) is rejected so parse_output falls
    through to its other extraction strategies.
    """
    if isinstance(obj, dict):
        if _is_citation_item(obj) or not any(k in obj for k in _SELECTION_KEYS):
            return None
        return obj
    if isinstance(obj, list):
        for item in obj:
            if (isinstance(item, dict) and not _is_citation_item(item)
                    and any(k in item for k in _SELECTION_KEYS)):
                return item
    return None


# C0 controls (minus \n\t) and unpaired UTF-16 surrogates in DELIVERED text:
# the selector occasionally emits garbage escapes like "frac" (meant
# \frac) -- unreconstructable, and control chars / lone surrogates can crash
# tokenizers downstream (see the step-6 lone-surrogate run kill), so drop them.
_CTRL_OR_LONE_SURROGATE_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
    r'|[\ud800-\udbff](?![\udc00-\udfff])'
    r'|(?<![\ud800-\udbff])[\udc00-\udfff]')


def _postprocess_selection(obj: dict) -> dict:
    """Sanitize the fields we deliver/score (never citation quotes -- those
    must stay verbatim for trace matching): restore newline-decoded
    \\n-commands, drop control chars and unpaired surrogates."""
    for k in _STR_KEYS:
        v = obj.get(k)
        if not isinstance(v, str):
            continue
        if "\n" in v:
            v = _restore_nl_commands(v)
        obj[k] = _CTRL_OR_LONE_SURROGATE_RE.sub("", v)
    return obj


def parse_output(raw: str):
    """Pull the JSON object out of <output>...</output>, then fall back to a
    brace-matched object and a fenced ```json block. Tolerates invalid LaTeX
    backslash escapes. Always returns a dict or None -- a JSON array from the
    selector is coerced to its first object. Returns (obj, error)."""
    m = OUTPUT_RE.search(raw)
    if m:
        block = m.group(1).strip()
        # strip a ```json fence if the model added one inside the tags
        block = re.sub(r"^```(?:json)?\s*|\s*```$", "", block.strip())
        obj = _as_selection_dict(_loads_lenient(block))
        if obj is not None:
            return _postprocess_selection(obj), None
        obj = _as_selection_dict(_extract_last_json_object(block))
        if obj is not None:
            return _postprocess_selection(obj), None
        err = "json decode failed (after escape repair)"
    else:
        err = "no <output> block"
    obj = _as_selection_dict(_extract_last_json_object(raw))
    if obj is not None:
        return _postprocess_selection(obj), None
    # hard fallback: field-by-field regex extraction that tolerates JSON the
    # parser rejects (e.g. unescaped LaTeX backslashes). Prefer the <output>
    # block when present, else scan the whole completion.
    obj = _hard_parse(m.group(1) if m else raw) or _hard_parse(raw)
    if obj is not None:
        return _postprocess_selection(obj), None
    return None, err


def hint_id_of(selection):
    """Selected hint id as a string, else None. Pool ids always contain a
    digit ("2.1"); the selector's explicit "nothing to select" answers come as
    "" or "none"/"null" -- map those to None so a decline can't be appended to
    ``completed``, priced as a real hint, or injected as a placeholder."""
    if not isinstance(selection, dict):
        return None
    hid = selection.get("hint_id")
    if hid is None:
        return None
    s = str(hid).strip()
    if not re.search(r"\d", s):
        return None
    return s
