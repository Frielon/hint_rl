"""Local copies of the offline selector prompt + tolerant <output> parser.

These were previously imported from ``${HINT_RL_HOME}/selector`` (``seletor_prompt.py``
and ``run_hint_selection_model.py``) via a sys.path hack. They are vendored here so
that everything the HPRL rollout needs lives inside ``script/hint_rl`` and the package
has no cross-folder import dependency.

Kept verbatim so the rollout uses the *same* prompt template and the *same* tolerant
<output> JSON parser the offline selector eval validated against. If the offline
selector changes, sync the relevant pieces here.
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

The current step is the **earliest step the student has not yet completed correctly**. Note that this may be *earlier* than where the student feels stuck: if the student's work first went wrong at an earlier step, anchor on that step, not on the point they report being stuck. Justify your choice with specific evidence from the trace, including any error you found in the student's earlier work.

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
    preserved verbatim in the decoded string)."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)


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
