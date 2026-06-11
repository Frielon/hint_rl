"""Prompt variants for the hint-selector, targeting the `jump_last` failure.

Failure being attacked (see unearned_final_step_reveals.README.md): with earlier
steps still unrevealed and the student having written <500 chars since the last
hint, the deployed selector picked the pool's FINAL step -- which carries the
literal answer for ~96% of pools -- with self-reported confidence 5.

Every variant keeps the same inputs (problem, trace, hints_filtered) and the
same <output> JSON schema, so results are directly comparable and a winning
variant can drop into ``utils.selector_prompt`` unchanged. Variants are defined
as exact-substring edits of the baseline template; ``_edit`` raises if an anchor
goes stale, so a baseline rewrite can't silently no-op an edit.

Registry: ``VARIANTS[name] -> Variant(description, trace_field, build)`` where
``build(problem, trace, hints) -> prompt``.
"""
from dataclasses import dataclass
from typing import Callable, Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seletor_prompt import selector_prompt  # baseline template lives there

# The baseline template text (selector_prompt fills {{...}} via .replace).
_BASE = selector_prompt("{{problem}}", "{{trace}}", "{{hints}}")


def _edit(template: str, anchor: str, replacement: str) -> str:
    """Replace an exact anchor substring; raise if the anchor is missing."""
    if anchor not in template:
        raise ValueError(f"prompt anchor not found:\n{anchor[:120]}...")
    return template.replace(anchor, replacement)


def _fill(template: str) -> Callable[[str, str, str], str]:
    def build(problem: str, trace: str, hints: str) -> str:
        return (template
                .replace("{{problem}}", problem)
                .replace("{{trace}}", trace)
                .replace("{{hints}}", hints))
    return build


@dataclass(frozen=True)
class Variant:
    description: str
    trace_field: str                      # parquet column used as {{trace}}
    build: Callable[[str, str, str], str]  # (problem, trace, hints) -> prompt


# --------------------------------------------------------------------------- #
# anchors in the baseline template (exact substrings)
# --------------------------------------------------------------------------- #
_A_STEP1_TAIL = (
    "The current step is the **earliest step the student has not yet completed "
    "correctly**."
)
_A_WORKFLOW2 = "## 2. Select a hint within that major step"


# --------------------------------------------------------------------------- #
# v1: earned-progress rules -- hint text is not student work
# --------------------------------------------------------------------------- #
_V1_RULES = """### Ground rules for crediting progress
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

"""

_V1 = _edit(_BASE, _A_STEP1_TAIL, _V1_RULES + _A_STEP1_TAIL)


# --------------------------------------------------------------------------- #
# v2: v1 + explicit gate on the final step (it usually contains the answer)
# --------------------------------------------------------------------------- #
_V2_GATE = """### Guard the final step
The last major step typically states the final answer. Selecting it gives the
answer away, so it is justified **only** when the student's own written work
has correctly completed every earlier candidate step. If you are at all unsure
whether an earlier candidate step is genuinely done, select that earlier step
instead. An unearned answer reveal is a much worse error than a hint that is
one step too early.

"""

_V2 = _edit(_V1, _A_WORKFLOW2, _V2_GATE + _A_WORKFLOW2)


# --------------------------------------------------------------------------- #
# v3: v2 + forced per-candidate verification checklist
# --------------------------------------------------------------------------- #
_V3_CHECKLIST = """Before choosing, go through **every candidate step in order** and give a
verdict: `completed by the student's own work: yes/no`, each with one line of
evidence quoted or paraphrased from the *student's* writing (not from hint
text). The current step is the **first candidate with verdict "no"**. Put this
checklist in `reasoning_of_major_step`.

"""

_V3 = _edit(_V2, _A_STEP1_TAIL, _V3_CHECKLIST + _A_STEP1_TAIL)


# --------------------------------------------------------------------------- #
# v4: v2 + per-step VERBATIM CITATIONS of student work (user idea 2026-06-10):
# to select step N, the selector must quote the student's own words that carry
# out every earlier candidate step, in a machine-checkable `completed_steps`
# JSON array. Fabricated/hint-derived quotes are detectable by substring match
# against the trace, so the agent loop can ENFORCE the justification.
# --------------------------------------------------------------------------- #
_A_GUARD = "### Guard the final step"
_A_MAJOR_ID = '"major_step_id": <step_id>,'

_V4_CITE_RULE = """### Cite your evidence
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

"""

_V4 = _edit(_V2, _A_GUARD, _V4_CITE_RULE + _A_GUARD)
_V4 = _edit(
    _V4,
    _A_MAJOR_ID,
    '"completed_steps": [{"step_id": <id of a candidate step earlier than your selection>, '
    '"quote": <verbatim excerpt from the student\'s own writing that carries out this step>, '
    '"why": <one line: why this excerpt completes the step>}, ...] '
    '(one entry for EVERY candidate step listed before your selected step; [] if you selected the earliest),\n'
    '  ' + _A_MAJOR_ID,
)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
VARIANTS: Dict[str, Variant] = {
    # reference points
    "as_seen": Variant(
        "deployed conditions: baseline prompt + hints-only trace (blind-trace bug)",
        "trace_as_seen", _fill(_BASE)),
    "baseline": Variant(
        "baseline prompt + intended full trace (fixed agent loop)",
        "trace_full", _fill(_BASE)),
    # candidates
    "v1_earned": Variant(
        "baseline + earned-progress rules (hint text is not student work)",
        "trace_full", _fill(_V1)),
    "v2_final_gate": Variant(
        "v1 + explicit guard on selecting the answer-bearing final step",
        "trace_full", _fill(_V2)),
    "v3_checklist": Variant(
        "v2 + forced per-candidate completed-yes/no checklist",
        "trace_full", _fill(_V3)),
    "v4_cite": Variant(
        "v2 + machine-checkable verbatim citations of student work per completed step",
        "trace_full", _fill(_V4)),
}


if __name__ == "__main__":
    # quick self-check: anchors resolve, templates fill, and variants differ
    for name, v in VARIANTS.items():
        p = v.build("P", "T", "H")
        assert "{{" not in p, name
        print(f"{name:15s} trace={v.trace_field:13s} len={len(p):5d}  {v.description}")
