"""Multi-round Template F selector prompt -- EDIT FREELY.

Adapts Template F to a multi-round tutoring setting: some hints are already
marked ``completed`` (given/verified in earlier rounds); the model must select
the next hint among the still-``pending`` ones, and cite any pending hints it
newly recognizes as achieved this round.

``build_prompt(problem, trace, hints)`` fills the {{problem}}/{{trace}}/{{hints}}
markers. ``render_hints_with_status(pool, completed)`` renders the pruned hint
pool with a per-hint ``status`` (completed | pending) for the {{hints}} slot.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


TEMPLATE_MULTI_F = r"""You are an expert math tutor helping a student over multiple rounds. In earlier rounds some hints were already given to the student and verified as achieved; those are marked `status: "completed"`. Your task is to choose the first hint that the student has not yet achieved, considering only the hints still marked `status: "pending"`, based on the student's reasoning trace.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

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
    """Render the (pruned) hint pool for the {{hints}} slot, tagging each hint
    with status completed|pending. Each hint is shown as {hint_id, hint, status}."""
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


def build_prompt(problem: str, trace: str, hints: str) -> str:
    """Fill the multi-round Template F placeholders."""
    return (TEMPLATE_MULTI_F
            .replace("{{problem}}", problem)
            .replace("{{trace}}", trace)
            .replace("{{hints}}", hints))
