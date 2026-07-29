"""Template F (v5_substep_cite) selector prompt -- EXPLICIT, EDIT FREELY.

Standalone copy of the v5_substep_cite prompt so it can be modified/tuned here
without touching the shared selector/prompt_variants.py. `build_prompt` fills the
three placeholders. To tune the prompt, edit TEMPLATE_F below; keep the
{{problem}}, {{trace}}, {{hints}} markers so build_prompt can fill them.
"""

TEMPLATE_F = r"""You are an expert math tutor. Your task is to choose the first hint that the student has not yet achieved based on the student's reasoning trace.

# Inputs

## Math problem
{{problem}}

## Student's current reasoning trace
{{trace}}

## Candidate hints
The hints are organized as an ordered list of major steps that build toward the solution. Each major step contains:
- `step_id`: the step's position in the solution.
- `purpose`: what this step accomplishes and why.
- `hints`: an ordered list of hints for the step.

{{hints}}

# Core rule:

Select the earliest substep hint, in order, whose mathematical idea is **not clearly achieved** in the student’s trace.

# Definitions:

* A hint is **achieved** only if the student’s trace clearly demonstrates the underlying mathematical idea of that hint.
* The student does not need to use the same wording as the hint.
* A hint is **not achieved** if the idea is missing, only partially present, contradicted, used incorrectly, or based on an unjustified guess.
* Do not skip an earlier unachieved hint even if the student has achieved later hints.
* For every hint that you mark as completed, you must provide an exact quote from the student’s trace proving completion.
* The quote must be copied character-for-character from the student’s trace.
* Preserve LaTeX markup exactly.
* Do not convert symbols to Unicode.
* Do not paraphrase the quote.
* The quote must be findable by exact string search in the student’s trace.
* If you cannot find an exact supporting quote, do not mark that hint as completed.

# Procedure:

1. Scan hints in order:

   * First by `step_id`.
   * Then by the order of hints within that step.

2. For each hint, decide whether the student achieved it:

   * If achieved, add it to `completed_hints` with exact quote evidence according to the following step 4.
   * If not achieved, stop immediately and select that hint.

3. Rephrase the selected hint only if necessary:

   * Keep the mathematical intent of the original hint unchanged.
   * Make it clear and helpful for the student.
   * Do not reveal the full solution unless the original hint already does.
   * If no rephrasing is needed, use the original hint text.

4. Write completed hints:

   * For each hint before the selected hint, include an entry in `completed_hints` with:
     - `hint_id`: the hint's id.
     - `quote`: the exact quote from the student's trace that demonstrates achievement.
     - `why`: a one-line explanation of why this quote demonstrates that the student achieved the hint.
   * For example, if the selected hint is "3.2", then `completed_hints` must include all hints from step 1 and step 2 and hint "3.1", but no hints from "3.2" or later.

# Output format
Return a single JSON object wrapped in <output> tags. The output must contains four fields: `hint_id` is the id of the selected hint (e.g., "2.1" or "2.2"), `hint` is its text, rephrased per step 3 if necessary, `completed_hints` is a list of hints that were achieved, and `reasoning_of_hint` is an explanation of why this hint was selected.

<output>
{
"completed_hints": [
{
"hint_id": "<id of a substep_hint before the selected hint_id, e.g. "1.1">",
"quote": "<a string copied character-for-character from the student's trace — LaTeX markup preserved exactly, no Unicode conversion, no paraphrase; must be findable by exact search in the trace>",
"why": "<one line, in your own words, explaining why this excerpt carries out this substep>"
} ## One entry for EACH hint that is achieved before the selected hint
],
"hint_id": "<hint_id of the selected hint>",
"reasoning_of_hint": "<why this hint was selected: explain why this is the first unachieved substep, or why the student needs this specific step-guidance hint>",
"hint": "<the selected hint text, rephrased according to step 3 if necessary>"
}
</output>
"""


def build_prompt(problem: str, trace: str, hints: str) -> str:
    """Fill the Template F placeholders with the problem, trace and hint pool."""
    return (TEMPLATE_F
            .replace("{{problem}}", problem)
            .replace("{{trace}}", trace)
            .replace("{{hints}}", hints))
