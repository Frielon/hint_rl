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