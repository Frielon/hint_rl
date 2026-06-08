# Wrong-Step Analysis — `selector_gpt-oss-20b`

Analysis of the **11 problems** (of 132) where GPT-OSS-20B's majority-voted *major step*
disagreed with the codex reference. The task each model performs: given the student's
reasoning trace and an ordered menu of solution steps, identify **the earliest step the
student has not yet completed**. For each case we read the problem, the student's trace,
and both sides' reasoning, then judged who is actually right.

## Bottom line

| Who is wrong | Count | Cases |
|---|---|---|
| **GPT-OSS-20B is wrong** | **9 / 11** | 1-100, 100-91, 102-9, 106-86, 11-14, 112-73, 114-38, 114-42, 116-81 |
| **Codex is wrong** | **2 / 11** | 101-67, 112-38 |

The wrong-step errors are overwhelmingly the **model's** fault, not the reference's. And
the failure has a single dominant cause.

## Root cause: the model trusts the student; codex audits the student

**GPT-OSS-20B's recurring error (9 cases): it takes the student's work at face value and
jumps to where the student *feels* stuck, without checking whether the earlier work is
correct.** Codex instead audits the earlier steps and correctly rewinds to the first one
that is actually broken.

This splits into two sub-patterns:

### A. Model believes the student's wrong conclusion (4 cases)
The student states a confident but *false* result; the model accepts it as "done" and
advances. Codex catches the error and stays on the broken step.

- **100-91** (cos(x/4)=cos x): Student dismisses the first solution family as having "no
  positive solution" — but it actually contributes **8 solutions** (negative k → positive x).
  Model trusts the dismissal and jumps to family 2 (step 3); codex correctly keeps them on
  step 2.
- **102-9** (max isotropic subspace of trace form): Student claims "products of skew-symmetric
  matrices are symmetric with zero trace" — **false** (skew is the *negative-definite* part,
  tr(A²)<0). Model treats the skew construction as valid and jumps to "prove the upper bound"
  (step 2); codex stays on step 1, the misunderstood decomposition.
- **114-42** (slow watch): Student misreads "loses 2.5 min/**day**" as "per real minute" and
  writes a meaningless `6150/2.5`. Model treats 6150 as a valid elapsed time and moves to
  step 2; codex stays on step 1, the unestablished rate relation.
- **116-81** (two blocks on a table): Student writes a mis-modeled system (one block height +
  a spurious √2 tilt) instead of two distinct side lengths. Model sees "two equations, just
  solve them" and jumps to the algebra (step 4); codex stays on step 1, the wrong setup.

### B. Model accepts a value the student "wrote down" without auditing correctness (3 cases)
The student produces a number/figure that *looks* like the step output but is wrong or
mislabeled; the model rewards the surface form.

- **11-14** (meeting-probability geometry): Student sets the girl's window to **7–9 AM**
  instead of **8–9 AM** — wrong rectangle. Model sees "a rectangle was drawn" and jumps to
  area computation (step 3); codex stays on step 1. (Notably, several model samples *noticed*
  the 7-vs-8 discrepancy in their hidden reasoning and waved it off anyway.)
- **112-73** (median raised to 85): Student's score distribution is miscounted (sums to 13,
  not 20; two 80s instead of five). The answer depends critically on there being five 80s.
  Model anchors on the correct-sounding "positions 10–11 are 80" and jumps to step 2; codex
  stays on step 1, the wrong cumulative counts.
- **114-38** (circle angle chase): Student computes "40°" but attaches it to ∠BAS — an
  impossible central angle (A, S, B collinear) — and never records AC's direction. Model
  credits the bare "40°" and the mention of AC⊥SD as completing step 1, advancing to step 2;
  codex stays on step 1.

### C. Model rewinds too far on a trivial omission (1 case)
The mirror image of the above, and still the model's fault:

- **106-86** (glue three cubes, min surface area): Student has computed the per-cube areas
  (6, 24, 54) and stated the maximize-contact principle, and is stuck bounding the contact —
  i.e. genuinely at step 3. The model fixates on the fact that the student never *literally
  wrote* the sum "84" and rewinds two steps to step 1. Codex correctly reads them at step 3.

### D. Model jumps to a later step on a not-yet-built foundation (1 case)
- **1-100** (volume of a folded net): Student has only computed planar areas and explicitly
  has no 3D model. The true step is step 1 (model the hexagon as a cube slice). The model's
  plurality picks step 3 (the half-cube symmetry argument) — and its own reasoning even
  admits "they have not yet considered how the net corresponds to a known solid," then picks
  step 3 anyway. Codex correct.

## The 2 cases where CODEX is wrong

Both are the *same* ambiguity, going the opposite direction from the model's usual error:
**how much credit does "recognizing" a step earn before any of its work is done?** Here
codex *over*-credits a passive mention and skips a genuinely-unfinished step 1; the model's
stricter "it isn't done until it's actually carried out" reading is correct.

- **101-67** (divisibility by 323): Student only *notes* "323 = 17·19" and *suspects* CRT,
  explicitly saying they "haven't been able to establish sufficient conditions" and have done
  zero residue work. Codex jumps to step 2 (compute mod 17); the model unanimously (16/16)
  stays on step 1 (establish the reduction). **Model correct** — naming a factorization is
  not completing the reduction.
- **112-38** (monochromatic-triangle probability): Student recognizes the 2¹⁰ sample space
  but never defines N or writes the probability as 1−N/1024, and his vertex remark is the
  *wrong* local condition. Codex generously calls step 1 "essentially done" and reads the
  garbled remark as step-2 work; the model majority (10/16) stays on the unfinished step 1.
  **Model correct.**

## Tension worth noting

Cases **101-67** and **112-38** (model right) versus **106-86** (codex right) are the same
*type* of judgment — "is a step complete when it's been gestured at but not formalized?" —
resolved in opposite directions. The distinction our analysts drew: in 106-86 the student
had done the step's *substantive* reasoning (and only omitted a trivial arithmetic sum),
whereas in 101-67 / 112-38 the substantive work (the reduction; defining N) was genuinely
absent. That is a defensible line, but it shows a few of these disagreements are
**rubric-boundary calls**, not clean right/wrong — the grading rubric for "completed" is
underspecified.

## Takeaways

1. **9 of 11 wrong-step cases are GPT-OSS-20B errors**; only 2 are codex errors. The
   reference is substantially more reliable on step identification.
2. **The model's dominant failure mode is trusting the student.** It does not verify the
   correctness of prior work — it locates where the student *says* they are stuck and serves
   a hint there. When the student has a wrong premise (≈7 of the 9), the model inherits it
   and skips past the actual error. A tutor must audit, not echo.
3. **Codex's rare errors are the opposite bias** (over-crediting a recognized-but-unexecuted
   step), and they cluster on an underspecified rubric question: when does "recognizing" a
   step count as "completing" it?
4. **Suggested fix for the selector:** add an explicit correctness-audit instruction —
   *"before accepting that a step is complete, verify the student's stated results for that
   step are actually correct; a confident but wrong conclusion means the step is NOT
   complete."* This directly targets the 7 face-value-trust failures.
