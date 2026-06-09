# Copyright 2026
#
# Shared hint-tool system-prompt rendering for HPRL.
#
# Both the offline data prep (prepare_hint_data.py) and the online dynamic-budget
# dataset (hint_dataset.HintBudgetDataset) need to render the *same* tool/budget
# instruction sentence -- the only thing that varies between them is the budget
# number B_q (static at prep time, dynamic per-epoch under the budget ratchet).
#
# Keeping the template here, and having both sides call render_system(), means
# the dynamic re-render at __getitem__ time is byte-identical to the baked prompt
# except for the budget digits, so turning the ratchet on/off never silently
# changes the prompt wording.

from __future__ import annotations

import re

# Default system preamble used when a sample carries no system message.
# NB: deliberately NOT "...solving problems step by step" -- a "step by step" /
# "think step by step" phrasing is a chain-of-thought trigger that makes the
# model push straight to \boxed{} and almost never emit <hint_call/> (offline
# bisect 2026-06-09: closer "step by step" 8-11% emit vs "briefly" ~28%). If a
# sample's own base_system carries such a tail (e.g. the DAPO user-message
# "Let's think step by step"), it will re-suppress emission -- strip it upstream.
DEFAULT_BASE_SYSTEM = "You are a math assistant."

# The tool/budget instruction appended to the base system prompt. Rendered via
# str.format(budget=...): ``{budget}`` is the per-sample placeholder and every
# literal brace is doubled (``\boxed{{...}}`` -> ``\boxed{...}``). Raw string so
# the ``\b`` in ``\boxed`` stays literal.
#
# Mechanism note: the hint is requested by emitting the sentinel ``<hint_call/>``
# on its own line (NOT a hermes ``request_hint`` tool call); the rollout side must
# detect that sentinel and inject the curated hint back as the next USER message.
TOOL_INSTRUCTION = r"""

A hint tool is available. Emitting the exact sentence <hint_call/> on its own line causes the system to inject one curated hint as the next user message. You may use the hint tool at most {budget} time(s) for this problem, so call it only when you are genuinely stuck.

Before each hint call, briefly describe your progress so far: what you have established and where you are stuck. This summary is what the hint is based on, so make it accurate. Then emit <hint_call/> on its own line.

Example:
User: How many positive integers less than 1000 are divisible by neither 5 nor 7?
Assistant: I want to count the integers from 1 to 999 that avoid both 5 and 7. It is easier to count those that ARE divisible by 5 or 7 and subtract, but I'm not sure how to combine the two divisibility counts without double-counting.
<hint_call/>
User (hint): Use inclusion-exclusion: |A or B| = |A| + |B| - |A and B|, where A is the multiples of 5 and B is the multiples of 7.
Assistant: So I need three counts up to 999. Multiples of 5: floor(999/5) = 199. Multiples of 7: floor(999/7) = 142. I have those, but I still need the overlap (multiples of both 5 and 7) and I'm not certain of the right modulus for it. That is a separate sticking point, so I'll ask for another hint rather than guess.
<hint_call/>
User (hint): Numbers divisible by both 5 and 7 are exactly the multiples of 35, so count floor(999/35) = 28.
Assistant: Then |A or B| = 199 + 142 - 28 = 313, so the count divisible by neither is 999 - 313 = 686. \boxed{{686}}

Now solve the user's problem in this style: reason briefly, summarize your progress, and emit <hint_call/> whenever you get stuck and need a hint. End your final answer in \boxed{{...}}.

IMPORTANT: After you emit <hint_call/>, STOP immediately and output nothing more. Do NOT write the hint yourself and do NOT keep solving. The system will send the hint as the next user message; only after it arrives do you continue (as shown in the example above)."""


def render_system(base_system: str | None, budget: int) -> str:
    """Render the full system prompt: ``base_system`` + the tool/budget sentence.

    ``base_system`` is the budget-free preamble (e.g. "You are a math expert.");
    ``budget`` is this sample's current B_q. Falls back to DEFAULT_BASE_SYSTEM
    when ``base_system`` is empty/None.

    ``budget <= 0`` means NO hints for this problem (the hard-problem-curriculum
    "easy" bucket, or a problem ratcheted to 0): the tool instruction is dropped
    entirely so the prompt never advertises a tool the rollout won't honor. This
    keeps prep (prepare_hint_data) and the dynamic re-render (HintBudgetDataset)
    consistent at budget 0.
    """
    base = base_system if base_system else DEFAULT_BASE_SYSTEM
    if int(budget) <= 0:
        return base
    return base + TOOL_INSTRUCTION.format(budget=int(budget))


# --- user-message budget reminder ------------------------------------------
# Placement is the dominant lever for hint-call emission. The budget fact in the
# SYSTEM prompt barely moves the policy, but appended as the LAST sentence of the
# USER message -- the final context before generation -- it ~doubles emission and
# ~triples the multi-hint-call rate (hint_call_test Round 10: 9%->25% multi-call).
#
# DAPO-style math datasets append a "Let's think step by step and output the final
# answer within \boxed{}" instruction to the user turn. That is a CoT trigger that
# both suppresses <hint_call/> and, sitting right before the reminder, blunts it
# (keeping it caps multi-call ~13% vs ~25% stripped). strip_user_cot_tail removes
# such a trailing instruction; \boxed{} is still mandated by the system prompt.
_USER_COT_TAIL_RE = re.compile(
    r"\s*(?:let'?s|let us|please|now|first,?)\b[^.]*\bstep[\s-]by[\s-]step\b.*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_user_cot_tail(user_text: str) -> str:
    """Remove a trailing DAPO-style "...think step by step ... \\boxed{}" instruction."""
    return _USER_COT_TAIL_RE.sub("", user_text).rstrip()


def render_user(user_base: str, budget: int) -> str:
    """Append the budget reminder as the last sentence of the user message.

    ``user_base`` should already be CoT-tail-stripped (see strip_user_cot_tail).
    No reminder is added when ``budget <= 0``: that sample has the hint tool
    disabled (render_system also drops the tool instruction), so a "no calls
    remaining" line would be misleading.
    """
    base = user_base.rstrip()
    if int(budget) <= 0:
        return base
    return base + "\n\n" + render_remaining_calls(int(budget))


def render_remaining_calls(remaining: int) -> str:
    """Render the one-sentence "N hint calls left" notice appended to a delivered hint.

    ``remaining`` is ``budget - hints_used_so_far`` *after* the just-delivered hint
    is counted, i.e. how many MORE times the policy may call the hint tool on this
    problem. Lives here next to ``render_system`` so the budget wording stays in one
    module; both rollout paths (hint_agent_loop.HintAgentLoop and the legacy
    hint_tool.HintTool) call it, so the notice reads identically either way.
    """
    remaining = max(int(remaining), 0)
    if remaining == 0:
        return "You have no hint calls remaining for this problem, so finish on your own."
    noun = "call" if remaining == 1 else "calls"
    return f"You have {remaining} hint {noun} remaining for this problem."
