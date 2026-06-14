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
#
# DELIBERATELY NO few-shot example (Round 14, 2026-06-10): any worked demo is
# parroted — the Round-13 "Step 1: ... Step 2: ..." demo had the live policy
# copying the skeleton in 78-86% of first turns with ever-shorter step
# statements. Demo-free, the style bias disappears (~5% step-form, all genuine),
# pre-1st-call CoT doubles (median 341 -> 607 tok), and the two protocol worries
# do NOT regress: sentinel format had 0 malformed attempts in 720 rollouts (the
# instruction sentence anchors it) and self-fabricated "User (hint)" turns
# stayed ~0 (the IMPORTANT-STOP sentence does that work). Cost: offline emission
# halves to ~20%, right at the floor — watch live emit before tightening
# further. The closer keeps "reason step by step": demo-free it is BETTER on
# both axes than "reason briefly" (emit 19.6% vs 16.7%, median 607 vs 374;
# hint_call_test/test_no_demo_round14.py, progress.md Round 14).
TOOL_INSTRUCTION = r"""

A hint tool is available. Emitting the exact sentence <hint_call/> on its own line causes the system to inject one curated hint as the next user message. You may use the hint tool at most {budget} time(s) for this problem, so call it only when you are genuinely stuck.

Before each hint call, briefly describe your progress so far: what you have established and where you are stuck. This summary is what the hint is based on, so make it accurate. Then emit <hint_call/> on its own line.

Now solve the user's problem: reason step by step, summarize your progress, and emit <hint_call/> whenever you get stuck and need a hint. End your final answer in \boxed{{...}}.

IMPORTANT: After you emit <hint_call/>, STOP immediately and output nothing more. Do NOT write the hint yourself and do NOT keep solving. The system will send the hint as the next user message; only after it arrives do you continue."""


def render_system(base_system: str | None, budget: int) -> str:
    """Render the full system prompt: ``base_system`` + the tool/budget sentence.

    ``base_system`` is the budget-free preamble (e.g. "You are a math expert.");
    ``budget`` is this sample's current B_q. Falls back to DEFAULT_BASE_SYSTEM
    when ``base_system`` is empty/None.

    ``budget <= 0`` means NO hints for this problem (the hard-problem-curriculum
    "easy" bucket, or a problem ratcheted to 0). The tool instruction is STILL
    rendered (with the budget shown as 0, i.e. "at most 0 time(s)"); the user
    message's reminder reads "no hint calls remaining, finish on your own". This
    keeps the budget-0 prompt byte-identical to a budget>0 prompt except for the
    budget digit, so the policy sees the SAME framing/closer ("reason step by
    step ... \boxed{...}") whether or not it has hints. Earlier we dropped the
    instruction at budget 0; that gave those rollouts a bare base prompt with no
    closer and a visibly different response-length distribution (longer, no
    \boxed nudge). prep (prepare_hint_data) and the dynamic re-render
    (HintBudgetDataset) both go through here, so they stay consistent at budget 0.
    """
    base = base_system if base_system else DEFAULT_BASE_SYSTEM
    return base + TOOL_INSTRUCTION.format(budget=max(int(budget), 0))


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
    At ``budget <= 0`` the reminder is still appended and reads "no hint calls
    remaining, finish on your own" (render_remaining_calls(0)) -- the budget-0
    user message thus matches the budget>0 layout exactly, mirroring render_system
    which also keeps the tool instruction at budget 0.
    """
    base = user_base.rstrip()
    return base + "\n\n" + render_remaining_calls(max(int(budget), 0))


def rerender_messages_for_budget(
    messages: list,
    base_system: str | None,
    user_base: str | None,
    budget: int,
) -> list:
    """Return a NEW message list re-rendered to advertise ``budget``.

    The single source of truth for re-rendering a prompt at a (ratcheted or probed)
    budget. Both the dynamic-budget dataset (HintBudgetDataset.__getitem__) and the
    k-pack probe expansion (HPRLRayPPOTrainer._hprl_expand_kpacks) call this so a probe
    pack at ``B-j`` is byte-identical to what the dataset would have rendered at ``B-j``.

    Rewrites (a shallow-copied list -- the caller's list is left untouched):
      * the leading system turn -> ``render_system(base_system, budget)`` (inserted if
        the prompt has no system turn), and
      * the LAST user turn -> ``render_user(user_base, budget)`` (only when ``user_base``
        is provided; mirrors prepare_hint_data, which stores the budget-free user text).

    ``base_system`` / ``user_base`` are the budget-free bases baked by prepare_hint_data
    into ``extra_info.hprl_system_base`` / ``extra_info.hprl_user_base``.
    """
    out = [dict(m) for m in (messages or [])]
    new_system = {"role": "system", "content": render_system(base_system, budget)}
    if out and out[0].get("role") == "system":
        out[0] = new_system
    else:
        out.insert(0, new_system)
    if user_base is not None:
        for j in range(len(out) - 1, -1, -1):
            if out[j].get("role") == "user":
                out[j] = {"role": "user", "content": render_user(user_base, budget)}
                break
    return out


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
