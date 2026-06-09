# Copyright 2026
#
# Hint-importance-weight computation for HPRL (paper Section 3/6).
#
# The trajectory reward subtracts the applied hints' importance weights from the
# correct reward:  R = R_acc - sum_k w_{j_k}  (penalty applied only when correct;
# see hint_reward.compute_score). This module computes those per-hint weights w
# from the ORIGINAL (difficulty-annotated) hint JSON.
#
# Design (all knobs are parameters, so they can be retuned WITHOUT regenerating
# the dataset):
#
#   total_penalty (default 1.8)
#       The penalties of ALL hints in a problem's pool sum to this. Independent
#       of hard_factor (which only redistributes, never changes the total).
#
#   hard_factor (default 1.5)
#       Difficulty weight = hard_factor ** level, level = {easy:0, moderate:1,
#       hard:2}. So each difficulty step up multiplies the weight by hard_factor
#       (harder -> larger penalty). Adjust this one number to make hard steps/
#       hints heavier or lighter.
#
#   guidance_difficulty (default "moderate")
#       The X.0 step-guidance hint carries no substep difficulty of its own, so
#       it is assigned this difficulty inside its step.
#
# Two-level split:
#   1. step penalty  = total_penalty * step_weight / sum(step_weights)
#   2. hint penalty  = step_penalty  * hint_weight / sum(hint_weights in step)
#      (the within-step coefficient is hint_weight / sum -> a fraction summing to 1)
# so sum over hints in a step == step penalty, and sum over all == total_penalty.
#
# Two penalty strategies consume this split (selected by ``strategy``; the agent
# loop reveals the matching granularity -- see hint_reward.compute_score):
#   * STRATEGY_HINT       charges level-2 per-hint penalties for the individual
#     hints applied (``applied_penalty``).
#   * STRATEGY_MAJOR_STEP charges level-1 step penalties directly, one per major
#     step the rollout was shown in full (``applied_step_penalty``).
#
# The ORIGINAL hint JSON (extra_info.hint_full, stored by prepare_hint_data.py)
# is a list of steps:
#   [{"step_id", "difficulty", "substeps": [{"substep_id": "1.1", "difficulty"}]}]
# Each step's hints are: the guidance hint "<step_id>.0" plus one hint per
# substep, keyed by its substep_id ("1.1", "1.2", ...) -- matching the hint_id
# the selector returns.

from __future__ import annotations

import json
from typing import Optional

# difficulty -> exponent applied to hard_factor. Editable if more levels appear.
DIFFICULTY_LEVEL = {"easy": 0, "moderate": 1, "hard": 2}
DEFAULT_TOTAL_PENALTY = 1.8
DEFAULT_HARD_FACTOR = 1.5
DEFAULT_GUIDANCE_DIFFICULTY = "moderate"

# --- hint-selection + penalty strategy ------------------------------------- #
# Two strategies share this penalty module (and the same difficulty-annotated
# hint pool); the canonical names live here so the agent loop and the reward read
# the same source of truth.
#   STRATEGY_HINT       -- the selector reveals ONE hint within the identified
#                          major step; that hint is excluded next call; the reward
#                          charges the per-hint penalty (applied_penalty).
#   STRATEGY_MAJOR_STEP -- the selector identifies a major step and ALL of its
#                          hints are revealed at once; the WHOLE step is recorded
#                          and excluded next call; the reward charges the step
#                          penalty directly (applied_step_penalty).
STRATEGY_HINT = "hint"
STRATEGY_MAJOR_STEP = "major_step"
_MAJOR_STEP_ALIASES = {"major_step", "major-step", "majorstep", "major", "step", "step_penalty"}


def normalize_strategy(strategy) -> str:
    """Canonicalize a strategy string to ``STRATEGY_HINT`` or ``STRATEGY_MAJOR_STEP``.

    Tolerant of case and a few spellings (``"step"``, ``"major"``, ...); anything
    unrecognized falls back to the default per-hint strategy.
    """
    s = str(strategy or "").strip().lower()
    return STRATEGY_MAJOR_STEP if s in _MAJOR_STEP_ALIASES else STRATEGY_HINT


def difficulty_weight(difficulty, hard_factor: float) -> float:
    """``hard_factor ** level(difficulty)``; unknown difficulty -> moderate."""
    level = DIFFICULTY_LEVEL.get(str(difficulty).strip().lower(), DIFFICULTY_LEVEL["moderate"])
    return float(hard_factor) ** level


def _as_steps(hint_full) -> list:
    """Normalize the original hint JSON to a list of step dicts."""
    pool = json.loads(hint_full) if isinstance(hint_full, str) else hint_full
    if isinstance(pool, list):
        return pool
    if isinstance(pool, dict):  # tolerate {"steps": [...]}
        return pool.get("steps", [])
    return []


def compute_step_penalties(
    hint_full,
    *,
    total_penalty: float = DEFAULT_TOTAL_PENALTY,
    hard_factor: float = DEFAULT_HARD_FACTOR,
) -> dict[str, float]:
    """Map every major-step ``step_id`` (as a string) to its step-level penalty.

    This is the FIRST level of the two-level split::

        step_penalty = total_penalty * step_weight / sum(step_weights)

    with ``step_weight = difficulty_weight(step.difficulty)``. ``sum(result.values())
    == total_penalty`` (up to float error), regardless of ``hard_factor``.

    The major-step strategy charges these directly (one whole major step revealed
    per hint call -> ``applied_step_penalty``); the per-hint strategy subdivides
    each step penalty further across the step's hints (see ``compute_hint_penalties``).
    """
    steps = _as_steps(hint_full)
    if not steps:
        return {}
    step_weights = [difficulty_weight(s.get("difficulty"), hard_factor) for s in steps]
    total_step_w = sum(step_weights) or 1.0
    return {
        str(s.get("step_id")): total_penalty * w / total_step_w
        for s, w in zip(steps, step_weights)
    }


def compute_hint_penalties(
    hint_full,
    *,
    total_penalty: float = DEFAULT_TOTAL_PENALTY,
    hard_factor: float = DEFAULT_HARD_FACTOR,
    guidance_difficulty: str = DEFAULT_GUIDANCE_DIFFICULTY,
) -> dict[str, float]:
    """Map every ``hint_id`` in the pool to its penalty weight.

    ``sum(result.values()) == total_penalty`` (up to float error), regardless of
    ``hard_factor``.
    """
    steps = _as_steps(hint_full)
    if not steps:
        return {}

    # first level: the per-step penalties (these sum to total_penalty).
    step_penalties = compute_step_penalties(
        hint_full, total_penalty=total_penalty, hard_factor=hard_factor
    )

    penalties: dict[str, float] = {}
    for step in steps:
        step_id = step.get("step_id")
        step_penalty = step_penalties.get(str(step_id), 0.0)

        # second level: split this step's penalty across its hints --
        # guidance "<step_id>.0" + one per substep.
        hint_diffs = [(f"{step_id}.0", guidance_difficulty)]
        for ss in step.get("substeps") or []:
            hint_diffs.append((str(ss.get("substep_id")), ss.get("difficulty")))

        hint_weights = [difficulty_weight(d, hard_factor) for _, d in hint_diffs]
        total_hint_w = sum(hint_weights) or 1.0
        for (hid, _), w_hint in zip(hint_diffs, hint_weights):
            penalties[str(hid)] = step_penalty * w_hint / total_hint_w

    return penalties


def applied_penalty(
    applied_hints: list,
    hint_full,
    *,
    total_penalty: float = DEFAULT_TOTAL_PENALTY,
    hard_factor: float = DEFAULT_HARD_FACTOR,
    guidance_difficulty: str = DEFAULT_GUIDANCE_DIFFICULTY,
) -> float:
    """Sum of penalties for the hints applied in a rollout (``sum_k w_{j_k}``)."""
    if not applied_hints or not hint_full:
        return 0.0
    penalties = compute_hint_penalties(
        hint_full,
        total_penalty=total_penalty,
        hard_factor=hard_factor,
        guidance_difficulty=guidance_difficulty,
    )
    total = 0.0
    for h in applied_hints:
        hid = h.get("hint_id") if isinstance(h, dict) else None
        if hid is not None:
            total += penalties.get(str(hid), 0.0)
    return total


def applied_step_penalty(
    applied_hints: list,
    hint_full,
    *,
    total_penalty: float = DEFAULT_TOTAL_PENALTY,
    hard_factor: float = DEFAULT_HARD_FACTOR,
) -> float:
    """Sum of STEP penalties for the major steps revealed in a rollout.

    The major-step strategy (STRATEGY_MAJOR_STEP) reveals one whole major step per
    hint call and records it on ``applied_hints[k]["major_step_id"]``; this charges
    that step's penalty (``compute_step_penalties``) directly -- "apply the step
    penalty directly". A step revealed more than once is charged only once (the
    next-call whole-step exclusion normally prevents repeats; this just guards the
    edge case), so the policy is penalized once per distinct step of the solution
    it was shown.

    Returns 0.0 when no steps were revealed or no ``hint_full`` is present.
    """
    if not applied_hints or not hint_full:
        return 0.0
    step_penalties = compute_step_penalties(
        hint_full, total_penalty=total_penalty, hard_factor=hard_factor
    )
    total = 0.0
    seen: set[str] = set()
    for h in applied_hints:
        sid = h.get("major_step_id") if isinstance(h, dict) else None
        if sid is None:
            continue
        sid = str(sid)
        if sid in seen:
            continue
        seen.add(sid)
        total += step_penalties.get(sid, 0.0)
    return total
