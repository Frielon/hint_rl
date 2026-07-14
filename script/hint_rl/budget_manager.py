# Copyright 2026
#
# Per-problem hint-budget ratchet for Hint Penalized RL (HPRL) -- paper Section 7.
#
# This module is the *pure, self-contained* budget bookkeeping for the downward
# ratchet. It owns two things and nothing else:
#
#   1. ``compute_downward_budget`` -- the rule that maps a problem's current
#      budget B_q and the group of rollout outcomes (correct? / #hints used) to
#      a (possibly lower) next budget.
#   2. ``BudgetManager`` -- a JSON-backed, problem_id-keyed store of the current
#      B_q for every problem, with ``update_group`` to fold a step's rollouts in
#      and ``get`` to read the budget back when the problem is next sampled.
#
# It is deliberately decoupled from verl: it knows nothing about DataProto,
# Ray, or the agent loop. The trainer-side wiring (collecting per-problem
# (correct, num_hints) from a batch, and injecting the updated B_q back into the
# next epoch's system prompt + tools_kwargs) lives elsewhere -- see the plan in
# the README / docs. This keeps the ratchet rule unit-testable in isolation.
#
# ---------------------------------------------------------------------------
# The downward rule (this module implements exactly this):
#
#   Let N            = total rollouts for the problem this step (e.g. rollout.n).
#       C            = number of those rollouts that reached the correct answer.
#       h_1..h_C     = the hint-call counts of the *correct* rollouts.
#       m            = min(h_1..h_C) -- fewest hints any correct rollout used.
#
#   * PRIMARY (frugal-success) rule: if some correct rollout solved with STRICTLY
#     FEWER hints than the current budget (m < B_q), set the new budget to m. The
#     problem is demonstrably solvable with m hints this step, so there is no
#     reason to keep granting more. This is UNGATED by the correct fraction -- a
#     single frugal success is sufficient evidence (aggressive by design).
#   * FALLBACK (reached only when there is no correct rollout, or every correct
#     rollout used the FULL budget, i.e. m == B_q): the (N/2)-th-smallest rule --
#       - If C < N/2 (fewer than half correct): the policy has not yet shown it can
#         reliably solve the problem at the current budget -> keep B_q unchanged.
#       - Else (C >= N/2): sort the correct rollouts' hint counts ASCENDING and take
#         the (N/2)-th smallest value v; set the new budget to v - decrement
#         (default 1). When the fallback runs, every correct rollout used the full
#         budget, so v == B_q and this simply squeezes B_q down by one.
#
#   Either way the result is clamped to [min_budget, current_budget]: this is the
#   *downward* ratchet, so it never raises B_q (the upward-on-plateau half of
#   Section 7 is a separate concern), and never drops below ``min_budget`` (default
#   0 -> the problem can ratchet all the way to fully-unaided).
# ---------------------------------------------------------------------------
#
# Usage as a library (trainer side):
#   bm = BudgetManager(path="budget_state.json", default_budget=8)
#   new_b = bm.update_group(problem_id, results=[(correct, n_hints), ...])
#   ...                    # later, when the problem is sampled again:
#   b_q = bm.get(problem_id)   # feed into the prompt + tools_kwargs
#
# Quick self-check:
#   python budget_manager.py --selftest

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

# A single rollout's outcome: (correct?, number of hint calls it used).
Result = Tuple[bool, int]


def load_budget_table(path: str) -> dict[str, int]:
    """Read just the ``{problem_id: B_q}`` table from a budget-state JSON file.

    The dataset-side reader (hint_dataset.HintBudgetDataset) uses this to look up
    the current per-problem budget without instantiating a full BudgetManager.
    Returns an empty dict if the file is missing or unreadable -- callers fall
    back to the per-row baked budget in that case.
    """
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return {str(k): int(v) for k, v in payload.get("budgets", {}).items()}
    except Exception:  # noqa: BLE001 -- missing/partial file -> empty table
        return {}


def get_create_budget(tools_kwargs, default: int, tool_name: str = "request_hint") -> int:
    """Read ``tools_kwargs[tool_name].create_kwargs.budget`` (``default`` if absent).

    The single reader for the per-row tool budget. Shared by the dataset (injection),
    the trainer (k-pack probe expansion), so they agree on what budget a row runs under.
    Defensive: returns ``default`` for any shape that isn't the expected nested dict.
    """
    if isinstance(tools_kwargs, dict):
        tool = tools_kwargs.get(tool_name)
        if isinstance(tool, dict):
            ck = tool.get("create_kwargs")
            if isinstance(ck, dict) and ck.get("budget") is not None:
                return int(ck["budget"])
    return int(default)


def set_create_budget(tools_kwargs, budget: int, tool_name: str = "request_hint") -> None:
    """Set ``tools_kwargs[tool_name].create_kwargs.budget = budget`` in place (no-op if
    the structure isn't the expected nested dict). The single writer, shared as above."""
    if not isinstance(tools_kwargs, dict):
        return
    tool = tools_kwargs.get(tool_name)
    if not isinstance(tool, dict):
        return
    ck = tool.get("create_kwargs")
    if isinstance(ck, dict):
        ck["budget"] = int(budget)


@dataclass
class BudgetUpdate:
    """The outcome of one downward-ratchet evaluation (for logging / debugging)."""

    old_budget: int
    new_budget: int
    n_total: int
    n_correct: int
    changed: bool
    # Which rule decided the new budget:
    #   "min_frugal" -- a correct rollout used < current_budget hints; the budget was
    #                   snapped to that minimum (the primary, ungated rule).
    #   "pivot"      -- fallback fired: new = (N/2-th smallest correct count) - decrement.
    #   "unchanged"  -- no correct rollout, or fewer than N/2 correct at the full budget.
    #   "raise"      -- ADAPTIVE mode, no correct rollout: budget RAISED by 1 (clamped to
    #                   max_budget). The only rule that increases B_q.
    #   "pivot_set"  -- ADAPTIVE mode, over half correct: new = the (N/2)-th smallest correct
    #                   hint count (NO decrement). See compute_adaptive_budget.
    #   "kpack"      -- the k-pack counterfactual-probe rule fired: new = the smallest
    #                   B' at which >= ``require_successes`` correct rollouts (pooled over
    #                   all probe packs) used <= B' hints (== the require_successes-th
    #                   smallest pooled correct hint count). See compute_kpack_budget.
    #   Any of the above may carry a "_held" SUFFIX (e.g. "pivot_set_held"): the rule
    #   chose a LOWER budget but the manager's allow_decrease=False veto held B_q at the
    #   current value (BudgetManager._hold_decrease). The decision stats (n_correct,
    #   pivot/min hint counts) are kept as the rule computed them.
    rule: str = "unchanged"
    # The (N/2)-th-SMALLEST correct hint count that drove a "pivot" decision (None
    # otherwise -- e.g. the fast path, or an unchanged budget).
    pivot_hint_count: Optional[int] = None
    # Fewest hints any CORRECT rollout used this step (None if none were correct).
    # Drives the "min_frugal" rule; recorded in every branch for logging.
    min_correct_hint_count: Optional[int] = None
    # k-pack only: the chosen frontier B' (the require_successes-th smallest pooled
    # correct hint count) that drove a "kpack" decision. None for every other rule.
    kpack_threshold: Optional[int] = None
    # k-pack only: how many distinct probe packs (budget levels) were pooled for this
    # decision. 1 means no probe ran (the k=1 path uses compute_downward_budget instead).
    kpack_num_packs: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "old_budget": self.old_budget,
            "new_budget": self.new_budget,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "changed": self.changed,
            "rule": self.rule,
            "pivot_hint_count": self.pivot_hint_count,
            "min_correct_hint_count": self.min_correct_hint_count,
            "kpack_threshold": self.kpack_threshold,
            "kpack_num_packs": self.kpack_num_packs,
        }


def compute_downward_budget(
    current_budget: int,
    results: Sequence[Result],
    *,
    min_budget: int = 0,
    decrement: int = 1,
) -> BudgetUpdate:
    """Compute the next (downward) budget for one problem from a group of rollouts.

    Args:
        current_budget: the budget B_q the rollouts were generated under.
        results: one ``(correct, num_hints)`` per rollout for THIS problem this
            step. ``num_hints`` is the number of hint calls the rollout actually
            consumed (== ``len(applied_hints)`` from HintTool, already capped at
            the budget by the tool).
        min_budget: floor on the budget (default 0 -> a problem may ratchet to
            fully unaided).
        decrement: how far below the pivot to set the new budget (default 1).

    Returns:
        a ``BudgetUpdate`` carrying the new budget and the decision metadata.

    The rule (documented in full at the top of this file):
        m = min hints over correct rollouts.
        m < current_budget   -> new = m            (PRIMARY frugal-success rule;
                                                    ungated by the correct fraction)
        otherwise            -> the (N/2)-th-smallest FALLBACK:
            C < N/2          -> unchanged
            C >= N/2         -> new = (N/2-th smallest correct hint count) - decrement
        everything clamped to [min_budget, current_budget].
    """
    n_total = len(results)
    correct_hint_counts = sorted(int(h) for ok, h in results if ok)
    n_correct = len(correct_hint_counts)
    # Fewest hints any CORRECT rollout used this step (None if none were correct).
    min_correct = correct_hint_counts[0] if n_correct else None

    def _clamp(b: int) -> int:
        # strictly-downward ratchet: never above current_budget, never below min_budget.
        return max(min_budget, min(b, current_budget))

    # PRIMARY (frugal-success) rule: if some correct rollout solved with STRICTLY
    # FEWER hints than the current budget, snap the budget straight down to that
    # most-frugal success -- the problem is demonstrably solvable with min_correct
    # hints, so there is no reason to keep granting more. Ungated by the correct
    # fraction: a single frugal success is sufficient evidence (aggressive by
    # design). Pre-empts the (N/2)-th-smallest fallback below.
    if min_correct is not None and min_correct < current_budget:
        new_budget = _clamp(min_correct)
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=new_budget,
            n_total=n_total,
            n_correct=n_correct,
            changed=(new_budget != current_budget),
            rule="min_frugal",
            pivot_hint_count=None,
            min_correct_hint_count=min_correct,
        )

    # FALLBACK (the prior mechanism): reached only when no rollout was correct, or
    # every correct rollout used the FULL budget (min_correct == current_budget).
    # Gate: change the budget only once at least half the rollouts are correct.
    # "C < N/2" with integer math is "2*C < N".
    if n_total == 0 or 2 * n_correct < n_total:
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=current_budget,
            n_total=n_total,
            n_correct=n_correct,
            changed=False,
            rule="unchanged",
            pivot_hint_count=None,
            min_correct_hint_count=min_correct,
        )

    # The (N/2)-th smallest correct hint count. 1-based rank N//2 -> 0-based index
    # N//2 - 1. n_correct >= N/2 guarantees the index is in range.
    rank = max(1, n_total // 2)
    pivot = correct_hint_counts[rank - 1]
    new_budget = _clamp(pivot - decrement)

    return BudgetUpdate(
        old_budget=current_budget,
        new_budget=new_budget,
        n_total=n_total,
        n_correct=n_correct,
        changed=(new_budget != current_budget),
        rule="pivot",
        pivot_hint_count=pivot,
        min_correct_hint_count=min_correct,
    )


def compute_adaptive_budget(
    current_budget: int,
    results: Sequence[Result],
    *,
    min_budget: int = 0,
    max_budget: int = 8,
) -> BudgetUpdate:
    """Two-sided budget rule (selected by ``ratchet_mode="adaptive"``).

    Unlike compute_downward_budget (strictly monotone-down), this can RAISE the budget
    when a problem has become unsolvable at its current budget:

      * NO correct rollout (C == 0)   -> RAISE by 1 (grant one more hint), clamped to
                                         ``max_budget``. The only branch that increases B_q.
      * OVER HALF correct (2*C > N)   -> set B_q to the (N/2)-th SMALLEST correct hint
                                         count -- the median-frugal success, with NO
                                         decrement -- clamped to [min_budget, max_budget].
      * otherwise (0 < C, 2*C <= N)   -> unchanged.

    Args:
        current_budget: the budget B_q the rollouts ran under this step.
        results: one ``(correct, num_hints)`` per rollout for THIS problem this step.
        min_budget: floor on the budget.
        max_budget: ceiling on the budget -- bounds the upward (raise) direction so a
            persistently-hard problem cannot grow without limit.

    Because a correct rollout's ``num_hints`` is capped at ``current_budget``, the
    ``pivot_set`` branch never exceeds the current budget (it lowers or holds); only the
    ``raise`` branch increases it.
    """
    n_total = len(results)
    correct_hint_counts = sorted(int(h) for ok, h in results if ok)
    n_correct = len(correct_hint_counts)
    min_correct = correct_hint_counts[0] if n_correct else None

    def _clamp(b: int) -> int:
        return max(min_budget, min(b, max_budget))

    # No correct rollout -> too hard at this budget: raise by 1.
    if n_correct == 0:
        new_budget = _clamp(current_budget + 1)
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=new_budget,
            n_total=n_total,
            n_correct=0,
            changed=(new_budget != current_budget),
            rule="raise",
            pivot_hint_count=None,
            min_correct_hint_count=None,
        )

    # Over half correct -> the (N/2)-th smallest correct hint count becomes the budget.
    # 1-based rank N//2 -> 0-based index N//2 - 1; 2*C > N guarantees it is in range.
    if n_total > 0 and 2 * n_correct > n_total:
        rank = max(1, n_total // 2)
        pivot = correct_hint_counts[rank - 1]
        new_budget = _clamp(pivot)
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=new_budget,
            n_total=n_total,
            n_correct=n_correct,
            changed=(new_budget != current_budget),
            rule="pivot_set",
            pivot_hint_count=pivot,
            min_correct_hint_count=min_correct,
        )

    # Some correct, but not over half -> hold.
    return BudgetUpdate(
        old_budget=current_budget,
        new_budget=current_budget,
        n_total=n_total,
        n_correct=n_correct,
        changed=False,
        rule="unchanged",
        pivot_hint_count=None,
        min_correct_hint_count=min_correct,
    )


def compute_kpack_budget(
    current_budget: int,
    results: Sequence[Result],
    *,
    min_budget: int = 0,
    require_successes: int = 2,
    num_packs: Optional[int] = None,
) -> BudgetUpdate:
    """The k-pack counterfactual-probe downward rule (paper §7 "double-rollout"/k-pack).

    Used INSTEAD of ``compute_downward_budget`` when a problem was probed this step --
    i.e. its rollouts were generated under MORE THAN ONE budget (the normal pack at
    ``B`` plus probe packs forced down to ``B-1 ... B-k+1``). ``results`` is the POOLED
    ``(correct, num_hints)`` over ALL of that problem's packs this step.

    Rule (user, 2026-06-13): gather every CORRECT rollout across all packs; find the
    smallest budget ``B'`` at which at least ``require_successes`` of them used ``<= B'``
    hints, and set the new budget to that ``B'``. Because a rollout that solved with
    ``h`` hints is genuine evidence the problem is doable with ``h`` (the probe packs
    were *forced* to a low budget, so a frugal solve there is real, not an artifact of a
    greedy policy that always spends its budget), this reads true need directly instead
    of the corrupted full-budget-usage signal.

    The smallest ``B'`` with ``>= require_successes`` correct rollouts at ``<= B'`` is
    exactly the ``require_successes``-th smallest pooled correct hint count: at that value
    the ``require_successes`` most-frugal correct rollouts all qualify, and below it fewer
    than ``require_successes`` do. ``require_successes >= 2`` is the robustness guard
    against a single answer-leak / fluke solve driving the budget down (see
    hprl-answer-leak-major-step); set it to 1 to recover the aggressive single-success
    behavior.

    Strictly downward: clamped to ``[min_budget, current_budget]``; never raises B_q. If
    fewer than ``require_successes`` correct rollouts exist (no corroborated frontier),
    the budget is HELD (rule "unchanged") -- there is no evidence one fewer hint works.

    Args:
        current_budget: the problem's current budget ``B`` (the MAX over its packs --
            the ceiling the ratchet is lowering from).
        results: pooled ``(correct, num_hints)`` over every pack this step.
        min_budget: floor (default 0 -> a problem may ratchet to fully unaided).
        require_successes: how many corroborating frugal successes are needed (default 2).
        num_packs: how many distinct budget levels were pooled (logging only).
    """
    n_total = len(results)
    correct_hint_counts = sorted(int(h) for ok, h in results if ok)
    n_correct = len(correct_hint_counts)
    min_correct = correct_hint_counts[0] if n_correct else None

    def _clamp(b: int) -> int:
        return max(min_budget, min(b, current_budget))

    require = max(1, int(require_successes))
    if n_correct >= require:
        # require-th smallest pooled correct hint count == smallest B' with >= require
        # correct rollouts at <= B'. (For require==1 this is the most-frugal success.)
        threshold = correct_hint_counts[require - 1]
        new_budget = _clamp(threshold)
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=new_budget,
            n_total=n_total,
            n_correct=n_correct,
            changed=(new_budget != current_budget),
            rule="kpack",
            pivot_hint_count=None,
            min_correct_hint_count=min_correct,
            kpack_threshold=threshold,
            kpack_num_packs=num_packs,
        )

    # Fewer than ``require`` corroborating successes -> no trustworthy frontier; hold.
    return BudgetUpdate(
        old_budget=current_budget,
        new_budget=current_budget,
        n_total=n_total,
        n_correct=n_correct,
        changed=False,
        rule="unchanged",
        pivot_hint_count=None,
        min_correct_hint_count=min_correct,
        kpack_threshold=None,
        kpack_num_packs=num_packs,
    )


class BudgetManager:
    """JSON-backed, problem_id-keyed store of the current per-problem budget B_q.

    The store survives across training steps / restarts so the ratchet is
    monotone over the whole run. It is intentionally tiny and process-local;
    the trainer is expected to own a single instance on the driver and call
    ``update_group`` once per (problem, step) after rewards are computed, then
    ``get`` when building the next epoch's prompts.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        default_budget: int = 8,
        min_budget: int = 0,
        decrement: int = 1,
        kpack_require_successes: int = 2,
        max_budget: Optional[int] = None,
        ratchet_mode: str = "downward",
        allow_decrease: bool = True,
    ):
        self.path = path
        self.default_budget = int(default_budget)
        self.min_budget = int(min_budget)
        self.decrement = int(decrement)
        # k-pack rule: corroborating frugal successes needed to ratchet (compute_kpack_budget).
        self.kpack_require_successes = int(kpack_require_successes)
        # Ceiling on B_q -- only used by the adaptive rule's upward (raise) branch so a
        # hard problem cannot grow without bound. Defaults to default_budget.
        self.max_budget = int(max_budget) if max_budget is not None else int(default_budget)
        # Single-pack ratchet rule: "downward" (compute_downward_budget, strictly down) or
        # "adaptive" (compute_adaptive_budget: raise on zero-correct, N/2-set on majority).
        self.ratchet_mode = str(ratchet_mode or "downward").strip().lower()
        # False -> ONE-SIDED (never-decrease) guard: any rule outcome that would LOWER
        # B_q is vetoed and held at the current budget (_hold_decrease; the update's rule
        # gains a "_held" suffix). Raises pass through. Applies to EVERY rule: under
        # "adaptive" it keeps only the raise-on-zero-correct branch (pivot_set is held);
        # under "downward" / the k-pack probe (which only lower) it freezes budgets.
        self.allow_decrease = bool(allow_decrease)
        # problem_id -> current budget B_q
        self._budgets: dict[str, int] = {}
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------------ #
    # read / write the per-problem budget
    # ------------------------------------------------------------------ #
    def get(self, problem_id: str, default: Optional[int] = None) -> int:
        """Return the current budget for ``problem_id`` (default if unseen)."""
        if default is None:
            default = self.default_budget
        return int(self._budgets.get(problem_id, default))

    def set(self, problem_id: str, budget: int) -> None:
        self._budgets[problem_id] = int(budget)

    def seed(self, budgets: dict) -> None:
        """Initialize budgets from a ``{problem_id: B_q}`` mapping (e.g. the
        dataset's per-problem starting budgets) without overwriting existing
        entries."""
        for pid, b in budgets.items():
            self._budgets.setdefault(str(pid), int(b))

    # ------------------------------------------------------------------ #
    # the ratchet
    # ------------------------------------------------------------------ #
    def _hold_decrease(self, upd: BudgetUpdate, current_budget: int) -> BudgetUpdate:
        """The ``allow_decrease=False`` veto: replace a would-be DECREASE with a hold.

        Applied AFTER the rule runs, so the returned update keeps the rule's decision
        stats (n_correct, pivot/min hint counts); the rule name gains a ``"_held"``
        suffix to mark the veto in logs/metrics. Raises and holds pass through, as does
        everything when ``allow_decrease`` is True (the default).
        """
        if self.allow_decrease or upd.new_budget >= current_budget:
            return upd
        return replace(upd, new_budget=current_budget, changed=False, rule=upd.rule + "_held")

    def update_group(
        self,
        problem_id: str,
        results: Sequence[Result],
        *,
        current_budget: Optional[int] = None,
    ) -> BudgetUpdate:
        """Fold one step's rollouts for ``problem_id`` into the store.

        ``current_budget`` defaults to the stored budget (or ``default_budget``
        if the problem is unseen). Returns the ``BudgetUpdate`` and persists the
        new budget into the store (call ``save`` to flush to disk).
        """
        if current_budget is None:
            current_budget = self.get(problem_id)
        if self.ratchet_mode == "adaptive":
            upd = compute_adaptive_budget(
                current_budget,
                results,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
            )
        else:
            upd = compute_downward_budget(
                current_budget,
                results,
                min_budget=self.min_budget,
                decrement=self.decrement,
            )
        upd = self._hold_decrease(upd, current_budget)
        self._budgets[problem_id] = upd.new_budget
        return upd

    def update_group_kpack(
        self,
        problem_id: str,
        results: Sequence[Result],
        *,
        current_budget: Optional[int] = None,
        num_packs: Optional[int] = None,
        require_successes: Optional[int] = None,
    ) -> BudgetUpdate:
        """Fold a PROBED problem's pooled (over all packs) rollouts through the k-pack rule.

        Use this when the problem was probed this step (rollouts under >1 budget). ``results``
        is the pooled ``(correct, num_hints)`` across every pack; ``current_budget`` defaults
        to the stored B_q (the ceiling). Returns the ``BudgetUpdate`` and persists the new
        budget. See ``compute_kpack_budget``.
        """
        if current_budget is None:
            current_budget = self.get(problem_id)
        upd = compute_kpack_budget(
            current_budget,
            results,
            min_budget=self.min_budget,
            require_successes=(self.kpack_require_successes if require_successes is None else require_successes),
            num_packs=num_packs,
        )
        upd = self._hold_decrease(upd, current_budget)
        self._budgets[problem_id] = upd.new_budget
        return upd

    # ------------------------------------------------------------------ #
    # persistence (atomic write)
    # ------------------------------------------------------------------ #
    def load(self, path: Optional[str] = None) -> None:
        path = path or self.path
        with open(path, "r") as f:
            payload = json.load(f)
        self._budgets = {str(k): int(v) for k, v in payload.get("budgets", {}).items()}
        meta = payload.get("meta", {})
        self.default_budget = int(meta.get("default_budget", self.default_budget))
        self.min_budget = int(meta.get("min_budget", self.min_budget))
        self.decrement = int(meta.get("decrement", self.decrement))
        self.kpack_require_successes = int(meta.get("kpack_require_successes", self.kpack_require_successes))

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("BudgetManager.save needs a path")
        payload = {
            "meta": {
                "default_budget": self.default_budget,
                "min_budget": self.min_budget,
                "decrement": self.decrement,
                "kpack_require_successes": self.kpack_require_successes,
            },
            "budgets": self._budgets,
        }
        # atomic: write to a temp file in the same dir, then rename.
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def __len__(self) -> int:
        return len(self._budgets)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    def chk(name, got, want):
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
        assert ok, name

    # ---- PRIMARY (frugal-success) rule: m < current_budget -> new = m --------
    # Ungated by the correct fraction: 3/8 correct, min correct count 1 -> snap to 1.
    r = compute_downward_budget(5, [(True, 1), (True, 2), (True, 3)] + [(False, 5)] * 5)
    chk("frugal: rule", r.rule, "min_frugal")
    chk("frugal: new budget = min correct", r.new_budget, 1)
    chk("frugal: min_correct recorded", r.min_correct_hint_count, 1)
    chk("frugal: no pivot", r.pivot_hint_count, None)
    chk("frugal: changed", r.changed, True)

    # A SINGLE frugal success (1/8 correct) is enough to ratchet down.
    r = compute_downward_budget(5, [(True, 2)] + [(False, 7)] * 7)
    chk("frugal: single success ratchets", r.new_budget, 2)
    chk("frugal: single success rule", r.rule, "min_frugal")

    # Frugal rule PRE-EMPTS the (N/2)-th-smallest pivot (min 1 beats old pivot 3->2).
    r = compute_downward_budget(
        5, [(True, 1), (True, 2), (True, 2), (True, 3), (True, 4), (True, 5), (False, 5), (False, 5)]
    )
    chk("frugal: pre-empts pivot", r.new_budget, 1)
    chk("frugal: pre-empts pivot rule", r.rule, "min_frugal")

    # Frugal result still clamps to min_budget.
    r = compute_downward_budget(5, [(True, 1)] * 8, min_budget=3)
    chk("frugal: floored at min_budget", r.new_budget, 3)
    r = compute_downward_budget(5, [(True, 0)] * 8, min_budget=0)
    chk("frugal: to zero (unaided)", r.new_budget, 0)

    # ---- FALLBACK ((N/2)-th-smallest): only when m == current_budget ---------
    # Every correct rollout used the FULL budget (min == 5): pivot squeeze, C >= N/2.
    r = compute_downward_budget(5, [(True, 5)] * 6 + [(False, 5)] * 2)
    chk("fallback: rule", r.rule, "pivot")
    chk("fallback: pivot count", r.pivot_hint_count, 5)
    chk("fallback: squeeze by decrement", r.new_budget, 4)

    # m == budget but fewer than half correct -> unchanged.
    r = compute_downward_budget(5, [(True, 5)] * 2 + [(False, 5)] * 6)
    chk("fallback: too-few-correct keeps budget", r.new_budget, 5)
    chk("fallback: too-few-correct unchanged", r.changed, False)
    chk("fallback: too-few-correct rule", r.rule, "unchanged")

    # no correct rollout at all -> unchanged, min_correct = None.
    r = compute_downward_budget(5, [(False, 2)] * 8)
    chk("no-correct keeps budget", r.new_budget, 5)
    chk("no-correct min_correct None", r.min_correct_hint_count, None)
    chk("no-correct rule", r.rule, "unchanged")

    # ---- clamp / monotonicity -------------------------------------------------
    # Counts exceeding the budget (shouldn't happen; defensive): min 8 is NOT < 1,
    # so the fallback runs and clamps the squeeze down to current_budget (1).
    r = compute_downward_budget(1, [(True, 8)] * 8)
    chk("monotone-down clamp", r.new_budget, 1)

    # all correct, all used 3 hints < budget 4 -> frugal -> new = 3.
    r = compute_downward_budget(4, [(True, 3)] * 16)
    chk("uniform-3 frugal", r.new_budget, 3)

    # ---- k-pack counterfactual-probe rule ------------------------------------
    # Two packs at B=5 and B=4 pooled. Correct hint counts: 4 (full B-pack), 4, 3 (probe).
    # require=2 -> 2nd smallest correct count = 4 -> ratchet 5 -> 4.
    r = compute_kpack_budget(5, [(True, 4), (True, 4), (True, 3), (False, 5)], num_packs=2)
    chk("kpack: rule", r.rule, "kpack")
    chk("kpack: 2nd-smallest frontier", r.new_budget, 4)
    chk("kpack: threshold recorded", r.kpack_threshold, 4)
    chk("kpack: num_packs recorded", r.kpack_num_packs, 2)
    chk("kpack: changed", r.changed, True)

    # A deep probe pack solving frugally pulls multiple levels in one update.
    # Pooled correct counts 1,1,5 -> 2nd smallest = 1 -> B 5 -> 1 (multi-level jump).
    r = compute_kpack_budget(5, [(True, 1), (True, 1), (True, 5)] + [(False, 5)] * 3, num_packs=3)
    chk("kpack: multi-level jump", r.new_budget, 1)
    chk("kpack: multi-level rule", r.rule, "kpack")

    # Only ONE correct rollout (a lone fluke/leak): require=2 not met -> HOLD.
    r = compute_kpack_budget(5, [(True, 2)] + [(False, 5)] * 7, num_packs=2)
    chk("kpack: single success holds", r.new_budget, 5)
    chk("kpack: single success unchanged", r.changed, False)
    chk("kpack: single success rule", r.rule, "unchanged")
    chk("kpack: single success min recorded", r.min_correct_hint_count, 2)

    # require_successes=1 recovers aggressive single-success (most-frugal) behavior.
    r = compute_kpack_budget(5, [(True, 2)] + [(False, 5)] * 7, require_successes=1)
    chk("kpack: require=1 single success ratchets", r.new_budget, 2)

    # Every correct rollout used the full budget (no frugal evidence) -> unchanged.
    r = compute_kpack_budget(5, [(True, 5)] * 4 + [(False, 5)] * 4, num_packs=2)
    chk("kpack: all-full-budget holds", r.new_budget, 5)
    chk("kpack: all-full-budget rule", r.rule, "kpack")
    chk("kpack: all-full-budget unchanged", r.changed, False)

    # No correct rollout at all -> hold, min_correct None.
    r = compute_kpack_budget(5, [(False, 1)] * 8, num_packs=2)
    chk("kpack: no-correct holds", r.new_budget, 5)
    chk("kpack: no-correct min None", r.min_correct_hint_count, None)

    # Clamp to min_budget: pooled frontier 0 but floor 2 -> 2.
    r = compute_kpack_budget(5, [(True, 0), (True, 0)], min_budget=2, num_packs=2)
    chk("kpack: floored at min_budget", r.new_budget, 2)
    # Probe down to fully unaided: two correct at 0 hints -> ratchet to 0.
    r = compute_kpack_budget(2, [(True, 0), (True, 0), (False, 2)], min_budget=0, num_packs=3)
    chk("kpack: to zero (unaided)", r.new_budget, 0)

    # ---- manager round-trip + persistence ------------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "budget_state.json")
        bm = BudgetManager(p, default_budget=8)
        bm.update_group("probA", [(True, 3)] * 16)  # min 3 < 8 -> frugal -> 3
        chk("manager stored", bm.get("probA"), 3)
        # k-pack manager path: pooled probe successes ratchet via the kpack rule.
        bm.update_group_kpack("probK", [(True, 2), (True, 4)], current_budget=6, num_packs=2)
        chk("manager kpack stored", bm.get("probK"), 4)  # 2nd-smallest of {2,4} = 4
        bm.save()
        bm2 = BudgetManager(p, default_budget=8)
        chk("manager reloaded", bm2.get("probA"), 3)
        chk("manager kpack reloaded", bm2.get("probK"), 4)
        chk("manager unseen default", bm2.get("probB"), 8)

    # ---- allow_decrease=False: one-sided (raise-only) manager guard -----------
    bm = BudgetManager(None, default_budget=8, max_budget=8, ratchet_mode="adaptive", allow_decrease=False)
    # the adaptive RAISE branch passes through untouched: zero-correct still grants +1.
    r = bm.update_group("q", [(False, 3)] * 8, current_budget=3)
    chk("no-decrease: raise passes", r.new_budget, 4)
    chk("no-decrease: raise rule", r.rule, "raise")
    # the pivot_set DOWN branch is vetoed: held at the current budget, stats kept.
    r = bm.update_group("q", [(True, 1)] * 6 + [(False, 4)] * 2, current_budget=4)
    chk("no-decrease: pivot_set held", r.new_budget, 4)
    chk("no-decrease: held rule", r.rule, "pivot_set_held")
    chk("no-decrease: held unchanged", r.changed, False)
    chk("no-decrease: held keeps pivot stat", r.pivot_hint_count, 1)
    chk("no-decrease: store holds too", bm.get("q"), 4)
    # the k-pack probe rule (pure decrease) is vetoed under the same flag.
    r = bm.update_group_kpack("q", [(True, 2), (True, 3)], current_budget=5, num_packs=2)
    chk("no-decrease: kpack held", r.new_budget, 5)
    chk("no-decrease: kpack held rule", r.rule, "kpack_held")
    # downward mode + no-decrease == frozen budgets (the frugal snap is vetoed as well).
    bm = BudgetManager(None, default_budget=8, ratchet_mode="downward", allow_decrease=False)
    r = bm.update_group("q", [(True, 1)] + [(False, 5)] * 7, current_budget=5)
    chk("no-decrease: downward frozen", r.new_budget, 5)
    chk("no-decrease: downward rule", r.rule, "min_frugal_held")
    # default (allow_decrease=True) still ratchets down -- the guard is opt-in.
    bm = BudgetManager(None, default_budget=8, max_budget=8, ratchet_mode="adaptive")
    r = bm.update_group("q", [(True, 1)] * 6 + [(False, 4)] * 2, current_budget=4)
    chk("decrease-allowed default: pivot_set applies", r.new_budget, 1)
    chk("decrease-allowed default: rule unsuffixed", r.rule, "pivot_set")

    print("all budget_manager self-tests passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the built-in unit checks")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
