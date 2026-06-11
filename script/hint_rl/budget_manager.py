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
#   Let h_1..h_C = the hint-call counts of the *correct* rollouts for the problem
#   this step (C = number of correct rollouts), and B_q = the current budget.
#
#   * If C == 0 (no correct rollout): the policy could not solve the problem at
#     the current budget -> keep B_q unchanged.
#   * Else, let m = min(h_1..h_C) be the FEWEST hints any correct rollout used:
#       - if m <  B_q: at least one rollout already succeeds with fewer hints than
#         the budget allows -> set the new budget to m (cap everyone at the best
#         rollout's hint usage next epoch).
#       - if m == B_q: even the most frugal success still needed the whole budget
#         -> squeeze by one, new budget = B_q - decrement (default decrement = 1).
#
#   The result is clamped to [min_budget, current_budget]: this is a STRICTLY
#   downward ratchet -- it never raises B_q (there is NO upward mechanism), and
#   never drops below ``min_budget`` (default 0 -> the problem can ratchet all
#   the way to fully-unaided).
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
from dataclasses import dataclass
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


@dataclass
class BudgetUpdate:
    """The outcome of one downward-ratchet evaluation (for logging / debugging)."""

    old_budget: int
    new_budget: int
    n_total: int
    n_correct: int
    changed: bool
    # The fewest hints any correct rollout used (min over correct rollouts) that
    # drove the decision -- or None when there was no correct rollout, so the
    # budget was left unchanged.
    min_correct_hint_count: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "old_budget": self.old_budget,
            "new_budget": self.new_budget,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "changed": self.changed,
            "min_correct_hint_count": self.min_correct_hint_count,
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
        decrement: how far to squeeze when the most frugal success still used the
            whole budget (the ``m == current_budget`` case; default 1).

    Returns:
        a ``BudgetUpdate`` carrying the new budget and the decision metadata.

    The rule is the one documented at the top of this file:
        C == 0             -> unchanged.
        m < current_budget -> new = m  (m = fewest hints any correct rollout used).
        m == current_budget-> new = current_budget - decrement.
        (result clamped to [min_budget, current_budget]; strictly downward.)
    """
    n_total = len(results)
    correct_hint_counts = [int(h) for ok, h in results if ok]
    n_correct = len(correct_hint_counts)

    # No correct rollout -> the policy could not solve it at this budget; hold.
    if n_correct == 0:
        return BudgetUpdate(
            old_budget=current_budget,
            new_budget=current_budget,
            n_total=n_total,
            n_correct=0,
            changed=False,
            min_correct_hint_count=None,
        )

    # Fewest hints any correct rollout used.
    m = min(correct_hint_counts)
    if m < current_budget:
        # Some success already needs fewer hints than the budget allows: cap at it.
        new_budget = m
    else:
        # m >= current_budget (normally == budget; > only from a stale higher gen
        # budget): even the most frugal success used the whole budget -> squeeze.
        new_budget = current_budget - decrement
    new_budget = max(min_budget, min(new_budget, current_budget))

    return BudgetUpdate(
        old_budget=current_budget,
        new_budget=new_budget,
        n_total=n_total,
        n_correct=n_correct,
        changed=(new_budget != current_budget),
        min_correct_hint_count=m,
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
    ):
        self.path = path
        self.default_budget = int(default_budget)
        self.min_budget = int(min_budget)
        self.decrement = int(decrement)
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
        upd = compute_downward_budget(
            current_budget,
            results,
            min_budget=self.min_budget,
            decrement=self.decrement,
        )
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

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("BudgetManager.save needs a path")
        payload = {
            "meta": {
                "default_budget": self.default_budget,
                "min_budget": self.min_budget,
                "decrement": self.decrement,
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

    # No correct rollout -> unchanged.
    r = compute_downward_budget(5, [(False, 5)] * 8)
    chk("no-correct keeps budget", r.new_budget, 5)
    chk("no-correct not changed", r.changed, False)

    # A single correct rollout below budget drives the new budget to its hint
    # count: budget 5, correct counts [2,3,4] -> min 2 < 5 -> new = 2.
    r = compute_downward_budget(
        5,
        [(True, 2), (True, 3), (True, 4), (False, 5), (False, 5)],
    )
    chk("min-below-budget min", r.min_correct_hint_count, 2)
    chk("min-below-budget new budget", r.new_budget, 2)

    # The most frugal success still used the whole budget -> squeeze by 1:
    # budget 5, correct counts all 5 -> min 5 == 5 -> new = 4.
    r = compute_downward_budget(5, [(True, 5)] * 4 + [(False, 5)] * 4)
    chk("min-equals-budget squeeze", r.new_budget, 4)

    # never increases above current (stale higher gen budget): min 8 >= 1 -> 1-1=0.
    r = compute_downward_budget(1, [(True, 8)] * 8)
    chk("monotone-down clamp", r.new_budget, 0)

    # floors at min_budget: correct with 0 hints -> min 0 < budget -> new 0.
    r = compute_downward_budget(5, [(True, 0)] * 8, min_budget=0)
    chk("min_budget floor", r.new_budget, 0)

    # all correct, all used 3 hints under budget 4 -> min 3 < 4 -> new = 3.
    r = compute_downward_budget(4, [(True, 3)] * 16)
    chk("uniform-3 new budget", r.new_budget, 3)

    # manager round-trip + persistence.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "budget_state.json")
        bm = BudgetManager(p, default_budget=8)
        bm.update_group("probA", [(True, 3)] * 16)  # min 3 < 8 -> 3
        chk("manager stored", bm.get("probA"), 3)
        bm.save()
        bm2 = BudgetManager(p, default_budget=8)
        chk("manager reloaded", bm2.get("probA"), 3)
        chk("manager unseen default", bm2.get("probB"), 8)

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
