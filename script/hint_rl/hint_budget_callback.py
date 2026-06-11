# Copyright 2026
#
# HintBudgetCallback -- the update side of the HPRL dynamic-budget ratchet.
#
# ``hprl_update_budgets`` is called once per training step by the HPRL trainer
# subclass (hprl_ray_trainer.HPRLRayPPOTrainer._update_actor) on the fully
# populated post-reward batch. It:
#
#   1. Groups the batch's rollouts by ``extra_info.problem_id`` -- with GRPO each
#      problem contributes exactly ``rollout.n`` rollouts (the group of N).
#   2. Builds ``(correct, num_hints)`` per rollout from the reward keys
#      ``acc`` / ``num_hints`` that hint_reward.compute_score emits.
#   3. Reads the budget those rollouts ACTUALLY ran under from
#      ``extra_info.tools_kwargs.request_hint.create_kwargs.budget`` (what the
#      dataset injected this epoch) -- not the live store, so the ratchet
#      evaluates against the true baseline.
#   4. Folds each group through ``BudgetManager.update_group`` (the downward rule
#      in budget_manager.compute_downward_budget) and atomically persists the new
#      table to the budget-state JSON the dataset reads.
#
# It returns a dict of scalar wandb metrics; the trainer merges them into the
# step's actor metrics. It never raises into the training loop -- the caller
# guards it -- but it is written defensively regardless.

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Optional

from budget_manager import BudgetManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_py(v):
    """Normalize numpy 0-d object arrays / scalars to plain Python objects."""
    if v is None:
        return None
    if hasattr(v, "item") and getattr(v, "shape", None) == ():
        return v.item()
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:  # noqa: BLE001
            return v
    return v


def _gen_budget_from_extra(extra_info: dict, default: int, tool_name: str = "request_hint") -> int:
    """Budget the rollouts ran under, read from extra_info.tools_kwargs."""
    tk = extra_info.get("tools_kwargs")
    if isinstance(tk, dict):
        tool = tk.get(tool_name)
        if isinstance(tool, dict):
            ck = tool.get("create_kwargs")
            if isinstance(ck, dict) and ck.get("budget") is not None:
                return int(ck["budget"])
    # fallback: the initial budget the data prep stored, else the manager default.
    if extra_info.get("hprl_init_budget") is not None:
        return int(extra_info["hprl_init_budget"])
    return int(default)


def hprl_update_budgets(
    batch,
    budget_mgr: BudgetManager,
    *,
    global_step: Optional[int] = None,
    tool_name: str = "request_hint",
) -> dict[str, float]:
    """Run the per-problem downward ratchet for one training step.

    Args:
        batch: the post-reward DataProto (needs non_tensor_batch keys
            ``extra_info``, ``acc``, ``num_hints``).
        budget_mgr: the driver-side BudgetManager (its JSON is what the dataset
            reads). Updated in place and saved.
        global_step: current step, for logging only.

    Returns:
        dict of scalar ``hprl/*`` metrics (empty if the batch lacks the keys).
    """
    ntb = batch.non_tensor_batch
    for key in ("extra_info", "acc", "num_hints"):
        if key not in ntb:
            logger.warning("hprl_update_budgets: batch missing non_tensor key %r; skipping ratchet", key)
            return {}

    extra = ntb["extra_info"]
    acc = ntb["acc"]
    num_hints = ntb["num_hints"]
    n = len(acc)

    # group rollouts by problem_id
    groups: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    gen_budget: dict[str, int] = {}
    for i in range(n):
        ei = _to_py(extra[i]) or {}
        pid = ei.get("problem_id")
        if pid is None:
            continue
        pid = str(pid)
        correct = float(_to_py(acc[i])) >= 0.5
        hcnt = int(round(float(_to_py(num_hints[i]))))
        groups[pid].append((correct, hcnt))
        if pid not in gen_budget:
            gen_budget[pid] = _gen_budget_from_extra(ei, budget_mgr.default_budget, tool_name)

    if not groups:
        logger.warning("hprl_update_budgets: no problem_id found in batch; skipping ratchet")
        return {}

    # fold each group through the downward rule
    updates = []
    for pid, results in groups.items():
        # monotone-down guard: never ratchet up from an already-lower stored B_q,
        # even if this batch ran under a (stale, higher) budget.
        cur = min(gen_budget[pid], budget_mgr.get(pid, gen_budget[pid]))
        upd = budget_mgr.update_group(pid, results, current_budget=cur)
        updates.append(upd)

    budget_mgr.save()

    # -------- metrics --------
    new_budgets = [u.new_budget for u in updates]
    n_changed = sum(1 for u in updates if u.changed)
    correct_fracs = [(u.n_correct / u.n_total) for u in updates if u.n_total]
    deltas = [(u.old_budget - u.new_budget) for u in updates]
    n_problems = len(updates)

    # "Active learning" problems: prompt-groups with BOTH a correct AND an
    # incorrect rollout (0 < C < N). Only these carry a correct-vs-wrong contrast
    # -- the groups the policy actually learns a correctness signal from; an
    # all-correct or all-wrong group has no such spread. (Under HPRL the reward is
    # continuous, so an all-correct group can still have non-zero GRPO variance via
    # differing hint/shape penalties; this is specifically the correct/incorrect-
    # mixed fraction.)
    n_active = sum(
        1
        for results in groups.values()
        if any(ok for ok, _ in results) and any(not ok for ok, _ in results)
    )

    # -------- effort-shaping signal (TRAINING scalar) -----------------------
    # hint_reward.compute_score returns hint_shape_sum (coeff-free shortfall sum)
    # and hint_shape_penalty (coeff-scaled, what's actually subtracted) per
    # rollout, but verl only auto-aggregates reward_extra_info to wandb on the
    # VALIDATION path -- where it's always 0 (val is single-turn / no hints). We
    # surface it as a TRAINING scalar here (best-effort; absent in legacy runs).
    # `*_hinted` averages only over rollouts that actually applied a hint -- the
    # meaningful "how shallow were the pre-hint turns" signal; the plain mean is
    # diluted by hint-free rollouts (which contribute 0).
    shape_sum_arr = ntb.get("hint_shape_sum")
    shape_pen_arr = ntb.get("hint_shape_penalty")
    shape_metrics = {}
    if shape_sum_arr is not None:
        sums = [float(_to_py(shape_sum_arr[i]) or 0.0) for i in range(n)]
        hinted = [s for i, s in enumerate(sums) if int(round(float(_to_py(num_hints[i])))) > 0]
        shape_metrics["hprl/hint_shape_sum_mean"] = float(sum(sums) / n) if n else 0.0
        shape_metrics["hprl/hint_shape_sum_mean_hinted"] = (
            float(sum(hinted) / len(hinted)) if hinted else 0.0
        )
    if shape_pen_arr is not None:
        pens = [float(_to_py(shape_pen_arr[i]) or 0.0) for i in range(n)]
        shape_metrics["hprl/hint_shape_penalty_mean"] = float(sum(pens) / n) if n else 0.0

    metrics = {
        "hprl/n_problems": float(n_problems),
        "hprl/budget_mean": float(sum(new_budgets) / n_problems),
        "hprl/budget_min": float(min(new_budgets)),
        "hprl/budget_max": float(max(new_budgets)),
        "hprl/budget_delta_mean": float(sum(deltas) / n_problems),
        "hprl/num_ratcheted": float(n_changed),
        "hprl/frac_ratcheted": float(n_changed / n_problems),
        "hprl/correct_frac_mean": float(sum(correct_fracs) / len(correct_fracs)) if correct_fracs else 0.0,
        # fraction of problems with mixed correct/incorrect rollouts this step.
        "hprl/active_learning_frac": float(n_active / n_problems),
        "hprl/num_active_learning": float(n_active),
        **shape_metrics,
    }

    # -------- selector health: applied vs failed hint calls -----------------
    # Total hints actually applied this step vs total hint calls that hit the
    # selector no-op fallback. hint_call_failed_frac near 1.0 == a selector
    # outage (the failure mode where hints silently degrade to no-ops). num_hints
    # is already validated present above; hint_call_failed is best-effort.
    applied_total = sum(int(round(float(_to_py(num_hints[i])))) for i in range(n))
    failed_arr = ntb.get("hint_call_failed")
    failed_total = (
        sum(int(round(float(_to_py(failed_arr[i]) or 0))) for i in range(len(failed_arr)))
        if failed_arr is not None
        else 0
    )
    call_total = applied_total + failed_total
    metrics["hprl/hint_calls_applied"] = float(applied_total)
    metrics["hprl/hint_calls_failed"] = float(failed_total)
    metrics["hprl/hint_call_failed_frac"] = float(failed_total / call_total) if call_total else 0.0
    if failed_total and applied_total == 0:
        logger.warning(
            "[HPRL step=%s] SELECTOR OUTAGE: %d hint call(s) ALL failed (0 applied) -- "
            "hints are degrading to no-ops; check selector reachability.",
            global_step,
            failed_total,
        )
    logger.warning(
        "[HPRL step=%s] problems=%d ratcheted=%d budget(mean/min/max)=%.2f/%d/%d",
        global_step,
        n_problems,
        n_changed,
        metrics["hprl/budget_mean"],
        int(metrics["hprl/budget_min"]),
        int(metrics["hprl/budget_max"]),
    )
    return metrics
