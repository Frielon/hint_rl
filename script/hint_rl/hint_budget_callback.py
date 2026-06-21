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
    kpack_cfg: Optional[dict] = None,
) -> dict[str, float]:
    """Run the per-problem downward ratchet for one training step.

    Args:
        batch: the post-reward DataProto (needs non_tensor_batch keys
            ``extra_info``, ``acc``, ``num_hints``).
        budget_mgr: the driver-side BudgetManager (its JSON is what the dataset
            reads). Updated in place and saved.
        global_step: current step, for logging only.
        kpack_cfg: the ``data.hprl.kpack`` knobs. When ``kpack.enable`` is set, a
            problem whose rollouts this step span MORE THAN ONE budget level (it was
            probed -- see HPRLRayPPOTrainer._hprl_expand_kpacks) is ratcheted by the
            pooled k-pack rule (budget_manager.compute_kpack_budget) instead of the
            single-pack downward rule. Every problem's pooled outcome is recorded as
            last-step stats for the NEXT step's probe gate regardless.

    Returns:
        dict of scalar ``hprl/*`` metrics (empty if the batch lacks the keys).
    """
    ntb = batch.non_tensor_batch
    for key in ("extra_info", "acc", "num_hints"):
        if key not in ntb:
            logger.warning("hprl_update_budgets: batch missing non_tensor key %r; skipping ratchet", key)
            return {}

    kpack_cfg = kpack_cfg or {}
    kpack_enable = bool(kpack_cfg.get("enable", False))

    extra = ntb["extra_info"]
    acc = ntb["acc"]
    num_hints = ntb["num_hints"]
    n = len(acc)

    # group rollouts by problem_id, POOLING across all of a problem's packs. Track the
    # set of distinct budget levels each problem ran under: >1 level == it was probed
    # this step (the k-pack rule applies); 1 level == an ordinary single-pack group.
    groups: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    pack_budgets: dict[str, set[int]] = defaultdict(set)
    # Pack-wise view (for the GRPO-honest active-learning metrics): each ``uid`` is one
    # GRPO group == one k-pack pack (verl normalizes advantages within a uid). We track
    # correctness per uid alongside the problem_id pooling, so we can report the contrast
    # GRPO ACTUALLY sees (within a pack) -- not just the pooled-by-problem view, which
    # flags a problem active even when its correct & wrong rollouts live in DIFFERENT
    # packs (a contrast no single group ever sees). ``uid`` is assigned on the training
    # path only; absent (validation / legacy / mock batches) -> pack-wise metrics skipped.
    uid_arr = ntb.get("uid")
    pack_correct: dict[str, list[bool]] = defaultdict(list)  # uid -> rollout correctness
    pack_to_pid: dict[str, str] = {}                         # uid -> problem_id
    for i in range(n):
        ei = _to_py(extra[i]) or {}
        pid = ei.get("problem_id")
        if pid is None:
            continue
        pid = str(pid)
        correct = float(_to_py(acc[i])) >= 0.5
        hcnt = int(round(float(_to_py(num_hints[i]))))
        groups[pid].append((correct, hcnt))
        pack_budgets[pid].add(_gen_budget_from_extra(ei, budget_mgr.default_budget, tool_name))
        if uid_arr is not None:
            uid = _to_py(uid_arr[i])
            if uid is not None:
                pack_correct[str(uid)].append(correct)
                pack_to_pid[str(uid)] = pid

    if not groups:
        logger.warning("hprl_update_budgets: no problem_id found in batch; skipping ratchet")
        return {}

    # fold each problem's pooled packs through the right rule.
    updates = []
    n_probed = 0
    n_probed_ratcheted = 0
    probed_delta_sum = 0
    for pid, results in groups.items():
        budgets = pack_budgets[pid]
        num_packs = len(budgets)
        # current budget == the ceiling the ratchet lowers from == MAX over the packs
        # (pack 0's budget B; the forced probe packs ran below it). Monotone-down guard:
        # never ratchet up from an already-lower stored B_q.
        ceiling = max(budgets) if budgets else budget_mgr.default_budget
        cur = min(ceiling, budget_mgr.get(pid, ceiling))
        # >1 distinct budget == this problem was split into probe packs this step -> use the
        # pooled k-pack rule; otherwise (all packs at the same budget, e.g. B already at the
        # floor) fall back to the single-pack downward rule.
        if kpack_enable and num_packs > 1:
            upd = budget_mgr.update_group_kpack(pid, results, current_budget=cur, num_packs=num_packs)
            n_probed += 1
            if upd.changed:
                n_probed_ratcheted += 1
            probed_delta_sum += upd.old_budget - upd.new_budget
        else:
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

    # Effective anti-suppression bonus actually paid out, averaged over rollouts that
    # applied a hint. The bonus is effort-gated (max(0, hint_call_reward - shape_penalty)),
    # so this DROPS toward 0 as front-loading rises -- the direct readout of whether the
    # gate is clawing the bonus back from shallow hinted failures. Diluted by correct-
    # hinted rollouts (bonus is incorrect-only -> they contribute 0); best-effort.
    bonus_arr = ntb.get("hint_call_bonus")
    if bonus_arr is not None:
        bonuses = [float(_to_py(bonus_arr[i]) or 0.0) for i in range(n)]
        hinted_bonus = [
            b for i, b in enumerate(bonuses)
            if int(round(float(_to_py(num_hints[i])))) > 0
        ]
        shape_metrics["hprl/hint_call_bonus_mean_hinted"] = (
            float(sum(hinted_bonus) / len(hinted_bonus)) if hinted_bonus else 0.0
        )

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

    # -------- pack-wise active learning (the GRPO-honest views) -------------------
    # active_learning_frac above POOLS a problem's packs, so it flags a problem active
    # whenever some correct AND some wrong rollout exist anywhere in the problem -- even
    # when they sit in DIFFERENT packs (e.g. one pack all-correct, another all-wrong).
    # GRPO never sees that contrast: advantages are normalized WITHIN each uid (pack).
    # Two honest views over the per-uid (pack) groups:
    #   * _packwise: fraction of PACKS whose own rollouts span correct & wrong (0<C<N).
    #   * _anypack:  fraction of PROBLEMS with AT LEAST ONE such internally-mixed pack
    #                -- the real per-problem learnability (>= it iff a pack truly mixes).
    # _anypack <= active_learning_frac always (an internally-mixed pack also mixes the
    # pool, but a mixed pool can come from uniform packs). With k-pack off there is one
    # uid per problem, so both collapse onto active_learning_frac. Skipped if no uid.
    if pack_correct:
        n_packs = len(pack_correct)
        active_pack_uids = [u for u, oks in pack_correct.items() if any(oks) and not all(oks)]
        n_active_packs = len(active_pack_uids)
        n_active_anypack = len({pack_to_pid[u] for u in active_pack_uids})
        metrics["hprl/active_learning_frac_packwise"] = float(n_active_packs / n_packs)
        metrics["hprl/num_active_packs"] = float(n_active_packs)
        metrics["hprl/num_packs"] = float(n_packs)
        metrics["hprl/active_learning_frac_anypack"] = float(n_active_anypack / n_problems)
        metrics["hprl/num_active_learning_anypack"] = float(n_active_anypack)

    # -------- k-pack probe ratchet: how many probed problems actually moved -------
    # n_probed = problems whose rollouts spanned >1 budget this step (the k-pack rule
    # ran). num_kpack_ratcheted / mean delta show whether the counterfactual probe is
    # finding real frontier (budget descends) vs holding (<2 corroborating successes).
    if kpack_enable:
        metrics["hprl/kpack_num_probed"] = float(n_probed)
        metrics["hprl/kpack_num_ratcheted"] = float(n_probed_ratcheted)
        metrics["hprl/kpack_ratcheted_frac"] = float(n_probed_ratcheted / n_probed) if n_probed else 0.0
        metrics["hprl/kpack_budget_delta_mean"] = float(probed_delta_sum / n_probed) if n_probed else 0.0

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
    # -------- protocol-violation rate: over-budget hint calls ---------------
    # Fraction of rollouts that emitted <hint_call/> after their budget was spent
    # and were floored to budget_exceeded_reward (acc=0, answer ungraded). A rising
    # value means the policy is still calling for help it cannot have.
    be_arr = ntb.get("hint_budget_exceeded")
    if be_arr is not None:
        be_total = sum(int(round(float(_to_py(be_arr[i]) or 0))) for i in range(len(be_arr)))
        metrics["hprl/hint_budget_exceeded"] = float(be_total)
        metrics["hprl/hint_budget_exceeded_frac"] = float(be_total / n) if n else 0.0

    # -------- out-of-hints rate: pool-exhausted hint calls ------------------
    # Hint calls that found an EMPTY candidate pool (every hint already surfaced) and
    # got the no-hint no-op instead of a selector round-trip. Kept DISTINCT from
    # hint_call_failed (a selector outage): a rising value here is benign and EXPECTED
    # under cumulative step-exclude once a rollout has revealed its last/highest step.
    # The frac is over ALL hint-call attempts (applied + failed + exhausted).
    exh_arr = ntb.get("hint_pool_exhausted")
    if exh_arr is not None:
        exhausted_total = sum(int(round(float(_to_py(exh_arr[i]) or 0))) for i in range(len(exh_arr)))
        attempts_total = call_total + exhausted_total
        metrics["hprl/hint_pool_exhausted"] = float(exhausted_total)
        metrics["hprl/hint_pool_exhausted_frac"] = (
            float(exhausted_total / attempts_total) if attempts_total else 0.0
        )

    # -------- selector latency (the RELIABLE timer; verl-native one is dropped) ----
    # Per-rollout selector time + call count come through extra_fields -> reward ->
    # ntb (the verl timing_s/agent_loop/hint_select/* is a flat 0.0 because
    # AgentLoopMetrics has no field for it). We report: total wall-equivalent selector
    # seconds this step, mean per rollout, mean per ACTUAL call, and the worst rollout.
    time_arr = ntb.get("hint_select_time")
    calls_arr = ntb.get("hint_select_calls")
    if time_arr is not None:
        times = [float(_to_py(time_arr[i]) or 0.0) for i in range(len(time_arr))]
        calls = (
            [int(round(float(_to_py(calls_arr[i]) or 0))) for i in range(len(calls_arr))]
            if calls_arr is not None
            else [0] * len(times)
        )
        total_calls = sum(calls)
        metrics["hprl/hint_select_time_sum"] = float(sum(times))
        metrics["hprl/hint_select_time_mean"] = float(sum(times) / len(times)) if times else 0.0
        metrics["hprl/hint_select_time_max"] = float(max(times)) if times else 0.0
        # mean latency per individual selector call -- the round-trip cost, undiluted
        # by hint-free rollouts (which contribute 0 time and 0 calls).
        metrics["hprl/hint_select_time_per_call"] = float(sum(times) / total_calls) if total_calls else 0.0

    # -------- box-then-call rate: did the hint-calling turn already box an answer? --
    # Per CALL: of all hint calls, the fraction emitted by a turn that already had a
    # \boxed{} (the front-loading / box-then-call pattern). Per ROLLOUT: the fraction
    # of rollouts (all, and of hint-USING ones) that did this at least once.
    tot_arr = ntb.get("hint_calls_total")
    box_arr = ntb.get("hint_calls_with_box")
    if tot_arr is not None and box_arr is not None:
        totals = [int(round(float(_to_py(tot_arr[i]) or 0))) for i in range(len(tot_arr))]
        boxes = [int(round(float(_to_py(box_arr[i]) or 0))) for i in range(len(box_arr))]
        sum_calls = sum(totals)
        sum_box = sum(boxes)
        n_hinting = sum(1 for t in totals if t > 0)
        n_box_roll = sum(1 for b in boxes if b > 0)
        # per HINT CALL
        metrics["hprl/hint_call_with_box_frac"] = float(sum_box / sum_calls) if sum_calls else 0.0
        # per ROLLOUT (over all rollouts, and over hint-using rollouts)
        metrics["hprl/hint_call_with_box_rollout_frac"] = float(n_box_roll / n) if n else 0.0
        metrics["hprl/hint_call_with_box_frac_of_hinting"] = (
            float(n_box_roll / n_hinting) if n_hinting else 0.0
        )

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
