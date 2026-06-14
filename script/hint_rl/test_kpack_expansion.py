# Copyright 2026
#
# Standalone checks for the verl-free halves of the k-pack budget split:
#   * hint_prompt.rerender_messages_for_budget  (prompt re-render at a pack budget)
#   * kpack_expand.render_variant_rows          (turn select_idxs'd rows into budget packs)
#   * the N -> N*k split + floor clamp (driven through a numpy mimic of select_idxs/concat)
#   * hint_budget_callback.hprl_update_budgets  (k-pack dispatch + pooling, via a mock batch)
#
# These exercise the subtle "deepcopy-before-mutate so the SOURCE rows stay untouched"
# logic and the per-problem pack split WITHOUT importing verl. Run:
#   python test_kpack_expansion.py

from __future__ import annotations

import numpy as np

from budget_manager import get_create_budget, set_create_budget
from hint_prompt import rerender_messages_for_budget
from kpack_expand import render_variant_rows


def chk(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    assert ok, name


def chk_true(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def _make_row(pid: str, budget: int, uid: str, index: int) -> dict:
    """A row shaped like HintBudgetDataset output (the non_tensor fields we touch)."""
    tk = {
        "request_hint": {
            "create_kwargs": {
                "problem": f"problem {pid}",
                "hints": "[]",
                "ground_truth": "42",
                "budget": int(budget),
            }
        }
    }
    extra_info = {
        "problem_id": pid,
        "hprl_system_base": "You are a math assistant.",
        "hprl_user_base": f"Compute the answer to {pid}.",
        "hprl_init_budget": int(budget),
        # extra_info keeps its OWN copy of tools_kwargs (a separate dict, as the dataset does).
        "tools_kwargs": {
            "request_hint": {"create_kwargs": dict(tk["request_hint"]["create_kwargs"])}
        },
    }
    raw_prompt = rerender_messages_for_budget(
        [{"role": "user", "content": f"Compute the answer to {pid}."}],
        extra_info["hprl_system_base"],
        extra_info["hprl_user_base"],
        budget,
    )
    return {
        "extra_info": extra_info,
        "tools_kwargs": tk,
        "raw_prompt": raw_prompt,
        "uid": uid,
        "index": int(index),
    }


def _as_nt(rows: list[dict]) -> dict:
    """Pack a list of row dicts into a non_tensor dict of object arrays (collate mimic)."""
    keys = rows[0].keys()
    nt = {}
    for key in keys:
        arr = np.empty(len(rows), dtype=object)
        arr[:] = [r[key] for r in rows]
        nt[key] = arr
    return nt


def _select_idxs(nt: dict, idxs) -> dict:
    """Mimic DataProto.select_idxs for non_tensor: val[idxs] (object arrays share refs)."""
    idxs = np.array(idxs, dtype=np.int64)
    return {key: arr[idxs] for key, arr in nt.items()}


def _concat(a: dict, b: dict) -> dict:
    """Mimic DataProto.concat for non_tensor: np.concatenate per key."""
    return {key: np.concatenate([a[key], b[key]], axis=0) for key in a}


def _budget_in_prompt(messages, budget) -> bool:
    sys = next(m["content"] for m in messages if m["role"] == "system")
    usr = next(m["content"] for m in messages if m["role"] == "user")
    return (f"at most {budget} time(s)" in sys) and (f"{budget} hint call" in usr)


def test_rerender():
    print("rerender_messages_for_budget:")
    base_msgs = [{"role": "user", "content": "Solve it."}]
    msgs = rerender_messages_for_budget(base_msgs, "You are a math assistant.", "Solve it.", 3)
    chk("inserts system turn", msgs[0]["role"], "system")
    chk_true("system advertises budget 3", "at most 3 time(s)" in msgs[0]["content"])
    chk_true("user reminder budget 3", "3 hint calls remaining" in msgs[-1]["content"])
    chk_true("input list not mutated", base_msgs == [{"role": "user", "content": "Solve it."}])
    # budget 0 still renders the template (reminder reads "no ... remaining").
    z = rerender_messages_for_budget(base_msgs, "You are a math assistant.", "Solve it.", 0)
    chk_true("budget 0 keeps template", "at most 0 time(s)" in z[0]["content"])
    chk_true("budget 0 user reminder", "no hint calls remaining" in z[-1]["content"])


def test_render_variant_rows_no_source_mutation():
    print("render_variant_rows (k=3, two variants from one source -- originals untouched):")
    rows = [_make_row("pA", 5, "uid-A", 10), _make_row("pB", 5, "uid-B", 11)]
    nt = _as_nt(rows)

    # snapshot the ORIGINAL row-0 invariants we must not disturb.
    orig0_budget = get_create_budget(nt["tools_kwargs"][0], -1)
    orig0_uid = nt["uid"][0]
    orig0_index = nt["index"][0]
    orig0_prompt_b = 5

    # probe row 0 at budgets [4, 3] (k=3): select_idxs shares refs with row 0.
    var_src, var_budget = [0, 0], [4, 3]
    variants = _select_idxs(nt, var_src)
    render_variant_rows(variants, var_budget)

    # --- the source row (row 0, the normal pack) is byte-for-byte unchanged ----------
    chk("source budget untouched (tools_kwargs)", get_create_budget(nt["tools_kwargs"][0], -1), orig0_budget)
    chk("source budget untouched (extra_info)", get_create_budget(nt["extra_info"][0]["tools_kwargs"], -1), orig0_budget)
    chk("source uid untouched", nt["uid"][0], orig0_uid)
    chk("source index untouched", nt["index"][0], orig0_index)
    chk_true("source prompt untouched", _budget_in_prompt(nt["raw_prompt"][0], orig0_prompt_b))
    chk_true("source extra_info has no probe marker", "hprl_probe_budget" not in nt["extra_info"][0])

    # --- variant 0 (budget 4) -------------------------------------------------------
    chk("variant0 tool budget", get_create_budget(variants["tools_kwargs"][0], -1), 4)
    chk("variant0 extra_info budget", get_create_budget(variants["extra_info"][0]["tools_kwargs"], -1), 4)
    chk("variant0 probe marker", variants["extra_info"][0]["hprl_probe_budget"], 4)
    chk_true("variant0 prompt at 4", _budget_in_prompt(variants["raw_prompt"][0], 4))
    chk("variant0 pid preserved", variants["extra_info"][0]["problem_id"], "pA")
    chk("variant0 index unique-neg", variants["index"][0], -1)

    # --- variant 1 (budget 3), from the SAME source, rendered independently ---------
    chk("variant1 tool budget", get_create_budget(variants["tools_kwargs"][1], -1), 3)
    chk("variant1 extra_info budget", get_create_budget(variants["extra_info"][1]["tools_kwargs"], -1), 3)
    chk_true("variant1 prompt at 3", _budget_in_prompt(variants["raw_prompt"][1], 3))
    chk("variant1 index unique-neg", variants["index"][1], -2)

    # --- fresh, distinct uids (each probe pack is its own GRPO group) ----------------
    chk_true("variant uids fresh & distinct", len({variants["uid"][0], variants["uid"][1], orig0_uid}) == 3)
    # the two variants' tool dicts are independent objects (no shared mutation).
    chk_true(
        "variant tool dicts independent",
        variants["tools_kwargs"][0] is not variants["tools_kwargs"][1],
    )


def test_end_to_end_split():
    print("end-to-end split surgery (every problem -> k packs, grow N -> N*k):")
    # 4 problems at budget 5, k=3 -> each splits into 3 packs at budgets {5,4,3}.
    k, min_budget = 3, 0
    rows = [_make_row(f"p{i}", 5, f"u{i}", i) for i in range(4)]
    nt = _as_nt(rows)
    n = len(rows)

    # mimic HPRLRayPPOTrainer._hprl_expand_kpacks: k-1 variants per problem.
    var_src, var_budget = [], []
    for i in range(n):
        B = get_create_budget(nt["tools_kwargs"][i], -1)
        for j in range(1, k):
            var_src.append(i)
            var_budget.append(max(min_budget, B - j))
    variants = _select_idxs(nt, var_src)
    render_variant_rows(variants, var_budget)
    expanded = _concat(nt, variants)  # [all originals] + variants

    # batch grew uniformly N -> N*k (the repeat factor n/k is what keeps the total fixed).
    chk("split: rows N -> N*k", len(expanded["uid"]), n * k)
    # all uids distinct -> N*k GRPO groups.
    chk("split: all uids distinct", len(set(expanded["uid"].tolist())), n * k)

    # every problem spans exactly the k budgets {5,4,3}.
    by_pid = {}
    for j in range(len(expanded["uid"])):
        pid = expanded["extra_info"][j]["problem_id"]
        b = get_create_budget(expanded["tools_kwargs"][j], -1)
        by_pid.setdefault(pid, set()).add(b)
    chk_true("split: every problem spans {5,4,3}", all(bs == {5, 4, 3} for bs in by_pid.values()))
    chk("split: k packs per problem", sorted(len(bs) for bs in by_pid.values()), [k] * n)

    # clamp at the floor: a problem already at budget 1 (k=3, min 0) -> packs {1,0,0}.
    rows2 = [_make_row("low", 1, "ulow", 0)]
    nt2 = _as_nt(rows2)
    vb = [max(min_budget, 1 - j) for j in range(1, k)]  # [0, 0]
    v2 = _select_idxs(nt2, [0, 0])
    render_variant_rows(v2, vb)
    exp2 = _concat(nt2, v2)
    budgets2 = sorted(get_create_budget(exp2["tools_kwargs"][j], -1) for j in range(len(exp2["uid"])))
    chk("split: floor clamp {1,0,0}", budgets2, [0, 0, 1])


class _MockBatch:
    """Minimal stand-in: hprl_update_budgets only reads ``batch.non_tensor_batch``."""

    def __init__(self, ntb):
        self.non_tensor_batch = ntb


def _ei(pid: str, budget: int) -> dict:
    return {
        "problem_id": pid,
        "tools_kwargs": {"request_hint": {"create_kwargs": {"budget": int(budget)}}},
    }


def _obj_arr(vals):
    a = np.empty(len(vals), dtype=object)
    a[:] = list(vals)
    return a


def test_callback_ratchet():
    print("hint_budget_callback.hprl_update_budgets (k-pack dispatch + pooling + gate stats):")
    import tempfile

    from budget_manager import BudgetManager
    from hint_budget_callback import hprl_update_budgets

    # pA: PROBED -> two packs (budget 5 and 4). Pooled correct hint counts {5,4,3};
    #     require_successes=2 -> 2nd-smallest = 4 -> ratchet 5 -> 4 via the k-pack rule.
    # pB: single pack at budget 8, a frugal correct at 3 -> downward rule -> 3.
    rows_ei = [_ei("pA", 5), _ei("pA", 5), _ei("pA", 4), _ei("pA", 4), _ei("pB", 8), _ei("pB", 8)]
    acc = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    num_hints = [5, 5, 4, 3, 3, 8]
    ntb = {"extra_info": _obj_arr(rows_ei), "acc": np.array(acc), "num_hints": np.array(num_hints)}

    with tempfile.TemporaryDirectory() as d:
        import os as _os

        bm = BudgetManager(_os.path.join(d, "s.json"), default_budget=8)
        bm.set("pA", 5)
        bm.set("pB", 8)
        metrics = hprl_update_budgets(
            _MockBatch(ntb), bm, global_step=7, kpack_cfg={"enable": True}
        )

        chk("callback: pA k-pack ratchet 5->4", bm.get("pA"), 4)
        chk("callback: pB downward frugal ->3", bm.get("pB"), 3)
        chk("callback: 1 problem probed", metrics["hprl/kpack_num_probed"], 1.0)
        chk("callback: 1 probed ratcheted", metrics["hprl/kpack_num_ratcheted"], 1.0)

    # kpack disabled -> probed problem falls back to the single-pack downward rule.
    with tempfile.TemporaryDirectory() as d:
        import os as _os

        bm = BudgetManager(_os.path.join(d, "s.json"), default_budget=8)
        bm.set("pA", 5)
        bm.set("pB", 8)
        m = hprl_update_budgets(_MockBatch(ntb), bm, global_step=7, kpack_cfg={"enable": False})
        # downward rule pools all of pA too (min frugal = 3) -> 3; no kpack metrics.
        chk("callback: kpack off -> downward on pA (min 3)", bm.get("pA"), 3)
        chk_true("callback: no kpack metric when off", "hprl/kpack_num_probed" not in m)


def main():
    test_rerender()
    test_render_variant_rows_no_source_mutation()
    test_end_to_end_split()
    test_callback_ratchet()
    print("all kpack expansion tests passed.")


if __name__ == "__main__":
    main()
