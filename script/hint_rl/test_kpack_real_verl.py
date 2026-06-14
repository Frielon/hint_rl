# Real-verl smoke test for the k-pack SPLIT (run with the verl env python):
#   /share5/users/xutao.ma/miniconda3/envs/verl/bin/python test_kpack_real_verl.py
#
# Unlike test_kpack_expansion.py (numpy mimic, verl-free), this drives the ACTUAL
# HPRLRayPPOTrainer config split + _get_gen_batch override against a real DataProto --
# exercising verl's select_idxs / concat / repeat and the _get_gen_batch pop -- so the
# verl-specific glue is execution-tested before a cluster run. It checks the headline
# invariant: total rollouts after the repeat == train_batch * (ORIGINAL rollout.n),
# split into k packs of rollout.n/k per problem.

from __future__ import annotations

import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto

from budget_manager import BudgetManager, get_create_budget
from hint_prompt import rerender_messages_for_budget
from hprl_ray_trainer import HPRLRayPPOTrainer


def _obj(vals):
    a = np.empty(len(vals), dtype=object)
    a[:] = list(vals)
    return a


def _row(pid, budget):
    ck = {"problem": f"p {pid}", "hints": "[]", "ground_truth": "42", "budget": int(budget)}
    tk = {"request_hint": {"create_kwargs": dict(ck)}}
    ei = {
        "problem_id": pid,
        "hprl_system_base": "You are a math assistant.",
        "hprl_user_base": f"Compute {pid}.",
        "tools_kwargs": {"request_hint": {"create_kwargs": dict(ck)}},
    }
    raw = rerender_messages_for_budget(
        [{"role": "user", "content": f"Compute {pid}."}], ei["hprl_system_base"], ei["hprl_user_base"], budget
    )
    return ei, tk, raw


def build_batch(pids_budgets):
    eis, tks, raws, uids, idxs = [], [], [], [], []
    for j, (pid, b) in enumerate(pids_budgets):
        ei, tk, raw = _row(pid, b)
        eis.append(ei); tks.append(tk); raws.append(raw)
        uids.append(f"uid-{j}"); idxs.append(j)
    n = len(pids_budgets)
    batch = DataProto.from_single_dict(
        {
            "dummy_tensor": torch.zeros(n, 1, dtype=torch.uint8),
            "extra_info": _obj(eis),
            "tools_kwargs": _obj(tks),
            "raw_prompt": _obj(raws),
            "uid": _obj(uids),
            "index": _obj(idxs),
            "data_source": _obj(["math_dapo"] * n),
            "reward_model": _obj([{"ground_truth": "42"} for _ in range(n)]),
        }
    )
    batch.meta_info["temperature"] = 1.0
    return batch


def make_trainer(*, rollout_n, ppo_mini_batch, k, enable=True, scale_mini_batch=True):
    """A HPRLRayPPOTrainer shell that bypasses __init__ (we only call the split methods)."""
    t = object.__new__(HPRLRayPPOTrainer)
    t.config = OmegaConf.create(
        {
            "data": {
                "hprl": {
                    "enable": enable,
                    "min_budget": 0,
                    "tool_name": "request_hint",
                    "kpack": {"enable": enable, "k": k, "scale_mini_batch": scale_mini_batch},
                }
            },
            "actor_rollout_ref": {
                "rollout": {"n": rollout_n},
                "actor": {"ppo_mini_batch_size": ppo_mini_batch},
            },
        }
    )
    t.global_steps = 1
    return t


def chk(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    assert ok, name


def chk_true(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def pid_budgets(batch):
    # read budgets the way the ratchet callback does -- from extra_info.tools_kwargs,
    # which survives _get_gen_batch's pop (top-level tools_kwargs is moved into gen_batch).
    by = {}
    for i in range(len(batch)):
        ei = batch.non_tensor_batch["extra_info"][i]
        by.setdefault(ei["problem_id"], set()).add(get_create_budget(ei.get("tools_kwargs"), -1))
    return by


def main():
    print("config split (_hprl_apply_kpack_split_config):")
    t = make_trainer(rollout_n=32, ppo_mini_batch=32, k=2)
    bm = BudgetManager(None, default_budget=8)
    t._budget_mgr = bm
    t._hprl_apply_kpack_split_config()
    chk("rollout.n 32 -> 16 (n/k)", int(t.config.actor_rollout_ref.rollout.n), 16)
    chk("ppo_mini_batch 32 -> 64 (xk)", int(t.config.actor_rollout_ref.actor.ppo_mini_batch_size), 64)
    chk("kpack_k recorded", t._kpack_k, 2)
    # idempotent: a second call does not divide again.
    t._hprl_apply_kpack_split_config()
    chk("idempotent rollout.n", int(t.config.actor_rollout_ref.rollout.n), 16)

    print("config split: rollout.n %% k != 0 raises:")
    t_bad = make_trainer(rollout_n=30, ppo_mini_batch=32, k=4)
    t_bad._budget_mgr = BudgetManager(None, default_budget=8)
    raised = False
    try:
        t_bad._hprl_apply_kpack_split_config()
    except ValueError as e:
        raised = "divisible" in str(e)
    chk_true("raises ValueError on 30 %% 4", raised)

    print("scale_mini_batch=false leaves ppo_mini_batch:")
    t2 = make_trainer(rollout_n=32, ppo_mini_batch=32, k=2, scale_mini_batch=False)
    t2._budget_mgr = BudgetManager(None, default_budget=8)
    t2._hprl_apply_kpack_split_config()
    chk("unscaled ppo_mini_batch", int(t2.config.actor_rollout_ref.actor.ppo_mini_batch_size), 32)

    print("real-verl _get_gen_batch split (every problem -> k packs; total preserved):")
    # 4 problems, rollout.n=32, k=2 -> pack size 16. After split: 8 rows; repeat(16) -> 128.
    batch = build_batch([(f"p{i}", 5) for i in range(4)])
    n0 = len(batch)
    tr = make_trainer(rollout_n=32, ppo_mini_batch=32, k=2)
    tr._budget_mgr = BudgetManager(None, default_budget=8)
    tr._hprl_apply_kpack_split_config()  # rollout.n -> 16
    pack = int(tr.config.actor_rollout_ref.rollout.n)
    gen_batch = tr._get_gen_batch(batch)

    chk("batch grew N -> N*k", len(batch), n0 * 2)
    chk("gen_batch grew N -> N*k", len(gen_batch), n0 * 2)
    chk("all uids distinct (N*k groups)", len(set(batch.non_tensor_batch["uid"].tolist())), n0 * 2)
    by = pid_budgets(batch)
    chk_true("every problem spans {5,4}", all(bs == {5, 4} for bs in by.values()))
    # THE headline invariant: total rollouts == N * original rollout.n (32), via N*k rows x (n/k) repeat.
    total = len(gen_batch.repeat(repeat_times=pack, interleave=True))
    chk("total rollouts == N * original n", total, n0 * 32)
    # batch.repeat stays aligned with gen_batch.repeat (the union in fit()).
    chk("batch.repeat aligns", len(batch.repeat(repeat_times=pack, interleave=True)), total)
    # gen_batch carries a budget-4 forced probe pack, prompt rendered at 4.
    gtk = gen_batch.non_tensor_batch["tools_kwargs"]
    graw = gen_batch.non_tensor_batch["raw_prompt"]
    pi = [i for i in range(len(gen_batch)) if get_create_budget(gtk[i], -1) == 4]
    chk_true("gen_batch has budget-4 pack", len(pi) >= 1)
    sys_txt = next(m["content"] for m in graw[pi[0]] if m["role"] == "system")
    chk_true("budget-4 pack prompt rendered at 4", "at most 4 time(s)" in sys_txt)

    print("rollout-dump robustness (short non_tensor key -> drop, not crash):")
    # Reproduce the production failure mode: verl's _write_generations builds `gts` via
    # `for item in batch`, which stops at the FIRST non_tensor column shorter than the
    # tensor batch -> a short gts then IndexErrors. Confirm the mechanism, then confirm
    # the _log_rollout_data fix (drop length-mismatched non_tensor keys) restores it.
    M = 8
    db = DataProto.from_single_dict(
        {
            "prompts": torch.zeros(M, 3, dtype=torch.long),
            "extra_info": _obj([{"problem_id": f"p{i}"} for i in range(M)]),
            "reward_model": _obj([{"ground_truth": "42"} for _ in range(M)]),
        }
    )
    short = np.empty(M - 2, dtype=object)
    short[:] = [{} for _ in range(M - 2)]
    db.non_tensor_batch["bad_key"] = short  # a malformed, length-(M-2) column
    chk("dump: short key makes `for item in batch` stop early", len(list(db)), M - 2)
    n = len(db)
    db.non_tensor_batch = {k: v for k, v in db.non_tensor_batch.items() if len(v) == n}
    chk("dump: dropping short keys restores full iteration", len(list(db)), M)

    print("validation guard (no uid -> no split):")
    vbatch = build_batch([("v0", 5), ("v1", 5)])
    vbatch.non_tensor_batch.pop("uid")
    tv = make_trainer(rollout_n=32, ppo_mini_batch=32, k=2)
    tv._budget_mgr = BudgetManager(None, default_budget=8)
    tv._kpack_k = 2
    _ = tv._get_gen_batch(vbatch)
    chk("val batch untouched (no uid)", len(vbatch), 2)

    print("real-verl k-pack split smoke test passed.")


if __name__ == "__main__":
    main()
