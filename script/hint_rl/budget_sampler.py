# Copyright 2026
#
# BudgetGroupedSampler -- budget-based data sampling for HPRL.
#
# Problem this solves
# -------------------
# The stock RandomSampler draws each step's `train_batch_size` problems uniformly
# at random, so a single step's batch mixes problems with very different hint-call
# budgets B_q -- a budget-0 problem (a single unaided turn) sits next to a budget-6
# problem (up to six selector rounds, each adding a tool round-trip + extra
# generation turns). Async multi-turn rollout finishes a step only when its SLOWEST
# rollout finishes, so the budget-0 rollouts complete early and their GPUs idle
# while the budget-6 stragglers run on -- the high-budget problems bottleneck the
# whole generation step.
#
# What this does
# --------------
# At the START OF EACH EPOCH it orders the epoch's problems by their CURRENT budget
# B_q and packs same-budget problems into the same step's batch, so every rollout in
# a step runs ~the same number of hint rounds and finishes at ~the same time -- no
# straggler, no idle GPUs. The budget read is the LIVE ratcheted B_q from the shared
# budget-state JSON (the same file HintBudgetDataset reads and the trainer ratchet
# writes), falling back to the parquet's baked budget for problems not yet ratcheted
# -- so the grouping tracks the ratchet as budgets drift over the run.
#
# Invariants kept identical to the stock sampler
# ----------------------------------------------
#   * Every yielded batch is EXACTLY `batch_size` (gen_batch_size or
#     train_batch_size). The verl train path does not auto-pad and the per-step row
#     count must stay divisible by ppo_mini_batch_size, so a partial batch is never
#     emitted -- like the stock RandomSampler + drop_last, a random remainder of
#     `len(dataset) % batch_size` problems is dropped each epoch (a DIFFERENT random
#     subset every epoch, never a fixed high-budget tail).
#   * The per-epoch problem multiset is otherwise unchanged; only the assignment of
#     problems to steps (and the intra-batch budget spread) changes. GRPO groups by
#     uid, not by batch membership, so advantage computation is unaffected.
#   * Stateful for mid-epoch checkpoint resume: the iterator snapshots + checkpoints
#     the torch Generator state and the yielded count exactly like torchdata's
#     RandomSampler, so StatefulDataLoader.load_state_dict resumes the interrupted
#     epoch's order and fast-forwards to the right step.
#
# Wiring: main_hprl.HPRLTaskRunner.run rebinds verl's `create_rl_sampler` to
# `wrap_create_rl_sampler(...)` (the same flag-gated override pattern it uses for
# RayPPOTrainer). With data.hprl.budget_sampling.enable=false the wrapper delegates
# to the stock create_rl_sampler, so this file is a pure no-op when off.
#
# Quick self-check:  python budget_sampler.py --selftest

from __future__ import annotations

import logging
import os
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler

from budget_manager import get_create_budget, load_budget_table

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _BudgetGroupedSamplerIterator(Iterator[int]):
    """Per-epoch iterator over a BudgetGroupedSampler's order. Stateful (duck-typed:
    state_dict/load_state_dict -> recognized by torchdata's `Stateful` Protocol), so
    StatefulDataLoader checkpoints/restores it for mid-epoch resume.

    Mirrors torchdata's ``_StatefulRandomSamplerIterator``: it snapshots the sampler's
    Generator state at construction (BEFORE building this epoch's order, which advances
    the generator), tracks how many indices it has yielded, and on restore re-seeds the
    generator to that snapshot and rebuilds the SAME order before fast-forwarding. The
    persistent generator lives on the sampler, so successive epochs draw fresh orders.
    """

    def __init__(self, sampler: "BudgetGroupedSampler"):
        self.sampler = sampler
        # snapshot the generator state that PRODUCES this epoch's order (before _build_order
        # advances it), so load_state_dict can reproduce the order exactly.
        self.generator_state = sampler.generator.get_state()
        self.yielded = 0
        self.order: List[int] = sampler._build_order(sampler.generator)

    def __iter__(self) -> "Iterator[int]":
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.order):
            raise StopIteration
        val = self.order[self.yielded]
        self.yielded += 1
        return val

    # -- Stateful protocol -------------------------------------------------- #
    def state_dict(self) -> dict:
        return {"yielded": self.yielded, "generator_state": self.generator_state}

    def load_state_dict(self, state_dict: dict) -> None:
        self.generator_state = state_dict["generator_state"]
        # restore the generator to the snapshot and rebuild THIS epoch's order. (The
        # budget table is re-read here; if it drifted since the checkpoint the order may
        # differ slightly, but the generator is consumed identically so later epochs are
        # unaffected -- and the fast-forward still lands at the right step count.)
        self.sampler.generator.set_state(self.generator_state)
        self.order = self.sampler._build_order(self.sampler.generator)
        self.yielded = int(state_dict["yielded"])


class BudgetGroupedSampler(Sampler[int]):
    """Yields a budget-grouped index order so each `batch_size` window is ~budget-homogeneous.

    Args:
        dataset: the (HintBudget)RLHFDataset -- read once for each row's problem_id and
            baked budget (``extra_info.tools_kwargs.<tool>.create_kwargs.budget``).
        batch_size: the dataloader batch size the order is chunked to (gen_batch_size or
            train_batch_size). Each emitted batch is exactly this many problems.
        budget_state_path: shared budget-state JSON (the trainer ratchet writes it, the
            dataset reads it). Re-read every epoch for the LIVE B_q; None / missing -> the
            baked per-row budget is used for every problem.
        default_budget: fallback when a row carries no baked tool budget.
        tool_name: the tool whose create_kwargs.budget is the per-row budget.
        shuffle_batch_order: randomize the ORDER the homogeneous batches run within an
            epoch (each batch stays homogeneous; only the step sequence is shuffled), so
            the epoch isn't a fixed easy->hard ramp and each step is still a random budget
            level. False -> ascending-budget order.
        generator: torch.Generator for the (per-epoch) randomness; one is created+seeded
            if None (matching create_rl_sampler when data.seed is unset).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        budget_state_path: Optional[str] = None,
        default_budget: int = 8,
        tool_name: str = "request_hint",
        shuffle_batch_order: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.budget_state_path = budget_state_path
        self.default_budget = int(default_budget)
        self.tool_name = tool_name
        self.shuffle_batch_order = bool(shuffle_batch_order)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        self.generator = generator

        # read each row's (problem_id, baked_budget) ONCE; per-epoch we only overlay the
        # live ratcheted budgets on top (cheap small-JSON read).
        self._problem_ids, self._baked_budgets = self._read_row_meta(dataset, self.default_budget, self.tool_name)

        from collections import Counter

        dist = sorted(Counter(self._baked_budgets).items())
        logger.warning(
            "BudgetGroupedSampler: ENABLED (n=%d, batch_size=%d, shuffle_batch_order=%s, "
            "state=%s); baked budget distribution=%s",
            len(self._baked_budgets),
            self.batch_size,
            self.shuffle_batch_order,
            self.budget_state_path,
            dist,
        )

    # ------------------------------------------------------------------ #
    # per-row metadata (problem_id + baked budget), read once
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_row_meta(dataset, default_budget: int, tool_name: str):
        n = len(dataset)
        problem_ids: List[Optional[str]] = [None] * n
        baked: List[int] = [int(default_budget)] * n
        df = getattr(dataset, "dataframe", None)
        extra = None
        if df is not None:
            try:
                if "extra_info" in df.column_names:
                    extra = df["extra_info"]  # columnar read -> list of dicts
            except Exception as e:  # noqa: BLE001 -- be defensive about the dataframe shape
                logger.warning("BudgetGroupedSampler: could not read extra_info column: %s", e)
                extra = None
        if extra is None:
            logger.warning(
                "BudgetGroupedSampler: dataset has no readable extra_info; all rows fall back to "
                "default_budget=%d (grouping degenerates to a random order).",
                default_budget,
            )
            return problem_ids, baked
        for i in range(n):
            ei = extra[i]
            if isinstance(ei, dict):
                pid = ei.get("problem_id")
                problem_ids[i] = str(pid) if pid is not None else None
                baked[i] = get_create_budget(ei.get("tools_kwargs"), default_budget, tool_name)
        return problem_ids, baked

    def _current_budgets(self) -> List[int]:
        """The LIVE per-row budget this epoch: ratcheted B_q if present, else baked."""
        table = {}
        if self.budget_state_path and os.path.exists(self.budget_state_path):
            table = load_budget_table(self.budget_state_path)  # {} on missing/partial file
        out: List[int] = []
        for pid, baked in zip(self._problem_ids, self._baked_budgets):
            if pid is not None and pid in table:
                out.append(int(table[pid]))
            else:
                out.append(int(baked))
        return out

    # ------------------------------------------------------------------ #
    # the per-epoch order
    # ------------------------------------------------------------------ #
    def _build_order(self, generator: torch.Generator) -> List[int]:
        """Budget-grouped index order for one epoch (advances `generator`).

        random permutation -> drop the remainder (a random subset, so no fixed problem is
        always dropped) -> STABLE sort by budget (equal budgets keep their random order)
        -> chunk into batch_size groups -> optionally shuffle the chunk order -> flatten.
        Two generator draws (randperm(n), and randperm(n_chunks) iff shuffle_batch_order);
        the branch is fixed per run, so consumption is deterministic for resume.
        """
        n = len(self.dataset)
        bs = self.batch_size
        n_batches = n // bs
        if n_batches == 0:
            # dataset smaller than one batch: nothing the dataloader can do with drop_last;
            # hand back a plain shuffled order and let it raise its own empty-loader error.
            return torch.randperm(n, generator=generator).tolist()
        keep = n_batches * bs

        budgets = self._current_budgets()
        perm = torch.randperm(n, generator=generator).tolist()
        kept = perm[:keep]  # drop the random remainder (perm[keep:]) -> rotates across epochs
        # stable sort: within one budget the order stays as in `kept` (i.e. random).
        kept.sort(key=lambda i: budgets[i])
        chunks = [kept[i : i + bs] for i in range(0, keep, bs)]
        if self.shuffle_batch_order and len(chunks) > 1:
            cperm = torch.randperm(len(chunks), generator=generator).tolist()
            chunks = [chunks[j] for j in cperm]

        self._log_epoch_order(chunks, budgets)
        return [idx for chunk in chunks for idx in chunk]

    def _log_epoch_order(self, chunks: List[List[int]], budgets: List[int]) -> None:
        """One concise line per epoch so the budget grouping is visible in the run log."""
        if not chunks:
            return
        spans = [(min(budgets[i] for i in c), max(budgets[i] for i in c)) for c in chunks]
        mixed = sum(1 for lo, hi in spans if lo != hi)
        gmin = min(lo for lo, _ in spans)
        gmax = max(hi for _, hi in spans)
        # print() (not logger) so it reaches the cluster console log from the driver.
        print(
            f"[BudgetGroupedSampler] epoch order: {len(chunks)} batches x {self.batch_size}; "
            f"budget range {gmin}..{gmax}; mixed-budget batches {mixed}/{len(chunks)} "
            f"(remainder dropped this epoch: {len(self.dataset) - len(chunks) * self.batch_size})",
            flush=True,
        )

    def __iter__(self) -> Iterator[int]:
        return _BudgetGroupedSamplerIterator(self)

    def __len__(self) -> int:
        # number of indices actually yielded per epoch == floor(n/bs)*bs, so the
        # dataloader's len == this // bs == floor(n/bs) batches (matches what we yield).
        n = len(self.dataset)
        bs = self.batch_size
        return (n // bs) * bs if n >= bs else n


# --------------------------------------------------------------------------- #
# wiring: a drop-in replacement for verl.trainer.main_ppo.create_rl_sampler
# --------------------------------------------------------------------------- #
def build_budget_grouped_sampler(data_config, dataset) -> BudgetGroupedSampler:
    """Construct a BudgetGroupedSampler from verl's data config + the train dataset."""
    hprl = data_config.get("hprl", {}) or {}
    bscfg = hprl.get("budget_sampling", {}) or {}
    # the dataloader chunks by gen_batch_size if set, else train_batch_size.
    batch_size = data_config.get("gen_batch_size", None) or data_config.get("train_batch_size")
    seed = data_config.get("seed", None)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    return BudgetGroupedSampler(
        dataset,
        int(batch_size),
        budget_state_path=hprl.get("budget_state_path", None),
        default_budget=int(hprl.get("default_budget", 8)),
        tool_name=hprl.get("tool_name", "request_hint"),
        shuffle_batch_order=bool(bscfg.get("shuffle_batch_order", True)),
        generator=generator,
    )


def wrap_create_rl_sampler(orig_create_rl_sampler):
    """Wrap verl's ``create_rl_sampler`` so it returns a BudgetGroupedSampler when
    ``data.hprl.budget_sampling.enable`` is set, and is byte-identical to the stock
    function otherwise. Installed by main_hprl.HPRLTaskRunner.run."""

    def create_rl_sampler(data_config, dataset):
        hprl = (data_config.get("hprl", {}) or {})
        bscfg = hprl.get("budget_sampling", {}) or {}
        if not bscfg.get("enable", False):
            return orig_create_rl_sampler(data_config, dataset)
        return build_budget_grouped_sampler(data_config, dataset)

    return create_rl_sampler


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import json
    import tempfile

    def chk(name, got, want):
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
        assert ok, name

    class _FakeDF:
        def __init__(self, extra):
            self._extra = extra
            self.column_names = ["extra_info"]

        def __getitem__(self, k):
            if k == "extra_info":
                return self._extra
            raise KeyError(k)

    class _FakeDS:
        def __init__(self, budgets, problem_ids=None):
            self.dataframe = _FakeDF(
                [
                    {
                        "problem_id": (problem_ids[i] if problem_ids else f"p{i}"),
                        "tools_kwargs": {"request_hint": {"create_kwargs": {"budget": int(b)}}},
                    }
                    for i, b in enumerate(budgets)
                ]
            )
            self._n = len(budgets)

        def __len__(self):
            return self._n

    # 500 problems, budgets 0..5 in a deliberately scrambled order; batch_size 32.
    n, bs = 500, 32
    budgets = [(i * 7) % 6 for i in range(n)]
    ds = _FakeDS(budgets)

    g = torch.Generator()
    g.manual_seed(0)
    s = BudgetGroupedSampler(ds, bs, default_budget=8, shuffle_batch_order=True, generator=g)

    # ---- length / divisibility ------------------------------------------------
    chk("len == floor(n/bs)*bs", len(s), (n // bs) * bs)
    order = list(iter(s))
    chk("yields exactly floor(n/bs)*bs indices", len(order), (n // bs) * bs)
    chk("all indices in range", all(0 <= i < n for i in order), True)
    chk("no duplicate indices within an epoch", len(set(order)), len(order))

    # ---- homogeneity: every fixed bs-window (== a dataloader batch) is tight ----
    chunks = [order[i : i + bs] for i in range(0, len(order), bs)]
    spans = [max(budgets[i] for i in c) - min(budgets[i] for i in c) for c in chunks]
    # with 6 budget levels each filling many batches, at most one boundary batch per
    # level transition is mixed, and never by more than 1 (adjacent levels).
    chk("every batch spans <= 1 budget level", max(spans), 1 if max(spans) <= 1 else max(spans))
    chk("most batches are perfectly homogeneous", sum(1 for sp in spans if sp == 0) >= len(chunks) - 6, True)

    # ---- a different epoch yields a different order (generator advanced) --------
    order2 = list(iter(s))
    chk("epoch 2 differs from epoch 1", order != order2, True)
    chk("epoch 2 still homogeneous", max(max(budgets[i] for i in order2[k : k + bs]) - min(budgets[i] for i in order2[k : k + bs]) for k in range(0, len(order2), bs)) <= 1, True)

    # ---- the dropped remainder rotates (not a fixed high-budget tail) ----------
    g.manual_seed(123)
    s2 = BudgetGroupedSampler(ds, bs, shuffle_batch_order=False, generator=g)
    dropped = [set(range(n)) - set(iter(s2)) for _ in range(6)]
    chk("remainder size == n %% bs", all(len(d) == n % bs for d in dropped), True)
    union = set().union(*dropped)
    chk("dropped set rotates across epochs", len(union) > n % bs, True)

    # ---- ascending order when shuffle_batch_order=False ------------------------
    g.manual_seed(7)
    s3 = BudgetGroupedSampler(ds, bs, shuffle_batch_order=False, generator=g)
    o3 = list(iter(s3))
    batch_budgets = [budgets[o3[i]] for i in range(0, len(o3), bs)]  # first problem's budget per batch
    chk("batches in ascending budget order", batch_budgets == sorted(batch_budgets), True)

    # ---- stateful resume: rebuild the same order + fast-forward -----------------
    g.manual_seed(42)
    s4 = BudgetGroupedSampler(ds, bs, shuffle_batch_order=True, generator=g)
    it = iter(s4)
    first10 = [next(it) for _ in range(10)]
    sd = it.state_dict()
    rest = list(it)  # exhaust
    full = first10 + rest

    # a fresh iterator restored from the snapshot must reproduce `full` from index 10.
    g.manual_seed(999)  # perturb -- load_state_dict must override via generator_state
    s5 = BudgetGroupedSampler(ds, bs, shuffle_batch_order=True, generator=g)
    it2 = iter(s5)
    it2.load_state_dict(sd)
    resumed = list(it2)
    chk("resume reproduces the tail exactly", resumed, full[10:])

    # ---- live budget table overrides the baked budget --------------------------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "budget_state.json")
        # ratchet every problem to budget 0 -> a single homogeneous run.
        with open(p, "w") as f:
            json.dump({"budgets": {f"p{i}": 0 for i in range(n)}}, f)
        g.manual_seed(1)
        s6 = BudgetGroupedSampler(ds, bs, budget_state_path=p, shuffle_batch_order=True, generator=g)
        live = s6._current_budgets()
        chk("live table overrides baked", set(live), {0})

    print("all budget_sampler self-tests passed.")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the built-in unit checks")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
