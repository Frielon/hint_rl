# Downward Budget Ratchet — HPRL §7 (IMPLEMENTED)

Per-problem hint budget `B_q` that **ratchets downward** as the policy proves it
can solve a problem with fewer hints. This is the downward half of paper
Section 7 (the upward-on-plateau half is deferred).

**Status: implemented**, gated behind the `data.hprl.enable` flag. Everything is
an *override* — a custom dataset class, a `RayPPOTrainer` subclass, and a recipe
entry that swaps it in — so **verl core is not edited**, and with the flag off
the code path is byte-for-byte stock verl. The pure rule + state store live in
[`budget_manager.py`](./budget_manager.py); this doc is the design + wiring map.

---

## 1. The downward rule

For one problem `q`, evaluated once per epoch over its group of rollouts:

- Let `N` = total rollouts for the problem this step (`rollout.n`, here 16).
- Let `C` = number of those rollouts that reached the correct answer.
- Let `h_1 … h_C` = the hint-call counts of the **correct** rollouts.

```
if 2·C < N:                      # fewer than half correct
    B_q  unchanged               # not yet reliable at this budget
else:                            # at least half correct
    asc   = sort(h_1 … h_C)      # ascending
    v     = asc[N//2 - 1]        # the (N/2)-th SMALLEST correct hint count
    B_q   = clamp(v - 1, min_budget, current_B_q)
```

`clamp(…, min_budget, current_B_q)` makes the ratchet **monotone down** (never
raises `B_q`) and floors it at `min_budget` (default `0` → a problem may ratchet
all the way to fully unaided, its terminal state).

**Worked example.** `N=8`, 6 correct with hint counts `[1,2,2,3,4,5]`:
`N/2 = 4` → 4th smallest = `3` → new `B_q = 3 - 1 = 2`. Intuition: at least half
the rollouts already succeed using `≤ 3` hints, so squeeze the budget just under
that to push the policy to solve one more step unaided next epoch.

---

## 2. Workflow

A closed loop per problem `q`, turning once per epoch (each time `q` is sampled).

```
                         ┌──────────────────────────────────────────────┐
                         │            budget_state.json                 │
                         │     { problem_id -> current B_q }   (driver) │  ← persists across
                         │   seeded from dataset's initial B_q on init  │    restarts (resume)
                         └──────────────────────────────────────────────┘
                            ▲  (5) save() updated B_q          │  (1) get(B_q)
                            │      per problem                 ▼
   ┌────────────────────────┴───────────┐      ┌────────────────────────────────────────────┐
   │  (4) POST-STEP RATCHET             │      │  (1) INJECTION  —  HintBudgetDataset       │
   │      hint_budget_callback          │      │       .__getitem__(row)                    │
   │                                    │      │                                            │
   │  group batch by problem_id         │      │  b = budget_mgr.get(problem_id)            │
   │  results = [(acc==1, num_hints),…] │      │  • system prompt: "{HINT_BUDGET}" → b      │
   │  gen_B = tools_kwargs.budget       │      │  • tools_kwargs…create_kwargs.budget = b   │
   │                                    │      └──────────────────────┬─────────────────────┘
   │  compute_downward_budget(          │                             │
   │     gen_B, results,                │                             ▼  prompt advertises b,
   │     min_budget=0, decrement=1)     │      ┌────────────────────────────────────────────┐
   │                                    │      │  (2) ROLLOUT  —  async agent loop          │
   │  ┌──────────────────────────────┐  │      │      n = 16 rollouts for this problem      │
   │  │ C = #correct,  N = #rollouts │  │      │                                            │
   │  │ if 2·C < N:  B_q unchanged   │  │      │  HintTool.execute():                       │
   │  │ else:                        │  │      │   • enforces cap: no-op once len(applied)≥b│
   │  │   asc = sort(correct hints)  │  │      │   • records applied_hints (per-rollout)    │
   │  │   v = asc[N//2 - 1]          │  │      └──────────────────────┬─────────────────────┘
   │  │   B_q = clamp(v - 1,         │  │                             │
   │  │        min_budget, gen_B)    │  │                             ▼
   │  └──────────────────────────────┘  │      ┌────────────────────────────────────────────┐
   │                                    │      │  (3) REWARD  —  HintRewardManager +        │
   │  budget_mgr.update_group(pid, …)   │      │       hint_reward.compute_score            │
   │  → log: mean/min B_q, #ratcheted   │◀─────│   per rollout: acc∈{0,1}, num_hints        │
   └────────────────────────────────────┘      │   (applied_hints merged into extra_info)   │
                            ▲                  └────────────────────────────────────────────┘
                            └──────────────── batch (acc, num_hints, problem_id, gen_B) ──────┘

   ════════════════════════════════════════════════════════════════════════════════════════
   NEXT EPOCH: q is sampled again → step (1) reads the *lowered* B_q → loop repeats,
   ratcheting down step-by-step until B_q hits min_budget=0 (problem becomes fully unaided).
```

---

## 3. Components & code mapping

Everything is gated by **`data.hprl.enable`** (run-script env `HPRL_ENABLE`).
The flag lives under `data.hprl` because verl passes the data config to the
dataset, while the trainer reads the same node as `config.data.hprl` — one
source of truth for both sides.

| Stage | Where it lives | Status |
|---|---|---|
| **State store** | `budget_state.json` via `BudgetManager` (+ `load_budget_table` reader) | ✅ `budget_manager.py` |
| **(1) Injection** | `HintBudgetDataset(RLHFDataset).__getitem__`: reads `B_q` from the budget JSON (mtime-cached, per-row baked fallback), re-renders the system prompt from `extra_info.hprl_system_base` via `render_system`, overrides budget in `tools_kwargs` **and** `extra_info`. No-op unless flag on / row has `hprl_system_base`. | ✅ `hint_dataset.py`, `hint_prompt.py`, `prepare_hint_data.py` |
| **(2) Rollout** | `HintTool` enforces `budget` and records `applied_hints` | unchanged |
| **(3) Reward** | `HintRewardManager` + `compute_score` emit `acc`, `num_hints`; `problem_id` rides `extra_info` | unchanged |
| **(4) Ratchet** | `HPRLRayPPOTrainer._update_actor` → `hprl_update_budgets`: group batch by `problem_id` → `compute_downward_budget` → `update_group`, surfaced as `hprl/*` wandb metrics | ✅ `hprl_ray_trainer.py`, `hint_budget_callback.py` |
| **(5) Persist** | `BudgetManager.save()` (atomic) once per step | ✅ `budget_manager.py` |
| **Wiring** | recipe entry: `HPRLTaskRunner` swaps `RayPPOTrainer`→`HPRLRayPPOTrainer`; config extends `ppo_trainer` | ✅ `main_hprl.py`, `config/hprl_trainer.yaml` |

> **No verl core edits.** Injection is wired via `data.custom_cls`; the trainer
> override is installed by rebinding the `RayPPOTrainer` module global inside the
> task runner (so the unmodified, evolving `TaskRunner.run()` builds the
> subclass). The chosen per-step seam is `_update_actor` — for GRPO it always
> runs and receives the fully-populated post-reward batch; budget metrics ride
> back on `actor_output.meta_info["metrics"]`.

---

## 4. Implementation steps

### Step 1 — make the budget dynamic in the dataset (injection side)
- **`prepare_hint_data.py`** keeps baking the static initial `B_q` (so the
  parquet still works flag-off) **and** now stores `extra_info.hprl_system_base`
  (the budget-free preamble) + `hprl_init_budget`. The tool/budget sentence is
  rendered by the shared `hint_prompt.render_system`, imported by both the data
  prep and the dataset so flag-on re-rendering is identical except for the digit.
  → **regenerate the parquet** (`python prepare_hint_data.py`) before a ratchet run.
- **`hint_dataset.py`: `HintBudgetDataset(RLHFDataset)`.** In `__getitem__`,
  after the base builds the row, read the **initial budget from the parquet**
  (`tools_kwargs.request_hint.create_kwargs.budget`, the baked `K_q` `B_q`) as
  the per-row fallback: `b = table.get(problem_id, baked_b)`. Then re-render the
  system prompt from `hprl_system_base` for `b` and overwrite the budget in
  `tools_kwargs` **and** `extra_info` (the latter is what the ratchet reads back
  as the budget the rollouts ran under). Rows without `hprl_system_base` (e.g.
  the unaided val set) pass through untouched. Wired via `data.custom_cls` in
  `config/hprl_trainer.yaml`.
  - **Initial load** (first epoch, or any problem not yet ratcheted): the
    `problem_id` is absent from `budget_state.json`, so `get` returns
    `baked_b` — that problem's own starting budget from the dataset. Once the
    ratchet records a lower value, `get` returns the lowered `B_q`. Using the
    per-row baked budget as the default (rather than a single global
    `default_budget`) means unseen problems start at their own `K_q`, and a
    partial state file is handled for free.
- **State channel = the JSON file**, read in `__getitem__` (with a small mtime
  cache). verl's DataLoader can use worker subprocesses that won't see
  driver-side in-memory mutations; the file is the robust channel. Budgets only
  change at step boundaries and a problem reappears once per epoch ≫ one step,
  so intra-step staleness is harmless.

### Step 2 — capture outcomes + ratchet after each step (update side)
Everything needed is already in the post-reward batch: `extra_info.problem_id`,
`acc` and `num_hints` (reward keys merged into `non_tensor_batch`), and the
budget the rollouts ran under (`extra_info.tools_kwargs…budget`, which the
dataset wrote and which `extra_info` is guaranteed to preserve through rollout).
- **`hint_budget_callback.hprl_update_budgets`** groups the batch by
  `problem_id`, builds `results=[(acc≥0.5, num_hints), …]`, and calls
  `bm.update_group(pid, results, current_budget=min(gen_B, stored))` (the `min`
  keeps the ratchet monotone even under dataloader prefetch lag), then
  `bm.save()`. Returns `hprl/*` scalar metrics.
- **`HPRLRayPPOTrainer._update_actor`** (flag-gated) calls it after the real
  actor update and merges the metrics into `actor_output.meta_info["metrics"]`.
  verl exposes no post-reward callback; `_update_actor` is the per-step driver
  method that, for GRPO, always runs and sees the fully-populated batch — chosen
  over `_log_rollout_data` (which is gated on `rollout_data_dir`). The subclass
  is installed by `main_hprl.HPRLTaskRunner` without copying `fit()`/`run()`.

### Step 3 — seeding & restart
The initial per-problem budget always comes from the **parquet** (the baked
`K_q`-based `B_q`), surfaced as the per-row default in `get` (Step 1) — so no
explicit seeding is required for correctness. `bm.seed({pid: baked_b})` at
trainer init is optional: it pre-populates `budget_state.json` so the starting
budgets show up in logs from step 0. `resume_mode=auto` restarts restore the
ratchet from the same file.

### Step 4 — observability
Log per step: mean/min `B_q` across problems, # problems ratcheted down this
step, and the histogram of `B_q`. `BudgetUpdate.as_dict()` already carries the
fields.

---

## 5. Correctness points / guards

- **`gen_B` = the budget the rollouts actually ran under** (read back from the
  batch's `tools_kwargs.budget`), not the current store value — so the ratchet
  evaluates against the right baseline even if anything raced.
- **State channel is the JSON file**, so it survives dataloader worker
  subprocesses and trainer restarts (`resume_mode=auto`).
- **One turn of the loop per epoch** — a problem is sampled once per epoch with
  `N=16` rollouts, which is the `N` in the rule.
- **Monotone down** — `clamp(…, min_budget, gen_B)` guarantees `B_q` never
  rises; terminal state is `B_q=0` (unaided GRPO problem).
- **`max_assistant_turns` (8)** stays `≥` any budget since budgets only fall —
  no change needed.
- This only handles the **downward** half of §7. The upward-on-plateau ratchet
  is a separate function in the same module later.
