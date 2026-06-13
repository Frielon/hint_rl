# Hint Penalized RL (HPRL) — training scripts

Implements Section 4 of `docs/hprl_v3.md`: a multi-turn GRPO run where the policy
requests a **hint** during rollout by emitting the sentinel `<hint_call/>`. Each
request is routed to a frozen **selector model** that picks the most useful
step-level hint from the problem's hint pool; every applied hint is recorded in
per-rollout state and (eventually) penalizes the trajectory reward.

Built on the verl async agent loop (a custom `AgentLoopBase` subclass,
`HintAgentLoop`). All verl integration is via overrides — a custom agent loop, a
custom dataset, and a `RayPPOTrainer` subclass — with **no edits to verl core**.

## Files (all live in this folder)

| File | Role |
|---|---|
| `prepare_hint_data.py` | Upgrades `dataset/dapo-3740-hint-verl-simplified.parquet` → `…-mt.parquet`: adds `agent_name="hint_agent"` (default; `--agent-name tool_agent` for the legacy path), the tool/budget system-prompt nudge, and `extra_info.tools_kwargs` carrying the per-sample hint pool + budget. |
| `hint_agent_loop.py` | **`HintAgentLoop`** (`agent_name="hint_agent"`) — the **`<hint_call/>` rollout**. Generates each turn to EOS, detects the sentinel `<hint_call/>`, calls the selector, and injects the hint as the next **user** message; enforces `B_q` and records `applied_hints`. |
| `hint_selector.py` | `HintSelector` (frozen-selector client, `from_env()`) + `build_trace` — the selector logic, factored out of the tool so the agent loop can use it without a verl tool. Also the candidate-pool filters (`exclude_applied_hints` / `exclude_applied_steps`) and the major-step helpers (`hints_for_step`, `format_step_hints`, `step_id_of`). |
| `hint_agent_config.yaml` | Registers `hint_agent → hint_agent_loop.HintAgentLoop` (verl `rollout.agent.agent_loop_config_path`). |
| `hint_tool.py` / `hint_tool_config.yaml` | **Legacy** hermes `request_hint` tool path (`agent_name="tool_agent"`). Superseded by the `<hint_call/>` loop; kept for comparison. |
| `hint_reward.py` | `compute_score`: outcome correctness (mathruler) **minus** the summed hint penalty (`R = R_acc − Σ wₖ`, penalty applied only when correct). Penalty knobs arrive as tunable `reward_kwargs`. |
| `hint_penalty.py` | Pure (verl-free) importance weights `wₖ` from the difficulty-annotated pool: `total_penalty` split across steps then hints by `hard_factor**difficulty_level`. Per-hint (`compute_hint_penalties` / `applied_penalty`) for the `hint` strategy, per-step (`compute_step_penalties` / `applied_step_penalty`) for `major_step`. |
| `hint_reward_manager.py` | `HintRewardManager`: merges the per-rollout `applied_hints` state into `extra_info` so the reward function sees it. |
| `hint_prompt.py` | Shared system-prompt renderer (`render_system`, `TOOL_INSTRUCTION`) used by both the data prep and the dynamic-budget dataset so the budget sentence is identical. |
| `budget_manager.py` | Pure downward-ratchet rule (`compute_downward_budget`) + JSON-backed per-problem budget store (`BudgetManager`). verl-free; unit-tested via `--selftest`. |
| `hint_dataset.py` | `HintBudgetDataset(RLHFDataset)` — **injection side** of the ratchet: reads each problem's current `B_q` from the budget-state JSON and re-renders the prompt + `tools_kwargs` budget. No-op unless `data.hprl.enable`. |
| `hint_budget_callback.py` | `hprl_update_budgets` — **update side**: groups a step's rollouts by `problem_id`, applies the downward rule, persists the new budgets. |
| `hprl_ray_trainer.py` | `HPRLRayPPOTrainer(RayPPOTrainer)` — flag-gated override of `_update_actor` that runs the ratchet each step. Identical to base when the flag is off. |
| `main_hprl.py` / `config/hprl_trainer.yaml` | Recipe entry: stock `TaskRunner`/`run_ppo` with the trainer swapped for `HPRLRayPPOTrainer`; config = `ppo_trainer` + the `data.hprl` knobs. |
| `run_hprl_qwen2.5_7b.sh` | Launch script (multi-turn GRPO + dynamic budget, derived from `script/run_grpo_qwen2.5_7b_npu.sh`). |
| `launch_hprl_cluster.sh` | 5-node cluster entrypoint (run on every pod): the selector node serves `gpt-oss-20b` via vLLM (DP=8); the other 4 nodes run `ray_cluster_launch.sh` → `run_hprl`. |

## Hint mechanism: the `<hint_call/>` sentinel

The policy requests a hint by emitting the sentinel **`<hint_call/>`** on its own
line (taught by the system prompt in `hint_prompt.TOOL_INSTRUCTION`). Generation
of each turn **still ends on EOS** — the sentinel is *not* a stop string.
`HintAgentLoop` then inspects the finished turn: if `<hint_call/>` is present and
budget remains, it asks the selector for a hint and injects it as the **next user
message**, then continues. The injected message is the hint text followed by a
one-sentence notice of how many hint calls remain (`hint_prompt.render_remaining_calls`,
`budget − hints_used`), so the policy can ration its remaining budget. This replaces the legacy hermes `request_hint`
function-tool call (which returned a *tool* message). Budget `B_q` and the
`applied_hints` state are tracked by the loop, so the reward + budget ratchet are
unchanged.

## Per-rollout state

The list of all hints applied in a rollout is kept on
`agent_data.extra_fields["applied_hints"]` (one object per trajectory). Each entry:

```json
{"call_index": 0, "hint_id": "2.1", "major_step_id": 2, "hint": "...", "confidence_of_hint": 4}
```

Under `HINT_STRATEGY=major_step` each entry instead records the whole revealed
step — `{"call_index": 0, "major_step_id": 2, "hint_ids": ["2.0","2.1","2.2"],
"hint": "<all step hints>", "confidence_of_major_step": 4}` — but it is still one
entry per call, so `num_hints` and the budget ratchet read it identically.

It flows out of the agent loop and `HintRewardManager` injects it into
`extra_info["applied_hints"]` for the reward.

### In the rollout log

`HPRLRayPPOTrainer._log_rollout_data` (flag-gated) splices the per-rollout state
into the dumped JSONL (`logs/<exp>/rollouts/<step>.jsonl`), on top of verl's
stock `input`/`output`/`score`/`gts` + reward keys (`acc`, `num_hints`, ...):

- **`applied_hints`** — the full structured list above (hint_id / major_step_id /
  confidence per applied hint), not just the `num_hints` count.
- **`hint_budget`** — the budget `B_q` that rollout actually ran under.

## How to run

1. Build the multi-turn dataset:
   ```bash
   python prepare_hint_data.py           # writes dataset/…-simplified-mt.parquet
   ```
2. Serve the frozen **selector model** on an OpenAI-compatible endpoint (e.g.
   `selector/run_eval_h100.sh` / an sglang server) and point the run at it:
   ```bash
   export SELECTOR_BASE_URL=http://localhost:30000/v1
   export SELECTOR_MODEL=Qwen3.5-27B
   ```
3. Launch (Ray cluster must be up, as for the plain GRPO run):
   ```bash
   ./run_hprl_qwen2.5_7b.sh
   ```

Validation runs on two held-out sets (`VAL_FILES`), neither carrying an
`agent_name`, so both run **single-turn / unaided** — measuring the policy's
hint-free capability:
- `aime2024.parquet` (30 problems) → `val-core/aime2024/*`;
- `dapo_sample_hard_100.parquet` (100 hard DAPO problems, zero `problem_id`
  overlap with the training set) → `val-core/math_dapo/*`.

### Cluster launch (5 nodes: 1 selector + 4 training)

Use `launch_hprl_cluster.sh` as the PyTorchJob entrypoint on **every** pod
(like `ray_cluster_launch.sh`). It routes by node `RANK`:

- **Selector node** (`SELECTOR_RANK`, default the last rank): serves
  `/share5/users/xutao.ma/model/gpt-oss-20b` via vLLM — data-parallel `DP=8`,
  `TP=1` (a small MoE, so 8 replicas behind one `/v1` endpoint maximizes
  throughput), `--reasoning-parser openai_gptoss`. It publishes its endpoint to a
  shared-FS rendezvous file (`logs/.selector_endpoint.<MASTER_PORT>`) so the
  training pods find it without DNS (or set `SELECTOR_HOST` to skip rendezvous).
- **Training nodes** (the other 4): read the endpoint, then run
  `ray_cluster_launch.sh` (`NNODES=4`) → `run_hprl_qwen2.5_7b.sh`, which forwards
  **all** selector call params into the rollout workers' Ray runtime env.

Selector call params (env → `HintSelector.from_env`, forwarded by `run_hprl`):
`SELECTOR_BASE_URL`, `SELECTOR_MODEL` (`gpt-oss-20b`), `SELECTOR_API_KEY`,
`SELECTOR_TEMPERATURE` (0.7), `SELECTOR_TOP_P` (1.0), `SELECTOR_MAX_TOKENS`
(16000), `SELECTOR_REQUEST_TIMEOUT_S` (600), `SELECTOR_MAX_RETRIES` (3). The
endpoint is published early, so training starts in parallel with model loading;
hint calls retry and degrade gracefully (unaided) if the selector is slow/down.

## Dynamic budget ratchet (paper §7) — the HPRL flag

The **downward** budget ratchet is implemented and gated behind one flag,
`data.hprl.enable` (default `true` in `config/hprl_trainer.yaml`; the run script
exposes it as `HPRL_ENABLE`). With it **off**, `HintBudgetDataset` and
`HPRLRayPPOTrainer` fall through to stock verl behavior — an ordinary
static-budget multi-turn GRPO run. With it **on**:

1. **Injection** (`hint_dataset.py`): each epoch, every problem's prompt + tool
   budget are set to its *current* `B_q`, read from the budget-state JSON.
2. **Ratchet** (`hprl_ray_trainer.py` → `hint_budget_callback.py`): after each
   step, rollouts are grouped by `problem_id` and `B_q` is lowered per the rule
   in `budget_manager.compute_downward_budget` (if any rollout is correct, drop
   to the fewest hints a correct rollout used — or one less when even the most
   frugal success used the whole budget; if none correct, hold). Strictly
   downward — there is no budget-raising mechanism.
3. **Persist**: the new table is atomically written back to the JSON the dataset
   reads. It survives restarts (`resume_mode=auto`).

Knobs (run-script env → config): `HPRL_MIN_BUDGET` (`data.hprl.min_budget`, floor,
default 0 → can reach fully unaided), `HPRL_DECREMENT` (`data.hprl.decrement`),
and `HPRL_DEFAULT_BUDGET`. An over-budget `<hint_call/>` (the policy emits the
sentinel after `B_q` is spent) is now ALWAYS a terminating **protocol violation**:
the rollout is flagged and floored to `budget_exceeded_reward`
(`custom_reward_function.reward_kwargs.budget_exceeded_reward`, default
`incorrect_reward`) with `acc=0` and the boxed answer **not graded** — even a
correct box earns nothing if the rollout ended on an illegal call. (The former
`terminate_on_budget_exceeded` terminate-vs-"nudge to finish" knob is retired.) See
`downward_budget_plan.md`.

All verl integration is via overrides (custom dataset class + trainer subclass +
recipe entry) — **no edits to verl core**.

## Hint-selection strategy (`HINT_STRATEGY`)

How a `<hint_call/>` is answered and penalized is selected by one parameter,
`HINT_STRATEGY` in `run_hprl_qwen2.5_7b.sh`. The **same** value is wired to both
the agent loop (`data.hprl.strategy` → `HintAgentLoop`) and the reward
(`reward_kwargs.hint_strategy` → `compute_score`); they **must agree** (the run
script passes one variable to both). Selection (what is excluded next call) and
penalty live at the **same granularity**:

| `HINT_STRATEGY` | Inject | Rollout state / next-call exclusion | Penalty |
|---|---|---|---|
| `hint` *(default)* | the **one** hint the selector picks inside the identified major step | that **hint** (`hint_id`) is excluded (`exclude_applied_hints`) | per-hint `Σ wₖ` (`applied_penalty`) |
| `major_step` | **all** hints of the major step the selector identifies | the **whole step** (`major_step_id`) goes into `applied_hints` and is excluded (`exclude_applied_steps`) | the **step penalty** directly (`applied_step_penalty`) |

In both modes the selector runs the *same* prompt (`utils.selector_prompt`) and
identifies the major step the student is stuck on; `major_step` simply reveals
the entire step (`hint_selector.hints_for_step` / `format_step_hints`) instead of
one hint, records the step rather than the hint, and so consumes one budget unit
per **step** revealed. Each call still appends one entry to `applied_hints`, so
`num_hints` and the budget ratchet are unchanged.

## Hint penalty (paper §3/§6)

Implemented in `hint_penalty.py`, applied by `hint_reward.compute_score`. The
penalties of **all** hints of a problem sum to `total_penalty` (default `1.8`),
split in two stages, both by difficulty weight `hard_factor**level`
(`easy=0, moderate=1, hard=2`):

1. **across steps** — `step_penalty = total_penalty · w_step / Σ w_step`
   (`compute_step_penalties`; the `major_step` strategy charges these directly);
2. **within a step** — `hint_penalty = step_penalty · w_hint / Σ w_hint`, where
   the `X.0` guidance hint is assigned `guidance_difficulty` (default `moderate`)
   (the `hint` strategy charges these).

Difficulties come from the original pool stored at `extra_info.hint_full`. All
three knobs are `reward_kwargs` (run-script env `HINT_PENALTY_TOTAL`,
`HINT_PENALTY_HARD_FACTOR`, `HINT_GUIDANCE_DIFFICULTY`) — **retunable without
regenerating the dataset**. The total is invariant to `hard_factor` (it only
redistributes). Reward: a wrong answer gets `incorrect_reward`; a correct answer
gets `correct_reward − Σ wₖ − shape` (penalties apply only when correct), where
`shape` is the effort-shaping penalty (`HINT_SHAPE_COEFF`, default 0.3): per
applied hint, `coeff · relu(mean_turn_len − pre_call_reasoning_len)/mean_turn_len`
summed over calls — it discourages calling a hint after too-little reasoning
(the front-loading pathology). The correct score is **floored at `incorrect_reward`
(−1)** so a hinted success never scores below a failure.

## Design notes

- The budget ratchet is **strictly downward by design — there is no upward /
  budget-raising mechanism.** A problem's `B_q` only ever falls (or holds), never
  rises; its terminal state is `B_q=0` (fully unaided).