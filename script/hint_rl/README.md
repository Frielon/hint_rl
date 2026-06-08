# Hint Penalized RL (HPRL) — training scripts

Implements Section 4 of `docs/hprl_v3.md`: a multi-turn tool-use GRPO run where
the policy can call a **hint tool** during rollout. Each hint call is routed to
a frozen **selector model** that picks the most useful step-level hint from the
problem's hint pool; every applied hint is recorded in per-rollout state and
(eventually) penalizes the trajectory reward.

Built on the multi-turn plumbing described in `docs/multiturn_tool_use_guide.md`
(verl async agent loop + a stateful `BaseTool`).

## Files (all live in this folder)

| File | Role |
|---|---|
| `prepare_hint_data.py` | Upgrades `dataset/dapo-3740-hint-verl-simplified.parquet` → `…-mt.parquet`: adds `agent_name="tool_agent"`, a tool/budget system-prompt nudge, and `extra_info.tools_kwargs` carrying the per-sample hint pool + budget. |
| `hint_tool.py` | `HintTool` (verl `BaseTool`). On each call: builds the trace from the live trajectory, calls the selector model (reusing `selector/seletor_prompt.py` + the parser in `selector/run_hint_selection_model.py`), records the hint in per-rollout state (`agent_data.extra_fields["applied_hints"]`), and enforces the per-problem budget `B_q`. |
| `hint_tool_config.yaml` | Declares the `request_hint` tool + selector endpoint config. |
| `hint_reward.py` | `compute_score`: outcome correctness (mathruler) × hint penalty. **The `hint_penalty()` hook is left blank** (returns `1.0`); fill it in to realize `R = R_acc · max(0, 1 − Σ wₖ)`. |
| `hint_reward_manager.py` | `HintRewardManager`: merges the per-rollout `applied_hints` state into `extra_info` so the reward function sees it. |
| `run_hprl_qwen2.5_7b.sh` | Launch script (multi-turn GRPO, derived from `script/run_grpo_qwen2.5_7b_npu.sh`). |

## Per-rollout state

The list of all hints applied in a rollout is kept on
`agent_data.extra_fields["applied_hints"]` (one object per trajectory; it
survives the per-call `create`/`release` tool lifecycle). Each entry:

```json
{"call_index": 0, "hint_id": "2.1", "major_step_id": 2, "hint": "...", "confidence_of_hint": 4}
```

It flows out of the agent loop and `HintRewardManager` injects it into
`extra_info["applied_hints"]` for the reward.

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

Validation (`aime2024.parquet`) has no `agent_name`, so it runs **single-turn /
unaided** — it measures the policy's hint-free capability.

## Not yet implemented (hooks left open)

- **`hint_penalty()`** in `hint_reward.py` — the multiplicative penalty / hint
  importance weights `wₖ` (paper §3, §6).
- **Budget ratcheting** (paper §7): `B_q` is currently *static* per problem
  (= number of major steps in its hint pool, capped at `--max-budget`), baked
  into the dataset. The downward-on-success / upward-on-plateau trainer-state
  ratchet is not wired.
