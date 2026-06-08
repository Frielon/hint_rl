# Converting `run_grpo_qwen2.5_7b_npu.sh` to multi-turn tool use

This guide explains how to turn the plain single-turn GRPO script
(`script/run_grpo_qwen2.5_7b_npu.sh`) into a **multi-turn tool-use** GRPO run in
verl: what the training/test data must look like, what tool config to add, and
exactly which script flags to change.

The canonical, currently-maintained reference in verl is the **ReTool recipe**
(`recipe/retool/run_qwen2_7b_dapo.sh`) — GRPO + multi-turn tools + vLLM +
Qwen2.5-7B, almost identical to our setup.

---

## How multi-turn tool use works in verl (the mental model)

verl drives multi-turn tool use through the **agent loop** running against an
**OpenAI-compatible async server** wrapped around the rollout engine. Importantly,
this works with **vLLM** — you do *not* have to switch to SGLang.

Three moving parts must be added:

1. **`rollout.mode=async`** + **`rollout.multi_turn.enable=True`** — turns on the agent loop.
2. **A tool config** (`tool_config_path` or `function_tool_path`) — defines the callable tool(s).
3. **Per-sample `agent_name: "tool_agent"`** in the dataset — routes each prompt through
   the tool-calling loop instead of single-turn generation.

The loop is: model generates → if it emits a `<tool_call>`, verl parses it (using
`multi_turn.format=hermes` for Qwen), executes the tool, injects the result as a
`tool`-role message, and feeds it back — repeating until the model stops calling
tools or hits `max_assistant_turns`. Loss is masked so **only assistant-generated
tokens are trained on** (tool outputs are excluded via delta-based tokenization).

---

## 1. Data format

The dataset stays a parquet of one row per prompt. Two things matter for multi-turn:

### Required columns

| Column | Type | Notes |
|---|---|---|
| `prompt` | `list[{"role","content"}]` | Chat messages (we already use `prompt_key=prompt`). System message should instruct the model to use the tool. |
| `data_source` | `str` | As today. |
| `reward_model` | `dict` | Must contain `ground_truth`; `compute_score` reads this. |
| `agent_name` | `str` | **NEW — must be `"tool_agent"`** so the row is routed to the tool agent loop. |
| `extra_info` | `dict` | Optional, but **required if the tool is stateful** (see below). |
| `ability` | `str` | As today. |

This is exactly the shape ReTool produces
(`recipe/retool/retool_dataset_utils.py:29`): it just adds
`"agent_name": "tool_agent"` to each row.

### The `agent_name` field is the key switch

Without `agent_name="tool_agent"` on a row, it falls back to
`default_agent_loop: single_turn_agent` (no tools). You can set the column per-row,
or set `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent` globally —
but the per-row column is the idiomatic way.

### Stateless vs. stateful tools — does the data need `tools_kwargs`?

- **Stateless tool** (e.g. a code interpreter / calculator — same behavior for every
  sample): **no extra data needed.** Just `agent_name`. This is the ReTool pattern.
- **Stateful tool** that needs per-sample data injected at creation (e.g. the old
  GSM8K reward tool, which needs `ground_truth` to score the model's answer): each
  row must add:

```python
"extra_info": {
    "need_tools_kwargs": True,
    "tools_kwargs": {
        "<tool_name>": {                       # must match the tool's schema name
            "create_kwargs": {"ground_truth": solution},
            # "execute_kwargs": {...},
            # "calc_reward_kwargs": {...},
            # "release_kwargs": {...},
        },
    },
}
```

This is the GSM8K-with-tools pattern
(`examples/data_preprocess/gsm8k_multiturn_w_tool.py:90`).

### Minimal preprocessing to upgrade the *existing* parquets

If `dapo-3740-hint-verl.parquet` / `aime2024.parquet` already have `prompt` +
`reward_model.ground_truth` (they do, since they train with plain GRPO today), you
only need to add `agent_name` and tweak the system prompt to mention the tool:

```python
import datasets

TOOL_HINT = (
    "\n\nYou may call the `code_interpreter` tool to run Python and check your work. "
    "Reason step by step; call the tool when useful; put the final answer in \\boxed{}."
)

def add_tool_fields(row):
    # ensure prompt is a chat list; nudge the model to use the tool
    msgs = row["prompt"]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] += TOOL_HINT
    else:
        msgs = [{"role": "system", "content": "You are a math expert." + TOOL_HINT}, *msgs]
    row["prompt"] = msgs
    row["agent_name"] = "tool_agent"     # <-- the switch
    return row

for split, path in [("train", "dapo-3740-hint-verl.parquet"), ("test", "aime2024.parquet")]:
    ds = datasets.Dataset.from_parquet(path)
    ds = ds.map(add_tool_fields)
    ds.to_parquet(path.replace(".parquet", "-mt.parquet"))
```

> `data.return_raw_chat=True` must be set (the current script doesn't) so the raw
> message list is preserved for the agent loop.

---

## 2. The tool config file (must be created)

You must decide **what tool** the model calls. Two routes:

### Option A — stateless function tool (simplest, no external server)

Create `tools/my_tools.py`:

```python
from verl.tools.function_tool import function_tool

@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a Python-style arithmetic expression.

    Args:
        expression: A Python-style arithmetic expression, e.g. "(3+4)*5".
    """
    return str(eval(expression, {"__builtins__": {}}, {}))
```

Then point at it with
`actor_rollout_ref.rollout.multi_turn.function_tool_path=tools/my_tools.py`.
No YAML, no server; the schema is inferred from the signature + docstring.

### Option B — code interpreter (ReTool-style, needs a sandbox server)

Create `tools/code_tool_config.yaml` (mirrors
`recipe/retool/sandbox_fusion_tool_config.yaml`):

```yaml
tools:
  - class_name: "recipe.retool.retool.CustomSandboxFusionTool"
    config:
      sandbox_fusion_url: "http://localhost:8080/run_code"
      num_workers: 64
      enable_global_rate_limit: true
      rate_limit: 64
      default_timeout: 30
      default_language: "python"
      type: native
    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
        description: "A tool for executing Python code."
        parameters:
          type: "object"
          properties:
            code: {type: "string", description: "The code to execute."}
          required: ["code"]
```

This requires running a Sandbox Fusion server separately. Point at it with
`multi_turn.tool_config_path=...`.

**Recommendation:** start with **Option A** (function tool) to validate the
multi-turn plumbing end-to-end, then move to a sandbox if you need real code execution.

---

## 3. Script modifications

Add these flags to the `ray job submit ... python3 -m verl.trainer.main_ppo`
invocation, and change two existing lines.

### Add near the top (variables)

```bash
max_turns=8                                   # cap on assistant/user turns
# Option A:
FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:-"${HINT_RL_HOME}/tools/my_tools.py"}
# Option B:
# TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-"${HINT_RL_HOME}/tools/code_tool_config.yaml"}
```

### Change these existing flags

```bash
# keep data.truncation='left', but ADD return_raw_chat:
data.return_raw_chat=True \

# response length: multi-turn accumulates tokens across turns — bump if memory allows
data.max_response_length=16384 \   # (was 8192)
```

### Add these NEW flags (the multi-turn block)

```bash
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.function_tool_path="${FUNCTION_TOOL_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
```

(For Option B, replace the `function_tool_path` line with
`actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}"`.)

### Point at the upgraded data

```bash
TRAIN_FILE=${TRAIN_FILE:-"${HINT_RL_HOME}/dataset/dapo-3740-hint-verl-mt.parquet"}
TEST_FILE=${TEST_FILE:-"${HINT_RL_HOME}/dataset/aime2024-mt.parquet"}
```

### What stays unchanged

Everything else works as-is: `rollout.name=vllm`, GRPO algorithm flags
(`adv_estimator`, `norm_adv_by_std_in_grpo`, clip-higher, no-KL), the
`custom_reward_function` (`compute_score` still receives the full final response and
`ground_truth`), FSDP/offload/sp_size, dynamic bsz, logging.

---

## Caveats to watch

1. **Async vLLM is required.** `rollout.mode=async` is mandatory for the agent loop.
   If the NPU/vLLM-Ascend build doesn't support async server mode, the agent loop
   won't run — verify on a tiny run first. (The script sets `trainer.device=cuda`
   despite the `_npu` filename, so on CUDA/vLLM this is well-supported.)
2. **Response length.** Multi-turn trajectories are long (prompt + N×(assistant +
   tool result)). The current `max_response_length=8192` may truncate; ReTool uses
   16384. Watch `data.truncation='error'` vs `'left'` — with multi-turn, overlong
   rollouts can abort the step.
3. **Tokenization sanity check.** Qwen2.5-Instruct is fine with `hermes` format. If
   you see "Inconsistent training and inference tokenization" warnings, set
   `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable`.
4. **System prompt must invite the tool.** If the prompt never tells the model a tool
   exists, it'll never call one and you've effectively got single-turn training. The
   data-prep `TOOL_HINT` above handles this.
5. **`max_tool_response_length`** defaults to 256 tokens (truncated `middle`). Raise it
   via `multi_turn.max_tool_response_length=...` if the tool returns longer output.

---

## References (in the verl tree)

- `recipe/retool/run_qwen2_7b_dapo.sh` — GRPO + multi-turn tools + vLLM (closest template).
- `recipe/retool/retool_dataset_utils.py` — data prep adding `agent_name`.
- `recipe/retool/sandbox_fusion_tool_config.yaml` — code-interpreter tool config.
- `examples/data_preprocess/gsm8k_multiturn_w_tool.py` — stateful tool data with `tools_kwargs`.
- `verl/trainer/config/rollout/rollout.yaml` — full `multi_turn` config schema.
- `verl/tools/function_tool.py` — `@function_tool` decorator (Option A).
- `docs/sglang_multiturn/multiturn.rst` — multi-turn rollout / tokenization docs.
