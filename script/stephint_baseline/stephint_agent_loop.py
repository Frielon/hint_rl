# Copyright 2026
#
# StepHintAgentLoop -- the STEPHINT (solution-prefix) rollout.
#
# Single-turn generation with an optional REFERENCE-SOLUTION PREFIX injected at
# the start of the assistant turn: the model is forced to continue from the
# first i segments of the dataset's segmented solution instead of starting from
# scratch. The per-rollout prefix text arrives as the ``stephint_prefix``
# non-tensor column, stamped onto the repeated prompt-group by
# stephint_fully_async.stamp_stephint_prefixes (rollouts j < r*(M-1) get
# segments 1..(j//r + 1); the rest get "" and behave exactly like verl's
# single_turn_agent).
#
# Token layout (the whole point of this loop):
#
#   prompt_ids    = chat-template prompt ONLY (no prefix) -> the "prompts"
#                   tensor is identical across the GRPO group.
#   response_ids  = prefix_ids + generated_ids   (prefix lives in the RESPONSE
#                   region, so the reward decodes prefix+continuation as one
#                   solution and the group shares one prompt shape)
#   response_mask = [0]*len(prefix_ids) + [1]*len(generated_ids)
#
# The 0-mask is what keeps the prefix gradient-free: GRPO broadcasts the scalar
# group advantage as ``score * response_mask`` and the actor aggregates the pg
# loss with masked_mean over the same mask, so prefix tokens contribute exactly
# nothing to the update while still conditioning the continuation. Rollout
# logprobs get 0.0 placeholders on the prefix positions (same convention as the
# auto-hint loop's injected user turns) -- inert under the loss mask, but they
# keep ``rollout_log_probs`` shape-aligned for bypass-mode old_log_probs.
#
# Generation is bounded with an explicit per-request max_tokens =
# response_length - len(prefix_ids), so prefix+generation can never outgrow the
# response budget (and never exceeds vLLM's max_model_len = prompt_length +
# response_length -- see the async context-overflow crash lore).
#
# Registered as agent_name "stephint_agent" (also via stephint_agent_config.yaml,
# imported from PYTHONPATH = script/stephint_baseline). No verl core edits.

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Never let a (pathologically long) prefix eat the whole generation budget: if
# fewer than this many tokens would remain for the continuation, drop the
# prefix and roll out from scratch instead (loud warning; the row is recorded
# with stephint_prefix_dropped=1). With the dolci stephint set (max prefix
# ~1.1k tokens vs response_length 16384) this never triggers.
_MIN_GEN_TOKENS = 256


@register("stephint_agent")
class StepHintAgentLoop(AgentLoopBase):
    """Single-turn rollout continuing from a loss-masked reference-solution prefix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. chat-template prompt (NO prefix -- identical across the group).
        prompt_ids = await self.apply_chat_template(messages)

        # 2. tokenize this rollout's solution prefix ("" -> from-scratch rollout).
        prefix_text = kwargs.get("stephint_prefix")
        prefix_text = str(prefix_text) if prefix_text is not None else ""
        prefix_dropped = 0
        prefix_ids: list[int] = []
        if prefix_text:
            prefix_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(prefix_text, add_special_tokens=False)
            )
            if len(prefix_ids) > self.response_length - _MIN_GEN_TOKENS:
                logger.warning(
                    "stephint prefix of %d tokens leaves < %d generation tokens "
                    "(response_length=%d); dropping the prefix for this rollout.",
                    len(prefix_ids),
                    _MIN_GEN_TOKENS,
                    self.response_length,
                )
                prefix_ids = []
                prefix_dropped = 1

        # 3. generate the continuation. COPY of sampling_params: generate() may
        # pop/mutate keys (max_tokens, logprobs, ...). The prefix rides in
        # prompt_ids for the engine, so partial-rollout resume (which replays
        # prompt_ids + generated-so-far) keeps it transparently.
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + prefix_ids,
                sampling_params={**sampling_params, "max_tokens": self.response_length - len(prefix_ids)},
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        # 4. response = prefix (mask 0, logprob 0.0 placeholders) + generation (mask 1).
        response_ids = prefix_ids + list(output.token_ids)
        response_mask = [0] * len(prefix_ids) + [1] * len(output.token_ids)
        response_logprobs = None
        if output.log_probs is not None:
            response_logprobs = [0.0] * len(prefix_ids) + list(output.log_probs)

        agent_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length] if response_logprobs is not None else None
            ),
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keep the single_turn/tool_agent schema + the stephint bookkeeping
        # (these land in non_tensor_batch via _postprocess -> rollout dumps).
        agent_output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "stephint_prefix_segments": (
                    int(kwargs.get("stephint_prefix_segments") or 0) if prefix_ids else 0
                ),
                "stephint_prefix_tokens": len(prefix_ids),
                "stephint_prefix_dropped": prefix_dropped,
            }
        )

        return agent_output
