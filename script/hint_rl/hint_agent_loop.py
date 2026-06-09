# Copyright 2026
#
# HintAgentLoop -- the ``<hint_call/>`` rollout mechanism for HPRL.
#
# Unlike the hermes tool path (ToolAgentLoop + request_hint as a verl tool), the
# policy here requests a hint by emitting the sentinel ``<hint_call/>`` (taught by
# the system prompt in hint_prompt.TOOL_INSTRUCTION). This loop:
#
#   * generates each assistant turn that STILL ENDS ON EOS -- we do NOT add the
#     sentinel (or any tool stop token) as a stop string; generation terminates
#     naturally and we inspect the finished turn afterward;
#   * DETECTS ``<hint_call/>`` in the decoded turn (this is the hint-call
#     detection). If present and budget remains, it asks the frozen selector for
#     a hint and injects it as the next USER message, then continues;
#   * enforces the per-problem budget B_q (from tools_kwargs, kept current by the
#     dynamic-budget dataset) and records each applied hint into
#     ``extra_fields["applied_hints"]`` -- the SAME state key HintTool used, so
#     the reward manager + budget ratchet are unchanged.
#
# Registered as agent_name "hint_agent" (also via hint_agent_config.yaml, which is
# what verl loads at AgentLoopWorker init). No verl core edits.

from __future__ import annotations

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import (
    SPEC_DECODE_EXTRA_KEYS,
    AgentState,
    ToolAgentLoop,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from hint_penalty import STRATEGY_MAJOR_STEP, normalize_strategy
from hint_prompt import render_remaining_calls
from hint_selector import (
    HintSelector,
    build_trace,
    exclude_applied_hints,
    exclude_applied_steps,
    format_step_hints,
    hint_id_of,
    hints_for_step,
    step_id_of,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The sentinel the policy emits (on its own line) to request a hint.
HINT_SENTINEL = "<hint_call/>"
# tools_kwargs key carrying this problem's create_kwargs (problem/hints/budget).
HINT_KWARGS_KEY = "request_hint"


@register("hint_agent")
class HintAgentLoop(ToolAgentLoop):
    """Agent loop whose hint trigger is the ``<hint_call/>`` sentinel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_sentinel = HINT_SENTINEL
        self.hint_kwargs_key = HINT_KWARGS_KEY
        # HPRL knobs (single source of truth: data.hprl, available as data_config).
        hprl = (self.data_config.get("hprl", {}) or {}) if self.data_config is not None else {}
        self.default_budget = int(
            hprl.get("default_budget", os.environ.get("HPRL_DEFAULT_BUDGET", self.max_assistant_turns or 8))
        )
        # When the policy emits <hint_call/> after the budget is spent:
        #   True  -> terminate the rollout;
        #   False -> inject a "no hints remaining, please finish" user message.
        self.terminate_on_budget_exceeded = bool(hprl.get("terminate_on_budget_exceeded", False))
        # Hint-selection + penalty strategy (single source of truth: data.hprl.strategy,
        # env HPRL_HINT_STRATEGY as a fallback). Must match the reward's hint_strategy.
        #   STRATEGY_HINT       -- inject ONE selector-chosen hint, exclude that hint
        #                          next call (per-hint penalty).
        #   STRATEGY_MAJOR_STEP -- inject ALL hints of the identified major step,
        #                          record + exclude the WHOLE step next call (step penalty).
        self.hint_strategy = normalize_strategy(
            hprl.get("strategy", os.environ.get("HPRL_HINT_STRATEGY", "hint"))
        )
        logger.warning("HintAgentLoop: hint strategy = %s", self.hint_strategy)
        self._selector = HintSelector.from_env()

    # NOTE: run() is intentionally NOT overridden. The inherited ToolAgentLoop.run
    # drives the state machine and dispatches to our overridden _handle_* methods
    # via normal method resolution -- re-wrapping it would double-apply the
    # @rollout_trace_op span.

    # ------------------------------------------------------------------ #
    # generation + hint-call DETECTION
    # ------------------------------------------------------------------ #
    async def _handle_generating_state(
        self, agent_data, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Generate one assistant turn (terminated by EOS) and decide what's next.

        This mirrors ToolAgentLoop._handle_generating_state EXCEPT:
          (1) it does NOT inject the tool-parser stop tokens -- the turn ends on
              EOS only (per the design: the generation/step token is unchanged);
          (2) the continuation decision is driven by detecting ``<hint_call/>`` in
              the finished turn instead of parsing hermes tool calls.
        """
        # (1) EOS only: intentionally NOT adding self.tool_parser.stop_token_ids.
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )

        # --- bookkeeping (faithful to the base loop) ----------------------
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        # Ensure EVERY rollout carries applied_hints from its first turn -- even
        # one that never emits <hint_call/>. Otherwise the key is absent for
        # hint-free rollouts, and DataProto.concat across agent-loop workers
        # asserts ("Key 'applied_hints' is not present ...") when some workers
        # have it and others don't. Set it AFTER the extra_fields bookkeeping
        # above so the first-turn `update(output.extra_fields)` is not skipped.
        agent_data.extra_fields.setdefault("applied_hints", [])
        # Per-rollout count of hint calls whose selector lookup FAILED (returned no
        # hint -> the policy got the no-op fallback). Initialized on every rollout
        # for the same DataProto.concat reason as applied_hints. Surfaced through
        # the reward fn and aggregated by hint_budget_callback so a selector outage
        # is VISIBLE in hprl/* metrics instead of silently degrading every call.
        agent_data.extra_fields.setdefault("hint_call_failed", 0)

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # --- hard caps (same as base) -------------------------------------
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # (2) hint-call DETECTION on the finished (EOS-terminated) turn.
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        if self.hint_sentinel in text:
            return AgentState.PROCESSING_TOOLS  # reuse this state slot for hint injection
        return AgentState.TERMINATED

    # ------------------------------------------------------------------ #
    # hint selection + inject as a USER message
    # ------------------------------------------------------------------ #
    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        """Select a hint (or a whole major step) and inject it as the next user turn.

        The injected user turn is masked 0 (untrained). The granularity depends on
        ``self.hint_strategy``:
          * ``STRATEGY_HINT``       -- one selector-chosen hint per call;
          * ``STRATEGY_MAJOR_STEP`` -- every hint of the selector-identified major
            step at once, with the whole step recorded + excluded next call.
        """
        agent_data.tool_calls = []  # not using hermes tools

        create_kwargs = {}
        tk = agent_data.tools_kwargs.get(self.hint_kwargs_key) if agent_data.tools_kwargs else None
        if isinstance(tk, dict):
            create_kwargs = tk.get("create_kwargs", {}) or {}
        budget = int(create_kwargs.get("budget", self.default_budget))

        applied = agent_data.extra_fields.setdefault("applied_hints", [])
        major_step_mode = self.hint_strategy == STRATEGY_MAJOR_STEP

        if len(applied) >= budget:
            # Budget spent. terminate_on_budget_exceeded decides what happens when
            # the policy still emits <hint_call/>:
            agent_data.metrics["hint_budget_exhausted"] = agent_data.metrics.get("hint_budget_exhausted", 0) + 1
            if self.terminate_on_budget_exceeded:
                return AgentState.TERMINATED
            # else: nudge the model to finish rather than inject a hint.
            hint_message = {
                "role": "user",
                "content": (
                    f"You have used all {budget} hint(s) for this problem. "
                    "No more hints are available; please finish your solution."
                ),
            }
        else:
            problem = create_kwargs.get("problem", "")
            hints_obj = create_kwargs.get("hints", "")
            # Exclude what's already been surfaced this rollout so the selector
            # cannot re-offer it: individual hints (hint strategy) or whole major
            # steps (major_step strategy). It must pick from the remaining candidates.
            if major_step_mode:
                hints_str = exclude_applied_steps(hints_obj, applied)
            else:
                hints_str = exclude_applied_hints(hints_obj, applied)
            trace = build_trace(agent_data.messages)

            # Time the (blocking, on-critical-path) call to the frozen gpt-oss
            # selector. simple_timer accumulates (+=) across every hint call in
            # this rollout -> per-sample TOTAL selector latency. Surfaced by
            # AgentLoopManager._performance_metrics as timing_s/agent_loop/
            # hint_select/{min,max,mean} (+ slowest/hint_select) in wandb.
            with simple_timer("hint_select", agent_data.metrics):
                selection, _raw, err = await self._selector.select(problem, trace, hints_str)
            if selection is None:
                logger.warning("hint selection failed (request=%s): %s", agent_data.request_id, err)
                agent_data.extra_fields["hint_call_failed"] = (
                    agent_data.extra_fields.get("hint_call_failed", 0) + 1
                )
                agent_data.metrics["hint_call_failed"] = agent_data.metrics.get("hint_call_failed", 0) + 1
                hint_message = {
                    "role": "user",
                    "content": "No hint is available right now; continue reasoning on your own.",
                }
            else:
                if major_step_mode:
                    content = self._record_major_step(selection, hints_obj, applied, budget, agent_data)
                else:
                    content = self._record_single_hint(selection, applied, budget, agent_data)
                hint_message = {"role": "user", "content": content}

        # inject the hint as a user turn, mirroring the base tool-response path.
        agent_data.messages.append(hint_message)
        response_ids = await self.apply_chat_template([hint_message], remove_system_prompt=True)

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)  # injected turn is not trained
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    # ------------------------------------------------------------------ #
    # per-strategy recording (mutates `applied` + metrics, returns the
    # user-message body that gets injected as the hint turn)
    # ------------------------------------------------------------------ #
    def _record_single_hint(self, selection, applied, budget, agent_data) -> str:
        """STRATEGY_HINT: record one selector-chosen hint; return the message body."""
        hint_text = (selection.get("hint") or "").strip()
        applied.append(
            {
                "call_index": len(applied),
                "hint_id": hint_id_of(selection),
                "major_step_id": selection.get("major_step_id"),
                "hint": hint_text,
                "confidence_of_hint": selection.get("confidence_of_hint"),
            }
        )
        agent_data.metrics["hint_call"] = agent_data.metrics.get("hint_call", 0) + 1
        # The remaining-calls notice lets the policy ration its budget; it is kept
        # OUT of applied[...]["hint"] so the recorded hint text (reward/penalty/
        # logging) stays the bare hint.
        body = hint_text or "(the selector returned an empty hint)"
        notice = render_remaining_calls(budget - len(applied))
        return f"{body}\n\n{notice}"

    def _record_major_step(self, selection, hints_obj, applied, budget, agent_data) -> str:
        """STRATEGY_MAJOR_STEP: reveal ALL hints of the selector-identified major step.

        Records the WHOLE step (by ``major_step_id``) into the rollout state so the
        next hint call excludes it (``exclude_applied_steps``) and the reward charges
        its step penalty (``applied_step_penalty``).
        """
        step_id = step_id_of(selection)
        step_hints = hints_for_step(hints_obj, step_id) if step_id is not None else []
        body = format_step_hints(step_hints)
        if not body:
            # The selector named a step but the pool had no hints for it (shouldn't
            # happen): fall back to its single rephrased hint so the call still
            # surfaces something. The step is still recorded below -> excluded next
            # call and penalized once.
            body = (selection.get("hint") or "").strip() or "(no hints available for this step)"
        applied.append(
            {
                "call_index": len(applied),
                "major_step_id": step_id,
                "hint_id": hint_id_of(selection),
                "hint_ids": [h.get("hint_id") for h in step_hints if isinstance(h, dict)],
                "hint": body,
                "confidence_of_major_step": selection.get("confidence_of_major_step"),
            }
        )
        agent_data.metrics["hint_call"] = agent_data.metrics.get("hint_call", 0) + 1
        agent_data.metrics["hint_step_revealed"] = agent_data.metrics.get("hint_step_revealed", 0) + 1
        notice = render_remaining_calls(budget - len(applied))
        lead = "Here are all the hints for the step you should focus on next:"
        return f"{lead}\n{body}\n\n{notice}"
