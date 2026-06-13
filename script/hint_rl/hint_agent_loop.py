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
#   * DETECTS a hint call ONLY when the finished turn ENDS with ``<hint_call/>``
#     alone on its own line (the taught "emit it on its own line and stop"
#     format). Inline / mid-reasoning sentinels are NOT requests -- a bare
#     substring match let the policy learn to spam the sentinel. If a real hint
#     call is seen and budget remains, it asks the frozen selector for a hint and
#     injects it as the next USER message, then continues;
#   * enforces the per-problem budget B_q (from tools_kwargs, kept current by the
#     dynamic-budget dataset) and records each applied hint into
#     ``extra_fields["applied_hints"]`` -- the SAME state key HintTool used, so
#     the reward manager + budget ratchet are unchanged.
#
# Registered as agent_name "hint_agent" (also via hint_agent_config.yaml, which is
# what verl loads at AgentLoopWorker init). No verl core edits.

from __future__ import annotations

import json
import logging
import os
import socket
import time
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

# --- DEBUG: per-selector-call dump ------------------------------------------ #
# Set HPRL_SELECTOR_DUMP_DIR to a writable dir to record EVERY hint-selection
# call -- the exact `trace` build_trace produced (the selector's whole view of
# student progress), the candidate steps, and the selector's parsed + raw output.
# This is how we VERIFY on a live run whether build_trace actually carries the
# student's reasoning or only the injected hints (the blind-trace path). Off
# (no-op) unless the env var is set, so production runs are unaffected. Each
# worker PROCESS writes its own file (host+pid) to avoid cross-process append
# races; concatenate them for analysis.
_SELECTOR_DUMP_DIR = os.environ.get("HPRL_SELECTOR_DUMP_DIR", "").strip()


def _dump_selector_call(record: dict) -> None:
    """Append one selector-call record as a JSON line to this process's dump file.

    No-op unless ``HPRL_SELECTOR_DUMP_DIR`` is set. Best-effort and fully guarded:
    a debug dump must NEVER break a rollout, so every error is swallowed.
    """
    if not _SELECTOR_DUMP_DIR:
        return
    try:
        os.makedirs(_SELECTOR_DUMP_DIR, exist_ok=True)
        path = os.path.join(
            _SELECTOR_DUMP_DIR, f"selector_calls.{socket.gethostname()}.{os.getpid()}.jsonl"
        )
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 -- never let debug dumping break training
        logger.warning("selector-call dump failed", exc_info=True)


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
        # NOTE: an over-budget <hint_call/> is ALWAYS a terminating protocol
        # violation now (see _handle_processing_tools_state) -- the rollout is
        # flagged for the floor reward and stopped immediately. The former
        # terminate_on_budget_exceeded knob (terminate vs "nudge the model to
        # finish") is retired: there is no longer a free chance to finish after an
        # illegal call. The config key is left in place for back-compat but no
        # longer changes behavior.
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
          (2) the continuation decision is driven by detecting a hint call --
              ``<hint_call/>`` alone on the finished turn's LAST line (see
              ``_is_hint_call``) -- instead of parsing hermes tool calls.
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
        # Per-rollout flag: set to 1 when the policy emits <hint_call/> AFTER its
        # budget is spent (an over-budget call). The reward short-circuits to the
        # floor score on this flag (hint_reward.compute_score) and the answer is NOT
        # graded. Initialized here for the same DataProto.concat reason as
        # applied_hints -- it must be present on every rollout, including those that
        # never go over budget.
        agent_data.extra_fields.setdefault("hint_budget_exceeded", 0)
        # Per-rollout selector latency: total seconds spent in the frozen selector
        # (sum over this rollout's hint calls) and the number of those calls. The
        # verl-native hint_select timer is dropped before aggregation (AgentLoopMetrics
        # has no field for it), so these extra_fields are the RELIABLE selector-cost
        # signal -- surfaced by the reward and averaged in hint_budget_callback to
        # hprl/hint_select_time_*. Initialized for the DataProto.concat reason as
        # applied_hints (present on every rollout, incl. hint-free ones -> 0.0/0).
        agent_data.extra_fields.setdefault("hint_select_time", 0.0)
        agent_data.extra_fields.setdefault("hint_select_calls", 0)
        # Box-then-call signal: number of hint calls this rollout made (every
        # <hint_call/> detected, served + over-budget terminal), and how many of
        # those were emitted by a turn that ALREADY contained a \boxed{...}. The
        # reward passes both through; hint_budget_callback derives the per-call rate
        # (with_box / total) and the per-rollout rate (rollouts with >=1 box-then-call).
        # Initialized for the same DataProto.concat reason as applied_hints.
        agent_data.extra_fields.setdefault("hint_calls_total", 0)
        agent_data.extra_fields.setdefault("hint_calls_with_box", 0)
        # Per-turn reasoning-token lengths for the reward's ORDER-AWARE effort-
        # shaping term (hint_reward.effort_shortfall_sum). Each turn is scored
        # against the SUFFIX MAXIMUM of this (ordered) list -- a turn is penalized
        # only to the extent that a LATER turn out-reasoned it, so rising/front-
        # loaded reasoning is penalized while reasoning hard early is free. ORDER
        # MATTERS: appended in chronological turn order, for EVERY assistant turn
        # (final solve turn + hint-free rollouts included). Initialized here for
        # the same DataProto.concat reason as applied_hints.
        agent_data.extra_fields.setdefault("turn_lens", [])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        # length of THIS assistant turn (== reasoning since the previous hint /
        # start, since each turn restarts after an injected hint user-message).
        agent_data.extra_fields["turn_lens"].append(len(output.token_ids))
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
        # A hint call is recognized ONLY when the turn ENDS with the sentinel on
        # its own line -- i.e. the policy emitted ``<hint_call/>`` alone on the
        # final line and then stopped (EOS). A plain substring match treated any
        # inline ``<hint_call/>`` buried mid-reasoning as a request, which the
        # policy learns to spam (occurrences/turn blow up late in training); the
        # taught format is "emit it alone on its own line and stop", so we key off
        # exactly that and let everything else terminate as a normal answer turn.
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)

        # BLIND-TRACE FIX: record this assistant turn in agent_data.messages so the
        # selector's build_trace(agent_data.messages) actually sees the student's
        # reasoning. The base loop tracks turns ONLY as tokens (prompt_ids); without
        # this, messages stays [system, problem] + injected hints, so build_trace
        # returns a hints-only trace and the selector picks steps blind to all
        # student work (verified: n_assistant_in_messages==0 on 285/285 live calls).
        # Safe: agent_data.messages feeds ONLY build_trace after startup -- the
        # trained sequence is built from prompt_ids/response_mask, and the initial
        # prompt is templated once in _handle_pending_state before any turn lands
        # here. Appended for EVERY assistant turn (incl. the one ending in
        # <hint_call/>, whose progress summary is exactly what the selector needs).
        agent_data.messages.append({"role": "assistant", "content": text})

        if self._is_hint_call(text):
            # Count the call and whether its emitting turn already boxed an answer
            # (the box-then-call pattern). ``text`` is THIS turn only, so ``\boxed{``
            # here means the model committed an answer and then asked for a hint in
            # the same turn. Counts every detected call -- served, selector-failed, and
            # the over-budget terminal one (all route through here before the budget
            # check in _handle_processing_tools_state).
            agent_data.extra_fields["hint_calls_total"] = (
                agent_data.extra_fields.get("hint_calls_total", 0) + 1
            )
            if "\\boxed{" in text:
                agent_data.extra_fields["hint_calls_with_box"] = (
                    agent_data.extra_fields.get("hint_calls_with_box", 0) + 1
                )
            return AgentState.PROCESSING_TOOLS  # reuse this state slot for hint injection
        return AgentState.TERMINATED

    def _is_hint_call(self, text: str) -> bool:
        """True iff the finished turn ends with the sentinel alone on its own line.

        Requires both signals the format teaches: the sentinel is on its OWN line
        (nothing but whitespace around it) and it is the LAST thing the policy
        emitted before stopping (EOS). Trailing whitespace/newlines are ignored;
        an inline or non-terminal ``<hint_call/>`` is NOT a hint call.
        """
        stripped = text.rstrip()
        last_nl = stripped.rfind("\n")
        last_line = stripped[last_nl + 1 :] if last_nl != -1 else stripped
        return last_line.strip() == self.hint_sentinel

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
            # Budget spent and the policy STILL emitted <hint_call/>: a hard protocol
            # violation (asking for help it cannot have). Flag the rollout so the
            # reward short-circuits to the floor score -- hint_reward.compute_score
            # reads extra_info["hint_budget_exceeded"] and returns budget_exceeded_reward
            # with acc=0, NOT grading the boxed answer (even a correct box earns nothing
            # if the rollout ended on an illegal call) -- and terminate immediately.
            # UNCONDITIONAL: an over-budget call always terminates + zeroes; the old
            # "nudge the model to finish" fallback is retired.
            agent_data.metrics["hint_budget_exhausted"] = agent_data.metrics.get("hint_budget_exhausted", 0) + 1
            agent_data.extra_fields["hint_budget_exceeded"] = 1
            return AgentState.TERMINATED
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
            call_index = len(applied)  # index of THIS hint call (before recording)

            # Time the (blocking, on-critical-path) call to the frozen gpt-oss
            # selector. Two timers, by necessity:
            #   * simple_timer writes agent_data.metrics["hint_select"], the verl-
            #     native path -> timing_s/agent_loop/hint_select/*. This is CURRENTLY
            #     DROPPED: AgentLoopMetrics (a pydantic model) has no hint_select field,
            #     so model_dump() strips it before _performance_metrics aggregates ->
            #     the wandb metric reads a flat 0.0. Kept so it starts working for free
            #     if verl ever adds the field.
            #   * perf_counter -> extra_fields["hint_select_time"] (accumulated +=
            #     across every call this rollout). extra_fields DOES survive to
            #     non_tensor_batch (same path as hint_call_failed), so this is the
            #     RELIABLE per-rollout selector latency the reward surfaces and
            #     hint_budget_callback aggregates to hprl/hint_select_time_*.
            _t0 = time.perf_counter()
            with simple_timer("hint_select", agent_data.metrics):
                selection, _raw, err = await self._selector.select(problem, trace, hints_str)
            select_latency_s = time.perf_counter() - _t0
            agent_data.extra_fields["hint_select_time"] = (
                agent_data.extra_fields.get("hint_select_time", 0.0) + select_latency_s
            )
            agent_data.extra_fields["hint_select_calls"] = (
                agent_data.extra_fields.get("hint_select_calls", 0) + 1
            )
            # Guard on dict, not just None: no selector output shape may crash the
            # rollout (a stray JSON array once killed a whole run via .get on a list).
            if not isinstance(selection, dict):
                logger.warning(
                    "hint selection failed (request=%s): %s",
                    agent_data.request_id,
                    err if selection is None else f"non-dict selection: {type(selection).__name__}",
                )
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

            # DEBUG: record this selector call (trace IN / selection OUT) so the
            # build_trace path can be verified on a live run. agent_data.messages
            # is captured BEFORE the hint turn below is appended -- i.e. exactly
            # what the selector saw. No-op unless HPRL_SELECTOR_DUMP_DIR is set;
            # covers both successful and failed (non-dict) selections.
            _dump_selector_call(
                {
                    "request_id": getattr(agent_data, "request_id", None),
                    "call_index": call_index,
                    "strategy": self.hint_strategy,
                    "budget": budget,
                    # per-call selector round-trip latency (s); summed across calls
                    # into extra_fields["hint_select_time"] for the wandb metric.
                    "select_latency_s": round(select_latency_s, 4),
                    "applied_step_ids_before": [h.get("major_step_id") for h in applied[:call_index]],
                    "n_assistant_in_messages": sum(
                        1 for m in agent_data.messages if m.get("role") == "assistant"
                    ),
                    "msg_roles": [m.get("role") for m in agent_data.messages],
                    "messages": [
                        {
                            "role": m.get("role"),
                            "content": (
                                f"<system prompt: {len(m.get('content') or '')} chars omitted>"
                                if m.get("role") == "system"
                                else m.get("content")
                            ),
                        }
                        for m in agent_data.messages
                    ],
                    "trace": trace,
                    "candidate_hints_str": hints_str,
                    "selection": selection if isinstance(selection, dict) else None,
                    "selector_raw": _raw,
                    "selector_err": err,
                }
            )

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
                # token length of the turn that emitted this <hint_call/> (==
                # reasoning since the previous hint). The reward's effort-shaping
                # term scores this against the rollout's mean turn length.
                "reasoning_len": len(agent_data.response_ids),
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
                # token length of the turn that emitted this <hint_call/> (==
                # reasoning since the previous hint). The reward's effort-shaping
                # term scores this against the rollout's mean turn length.
                "reasoning_len": len(agent_data.response_ids),
            }
        )
        agent_data.metrics["hint_call"] = agent_data.metrics.get("hint_call", 0) + 1
        agent_data.metrics["hint_step_revealed"] = agent_data.metrics.get("hint_step_revealed", 0) + 1
        notice = render_remaining_calls(budget - len(applied))
        lead = "Here are all the hints for the step you should focus on next:"
        return f"{lead}\n{body}\n\n{notice}"
