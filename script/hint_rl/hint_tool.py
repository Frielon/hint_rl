# Copyright 2026
#
# HintTool -- the hint-tool used by Hint Penalized RL (HPRL).
#
# This is a stateful verl ``BaseTool`` (it subclasses BaseTool and uses the
# full create/execute/release lifecycle) that the policy calls during a
# multi-turn rollout. On each call the tool:
#
#   1. Reads the per-sample hint pool / problem / ground-truth / budget that the
#      dataset injected via ``extra_info.tools_kwargs.<tool>.create_kwargs``.
#   2. Builds the student's current reasoning *trace* from the live trajectory
#      (``agent_data.messages``).
#   3. Routes (problem, trace, hint-pool) to a frozen, off-policy **selector
#      model** served on an OpenAI-compatible endpoint, reusing the exact
#      selector prompt + output parser the offline selector eval uses.
#   4. Records the selected hint into a per-rollout **state** kept on
#      ``agent_data.extra_fields["applied_hints"]`` -- the full list of hints
#      applied so far in this rollout -- and surfaces only the hint text back to
#      the policy.
#   5. Enforces the per-problem call-count budget ``B_q``: once the policy has
#      called the tool ``B_q`` times, further calls return a no-op response.
#
# The per-rollout ``applied_hints`` state flows out of the agent loop as a tool
# extra-field and is merged into ``extra_info`` by the reward manager
# (hint_reward_manager.HintRewardManager), where the trajectory reward applies
# the (currently-blank) hint penalty -- see hint_reward.py.
#
# verl creates a *fresh* BaseTool instance lifecycle (create -> execute ->
# release) on *every* tool call, so cross-call state can NOT live on the tool
# instance; it lives on ``agent_data`` (one object per rollout), which is the
# documented place for "tool session data".

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# --------------------------------------------------------------------------- #
# Reuse the offline selector prompt + parser verbatim.
#
# These live under ``${HINT_RL_HOME}/selector``. We add that directory to
# sys.path (derived from HINT_RL_HOME, with a fallback relative to this file)
# so the rollout uses the *same* prompt template and the *same* tolerant
# <output> JSON parser the offline selector eval validated against.
# --------------------------------------------------------------------------- #
def _add_selector_to_path() -> None:
    candidates = []
    hint_rl_home = os.environ.get("HINT_RL_HOME")
    if hint_rl_home:
        candidates.append(os.path.join(hint_rl_home, "selector"))
    # fallback: this file is at <HINT_RL_HOME>/script/hint_rl/hint_tool.py
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "selector")))
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


_add_selector_to_path()
from seletor_prompt import selector_prompt  # noqa: E402
from run_hint_selection_model import hint_id_of, parse_output  # noqa: E402


class HintTool(BaseTool):
    """Frozen-selector-backed hint tool for HPRL.

    Config keys (from the ``config:`` block of the tool YAML):
        selector_base_url:  OpenAI-compatible endpoint of the served selector
                            model (e.g. "http://localhost:30000/v1").
        selector_model:     served-model-name on that endpoint.
        selector_api_key:   API key (sglang/vLLM accept any non-empty string).
        temperature/top_p/max_tokens: selector sampling params.
        request_timeout_s:  per-call HTTP timeout.
        max_retries:        retries on selector call / parse failure.
        default_budget:     fallback call-count budget when a sample does not
                            carry one in create_kwargs.

    Per-sample data (from ``create_kwargs``, injected by prepare_hint_data.py):
        problem:        the problem statement string.
        hints:          the hint pool, as a JSON string (the dataset's
                        ``extra_info.hint``) or an already-parsed object.
        ground_truth:   gold answer (unused by selection; kept for parity / hooks).
        budget:         this sample's call-count budget B_q.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.selector_base_url = config.get("selector_base_url", "http://localhost:30000/v1")
        self.selector_model = config.get("selector_model", "Qwen3.5-27B")
        self.selector_api_key = config.get("selector_api_key", "EMPTY")
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.95))
        self.max_tokens = int(config.get("max_tokens", 16000))
        self.request_timeout_s = float(config.get("request_timeout_s", 600.0))
        self.max_retries = int(config.get("max_retries", 3))
        self.default_budget = int(config.get("default_budget", 8))
        # Per-instance create_kwargs store. Each tool call gets its own
        # instance_id (create is called per call), so this only ever holds the
        # data for the single in-flight call.
        self._instances: dict[str, dict] = {}
        self._client = None  # lazily created AsyncOpenAI client

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def create(
        self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = uuid4().hex
        self._instances[instance_id] = dict(create_kwargs or {})
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        create_kwargs = self._instances.get(instance_id, {})
        agent_data = kwargs.get("agent_data", None)

        # ---- per-rollout state: the list of all applied hints ------------
        state = self._rollout_state(agent_data)
        applied = state["applied_hints"]
        budget = int(create_kwargs.get("budget", self.default_budget))

        # ---- budget enforcement: no-op once the budget is spent ----------
        if len(applied) >= budget:
            msg = (
                f"Hint budget exhausted: you have already used your {budget} "
                f"hint call(s) for this problem. Continue solving on your own."
            )
            return ToolResponse(text=msg), 0.0, {"hint_budget_exhausted": 1}

        # ---- build inputs for the selector model -------------------------
        problem = create_kwargs.get("problem", "")
        hints_obj = create_kwargs.get("hints", "")
        hints_str = hints_obj if isinstance(hints_obj, str) else json.dumps(hints_obj, ensure_ascii=False)
        trace = self._build_trace(agent_data, applied)

        prompt = selector_prompt(problem, trace, hints_str)

        # ---- call the frozen selector model ------------------------------
        selection, raw, err = await self._call_selector(prompt)

        if selection is None:
            # selection failed; do NOT consume the budget for a failed call,
            # just tell the model the hint is unavailable.
            logger.warning("hint selection failed (instance=%s): %s", instance_id, err)
            return (
                ToolResponse(text="No hint is available right now; continue reasoning on your own."),
                0.0,
                {"hint_selection_failed": 1},
            )

        hint_text = (selection.get("hint") or "").strip()
        hint_id = hint_id_of(selection)

        # ---- record the applied hint into the per-rollout state ----------
        applied.append(
            {
                "call_index": len(applied),
                "hint_id": hint_id,
                "major_step_id": selection.get("major_step_id"),
                "hint": hint_text,
                "confidence_of_hint": selection.get("confidence_of_hint"),
            }
        )

        response_text = f"Hint: {hint_text}" if hint_text else "Hint: (the selector returned an empty hint)"
        metrics = {
            "hint_call": 1,
            "hint_id": hint_id,
            "hints_used": len(applied),
            "budget": budget,
        }
        # tool step-reward is 0.0; the hint penalty is multiplicative on the
        # final accuracy reward and is applied in the reward function instead.
        return ToolResponse(text=response_text), 0.0, metrics

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rollout_state(agent_data) -> dict:
        """Return (creating if needed) this rollout's hint state.

        Stored on ``agent_data.extra_fields["applied_hints"]`` so it (a) persists
        across the per-call create/release lifecycle and (b) is carried out of
        the agent loop into the batch, where the reward manager reads it.
        """
        if agent_data is None:
            # No trajectory object (should not happen in the agent loop); fall
            # back to an ephemeral state so execute() still works.
            return {"applied_hints": []}
        ef = agent_data.extra_fields
        if "applied_hints" not in ef:
            ef["applied_hints"] = []
        return {"applied_hints": ef["applied_hints"]}

    @staticmethod
    def _build_trace(agent_data, applied: list[dict]) -> str:
        """Render the student's reasoning trace for the selector.

        We feed the selector the assistant-generated reasoning so far (the
        student's work), with any already-surfaced hints inlined where they were
        injected, so the selector can see what guidance the student already has.
        """
        if agent_data is None:
            return ""
        parts: list[str] = []
        for m in agent_data.messages:
            role = m.get("role")
            content = m.get("content")
            if not content:
                continue
            if role == "assistant":
                parts.append(str(content))
            elif role == "tool":
                # a previously surfaced hint
                parts.append(f"[hint given] {content}")
        trace = "\n\n".join(parts).strip()
        if not trace:
            trace = "(The student has not written any reasoning yet.)"
        return trace

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.selector_base_url, api_key=self.selector_api_key)
        return self._client

    async def _call_selector(self, prompt: str):
        """Call the selector endpoint, parse <output>. Returns (selection, raw, err)."""
        client = self._get_client()
        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.selector_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.request_timeout_s,
                )
                raw = resp.choices[0].message.content or ""
                selection, perr = parse_output(raw)
                if selection is not None:
                    return selection, raw, None
                last_err = f"parse failed: {perr}"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
        return None, None, last_err
