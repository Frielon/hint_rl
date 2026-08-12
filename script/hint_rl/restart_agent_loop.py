# Copyright 2026
#
# RestartHintAgentLoop -- TERMINATE-AND-RESTART hint delivery for HPRL
# ("segment-chain" mode; devlog 2026-08-04).
#
# Same science as the AUTO-HINT (push-hint) loop -- the policy is never told
# about hints; the LOOP grades each attempt and decides when to hint -- but the
# hint is DELIVERED differently. AutoHintAgentLoop appends the hint as the next
# user turn of one ever-growing conversation, so a budget-k problem accumulates
# a 20-40k-token context whose length drives the known pathologies (post-hint
# effort collapse, depth-ramping verbatim replay, KV pressure). This loop
# instead ENDS the failed attempt and starts a FRESH single-turn rollout:
#
#   1. generate one attempt from the current segment prompt (EOS-terminated);
#   2. no boxed answer / per-turn cap hit -> terminate through the same
#      floor-reward path as the auto-hint loop (chain over);
#   3. correct -> terminate (the chain's final segment);
#   4. wrong + budget remains -> ask the multi-round selector for the next hint,
#      then TERMINATE THIS SEGMENT and start a new one whose prompt is
#          [system (unchanged)]
#          [user: original problem
#                 + "You have attempted this problem before and you made the
#                    following verified progress, which is correct: ..." (the v2
#                    recap list; a progress-FREE restart carries JUST the hint
#                    block, no preamble)
#                 + the new hint]
#      (selector_multi.format_restart_hint_message). Each segment's context is
#      prompt + ONE attempt -- never the accumulated transcript. POOL WORDING
#      (HPRL_RESTART_POOL_WORDING, default true): recap lines and the new hint
#      use the hint set's ORIGINAL sentences, not the selector's student-notation
#      rephrasings -- those adapt wording to a live trace, and a fresh prompt has
#      no trace in context. false restores selector wording (multi-turn parity).
#   5. stop on correct, budget spent, pool exhausted, or selector failure --
#      exactly the auto-hint loop's terminal semantics, per segment.
#
# WHAT THE TRAINED ROW IS (v0 -- "final segment only"): verl's agent-loop
# contract is one (prompt, response) row per rollout, so the row returned is the
# chain's FINAL segment: prompt = that segment's fresh prompt (recap + hint
# included), response = that segment's attempt only. Earlier segments are
# ARCHIVED into ``extra_fields["restart_segments"]`` (text + states + per-segment
# step_adv_turns/disable_spans; token ids only under HPRL_RESTART_ARCHIVE_IDS)
# -- they reach the rollout dump for analysis but carry NO gradient. Chain-level
# reward is unchanged: ``applied_hints`` accumulates across segments, so
# hint_reward.compute_score prices the whole chain on the final row, and the
# budget ratchet sees one row per chain exactly as before. Expanding archived
# segments into their own training rows (kpack-style batch surgery + state-priced
# advantages) is the documented follow-up, NOT done here.
#
# Per-segment metadata is archived-and-RESET at each restart so the final row's
# ``step_adv_turns`` / ``disable_spans`` / ``turn_lens`` describe only its own
# segment (response-relative coords stay valid). The final successful/terminal
# segment therefore never carries a foreign disable span, and the verified-prefix
# mask (auto_hint_mask) is a structural no-op on restart rows.
#
# STEP-ADV COMPATIBILITY (whole_turn) via PHANTOM fail records. The trainer's
# per-group value fit (step_advantage.compute_state_values) reads each row's
# ``step_adv_turns`` for the chain's failure set H_i (-> F_k) and final state
# (-> D_k). Archiving earlier segments would hide their failures and undercount
# F_k, flattening V. So at every restart the loop keeps a ZERO-TOKEN-SPAN record
# ``[0, 0, 0, ss, se, is_fail=1]`` in the live list: assign_row_advantages paints
# nothing for an empty span, but the record enters fail_states_list/final_states
# -- the fit sees the SAME (H_i, V_i) a multi-turn row would, V is identical, and
# the final segment's whole-turn advantage A = r_se + V[se+1] - V[ss] (fail) /
# V[se] - V[ss] (solve) EQUALS the multi-turn final turn's (the restart chain
# walks the same state trajectory: fail at s_k -> next segment resumes at
# s_{k+1}). v0 thus trains the final segment with the CORRECT state pricing; the
# intermediate turns' advantage terms are simply absent until the
# expand-segments-to-rows follow-up (a subset gradient, not a biased one).
# Verified numerically in test_restart_chain.test_step_adv_value_parity.
#
# ``agent_data.messages`` deliberately KEEPS the full logical transcript
# (assistant attempts + the injected hint blocks) even though prompt_ids are
# rebuilt fresh each segment: messages feed ONLY build_trace / _is_correct /
# the selector dumps, so the selector sees the SAME trace it sees in multi-turn
# mode -- selector behavior is a controlled variable in the restart-vs-multi-turn
# A/B. The trained sequence comes from prompt_ids/response_mask alone.
#
# ACTIVATION -- registry swap, no dataset rebuild, no shared-path edit:
#   AGENT_LOOP_CONFIG_PATH=hint_agent_config_restart.yaml
# maps the dataset's existing agent_name="auto_hint" rows to this class
# (run_auto_hint_qwen3_8b_base_async_restart.sh does this). The default
# hint_agent_config.yaml keeps auto_hint -> AutoHintAgentLoop byte-identical.
#
# Prompt-length: fresh prompts carry problem + recap + hint, so the restart
# launcher raises data.max_prompt_length (2048 -> 4096). A rendered restart
# prompt that would exceed rollout.prompt_length is NOT emitted (verl would
# left-truncate away the system prompt + problem): the chain ends instead, the
# hint is not charged, and ``restart_prompt_overflow`` counts the event.
#
# REFERENCE-PREFIX delivery (HPRL_RESTART_REFERENCE_PREFIX, default off; needs a
# ``hint_reference`` parquet + HintBudgetDataset's create_kwargs plumbing). A
# restart delivers progress + hint as an ASSISTANT-TURN PREFIX instead of the
# recap+hint user message: the fresh prompt is the ORIGINAL [system, user
# (problem)] alone, and the completed hints' reference solutions (pool order) +
# the newly selected hint's reference are glued (selector_multi.
# build_reference_prefix) and prepended to the assistant turn -- the model
# CONTINUES a verified partial solution that ends at the step to do next. The
# prefix tokens live in the RESPONSE region with response_mask=0 (the injected-
# user-turn convention): attended, never trained, never painted -- every turn
# record's span starts at len(response_mask) - len(response_ids) == prefix_len,
# so the turn advantage is computed AS USUAL and lands only on the model-
# generated tokens. Selection, ``applied_hints`` (penalty), ``completed``,
# phantom fail records and the archive flow are IDENTICAL to the user-message
# delivery; ``messages`` records the delivery as an injected user turn
# (format_restart_reference_message) for build_trace/dump parity. The prefix
# consumes response-length budget: a prefix that would leave less than one full
# per-segment cap (max_turn_tokens) of generation room -- or a hint whose
# reference is missing (X.0 guidance, or a parquet without hint_reference) --
# FALLS BACK to the user-message restart for that round (``restart_ref_fallback``
# counts it; the hint is still delivered + charged). train_segments=all archives
# carry the full response region + ``prefix_len`` so restart_expand keeps the
# prefix untrained on segment rows too.

from __future__ import annotations

import logging
import os
import time

from verl.experimental.agent_loop.agent_loop import _agent_loop_registry, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.utils.profiler import simple_timer

from auto_hint_agent_loop import AutoHintAgentLoop, _extract_boxed
from hint_agent_loop import _as_bool, _dump_selector_call, _strip_lone_surrogates
from hint_selector import _is_selector_decline, build_trace, hint_id_of
from selector_multi import (
    build_reference_prefix,
    format_restart_hint_message,
    format_restart_reference_message,
    locate_quote_end,
    parse_reference_texts,
    pending_hint_ids,
    pool_hint_texts,
    resolve_selected_hint,
    update_progress_state,
)
from step_advantage import prefix_state as _prefix_state

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("restart_hint")
class RestartHintAgentLoop(AutoHintAgentLoop):
    """Push-hint rollout that restarts from a fresh recap+hint prompt instead of
    growing one multi-turn context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Knob channel: env vars, forwarded to the agent-loop workers by
        # run_hprl_async.sh's HPRL_RESTART_ENV runtime-env block. (Registry-yaml
        # extra keys reach only the FIRST rollout -- the module-tail registry
        # re-pin keeps only _target_ -- so never put knobs in the yaml.)
        #
        # Archive each segment's exact token ids alongside its text (bloats the
        # rollout dump; needed only by the future expand-segments-to-rows pass).
        self.restart_archive_ids = _as_bool(
            kwargs.get("archive_token_ids", os.environ.get("HPRL_RESTART_ARCHIVE_IDS", "false"))
        )
        # POOL WORDING (default ON in restart mode): deliver the ORIGINAL hint
        # sentences from the hint set -- for both the recap lines and the new
        # hint -- instead of the selector's rephrasings. The selector's
        # student-notation adaptation exists to splice into a live reasoning
        # trace; a fresh restart prompt has no trace in context, so the
        # canonical pool wording is used. Selection/verification is unchanged
        # (the selector still picks the hint and marks completed ones); only the
        # delivered TEXT differs. false -> selector wording, as multi-turn.
        self.restart_pool_wording = _as_bool(
            kwargs.get("pool_wording", os.environ.get("HPRL_RESTART_POOL_WORDING", "true"))
        )
        # TRAIN SEGMENTS ("final" | "all"). "all" = EVERY turn of the chain gets
        # gradient: each archived segment's exact tensors (prompt ids, response
        # ids, sampling logprobs, its step-adv record) ride
        # ``extra_fields["restart_segment_rows"]`` so the trainer
        # (hprl_ray_trainer + restart_expand) can materialize one training row
        # per segment, painted A = r_se + V[se+1] - V[ss] from the group fit.
        # "final" (code default) = v0 final-segment-only. The restart launchers
        # export "all". Key is kept OUT of the rollout dump (raw token ids).
        self.restart_train_segments = str(
            kwargs.get("train_segments", os.environ.get("HPRL_RESTART_TRAIN_SEGMENTS", "final"))
        ).strip().lower()
        # REFERENCE-PREFIX delivery (default OFF; see module header): restarts
        # prepend the completed steps' reference solutions + the new hint's
        # reference to the ASSISTANT turn (response_mask=0, untrained) instead of
        # building a recap+hint user message. Needs create_kwargs["hint_reference"]
        # (a hint-reference parquet + HintBudgetDataset's env-gated plumbing);
        # without it every restart falls back to the user-message delivery.
        self.restart_reference_prefix = _as_bool(
            kwargs.get("reference_prefix", os.environ.get("HPRL_RESTART_REFERENCE_PREFIX", "false"))
        )
        logger.warning(
            "RestartHintAgentLoop: terminate-and-restart (segment-chain) mode ACTIVE "
            "(pool_wording=%s, train_segments=%s, archive_token_ids=%s, step_adv=%s, "
            "reference_prefix=%s). "
            "train_segments=all -> every segment becomes a training row (trainer-side "
            "expansion); final -> v0 final-segment row only. Phantom fail records keep "
            "the group value fit chain-exact either way (see module header).",
            self.restart_pool_wording,
            self.restart_train_segments,
            self.restart_archive_ids,
            self.step_adv_enable,
            self.restart_reference_prefix,
        )

    # ------------------------------------------------------------------ #
    # base-conversation stash + per-rollout key init
    # ------------------------------------------------------------------ #
    async def _handle_pending_state(self, agent_data, sampling_params) -> AgentState:
        state = await super()._handle_pending_state(agent_data, sampling_params)
        # The original [system, user(problem)] conversation, kept for rebuilding
        # every fresh segment prompt. A plain attribute (NOT extra_fields): it
        # must not ride to non_tensor_batch / the dumps.
        agent_data._restart_base_messages = [dict(m) for m in agent_data.messages]
        # Reference-prefix bookkeeping, plain attributes for the same reason:
        # the CURRENT segment's untrained assistant-prefix length (0 = none; the
        # first segment never has one) and the per-rollout {hint_id: reference}
        # cache (parsed lazily from create_kwargs on the first restart).
        agent_data._restart_prefix_len = 0
        agent_data._restart_ref_texts = None
        return state

    async def _handle_generating_state(
        self, agent_data, sampling_params, ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        # Restart-mode keys, present on EVERY rollout (DataProto.concat needs
        # key-consistent extra_fields across all rows of a run -- same rationale
        # as the parent's setdefault block).
        ef = agent_data.extra_fields
        ef.setdefault("restart_segments", [])
        ef.setdefault("n_restarts", 0)
        ef.setdefault("restart_prompt_overflow", 0)
        # Reference-prefix counters, present on every rollout of a restart run
        # (key-consistency again): restarts that had to FALL BACK to the
        # user-message delivery, and the total untrained prefix tokens injected.
        # Both stay 0 when reference_prefix is off.
        ef.setdefault("restart_ref_fallback", 0)
        ef.setdefault("restart_ref_prefix_tokens", 0)
        if self.restart_train_segments == "all":
            ef.setdefault("restart_segment_rows", [])
        return state

    # ------------------------------------------------------------------ #
    # fresh-prompt rendering
    # ------------------------------------------------------------------ #
    def _restart_hint_block(self, hint_text: str, progress_state) -> str:
        """The recap+hint block appended to the problem in a fresh prompt (and
        recorded into ``messages`` as the injected user turn, for trace parity)."""
        return format_restart_hint_message(
            hint_text,
            progress_state,
            include_progress=self.progress_message,
        )

    async def _render_restart_prompt(self, agent_data, block):
        """(messages, prompt_ids) of the next segment's fresh prompt.

        system turn verbatim from the original conversation; user turn = the
        original problem text + the recap/hint block (``block`` None/empty -> the
        BARE problem, the reference-prefix mode's fresh prompt). Tokenized exactly
        like the initial prompt (_handle_pending_state: full template + generation
        tag, through the surrogate-scrubbing apply_chat_template override).
        """
        base = getattr(agent_data, "_restart_base_messages", None) or agent_data.messages
        msgs = []
        user_done = False
        for m in base:
            role = m.get("role")
            if role == "system" and not msgs:
                msgs.append(dict(m))
            elif role == "user" and not user_done:
                user_done = True
                content = str(m.get("content") or "").rstrip()
                msgs.append({"role": "user", "content": content + ("\n\n" + block if block else "")})
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(msgs, tools=schemas)
        return msgs, prompt_ids

    # ------------------------------------------------------------------ #
    # reference-prefix delivery helpers
    # ------------------------------------------------------------------ #
    _ref_missing_warned = False  # class-level: warn ONCE per worker process

    def _reference_texts(self, agent_data) -> dict:
        """{hint_id: reference} for this problem, parsed once per rollout from
        create_kwargs["hint_reference"] (plumbed by HintBudgetDataset when
        HPRL_RESTART_REFERENCE_PREFIX is on). {} when the parquet carries no
        usable field -- every restart then falls back to the user-message
        delivery (warned once per worker, counted per chain)."""
        cached = getattr(agent_data, "_restart_ref_texts", None)
        if cached is not None:
            return cached
        refs = parse_reference_texts(self._create_kwargs(agent_data).get("hint_reference"))
        if not refs and not RestartHintAgentLoop._ref_missing_warned:
            RestartHintAgentLoop._ref_missing_warned = True
            logger.warning(
                "RestartHintAgentLoop: reference_prefix is ON but this problem has no usable "
                "create_kwargs['hint_reference'] (missing field, bad JSON, or a parquet without "
                "references). Affected restarts FALL BACK to the user-message delivery "
                "(restart_ref_fallback counts them). Warned once per worker."
            )
        agent_data._restart_ref_texts = refs
        return refs

    def _reference_prefix_ids(self, prefix_text: str) -> list:
        """Tokenize the assistant-turn prefix (no special tokens -- it continues
        the generation-tagged fresh prompt), surrogate-scrubbed like every other
        injected text (the lone-surrogate tokenizer crash, hint_agent_loop)."""
        return list(self.tokenizer.encode(_strip_lone_surrogates(prefix_text), add_special_tokens=False))

    # ------------------------------------------------------------------ #
    # segment archive
    # ------------------------------------------------------------------ #
    def _archive_segment(
        self, agent_data, turn_text: str, state_start: int, state_end: int,
        next_hint_id, next_block: str, boundary: int,
    ) -> None:
        """Record the finished (failed) segment and RESET the per-segment metadata
        so the eventual final row describes only its own segment.

        The FINAL segment is never archived -- it IS the trained row (its text is
        the dump's ``output``); analysis code reconstructs the full chain as
        restart_segments + the row. ``disable_spans`` here is the segment's OWN
        would-be verified-prefix mask ([boundary, seg_end) in segment-relative
        coords) -- the live extra_fields copy is kept empty by design.

        ``step_adv_turns``: the segment's REAL records move into the archive; the
        zero-span PHANTOM fail records (see module header) stay in the live list
        -- they carry the chain's failure history to the trainer's value fit --
        and the caller appends this segment's own phantom right after.
        """
        ef = agent_data.extra_fields
        seg_end = len(agent_data.response_mask)
        live_turns = ef.get("step_adv_turns", []) or []
        # THIS segment's own untrained reference prefix (reference_prefix mode;
        # 0 in user-message delivery and always on segment 1). Its generated span
        # is [plen, seg_end) -- every record below must exclude the prefix.
        plen = int(getattr(agent_data, "_restart_prefix_len", 0) or 0)

        def _is_phantom(t) -> bool:
            return int(t[0]) == 0 and int(t[1]) == 0 and int(t[2]) == 0

        seg = {
            "seg_index": len(ef.get("restart_segments", [])),
            "n_tokens": len(agent_data.response_ids),
            "prefix_tokens": plen,
            "pred": _extract_boxed(turn_text),
            "state_start": int(state_start),
            "state_end": int(state_end),
            "boundary": int(boundary),
            "next_hint_id": (str(next_hint_id) if next_hint_id is not None else None),
            # the exact user block recorded for the NEXT segment's delivery
            # (recap+hint appended to its prompt, or -- reference_prefix mode --
            # the format_restart_reference_message record of its assistant prefix)
            "next_user_message": next_block,
            "response": turn_text,
            "turn_lens": list(ef.get("turn_lens", []) or []),
            "step_adv_turns": [list(t) for t in live_turns if not _is_phantom(t)],
            "disable_spans": ([[int(boundary), int(seg_end)]] if boundary < seg_end else []),
        }
        if self.restart_archive_ids:
            n_resp = len(agent_data.response_mask)
            seg["prompt_ids"] = list(agent_data.prompt_ids[: len(agent_data.prompt_ids) - n_resp])
            # the full RESPONSE REGION (reference prefix included), so prompt_ids
            # + response_ids reassemble the segment's exact live context. ==
            # agent_data.response_ids whenever there is no prefix.
            seg["response_ids"] = list(agent_data.prompt_ids[len(agent_data.prompt_ids) - n_resp:])
        ef.setdefault("restart_segments", []).append(seg)
        # train_segments=all: the exact tensors the trainer needs to materialize
        # this segment as its OWN training row. Separate key from the (dumped)
        # restart_segments so raw token ids never bloat the rollout jsonl.
        if self.restart_train_segments == "all":
            n_resp = len(agent_data.response_mask)
            ef.setdefault("restart_segment_rows", []).append({
                "prompt_ids": list(agent_data.prompt_ids[: len(agent_data.prompt_ids) - n_resp]),
                # full response region (prefix + generation): the sampling
                # logprobs below align with it (prefix rides 0.0s), and
                # restart_expand re-zeroes the loss mask over prefix_len.
                "response_ids": list(agent_data.prompt_ids[len(agent_data.prompt_ids) - n_resp:]),
                "logprobs": list(agent_data.response_logprobs or []),
                "prefix_len": plen,
                # the segment's step-adv record, segment-relative coords over the
                # GENERATED span [plen, n_resp): an archived segment always
                # FAILED its next hint (that is why the chain restarted),
                # advancing state_start -> state_end.
                "turns": [[plen, int(boundary), n_resp, int(state_start), int(state_end), 1]],
            })
        ef["step_adv_turns"] = [list(t) for t in live_turns if _is_phantom(t)]
        ef["disable_spans"] = []
        ef["turn_lens"] = []

    # ------------------------------------------------------------------ #
    # selector call + terminate-and-restart (replaces the user-turn injection)
    # ------------------------------------------------------------------ #
    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        """Pick the next hint (multi-round selector), then END this segment and
        start a fresh one from a recap+hint prompt.

        Adapted copy of AutoHintAgentLoop._handle_processing_tools_state: the
        selector protocol, bookkeeping, and every TERMINAL branch (pool
        exhausted / selector failed / selector declined / budget-spent label)
        are kept 1:1 -- those all end the chain on the CURRENT segment, which
        then is the trained row. Only the served-hint tail differs: instead of
        appending the hint as a user turn to the live context, the segment is
        archived and prompt_ids / response state are rebuilt from scratch.
        """
        agent_data.tool_calls = []  # not using hermes tools

        ck = self._create_kwargs(agent_data)
        budget = int(ck.get("budget", self.default_budget))
        problem = ck.get("problem", "")
        pool = self._pool(agent_data)
        applied = agent_data.extra_fields.setdefault("applied_hints", [])
        completed = agent_data.extra_fields.setdefault("auto_hint_completed", [])
        hint_order = pending_hint_ids(pool, [])

        # span of the segment's attempt that just ended (wrong) -- response-
        # relative coords are segment-relative here (the mask resets per segment).
        turn_len = len(agent_data.response_ids)
        turn_end = len(agent_data.response_mask)
        turn_start = turn_end - turn_len
        turn_text = agent_data.messages[-1].get("content", "") if agent_data.messages else ""

        # Pool order + the pre-segment state: computed UNCONDITIONALLY (the
        # parent gates them on step_adv) -- the segment archive records them.
        order = hint_order
        pre_state = _prefix_state(order, completed)
        terminal = self.step_adv_enable and len(applied) >= budget

        # Pool exhausted: nothing pending to give -> the chain ends here.
        pending = pending_hint_ids(pool, completed)
        if not pending:
            agent_data.metrics["hint_pool_exhausted"] = agent_data.metrics.get("hint_pool_exhausted", 0) + 1
            agent_data.extra_fields["hint_pool_exhausted"] = (
                agent_data.extra_fields.get("hint_pool_exhausted", 0) + 1
            )
            if self.step_adv_enable:
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_end, turn_end, pre_state, pre_state, is_fail=0
                )
            return AgentState.TERMINATED

        trace = build_trace(agent_data.messages)
        _t0 = time.perf_counter()
        with simple_timer("hint_select", agent_data.metrics):
            selection, _raw, err, _prompt = await self._selector.select_multi(problem, trace, pool, completed)
        select_latency_s = time.perf_counter() - _t0
        agent_data.extra_fields["hint_select_time"] = (
            agent_data.extra_fields.get("hint_select_time", 0.0) + select_latency_s
        )
        agent_data.extra_fields["hint_select_calls"] = (
            agent_data.extra_fields.get("hint_select_calls", 0) + 1
        )

        # Selector failure (no usable dict): the chain ends on this segment.
        if not isinstance(selection, dict):
            agent_data.extra_fields["hint_call_failed"] = (
                agent_data.extra_fields.get("hint_call_failed", 0) + 1
            )
            agent_data.metrics["hint_call_failed"] = agent_data.metrics.get("hint_call_failed", 0) + 1
            logger.warning("restart-hint selection failed (request=%s): %s", agent_data.request_id, err)
            self._dump(agent_data, turn_start, turn_end, problem, trace, completed, None, err,
                       selector_raw=_raw, selector_prompt=_prompt)
            if self.step_adv_enable:
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_end, turn_end, pre_state, pre_state, is_fail=0
                )
            return AgentState.TERMINATED

        # Shape-guard every field we read (stray selector JSON shapes must never
        # crash a rollout -- same guards as the parent).
        _h = selection.get("hint")
        hint_text = _h.strip() if isinstance(_h, str) else ""
        hid = hint_id_of(selection)
        completed_hints = selection.get("completed_hints")
        if not isinstance(completed_hints, list):
            completed_hints = []
        progress_state = agent_data.extra_fields.setdefault("auto_hint_progress", [])
        progress_state = update_progress_state(
            progress_state,
            completed_hints,
            hint_order=hint_order,
        )
        agent_data.extra_fields["auto_hint_progress"] = progress_state

        hid, hint_text, _resolve_flags = resolve_selected_hint(
            hid, hint_text, pool, pending,
            applied_ids=[a.get("hint_id") for a in applied],
        )
        for _flag in _resolve_flags:
            _k = "hint_" + _flag
            agent_data.extra_fields[_k] = agent_data.extra_fields.get(_k, 0) + 1
            agent_data.metrics[_k] = agent_data.metrics.get(_k, 0) + 1

        # Selector DECLINED ("all pending hints already achieved"): benign stop.
        if _is_selector_decline(selection, _raw):
            agent_data.metrics["hint_selector_declined"] = (
                agent_data.metrics.get("hint_selector_declined", 0) + 1
            )
            agent_data.extra_fields["hint_selector_declined"] = (
                agent_data.extra_fields.get("hint_selector_declined", 0) + 1
            )
            if self.step_adv_enable:
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_end, turn_end, pre_state, pre_state, is_fail=0
                )
            self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                       completed_hints=completed_hints,
                       selector_raw=_raw, selector_prompt=_prompt)
            return AgentState.TERMINATED

        # Verified-prefix boundary of THIS segment. Deliberately NOT appended to
        # the live ``disable_spans`` (the parent does): this segment is about to
        # be archived, and the eventual final row must not inherit a foreign
        # span. The archive stores it instead (seg["disable_spans"]).
        boundary = self._verified_boundary(turn_text, completed_hints, turn_start, turn_len)

        # The failed state this segment ends at (parent's step-adv arithmetic,
        # kept unconditionally for the archive record).
        try:
            fail_state = order.index(str(hid))
        except ValueError:
            fail_state = pre_state
        state_end = max(pre_state, fail_state)
        if self.step_adv_enable:
            self._record_step_adv_turn(
                agent_data, turn_start, boundary, turn_end, pre_state, state_end, is_fail=1
            )

        # CITATION FOUND RATE (chain-cumulative, same as the parent).
        for c in completed_hints:
            q = c.get("quote") if isinstance(c, dict) else None
            if isinstance(q, str) and q.strip():
                agent_data.extra_fields["cite_quotes_total"] = (
                    agent_data.extra_fields.get("cite_quotes_total", 0) + 1
                )
                if locate_quote_end(q, trace, self.auto_hint_fuzzy_threshold) is not None:
                    agent_data.extra_fields["cite_quotes_found"] = (
                        agent_data.extra_fields.get("cite_quotes_found", 0) + 1
                    )

        # TERMINAL label (step-adv, budget already spent): no hint, chain over.
        if terminal:
            self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                       boundary=boundary, completed_hints=completed_hints,
                       selector_raw=_raw, selector_prompt=_prompt)
            return AgentState.TERMINATED

        # ---- RESTART tail (replaces the parent's user-turn injection) --------
        # This round's newly self-completed hint ids (needed both for the recap /
        # reference glue below and, later, the ``completed`` status update).
        newly = [
            str(c.get("hint_id"))
            for c in completed_hints
            if isinstance(c, dict) and c.get("hint_id") is not None
        ]
        # The authoritative completed set at THIS delivery (prior ``completed`` +
        # this round's ``newly``, minus the hint being given now) -- drives both
        # the pool-worded recap and the reference-prefix glue, in pool order.
        done_ids = {str(x) for x in completed} | set(newly)
        done_ids.discard(str(hid))

        # POOL WORDING (restart default -- see __init__): the delivered hint and
        # the recap lines use the hint set's ORIGINAL sentences, rebuilt from
        # ``done_ids`` in pool order -- the same ids/order the selector-worded
        # progress_state would carry, just with canonical text. Falls back to the
        # selector's text for an id the pool cannot resolve (should not happen
        # after resolve_selected_hint).
        deliver_text = hint_text
        recap_state = progress_state
        if self.restart_pool_wording:
            pool_texts = pool_hint_texts(pool)
            _pt = pool_texts.get(str(hid)) if hid is not None else None
            if isinstance(_pt, str) and _pt.strip():
                deliver_text = _pt.strip()
            recap_state = [
                {"hint_id": h, "text": str(pool_texts.get(h, "")).strip()}
                for h in hint_order
                if h in done_ids and str(pool_texts.get(h, "")).strip()
            ]

        # REFERENCE-PREFIX delivery (see module header): glue the completed
        # steps' reference solutions + the selected hint's reference and prepend
        # them to the next segment's ASSISTANT turn over the BARE problem prompt.
        # Falls back to the user-message delivery below when the selected hint
        # has no reference (X.0 guidance / a parquet without hint_reference), or
        # the prefix would not leave a full per-segment cap of generation room,
        # or (degenerate) the bare prompt itself no longer fits.
        prefix_ids: list = []
        block = None
        new_msgs = None
        new_prompt_ids = None
        if self.restart_reference_prefix:
            prefix_text = build_reference_prefix(
                hint_order, done_ids, hid, self._reference_texts(agent_data)
            )
            if prefix_text is not None:
                base_msgs, base_prompt_ids = await self._render_restart_prompt(agent_data, None)
                cand_ids = self._reference_prefix_ids(prefix_text)
                # The prefix rides the RESPONSE budget: require it to leave at
                # least one full per-segment cap of room, so the continuation is
                # never squeezed below what every other segment gets (and a cap
                # hit keeps its per_turn classification -- classify_length_cut
                # flips to the neutral "global" label once room < max_turn_tokens).
                room_after = self.response_length - len(cand_ids)
                need = self.max_turn_tokens if (self.max_turn_tokens or 0) > 0 else 1
                if len(base_prompt_ids) < self.prompt_length and room_after >= need:
                    new_msgs, new_prompt_ids = base_msgs, base_prompt_ids
                    prefix_ids = cand_ids
                    block = format_restart_reference_message(prefix_text)
            if not prefix_ids:
                # Delivery still happens -- via the user-message path below; the
                # hint is charged as usual. Counted so a fallback-heavy run
                # (wrong parquet, tight response budget) is visible.
                agent_data.extra_fields["restart_ref_fallback"] = (
                    agent_data.extra_fields.get("restart_ref_fallback", 0) + 1
                )
                agent_data.metrics["restart_ref_fallback"] = (
                    agent_data.metrics.get("restart_ref_fallback", 0) + 1
                )

        # USER-MESSAGE delivery (the default path, and the reference-prefix
        # fallback). Render the fresh prompt FIRST and guard its length: an
        # over-long prompt is LEFT-TRUNCATED inside AgentLoopBase.
        # apply_chat_template (system + problem cut away, recap kept) and comes
        # back at exactly prompt_length -- so the guard must be >=, not >.
        # Refuse to restart instead: the chain ends on this segment and the hint
        # is NOT charged (mirrors the pool-exhausted benign stop). The >= also
        # refuses an exactly-at-cap prompt (indistinguishable from a clipped
        # one); that lost margin is one token and the conservative side is the
        # safe one.
        if new_prompt_ids is None:
            block = self._restart_hint_block(deliver_text, recap_state)
            new_msgs, new_prompt_ids = await self._render_restart_prompt(agent_data, block)
            if len(new_prompt_ids) >= self.prompt_length:
                agent_data.extra_fields["restart_prompt_overflow"] = (
                    agent_data.extra_fields.get("restart_prompt_overflow", 0) + 1
                )
                agent_data.metrics["restart_prompt_overflow"] = (
                    agent_data.metrics.get("restart_prompt_overflow", 0) + 1
                )
                logger.warning(
                    "restart prompt overflow (request=%s): %d > prompt_length %d -- chain ends, hint not charged",
                    agent_data.request_id, len(new_prompt_ids), self.prompt_length,
                )
                self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                           boundary=boundary, completed_hints=completed_hints,
                           selector_raw=_raw, selector_prompt=_prompt)
                return AgentState.TERMINATED

        # Record the GIVEN hint for the per-hint penalty (chain-cumulative).
        # ``hint`` = the text actually DELIVERED (pool sentence under
        # pool_wording; the selector's rephrasing otherwise).
        applied.append(
            {
                "call_index": len(applied),
                "hint_id": hid,
                "major_step_id": (str(hid).split(".")[0] if hid is not None else None),
                "hint": deliver_text,
                "reasoning_len": turn_len,
            }
        )
        agent_data.metrics["hint_call"] = agent_data.metrics.get("hint_call", 0) + 1

        # Mark the given hint AND newly self-completed hints as completed for the
        # next round's status rendering (deduped, discovery order; ``newly``
        # computed above, before the recap render).
        for cid in newly + ([str(hid)] if hid is not None else []):
            if cid not in completed:
                completed.append(cid)

        self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                   boundary=boundary, completed_hints=completed_hints,
                   selector_raw=_raw, selector_prompt=_prompt)

        # Archive the finished segment (uses response_ids/prompt_ids -- must run
        # BEFORE the reset below), then keep the transcript: the hint block joins
        # ``messages`` as the injected user turn so build_trace shows the selector
        # the same conversation it would see in multi-turn mode.
        self._archive_segment(agent_data, turn_text, pre_state, state_end, hid, block, boundary)
        if self.step_adv_enable:
            # PHANTOM fail record for the segment just archived: zero token span
            # (never painted) carrying (ss, failed-state, is_fail=1), so the
            # trainer's value fit counts this failure in F_k exactly as the
            # multi-turn row would (see module header).
            self._record_step_adv_turn(agent_data, 0, 0, 0, pre_state, state_end, is_fail=1)
        agent_data.messages.append({"role": "user", "content": block})
        update_progress_state(
            progress_state,
            selected_hint_id=hid,
            selected_hint=deliver_text,
            hint_order=hint_order,
        )

        # FRESH SEGMENT: replace the live context wholesale. run() finalizes the
        # row as prompt_ids[-len(response_mask):] vs the rest, so after this the
        # eventual row is (fresh prompt, final segment's attempt) -- earlier
        # segments exist only in the archive. In reference-prefix mode the glued
        # reference tokens go AFTER the generation tag as the start of the
        # assistant turn: RESPONSE-region tokens with mask 0 (the injected-user-
        # turn convention -- attended, never trained, never painted; 0.0
        # logprobs keep the sampling-logprob alignment). Generation continues
        # from them, and every later turn record's span starts past them
        # (turn_start = len(response_mask) - len(response_ids) == prefix_len).
        lp_on = bool(agent_data.response_logprobs)  # read BEFORE the reset below
        agent_data.prompt_ids = list(new_prompt_ids) + list(prefix_ids)
        agent_data.response_ids = []
        agent_data.response_mask = [0] * len(prefix_ids)
        agent_data.response_logprobs = [0.0] * len(prefix_ids) if (lp_on and prefix_ids) else []
        agent_data._restart_prefix_len = len(prefix_ids)
        if prefix_ids:
            agent_data.extra_fields["restart_ref_prefix_tokens"] = (
                agent_data.extra_fields.get("restart_ref_prefix_tokens", 0) + len(prefix_ids)
            )
        agent_data.extra_fields["n_restarts"] = (
            agent_data.extra_fields.get("n_restarts", 0) + 1
        )
        agent_data.user_turns += 1
        return AgentState.GENERATING

    # ------------------------------------------------------------------ #
    # selector-call debug dump (mode-tagged copy of the parent's)
    # ------------------------------------------------------------------ #
    def _dump(self, agent_data, turn_start, turn_end, problem, trace, completed,
              selection, err, *, boundary=None, completed_hints=None,
              selector_raw=None, selector_prompt=None) -> None:
        """Parent's per-selector-call dump with mode="restart_hint" + chain
        counters, so the viewer can tell restart chains from multi-turn rollouts."""
        _dump_selector_call(
            {
                "mode": "restart_hint",
                "request_id": getattr(agent_data, "request_id", None),
                "call_index": len(agent_data.extra_fields.get("applied_hints", [])),
                "n_restarts": agent_data.extra_fields.get("n_restarts", 0),
                "turn_span": [turn_start, turn_end],
                "boundary": boundary,
                "completed_status_before": list(completed),
                "n_assistant_in_messages": sum(
                    1 for m in agent_data.messages if m.get("role") == "assistant"
                ),
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
                "problem": problem,
                "trace": trace,
                "selector_prompt": selector_prompt,
                "selection": selection if isinstance(selection, dict) else None,
                "completed_hints": completed_hints,
                "progress_message_enabled": self.progress_message,
                "progress_state": list(agent_data.extra_fields.get("auto_hint_progress", [])),
                "selector_raw": selector_raw,
                "selector_err": err,
            }
        )


# --------------------------------------------------------------------------- #
# REGISTRY RE-PIN -- guards the yaml swap against the parent's import side
# effect. The restart registry (hint_agent_config_restart.yaml) maps agent_name
# "auto_hint" to THIS class at AgentLoopWorker init; hydra's FIRST instantiation
# then imports this module, and the ``from auto_hint_agent_loop import ...``
# above executes the parent's ``@register("auto_hint")`` decorator -- CLOBBERING
# the yaml mapping back to AutoHintAgentLoop for every subsequent rollout of the
# worker. Cluster smoke #1 (run 20260804-192840) died exactly this way: rollout
# 1 per worker ran restart (rows carry restart_* keys), the rest ran multi-turn
# (no keys) -> DataProto.concat key assert on the first trainer batch. Re-pin
# here, AFTER the parent import: this module is only imported when restart mode
# is active (nothing references it in a default run), so the default registry is
# untouched. CONSEQUENCE: agent_name "auto_hint" and "restart_hint" cannot be
# mixed within one run once this module loads -- one run = one delivery mode.
_agent_loop_registry["auto_hint"] = {
    "_target_": f"{RestartHintAgentLoop.__module__}.{RestartHintAgentLoop.__qualname__}"
}
