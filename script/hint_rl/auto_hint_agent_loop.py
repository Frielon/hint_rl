# Copyright 2026
#
# AutoHintAgentLoop -- the AUTO-HINT (push-hint) rollout for HPRL.
#
# A different control flow from the <hint_call/> loop (hint_agent_loop.HintAgentLoop):
# the policy is NOT taught to ask for hints. It runs the ordinary single-turn math
# prompt ("reason step by step, give the answer in \boxed{}", as in
# dataset/dapo-3139-single-turn.parquet). The LOOP, not the model, decides when to
# hint:
#
#   1. generate one answer turn (terminated by EOS);
#   2. grade its boxed answer with the SAME grader the reward uses;
#   3. if CORRECT -> terminate (this is the ending turn);
#   4. if WRONG and fewer than B_q hints have been injected -> ask the frozen
#      selector (MULTI-ROUND Template F, selector_multi) for the next hint, inject
#      it as the next USER message, and continue from step 1;
#   5. stop when correct, when the budget B_q is spent, or when the pool of pending
#      hints is exhausted.
#
# Hints injected this way are recorded into ``applied_hints`` -- the SAME state key
# HintAgentLoop uses -- so the reward (hint_reward.compute_score) penalty and the
# budget ratchet are unchanged. The reward must run with the hint-call bonus and
# the effort-shape penalty DISABLED (hint_call_reward=0, hint_shape_coeff=0): this
# mode has no <hint_call/> emission, so those terms have nothing to act on.
#
# VERIFIED-PREFIX gradient mask: each WRONG (hinted) turn's selector call returns
# ``completed_hints`` -- pending hints the student newly achieved that round, each
# with a verbatim ``quote``. We fuzzy-locate the LAST such quote in that turn's text
# and record a ``disable_spans`` range = (last-verified-token .. end-of-turn). The
# trainer (HPRLRayPPOTrainer._update_actor, auto_hint_mask) zeroes those ranges out
# of the loss mask ONLY for positive-advantage rollouts, so a better-than-average
# rollout reinforces only the selector-verified reasoning, not the unverified tail
# the hint then corrected. The ending turn carries no span -> always fully promoted.
#
# Registered as agent_name "auto_hint" (also via hint_agent_config.yaml). No verl
# core edits; flag-gated by the data carrying agent_name="auto_hint" +
# data.hprl.auto_hint.enable (trainer side).

from __future__ import annotations

import logging
import os
import time
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import (
    SPEC_DECODE_EXTRA_KEYS,
    AgentState,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from hint_agent_loop import HintAgentLoop, _dump_selector_call
from hint_selector import build_trace, hint_id_of
from selector_multi import (
    locate_quote_end,
    pending_hint_ids,
    prune_hint_pool,
    resolve_selected_hint,
)
from step_advantage import classify_length_cut as _classify_length_cut, prefix_state as _prefix_state

# Same boxed-answer grader the reward uses, so the loop's "is it correct yet?"
# decision matches how the rollout is ultimately scored. Guarded so the module
# still imports where mathruler is absent (the loop then treats every answer as
# wrong and injects hints up to budget -- loud warning emitted in __init__).
try:
    from mathruler.grader import extract_boxed_content as _extract_boxed, grade_answer as _grade_answer

    _GRADER_OK = True
except Exception:  # noqa: BLE001
    _GRADER_OK = False

    def _extract_boxed(_s):  # type: ignore[misc]
        return "None"

    def _grade_answer(_p, _g):  # type: ignore[misc]
        return False

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("auto_hint")
class AutoHintAgentLoop(HintAgentLoop):
    """Push-hint rollout: the loop injects a hint whenever the answer is still wrong."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hprl = (self.data_config.get("hprl", {}) or {}) if self.data_config is not None else {}
        ah = hprl.get("auto_hint", {}) or {}
        # Min in-order loose-char coverage to accept a completed-hint quote as
        # located in the turn's text (selector_multi.locate_quote_end). Lower than
        # the citation-CHECK threshold (0.85): here a near-match still legitimately
        # bounds the verified prefix, and a miss only makes the mask MORE conservative.
        self.auto_hint_fuzzy_threshold = float(
            ah.get("fuzzy_threshold", os.environ.get("HPRL_AUTO_HINT_FUZZY", "0.8"))
        )
        # STEP-LEVEL advantage mode (data.hprl.auto_hint.step_adv.enable): when on, the
        # loop records per-turn ``step_adv_turns`` segments (verified prefix a_C / failed
        # tail a_I, with the state reached) and LABELS the terminal turn of an
        # over-budget incorrect rollout (an extra selector call, no hint injected) so the
        # trainer can compute value-based per-token advantages (step_advantage.py). Off ->
        # the loop only records disable_spans for the verified-prefix MASK (auto_hint_mask).
        sa = ah.get("step_adv", {}) or {}
        self.step_adv_enable = str(
            sa.get("enable", os.environ.get("HPRL_STEP_ADV", "false"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Prune the X.0 step-guidance hints from the pool before it ever reaches the
        # selector (prune_hint_pool), so the rollout presents the SAME substep-only
        # pools the offline selector eval (multi-cite-gpt-eval) was built and scored
        # on. Applied at the single read point (_pool), so the selector prompt, the
        # pending/exhaustion check, and the step-adv hint order all see one pruned
        # pool consistently. Default off -> the full pool (with X.0) is used as before.
        self.prune_guidance = str(
            ah.get("prune_guidance", os.environ.get("HPRL_PRUNE_GUIDANCE", "false"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        # PER-TURN generation-length cap (data.hprl.auto_hint.max_turn_tokens; env
        # HPRL_MAX_TURN_TOKENS). When > 0, each turn's generation is bounded to
        # min(max_turn_tokens, remaining global room) via the per-request max_tokens --
        # an INDEPENDENT per-turn limit on top of the global response_length budget. A
        # turn that hits this cap WITHOUT an EOS, with global budget still remaining, is
        # treated as having FAILED to finish its next hint step: the WHOLE turn becomes
        # the a_I failed tail (boundary == turn_start, no verified prefix) failing the
        # next pending hint h_{k+1} at the rollout's current state S_k, the rollout is
        # flagged length_truncated (floor reward -> acc=0 -> incorrect), and it
        # terminates (no selector call, no hint injected). A turn that instead runs into
        # the GLOBAL response_length ceiling keeps the prior behavior (its tail carries
        # advantage 0). 0 (default) -> no per-turn cap; only the global one applies.
        self.max_turn_tokens = int(
            ah.get("max_turn_tokens", os.environ.get("HPRL_MAX_TURN_TOKENS", "0")) or 0
        )
        if not _GRADER_OK:
            logger.warning(
                "AutoHintAgentLoop: mathruler grader UNAVAILABLE -- every answer will be "
                "treated as wrong and hints injected up to budget. Install mathruler."
            )
        logger.warning(
            "AutoHintAgentLoop: auto-hint rollout active (fuzzy_threshold=%.2f, step_adv=%s, "
            "prune_guidance=%s, max_turn_tokens=%d)",
            self.auto_hint_fuzzy_threshold,
            self.step_adv_enable,
            self.prune_guidance,
            self.max_turn_tokens,
        )

    # ------------------------------------------------------------------ #
    # small accessors over this problem's create_kwargs
    # ------------------------------------------------------------------ #
    def _create_kwargs(self, agent_data) -> dict:
        tk = agent_data.tools_kwargs.get(self.hint_kwargs_key) if agent_data.tools_kwargs else None
        return (tk.get("create_kwargs", {}) or {}) if isinstance(tk, dict) else {}

    def _pool(self, agent_data) -> Any:
        """This problem's hint pool, with X.0 step-guidance hints pruned when
        ``prune_guidance`` is set. The single chokepoint for reading ``hints`` so the
        selector prompt, pending/exhaustion check, and step-adv hint order all operate
        on one consistently-(un)pruned pool."""
        pool = self._create_kwargs(agent_data).get("hints", "")
        return prune_hint_pool(pool) if self.prune_guidance else pool

    def _is_correct(self, agent_data) -> bool:
        """Grade the rollout's boxed answer so far (all assistant turns joined),
        exactly as the reward will. False (so the loop keeps hinting) when the
        grader is unavailable."""
        if not _GRADER_OK:
            return False
        gt = self._create_kwargs(agent_data).get("ground_truth", "")
        solution = "\n".join(
            str(m.get("content") or "")
            for m in agent_data.messages
            if m.get("role") == "assistant"
        )
        pred = _extract_boxed(solution)
        return pred != "None" and _grade_answer(pred, gt)

    # ------------------------------------------------------------------ #
    # generation + grade-driven continuation
    # ------------------------------------------------------------------ #
    async def _handle_generating_state(
        self, agent_data, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Generate one answer turn (EOS-terminated), then decide: correct -> stop;
        wrong & budget remains -> go inject a hint (PROCESSING_TOOLS); else stop."""
        # (1) EOS only -- no tool/stop tokens injected (the policy is unaware of hints).
        # PER-TURN length cap: ``room`` = global response tokens left BEFORE this turn;
        # classify_length_cut (below) uses it with the per-turn cap to label how the turn
        # was cut (vLLM reports EOS and a length cut alike, so we go by token count). When
        # the cap is on we bound THIS turn to min(max_turn_tokens, room) via the per-request
        # max_tokens, passed as a COPY so generate() (which pops/mutates max_tokens,
        # logprobs, ...) cannot corrupt the rollout-shared sampling_params reused every turn.
        room = self.response_length - len(agent_data.response_mask)
        if self.max_turn_tokens and self.max_turn_tokens > 0 and not ignore_termination:
            turn_cap = max(1, min(self.max_turn_tokens, room))
            sampling_params = {**sampling_params, "max_tokens": turn_cap}
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

        # Every rollout must carry the SAME extra_fields keys so DataProto.concat
        # across agent-loop workers doesn't assert. Init the hint-state keys the
        # reward / budget callback read (mostly inert here -- there is no
        # <hint_call/> -- but kept so hint_reward.compute_score + hint_budget_callback
        # run UNCHANGED), plus the two auto-hint keys:
        #   * disable_spans       -- [start,end) response-token ranges to drop from the
        #     loss mask under POSITIVE advantage (the verified-prefix mask).
        #   * auto_hint_completed -- hint ids already given/verified this rollout, for
        #     the multi-round selector's completed|pending status rendering.
        agent_data.extra_fields.setdefault("applied_hints", [])
        agent_data.extra_fields.setdefault("hint_call_failed", 0)
        agent_data.extra_fields.setdefault("hint_budget_exceeded", 0)
        # Per-rollout flag: set to 1 when the assistant generation ran into the hard
        # response-length cap (truncated mid-answer, no EOS). The reward short-circuits
        # to the floor (minimum) score on this flag, same as hint_budget_exceeded, so a
        # run-on rollout that never emits a boxed answer is scored as the worst outcome
        # rather than an ordinary (and sometimes accidentally-boxed) failure.
        agent_data.extra_fields.setdefault("length_truncated", 0)
        # Per-rollout flag: 1 when this rollout ended on a PER-TURN length-cap cut
        # (max_turn_tokens hit with global budget remaining) -- the subset of
        # length_truncated attributable to the per-turn cap rather than the global
        # ceiling. Surfaced (vs an agent_data.metrics counter, which AgentLoopMetrics
        # drops) so hint_budget_callback can report hprl/turn_truncated_frac. Initialized
        # on every rollout for the DataProto.concat key-consistency reason as applied_hints.
        agent_data.extra_fields.setdefault("turn_truncated", 0)
        agent_data.extra_fields.setdefault("hint_select_time", 0.0)
        agent_data.extra_fields.setdefault("hint_select_calls", 0)
        agent_data.extra_fields.setdefault("hint_calls_total", 0)
        agent_data.extra_fields.setdefault("hint_calls_with_box", 0)
        # resolve_selected_hint reconciliation counters -- initialized on EVERY
        # rollout (DataProto.concat key-consistency, same as the keys above);
        # incremented flag-wise at the selection site.
        agent_data.extra_fields.setdefault("hint_id_remapped", 0)
        agent_data.extra_fields.setdefault("hint_id_out_of_pool", 0)
        agent_data.extra_fields.setdefault("hint_text_from_pool", 0)
        agent_data.extra_fields.setdefault("hint_repeat", 0)
        agent_data.extra_fields.setdefault("turn_lens", [])
        agent_data.extra_fields.setdefault("hint_pool_exhausted", 0)
        agent_data.extra_fields.setdefault("finalized_incorrect", 0)
        agent_data.extra_fields.setdefault("final_hint_step", "")
        agent_data.extra_fields.setdefault("final_hint_id", "")
        agent_data.extra_fields.setdefault("final_hint_exhausted", 0)
        agent_data.extra_fields.setdefault("disable_spans", [])
        agent_data.extra_fields.setdefault("auto_hint_completed", [])
        # STEP-LEVEL advantage segments: per turn, a [turn_start, boundary, turn_end,
        # state_start, state_end, is_fail] record (response-relative token coords +
        # the states the turn moved between). Consumed by step_advantage.py when
        # data.hprl.auto_hint.step_adv.enable is set; inert otherwise.
        agent_data.extra_fields.setdefault("step_adv_turns", [])
        # citation found rate: completed_hints quotes seen / fuzzy-located in the trace.
        agent_data.extra_fields.setdefault("cite_quotes_total", 0)
        agent_data.extra_fields.setdefault("cite_quotes_found", 0)

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.extra_fields["turn_lens"].append(len(output.token_ids))
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # --- hard caps. TWO distinct length cuts, by WHICH limit bound the turn (the cut
        #     reason is invisible in stop_reason -- vLLM reports EOS and a length cut alike
        #     as "completed" -- so classify_length_cut goes by token count). The PER-TURN
        #     cut wins the tie room == max_turn_tokens, so a turn cut at length ==
        #     max_turn_tokens is punished even when it coincides with the global ceiling;
        #     only a STRICTLY smaller global room (room < max_turn_tokens) is neutral. -----
        cut = None if ignore_termination else _classify_length_cut(
            len(output.token_ids), self.max_turn_tokens, room
        )
        # (a) PER-TURN cap bound the turn and it ran to that cap -> treat the turn as
        #     FAILING to finish its next hint step. Flag length_truncated (reward floors to
        #     the minimum, acc=0 -> incorrect), and in step-adv mode charge the WHOLE turn
        #     as the a_I failed tail (boundary == turn_start, no verified prefix) failing
        #     the next pending hint h_{k+1} at the rollout's current state S_k = pre_state
        #     (ss == se == pre_state, so that hint joins H). Generalizes the FIRST-turn
        #     global-cap case (b) -- which hardcodes pre_state 0 -- to any turn/state. Then
        #     terminate WITHOUT a selector call or hint injection.
        if cut == "per_turn":
            agent_data.extra_fields["length_truncated"] = 1
            agent_data.extra_fields["turn_truncated"] = 1
            if self.step_adv_enable:
                turn_end = len(agent_data.response_mask)
                turn_start = turn_end - len(agent_data.response_ids)
                order = pending_hint_ids(self._pool(agent_data), [])
                pre_state = _prefix_state(
                    order, agent_data.extra_fields.get("auto_hint_completed", [])
                )
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_start, turn_end, pre_state, pre_state, is_fail=1
                )
            return AgentState.TERMINATED
        # (b) GLOBAL response-length ceiling bound the turn (no per-turn cap, or the
        #     remaining room was STRICTLY smaller than max_turn_tokens so the turn ran out
        #     of TOTAL budget, not its own step cap). Flag length_truncated; in step-adv a
        #     FIRST-turn cut is charged as failing h_0 at S_0 (it never reached step 1),
        #     while a LATER-turn cut records NO segment, so its tail keeps advantage 0 --
        #     the global ceiling is not the turn's "fault" (earlier turns already scored).
        if cut == "global":
            agent_data.extra_fields["length_truncated"] = 1
            if self.step_adv_enable and agent_data.assistant_turns == 1:
                turn_end = len(agent_data.response_mask)
                turn_start = turn_end - len(agent_data.response_ids)
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_start, turn_end, 0, 0, is_fail=1
                )
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Record this turn so the selector's build_trace + grading see it (the base
        # loop tracks turns only as tokens; build_trace reads messages).
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        agent_data.messages.append({"role": "assistant", "content": text})

        # (2) grade -> (3)/(4) continuation decision.
        if self._is_correct(agent_data):
            if self.step_adv_enable:
                # SOLVING turn: the whole turn is verified progress to the final state
                # S_K (no failed-tail a_I). state_start = how far the rollout already was.
                turn_end = len(agent_data.response_mask)
                turn_start = turn_end - len(agent_data.response_ids)
                order = pending_hint_ids(self._pool(agent_data), [])
                ss = _prefix_state(order, agent_data.extra_fields.get("auto_hint_completed", []))
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_end, turn_end, ss, len(order), is_fail=0
                )
            return AgentState.TERMINATED  # ending turn (solved) -> full promote
        budget = int(self._create_kwargs(agent_data).get("budget", self.default_budget))
        if len(agent_data.extra_fields.setdefault("applied_hints", [])) >= budget:
            if self.step_adv_enable:
                # Budget spent but still wrong: in step-adv mode we route to the selector
                # to LABEL this terminal turn (the processing-tools handler sees the spent
                # budget, records the failed-step segment, and terminates WITHOUT injecting
                # a hint). Legacy mode just terminates (the ending turn is full-promoted).
                return AgentState.PROCESSING_TOOLS
            return AgentState.TERMINATED  # ending turn (budget spent) -> full promote
        return AgentState.PROCESSING_TOOLS  # wrong + budget remains -> inject a hint

    # ------------------------------------------------------------------ #
    # selector call + hint injection + verified-prefix boundary
    # ------------------------------------------------------------------ #
    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        """Pick the next hint (multi-round selector), record the verified-prefix
        boundary for the just-finished WRONG turn, and inject the hint as a user turn.

        In STEP-ADV mode this also (a) records the per-turn ``step_adv_turns`` segment
        (a_C verified prefix advancing pre_state -> the failed hint's pool index, a_I
        failed tail) and (b) when the budget is already spent, treats the call as a
        TERMINAL label: it queries the selector to identify the failed step but injects
        no hint and charges no applied-hint penalty.
        """
        agent_data.tool_calls = []  # not using hermes tools

        ck = self._create_kwargs(agent_data)
        budget = int(ck.get("budget", self.default_budget))
        problem = ck.get("problem", "")
        pool = self._pool(agent_data)
        applied = agent_data.extra_fields.setdefault("applied_hints", [])
        completed = agent_data.extra_fields.setdefault("auto_hint_completed", [])

        # span of the turn that just ended (wrong) -- response-relative token coords.
        turn_len = len(agent_data.response_ids)
        turn_end = len(agent_data.response_mask)
        turn_start = turn_end - turn_len
        turn_text = agent_data.messages[-1].get("content", "") if agent_data.messages else ""

        # STEP-ADV: the pool's full hint order + the state the rollout was at BEFORE
        # this turn (state_start). A spent budget makes this a TERMINAL label (query the
        # selector for the failed step, but give no hint and charge no penalty).
        order = pending_hint_ids(pool, []) if self.step_adv_enable else []
        pre_state = _prefix_state(order, completed) if self.step_adv_enable else 0
        terminal = self.step_adv_enable and len(applied) >= budget

        # Pool exhausted: no pending hint left to give -> nothing more we can do, so
        # this wrong turn becomes the ending turn (terminate, full promote). Counted
        # for the hprl/hint_pool_exhausted metric. In step-adv this is a zero-advantage
        # no-progress, no-fail segment (state_start == state_end, no a_I).
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

        # Selector failure (no usable dict): log it and terminate -- this wrong turn
        # is the ending turn (no span). Never let a selector outage corrupt training
        # by guessing a boundary. Distinct from pool-exhaustion above (a benign stop).
        if not isinstance(selection, dict):
            agent_data.extra_fields["hint_call_failed"] = (
                agent_data.extra_fields.get("hint_call_failed", 0) + 1
            )
            agent_data.metrics["hint_call_failed"] = agent_data.metrics.get("hint_call_failed", 0) + 1
            logger.warning("auto-hint selection failed (request=%s): %s", agent_data.request_id, err)
            self._dump(agent_data, turn_start, turn_end, problem, trace, completed, None, err,
                       selector_raw=_raw, selector_prompt=_prompt)
            if self.step_adv_enable:
                # no trustworthy boundary -> a zero-advantage no-fail segment (don't
                # guess a failed step on a selector outage).
                self._record_step_adv_turn(
                    agent_data, turn_start, turn_end, turn_end, pre_state, pre_state, is_fail=0
                )
            return AgentState.TERMINATED

        # Shape-guard every field we read (a selector may emit odd JSON; a stray
        # array/int once killed a whole run). hint must be a str; completed_hints a list.
        _h = selection.get("hint")
        hint_text = _h.strip() if isinstance(_h, str) else ""
        hid = hint_id_of(selection)
        completed_hints = selection.get("completed_hints")
        if not isinstance(completed_hints, list):
            completed_hints = []

        # Reconcile (id, text) against the pool actually OFFERED this call (the
        # ``pending`` ids). A mislabeled id poisons ``completed``/step-adv/penalty
        # state and lets the truly-given hint be re-selected (verbatim duplicate
        # hints in 2.4% of the 20260723 dolci run's hinted rollouts); an empty
        # text with a valid id resolves to the pool's authoritative text instead
        # of the injected placeholder.
        hid, hint_text, _resolve_flags = resolve_selected_hint(
            hid, hint_text, pool, pending,
            applied_ids=[a.get("hint_id") for a in applied],
        )
        for _flag in _resolve_flags:
            _k = "hint_" + _flag  # hint_id_remapped / hint_id_out_of_pool / hint_text_from_pool / hint_repeat
            agent_data.extra_fields[_k] = agent_data.extra_fields.get(_k, 0) + 1
            agent_data.metrics[_k] = agent_data.metrics.get(_k, 0) + 1

        # Selector DECLINED: no id and no text even after pool reconciliation
        # (the "all pending hints already achieved" answer: hint_id "" + hint "").
        # Before this guard, that answer was injected as the "(the selector
        # returned an empty hint)" placeholder, charged the per-hint penalty and
        # appended "" to ``completed``. Treat it like pool-exhaustion instead: a
        # benign stop -- no injection, no penalty, a zero-advantage no-fail
        # segment (the selector gave us no failed hint to price).
        if hid is None and not hint_text:
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

        # Verified-prefix boundary for THIS turn: keep tokens up to the last
        # selector-verified sentence in the turn; disable the (unverified) tail.
        boundary = self._verified_boundary(turn_text, completed_hints, turn_start, turn_len)
        disable_spans = agent_data.extra_fields.setdefault("disable_spans", [])
        if boundary < turn_end:
            disable_spans.append([int(boundary), int(turn_end)])

        # STEP-ADV segment for THIS (failed) turn: a_C [turn_start, boundary) advances
        # pre_state -> the failed hint's pool index (state_end); a_I [boundary, turn_end)
        # is the unverified tail that attempted and failed hint ``state_end``.
        if self.step_adv_enable:
            try:
                fail_state = order.index(str(hid))
            except ValueError:
                fail_state = pre_state
            state_end = max(pre_state, fail_state)
            self._record_step_adv_turn(
                agent_data, turn_start, boundary, turn_end, pre_state, state_end, is_fail=1
            )

        # CITATION FOUND RATE: of the selector's completed_hints quotes, how many
        # fuzzy-locate in the FULL trace it was shown (build_trace output) -- the
        # selector-citation-honesty signal (a low rate = fabricated/paraphrased
        # quotes, so the verified-prefix mask anchors on nothing). Counted against
        # the whole trace (not just this turn) to match the selector's view; summed
        # per rollout, aggregated to auto_hint/cite_found_rate by the trainer.
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

        # TERMINAL label (step-adv, budget already spent): the failed-step segment is
        # recorded above; do NOT give the hint (no applied-hint penalty, no injection).
        if terminal:
            self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                       boundary=boundary, completed_hints=completed_hints,
                       selector_raw=_raw, selector_prompt=_prompt)
            return AgentState.TERMINATED

        # Record the GIVEN hint for the per-hint penalty (strategy="hint").
        applied.append(
            {
                "call_index": len(applied),
                "hint_id": hid,
                "major_step_id": (str(hid).split(".")[0] if hid is not None else None),
                "hint": hint_text,
                "reasoning_len": turn_len,
            }
        )
        agent_data.metrics["hint_call"] = agent_data.metrics.get("hint_call", 0) + 1

        # Mark the given hint AND the newly self-completed hints as completed for the
        # next round's status rendering (deduped, preserving discovery order).
        newly = [
            str(c.get("hint_id"))
            for c in completed_hints
            if isinstance(c, dict) and c.get("hint_id") is not None
        ]
        for cid in newly + ([str(hid)] if hid is not None else []):
            if cid not in completed:
                completed.append(cid)

        self._dump(agent_data, turn_start, turn_end, problem, trace, completed, selection, err,
                   boundary=boundary, completed_hints=completed_hints,
                   selector_raw=_raw, selector_prompt=_prompt)

        # Inject the hint as a user turn (masked 0 -- never trained).
        hint_message = {"role": "user", "content": self._format_hint(hint_text)}
        agent_data.messages.append(hint_message)
        response_ids = await self.apply_chat_template([hint_message], remove_system_prompt=True)
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _record_step_adv_turn(
        self, agent_data, turn_start: int, boundary: int, turn_end: int,
        state_start: int, state_end: int, *, is_fail: int,
    ) -> None:
        """Append one [turn_start, boundary, turn_end, state_start, state_end, is_fail]
        segment for the step-level advantage (step_advantage.py). Coords are
        response-relative tokens; states are pool-order indices (0 == nothing done)."""
        agent_data.extra_fields.setdefault("step_adv_turns", []).append(
            [int(turn_start), int(boundary), int(turn_end),
             int(state_start), int(state_end), int(is_fail)]
        )

    def _verified_boundary(self, turn_text: str, completed_hints, turn_start: int, turn_len: int) -> int:
        """Response-token index up to which THIS turn is selector-verified.

        Fuzzy-locates each completed-hint ``quote`` in the turn's text and takes the
        furthest char end; maps that to a token index by character fraction within
        the turn. Returns ``turn_start`` when no quote locates -- i.e. nothing in the
        turn is verified, so the whole turn is disabled under positive advantage.
        """
        end_char = -1
        for c in completed_hints:
            if not isinstance(c, dict):
                continue
            q = c.get("quote")
            if not isinstance(q, str):
                continue
            e = locate_quote_end(q, turn_text, self.auto_hint_fuzzy_threshold)
            if e is not None and e > end_char:
                end_char = e
        if end_char < 0:
            return turn_start
        frac = min(1.0, end_char / max(1, len(turn_text)))
        return turn_start + int(round(frac * turn_len))

    @staticmethod
    def _format_hint(hint_text: str) -> str:
        """Wrap the selector hint as a user turn that asks the model to continue.

        Deliberately contains NO literal ``\\boxed{}``: the reward extracts the final
        answer from the decoded FULL response (assistant turns + these injected user
        turns), so a ``\\boxed{}`` here could be mistaken for the answer. The system
        prompt already mandates the boxed final answer.
        """
        body = hint_text or "(the selector returned an empty hint)"
        return (
            f"Here is a hint to help you make progress:\n{body}\n\n"
            "Using this hint, continue your reasoning and give your final boxed answer."
        )

    def _dump(self, agent_data, turn_start, turn_end, problem, trace, completed,
              selection, err, *, boundary=None, completed_hints=None,
              selector_raw=None, selector_prompt=None) -> None:
        """Reuse the HintAgentLoop per-selector-call debug dump (no-op unless
        HPRL_SELECTOR_DUMP_DIR is set), with the auto-hint-specific extras.

        ``selector_prompt`` is the EXACT prompt sent to the selector and
        ``selector_raw`` its raw text output, so the rollout viewer can show both
        for an injected-hint turn. ``messages`` is dumped (assistant turns through
        the calling turn, system content elided) so build_selector_index.py can
        content-key auto-hint records the same way it keys the <hint_call/> ones --
        without it the indexer sees no assistant trace and can't join the call."""
        _dump_selector_call(
            {
                "mode": "auto_hint",
                "request_id": getattr(agent_data, "request_id", None),
                "call_index": len(agent_data.extra_fields.get("applied_hints", [])),
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
                # problem + trace + the rendered pool make up selector_prompt; it is
                # dumped whole so the viewer shows the verbatim prompt that was sent.
                "selector_prompt": selector_prompt,
                "selection": selection if isinstance(selection, dict) else None,
                "completed_hints": completed_hints,
                "selector_raw": selector_raw,
                "selector_err": err,
            }
        )
