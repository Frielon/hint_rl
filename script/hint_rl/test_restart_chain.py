#!/usr/bin/env python3
# Copyright 2026
#
# Tests for the TERMINATE-AND-RESTART (segment-chain) hint delivery
# (restart_agent_loop.RestartHintAgentLoop + selector_multi.format_restart_hint_message).
#
# Two layers:
#   * message-renderer tests -- stdlib-only (selector_multi), always run;
#   * chain-mechanics tests -- drive the REAL loop class through scripted
#     3-segment chains with a stub vLLM server + stub selector. These need verl
#     (+ the Qwen3 tokenizer on disk) and SKIP cleanly when absent; run them
#     under the verl conda env:
#       /share5/users/xutao.ma/miniconda3/envs/verl/bin/python test_restart_chain.py
#
# Run:  python test_restart_chain.py   (or pytest test_restart_chain.py)
from __future__ import annotations

import asyncio
import os

from selector_multi import format_restart_hint_message

# --------------------------------------------------------------------------- #
# renderer tests (stdlib-only)
# --------------------------------------------------------------------------- #
INVALIDATION = (
    "You have attempted this problem before. That attempt did not reach a "
    "correct answer: its boxed answer is incorrect and is no longer valid."
)
# with-progress opener (wording pinned 2026-08-04): one merged sentence, no
# explicit invalidation -- the failed attempt is not in the fresh context.
PROGRESS_OPENER = (
    "You have attempted this problem before and you made the following "
    "verified progress, which is correct:"
)


def test_restart_message_with_progress():
    progress = [
        {"hint_id": "1.1", "text": "You set k = floor(sqrt(n))."},
        {"hint_id": "1.2", "text": "You bounded k^2 <= n < (k+1)^2."},
    ]
    msg = format_restart_hint_message("Count the complement.", progress, include_progress=True)
    assert msg.startswith(PROGRESS_OPENER + "\n1. You set k = floor(sqrt(n))."), msg[:160]
    assert "2. You bounded k^2 <= n < (k+1)^2." in msg
    assert "incorrect and is no longer valid" not in msg, "with-progress variant carries no invalidation"
    assert "Here is a hint to help you make further progress:\nCount the complement." in msg
    assert "continue your reasoning from the verified progress" in msg
    # keeps the parent invariant: no literal \boxed{} in an injected block
    assert "\\boxed{" not in msg


def test_restart_message_without_progress():
    # progress-free variant (wording revised 2026-08-04): JUST the hint block --
    # no invalidation preamble, no mention of the prior attempt.
    msg = format_restart_hint_message("Try small cases.", [], include_progress=True)
    assert msg.startswith("Here is a hint to help you make progress:\nTry small cases."), msg[:120]
    assert INVALIDATION not in msg and "You have attempted" not in msg
    assert "verified progress" not in msg
    assert "reason step by step from the beginning" in msg
    # include_progress=False drops the recap even when progress exists
    msg2 = format_restart_hint_message("Try small cases.", [{"hint_id": "1.1", "text": "x"}],
                                       include_progress=False)
    assert "verified progress" not in msg2


def test_restart_message_empty_hint_placeholder():
    msg = format_restart_hint_message("", [], include_progress=True)
    assert "(the selector returned an empty hint)" in msg


def test_step_adv_value_parity():
    """PROOF of restart<->multi-turn step-adv compatibility (whole_turn mode).

    The same 3-chain group is encoded twice: as multi-turn rows (every turn's
    segment present) and as restart rows (final segment + zero-span PHANTOM fail
    records for the archived segments). The per-group value fit must be
    IDENTICAL (phantoms supply the same H_i/V_i), and each restart row's painted
    advantage must equal the multi-turn row's final-turn advantage exactly.
    """
    import numpy as np

    from step_advantage import apply_step_level_advantages, read_turn_advantages

    K, pvec = 2, [0.4, 0.3]
    kwargs = dict(
        correct_per_row=[True, True, False],
        penalty_per_row=[pvec] * 3,
        K_per_row=[K] * 3,
        terminal_value=1.0,
        zero_if_no_correct=True,
        adv_scale=1.0,
        normalize=False,
        overlong_penalty=0.0,
        whole_turn=True,
    )
    # chain A: unhinted solve (10 tok). B: fail h0 (6 tok) -> solve (8 tok).
    # C: fail h0 (5 tok) -> terminal fail h1 (7 tok), incorrect.
    multi_turns = [
        [[0, 10, 10, 0, 2, 0]],
        [[0, 0, 6, 0, 0, 1], [6, 14, 14, 1, 2, 0]],
        [[0, 0, 5, 0, 0, 1], [5, 5, 12, 1, 1, 1]],
    ]
    restart_turns = [
        [[0, 10, 10, 0, 2, 0]],
        [[0, 0, 0, 0, 0, 1], [0, 8, 8, 1, 2, 0]],          # phantom + final segment
        [[0, 0, 0, 0, 0, 1], [0, 0, 7, 1, 1, 1]],          # phantom + terminal label
    ]
    adv_m = np.zeros((3, 14)); mask_m = np.zeros((3, 14))
    for i, L in enumerate((10, 14, 12)):
        mask_m[i, :L] = 1
    adv_r = np.zeros((3, 10)); mask_r = np.zeros((3, 10))
    for i, L in enumerate((10, 8, 7)):
        mask_r[i, :L] = 1

    gv_m, gv_r = {}, {}
    _, _, st_m = apply_step_level_advantages(
        adv_m, adv_m, mask_m, ["p"] * 3, multi_turns, group_values_out=gv_m, **kwargs)
    _, _, st_r = apply_step_level_advantages(
        adv_r, adv_r, mask_r, ["p"] * 3, restart_turns, group_values_out=gv_r, **kwargs)

    # audit channel: fitted V identical across encodings and equal to the hand
    # recursion (V2=1, V1=1-0.3/3=0.9, V0=0.9-2*0.4/3).
    V_expect = [0.9 - 0.8 / 3, 0.9, 1.0]
    assert gv_m["p"] == gv_r["p"]
    assert all(abs(a - b) < 1e-9 for a, b in zip(gv_r["p"], V_expect)), gv_r["p"]

    # audit channel: tensor-readback per-turn advantages -- phantoms read None,
    # painted turns read [head, tail] with head == tail in whole_turn mode.
    ta = read_turn_advantages(adv_r, restart_turns)
    assert ta[0] == [[adv_r[0, 0], adv_r[0, 0]]] and abs(ta[0][0][0] - (1.0 - V_expect[0])) < 1e-9
    assert ta[1][0] is None and abs(ta[1][1][0] - 0.1) < 1e-9 and ta[1][1][0] == ta[1][1][1]
    assert ta[2][0] is None and abs(ta[2][1][0] - (-0.2)) < 1e-9

    # identical value fit: V[0] mean matches (V2=1, V1=1-0.3/3=0.9, V0=0.9-0.8/3)
    assert abs(st_m["step_adv/value_s0_mean"] - st_r["step_adv/value_s0_mean"]) < 1e-9
    assert abs(st_r["step_adv/value_s0_mean"] - (0.9 - 0.8 / 3)) < 1e-9

    # final-segment advantages equal across encodings, token for token
    assert np.allclose(adv_r[0, :10], adv_m[0, :10])            # A: V2-V0
    assert np.allclose(adv_r[1, :8], adv_m[1, 6:14])            # B solve: V2-V1
    assert np.allclose(adv_r[2, :7], adv_m[2, 5:12])            # C fail h1: r1+V2-V1
    assert abs(adv_r[1, 0] - 0.1) < 1e-9 and abs(adv_r[2, 0] - (-0.2)) < 1e-9
    # phantoms painted nothing; padding untouched
    assert np.all(adv_r[1, 8:] == 0.0) and np.all(adv_r[2, 7:] == 0.0)


# --------------------------------------------------------------------------- #
# chain-mechanics tests (need verl + the Qwen3 tokenizer; skip cleanly if absent)
# --------------------------------------------------------------------------- #
# pristine dir first: the -hprl derived dir is per-file symlinks that resolve on
# the PODS' mount root (/xutao/...) and dangle on the dev box; the tokenizer is
# identical either way (-hprl only patches generation_config.json).
_QWEN3_DIRS = (
    "/share5/users/xutao.ma/model/Qwen3-8B-Base",
    "/share5/users/xutao.ma/model/Qwen3-8B-Base-hprl",
)


def _verl_fixtures():
    """(tokenizer, make_loop) or None when verl / the tokenizer is unavailable."""
    try:
        import verl  # noqa: F401
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] verl env unavailable: {e}")
        return None
    tokenizer = None
    for tok_dir in _QWEN3_DIRS:
        if not os.path.isdir(tok_dir):
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
            break
        except Exception as e:  # noqa: BLE001 -- e.g. dangling pod-mount symlinks
            print(f"  [warn] tokenizer load failed from {tok_dir}: {e}")
    if tokenizer is None:
        print(f"  [skip] Qwen3 tokenizer not loadable from {_QWEN3_DIRS}")
        return None

    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from restart_agent_loop import RestartHintAgentLoop

    def make_loop(server, prompt_length=1024, response_length=512, step_adv=False, **loop_kwargs):
        cfg = OmegaConf.create({
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "multi_turn": {
                        "max_user_turns": 10,
                        "max_assistant_turns": 10,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 2048,
                        "tool_response_truncate_side": "middle",
                        "format": "hermes",
                    },
                },
            },
            "data": {
                "hprl": {
                    "default_budget": 6,
                    "strategy": "hint",
                    "step_exclude_mode": "applied",
                    "finalize_incorrect": False,
                    "auto_hint": {
                        "enable": True,
                        "fuzzy_threshold": 0.8,
                        "progress_message": True,
                        "prune_guidance": False,
                        "max_turn_tokens": 0,
                        "step_adv": {"enable": bool(step_adv)},
                    },
                },
            },
        })
        loop = RestartHintAgentLoop(
            trainer_config=DictConfigWrap(cfg),
            server_manager=server,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(cfg.data),
            tools=ToolListWrap([]),
            **loop_kwargs,
        )
        return loop

    return tokenizer, make_loop


class _ScriptedServer:
    """server_manager stub: returns the scripted attempt texts in order."""

    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.calls = 0
        self.prompt_lens = []

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        from verl.workers.rollout.replica import TokenOutput

        self.prompt_lens.append(len(prompt_ids))
        text = self.texts[self.calls]
        self.calls += 1
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), num_preempted=0)


class _ScriptedSelector:
    """HintSelector stub: returns the scripted selections in order."""

    def __init__(self, selections):
        self.selections = list(selections)
        self.calls = []

    async def select_multi(self, problem, trace, pool, completed):
        self.calls.append({"problem": problem, "trace": trace, "completed": list(completed)})
        sel = self.selections[len(self.calls) - 1]
        return sel, "<raw>", None, "<prompt>"


_POOL = (
    '{"steps": [{"step_id": 1, "purpose": "setup", "hints": ['
    '{"hint_id": "1.1", "hint": "Use inclusion-exclusion."},'
    '{"hint_id": "1.2", "hint": "Count the complement."}]}]}'
)
_PROBLEM = "How many integers in [1,100] are divisible by 2 or 5?"
_RAW_PROMPT = [
    {"role": "system", "content": "You are a helpful assistant. Solve the math problem given by "
                                  "the user, reasoning step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": _PROBLEM + " Let's think step by step and output the final answer "
                                           "within \\boxed{}."},
]


def _tools_kwargs(budget):
    return {"request_hint": {"create_kwargs": {
        "problem": _PROBLEM, "hints": _POOL, "ground_truth": "60", "budget": budget,
    }}}


def _run(loop_obj, **kwargs):
    return asyncio.get_event_loop().run_until_complete(
        loop_obj.run({"temperature": 1.0}, **kwargs)
    )


def test_chain_three_segments(fix):
    tokenizer, make_loop = fix
    texts = [
        "Count multiples of 2: 50. I think the answer is \\boxed{50}.",       # wrong
        "Also multiples of 5: 20, so maybe \\boxed{70}.",                      # wrong
        "Inclusion-exclusion: 50 + 20 - 10 = 60. \\boxed{60}.",               # correct
    ]
    server = _ScriptedServer(tokenizer, texts)
    loop_obj = make_loop(server)
    selector = _ScriptedSelector([
        {"hint_id": "1.1", "hint": "Use inclusion-exclusion.", "completed_hints": []},
        {"hint_id": "1.2", "hint": "Count the complement.", "completed_hints": []},
    ])
    loop_obj._selector = selector

    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_tools_kwargs(4), extra_info={})

    seg2_ids = tokenizer.encode(texts[2], add_special_tokens=False)
    assert out.response_ids == seg2_ids, "row response must be the FINAL segment only"
    assert out.response_mask == [1] * len(seg2_ids)
    assert out.response_logprobs == [-0.1] * len(seg2_ids), "logprobs must align with the final segment"

    prompt_text = tokenizer.decode(out.prompt_ids, skip_special_tokens=True)
    assert _PROBLEM in prompt_text
    assert "You have attempted this problem before" in prompt_text
    assert "Count the complement." in prompt_text, "fresh prompt must carry the LATEST hint"
    assert "Use inclusion-exclusion." in prompt_text, "recap must list the earlier given hint"
    assert texts[0] not in prompt_text and texts[1] not in prompt_text, \
        "earlier attempts must NOT be in the fresh context"

    ef = out.extra_fields
    assert [a["hint_id"] for a in ef["applied_hints"]] == ["1.1", "1.2"]
    assert ef["n_restarts"] == 2 and ef["restart_prompt_overflow"] == 0
    segs = ef["restart_segments"]
    assert len(segs) == 2
    assert [s["pred"] for s in segs] == ["50", "70"]
    assert [s["next_hint_id"] for s in segs] == ["1.1", "1.2"]
    assert segs[0]["response"] == texts[0] and segs[1]["response"] == texts[1]
    # first restart: nothing verified yet -> progress-free variant (hint block only);
    # second restart: recap-carrying variant with the merged opener.
    assert segs[0]["next_user_message"].startswith("Here is a hint to help you make progress:")
    assert "You have attempted this problem before" in segs[1]["next_user_message"]
    assert ef["step_adv_turns"] == [] and ef["disable_spans"] == [], \
        "final row must not inherit archived per-segment metadata (step_adv off)"
    assert ef["turn_lens"] == [len(seg2_ids)], "turn_lens must describe only the final segment"

    # selector saw the FULL transcript (multi-turn parity), not just the last segment
    assert texts[0] in selector.calls[1]["trace"]
    assert selector.calls[1]["completed"] == ["1.1"]

    # per-segment prompts stayed short: each generate call's context is one fresh prompt
    assert len(server.prompt_lens) == 3
    assert server.prompt_lens[1] < server.prompt_lens[0] + 400, "segment 2 prompt must not accumulate segment 1"

    # the transcript keeps everything for build_trace/dumps
    roles = [m["role"] for m in (_RAW_PROMPT + [])]  # original list untouched
    assert roles == ["system", "user"]


def test_registry_repin_survives_parent_import(fix):
    """Cluster smoke #1 regression (run 20260804-192840): the yaml swap maps
    auto_hint -> RestartHintAgentLoop at worker init, but importing this module
    (hydra's first instantiation) executes the parent's @register("auto_hint")
    decorator, which CLOBBERED the mapping back to AutoHintAgentLoop for every
    later rollout -> mixed-mode rows -> DataProto.concat key assert. The module
    now re-pins the entry after the parent import; fresh-interpreter subprocesses
    verify both the re-pin and that a default (parent-only) import is untouched."""
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    # (a) restart run: yaml swap, then first import (parent clobbers, module re-pins).
    code = (
        "from verl.experimental.agent_loop.agent_loop import _agent_loop_registry\n"
        "_agent_loop_registry['auto_hint'] = {'_target_': 'restart_agent_loop.RestartHintAgentLoop'}\n"
        "import restart_agent_loop\n"
        "t = str(_agent_loop_registry['auto_hint']['_target_'])\n"
        "assert t.endswith('RestartHintAgentLoop'), t\n"
        "print('REPIN_OK', t)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=here)
    assert "REPIN_OK" in r.stdout, f"stdout={r.stdout!r}\nstderr={r.stderr[-2000:]}"
    # (b) default run: importing only the parent leaves auto_hint -> AutoHintAgentLoop.
    code2 = (
        "from verl.experimental.agent_loop.agent_loop import _agent_loop_registry\n"
        "import auto_hint_agent_loop\n"
        "t = str(_agent_loop_registry['auto_hint']['_target_'])\n"
        "assert t.endswith('AutoHintAgentLoop'), t\n"
        "print('DEFAULT_OK', t)\n"
    )
    r2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True, cwd=here)
    assert "DEFAULT_OK" in r2.stdout, f"stdout={r2.stdout!r}\nstderr={r2.stderr[-2000:]}"


def test_chain_step_adv_phantom_records(fix):
    """step_adv=True: the live row carries zero-span phantom fail records for the
    archived segments + the final segment's real record; archives keep the real
    per-segment records."""
    tokenizer, make_loop = fix
    texts = [
        "Count multiples of 2: 50. I think the answer is \\boxed{50}.",
        "Also multiples of 5: 20, so maybe \\boxed{70}.",
        "Inclusion-exclusion: 50 + 20 - 10 = 60. \\boxed{60}.",
    ]
    server = _ScriptedServer(tokenizer, texts)
    loop_obj = make_loop(server, step_adv=True)
    loop_obj._selector = _ScriptedSelector([
        {"hint_id": "1.1", "hint": "Use inclusion-exclusion.", "completed_hints": []},
        {"hint_id": "1.2", "hint": "Count the complement.", "completed_hints": []},
    ])
    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_tools_kwargs(4), extra_info={})

    n0, n1, n2 = (len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    ef = out.extra_fields
    # live list: two phantoms (fail h0 at state 0, fail h1 at state 1) + the
    # solving segment's record (state 2 -> K=2), coords relative to the final row.
    assert ef["step_adv_turns"] == [
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1],
        [0, n2, n2, 2, 2, 0],
    ], ef["step_adv_turns"]
    # archives kept the real (painted-coords) records of their own segments.
    segs = ef["restart_segments"]
    assert segs[0]["step_adv_turns"] == [[0, 0, n0, 0, 0, 1]]
    assert segs[1]["step_adv_turns"] == [[0, 0, n1, 1, 1, 1]]
    assert segs[0]["disable_spans"] == [[0, n0]] and segs[1]["disable_spans"] == [[0, n1]]
    assert ef["disable_spans"] == []


_POOL3 = (
    '{"steps": [{"step_id": 1, "purpose": "setup", "hints": ['
    '{"hint_id": "1.1", "hint": "Use inclusion-exclusion."},'
    '{"hint_id": "1.2", "hint": "Count the complement."},'
    '{"hint_id": "1.3", "hint": "Add back the double-subtracted terms."}]}]}'
)


def _pool3_kwargs(budget):
    return {"request_hint": {"create_kwargs": {
        "problem": _PROBLEM, "hints": _POOL3, "ground_truth": "60", "budget": budget,
    }}}


_REPHRASED_SELECTIONS = [
    {"hint_id": "1.1", "hint": "REPHRASED-A: think about overlaps.", "completed_hints": []},
    {"hint_id": "1.3", "hint": "REPHRASED-C: re-add what you subtracted twice.",
     "completed_hints": [{"hint_id": "1.2", "quote": "multiples of 5",
                          "why": "counted the complement",
                          "progress": "REPHRASED-PROGRESS: you counted the complement."}]},
]
_TEXTS3 = [
    "Count multiples of 2: 50. I think the answer is \\boxed{50}.",
    "Also multiples of 5: 20, so maybe \\boxed{70}.",
    "Inclusion-exclusion: 50 + 20 - 10 = 60. \\boxed{60}.",
]


def test_pool_wording_delivery(fix):
    """pool_wording (restart default ON): recap lines + the delivered hint use
    the hint set's ORIGINAL sentences; the selector's rephrasings never reach
    the prompt. Chain: hint 1.1 given, 1.2 newly self-completed, 1.3 selected."""
    tokenizer, make_loop = fix
    server = _ScriptedServer(tokenizer, _TEXTS3)
    loop_obj = make_loop(server)
    loop_obj._selector = _ScriptedSelector([dict(s) for s in _REPHRASED_SELECTIONS])
    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_pool3_kwargs(4), extra_info={})

    prompt_text = tokenizer.decode(out.prompt_ids, skip_special_tokens=True)
    # recap = pool sentences of the given 1.1 + newly-completed 1.2, pool order,
    # excluding the currently-selected 1.3 (which is the hint section instead).
    assert "1. Use inclusion-exclusion." in prompt_text
    assert "2. Count the complement." in prompt_text
    assert "Here is a hint to help you make further progress:\nAdd back the double-subtracted terms." \
        in prompt_text
    assert "REPHRASED" not in prompt_text, "selector rephrasings must not reach the prompt"
    ef = out.extra_fields
    assert [a["hint"] for a in ef["applied_hints"]] == [
        "Use inclusion-exclusion.", "Add back the double-subtracted terms."]
    assert "Add back the double-subtracted terms." in ef["restart_segments"][1]["next_user_message"]


def test_pool_wording_off_uses_selector_text(fix):
    """pool_wording=False restores the selector's rephrasings (multi-turn parity)."""
    tokenizer, make_loop = fix
    server = _ScriptedServer(tokenizer, _TEXTS3)
    loop_obj = make_loop(server, pool_wording=False)
    loop_obj._selector = _ScriptedSelector([dict(s) for s in _REPHRASED_SELECTIONS])
    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_pool3_kwargs(4), extra_info={})

    prompt_text = tokenizer.decode(out.prompt_ids, skip_special_tokens=True)
    assert "REPHRASED-C: re-add what you subtracted twice." in prompt_text
    assert "REPHRASED-A: think about overlaps." in prompt_text          # given hint, in recap
    assert "REPHRASED-PROGRESS: you counted the complement." in prompt_text  # selector progress
    ef = out.extra_fields
    assert [a["hint"] for a in ef["applied_hints"]] == [
        "REPHRASED-A: think about overlaps.", "REPHRASED-C: re-add what you subtracted twice."]


def test_chain_budget_exhausted(fix):
    tokenizer, make_loop = fix
    texts = [
        "First try \\boxed{50}.",   # wrong -> hint 1.1 -> restart
        "Second try \\boxed{70}.",  # wrong, budget spent -> chain ends here
    ]
    server = _ScriptedServer(tokenizer, texts)
    loop_obj = make_loop(server)
    loop_obj._selector = _ScriptedSelector([
        {"hint_id": "1.1", "hint": "Use inclusion-exclusion.", "completed_hints": []},
    ])
    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_tools_kwargs(1), extra_info={})

    seg1_ids = tokenizer.encode(texts[1], add_special_tokens=False)
    assert out.response_ids == seg1_ids, "row must be the last (failed) segment"
    ef = out.extra_fields
    assert len(ef["applied_hints"]) == 1 and ef["n_restarts"] == 1
    assert len(ef["restart_segments"]) == 1
    prompt_text = tokenizer.decode(out.prompt_ids, skip_special_tokens=True)
    assert "Use inclusion-exclusion." in prompt_text
    assert server.calls == 2


def test_prompt_overflow_ends_chain_without_charging(fix):
    tokenizer, make_loop = fix
    text0 = "First try \\boxed{50}."
    # initial prompt fits; the restart prompt (recap + hint) must NOT.
    probe = _ScriptedServer(tokenizer, [text0])
    initial_len = None

    async def _measure():
        nonlocal initial_len
        lp = make_loop(probe)
        ids = await lp.apply_chat_template(_RAW_PROMPT, tools=[])
        initial_len = len(ids)

    asyncio.get_event_loop().run_until_complete(_measure())

    server = _ScriptedServer(tokenizer, [text0])
    loop_obj = make_loop(server, prompt_length=initial_len + 10)
    loop_obj._selector = _ScriptedSelector([
        {"hint_id": "1.1", "hint": "Use inclusion-exclusion, and be careful with the overlap term.",
         "completed_hints": []},
    ])
    out = _run(loop_obj, raw_prompt=_RAW_PROMPT, tools_kwargs=_tools_kwargs(4), extra_info={})

    seg0_ids = tokenizer.encode(text0, add_special_tokens=False)
    assert out.response_ids == seg0_ids, "chain must end on the un-restarted segment"
    ef = out.extra_fields
    assert ef["restart_prompt_overflow"] == 1
    assert ef["applied_hints"] == [], "an undeliverable hint must NOT be charged"
    assert ef["n_restarts"] == 0 and ef["restart_segments"] == []
    prompt_text = tokenizer.decode(out.prompt_ids, skip_special_tokens=True)
    assert "You have attempted this problem before" not in prompt_text


def main():
    passed = 0
    for fn in (test_restart_message_with_progress, test_restart_message_without_progress,
               test_restart_message_empty_hint_placeholder, test_step_adv_value_parity):
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")

    fix = _verl_fixtures()
    if fix is None:
        print(f"{passed} renderer/parity tests passed; chain-mechanics tests SKIPPED (no verl env)")
        return
    for fn in (test_registry_repin_survives_parent_import,
               test_chain_three_segments, test_chain_step_adv_phantom_records,
               test_pool_wording_delivery, test_pool_wording_off_uses_selector_text,
               test_chain_budget_exhausted,
               test_prompt_overflow_ends_chain_without_charging):
        fn(fix)
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"all {passed} tests passed")


if __name__ == "__main__":
    main()
