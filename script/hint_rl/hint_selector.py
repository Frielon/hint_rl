# Copyright 2026
#
# HintSelector -- the frozen-selector client + trace builder for HPRL, factored
# out so it can be used WITHOUT a verl tool.
#
# In the original design the selector was called from inside HintTool (a verl
# BaseTool). The ``<hint_call/>`` mechanism doesn't use a verl tool at all -- the
# agent loop (hint_agent_loop.HintAgentLoop) detects the sentinel and calls the
# selector directly -- so the selector logic lives here, independent of the tool
# lifecycle. It reuses the exact offline selector prompt + tolerant <output>
# parser (under ``${HINT_RL_HOME}/selector``), same as HintTool.

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any, Optional


# Offline selector prompt + tolerant <output> parser, vendored locally in
# utils.py (same as hint_tool.py) so there is no cross-folder import dependency.
from utils import selector_prompt, hint_id_of, parse_output


def exclude_applied_hints(hints_obj, applied: list[dict]) -> str:
    """Return the hint pool as a JSON string with already-applied hints removed.

    The pool is ``{"steps": [{"step_id", "purpose", "hints": [{"hint_id", ...}]}]}``.
    Every candidate hint whose ``hint_id`` already appears in ``applied`` is dropped
    (so the selector cannot re-offer it), and any major step left with no remaining
    hints is dropped too. On any parse problem the pool is passed through unfiltered.
    """
    applied_ids = {str(h.get("hint_id")) for h in applied if h.get("hint_id") is not None}

    try:
        pool = json.loads(hints_obj) if isinstance(hints_obj, str) else hints_obj
    except Exception:  # noqa: BLE001 -- malformed pool: pass it through unfiltered
        return hints_obj if isinstance(hints_obj, str) else json.dumps(hints_obj, ensure_ascii=False)

    if not applied_ids or not isinstance(pool, dict) or "steps" not in pool:
        return pool if isinstance(pool, str) else json.dumps(pool, ensure_ascii=False)

    new_steps = []
    for step in pool.get("steps", []):
        kept = [h for h in step.get("hints", []) if str(h.get("hint_id")) not in applied_ids]
        if kept:
            new_steps.append({**step, "hints": kept})
    return json.dumps({**pool, "steps": new_steps}, ensure_ascii=False)


def exclude_applied_steps(hints_obj, applied: list[dict]) -> str:
    """Return the hint pool as a JSON string with whole already-revealed MAJOR STEPS removed.

    The major-step strategy reveals an ENTIRE major step per hint call and records
    it (by ``major_step_id``) in the rollout state, so the next selector prompt must
    drop every step already revealed -- not just the individual hints. Any step
    whose ``step_id`` matches an applied ``major_step_id`` is removed wholesale, so
    the selector is forced to pick a NEW major step. On any parse problem the pool
    is passed through unfiltered.
    """
    used_steps = {str(h.get("major_step_id")) for h in applied if h.get("major_step_id") is not None}

    try:
        pool = json.loads(hints_obj) if isinstance(hints_obj, str) else hints_obj
    except Exception:  # noqa: BLE001 -- malformed pool: pass it through unfiltered
        return hints_obj if isinstance(hints_obj, str) else json.dumps(hints_obj, ensure_ascii=False)

    if not used_steps or not isinstance(pool, dict) or "steps" not in pool:
        return pool if isinstance(pool, str) else json.dumps(pool, ensure_ascii=False)

    new_steps = [s for s in pool.get("steps", []) if str(s.get("step_id")) not in used_steps]
    return json.dumps({**pool, "steps": new_steps}, ensure_ascii=False)


def hints_for_step(hints_obj, step_id) -> list[dict]:
    """Return the full ordered hint list of the major step ``step_id`` from the pool.

    Used by the major-step strategy to surface EVERY hint in the step the selector
    identified (guidance ``X.0`` first, then the substep hints ``X.1``, ``X.2`` ...,
    in pool order). Returns ``[]`` if the pool can't be parsed or the step isn't found.
    """
    try:
        pool = json.loads(hints_obj) if isinstance(hints_obj, str) else hints_obj
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(pool, dict):
        return []
    target = str(step_id)
    for step in pool.get("steps", []):
        if str(step.get("step_id")) == target:
            return list(step.get("hints", []) or [])
    return []


def format_step_hints(hints: list[dict]) -> str:
    """Render a major step's hints as a numbered list for injection into the rollout.

    Keeps just the hint text (the policy doesn't need the ``type``/``hint_id`` keys),
    one per line in pool order. Empty hint texts are skipped; numbering stays gapless.
    """
    lines: list[str] = []
    for h in hints:
        text = (h.get("hint") if isinstance(h, dict) else str(h)) or ""
        text = text.strip()
        if text:
            lines.append(f"{len(lines) + 1}. {text}")
    return "\n".join(lines)


def step_id_of(selection) -> Optional[str]:
    """The major-step id of a selector result, as a string (or None).

    Prefers ``major_step_id``; falls back to the integer part of ``hint_id``
    (``"2.1" -> "2"``) when the selector returned a hint id but no explicit step id.
    """
    if not isinstance(selection, dict):
        return None
    sid = selection.get("major_step_id")
    if sid is not None:
        return str(sid)
    hid = selection.get("hint_id")
    if hid is not None:
        return str(hid).split(".")[0]
    return None


def build_trace(messages: list[dict]) -> str:
    """Render the student's reasoning trace for the selector.

    The conversation is: system, user(problem), assistant(reasoning [+<hint_call/>]),
    user(hint), assistant(...), .... We feed the selector the assistant reasoning
    so far plus any previously injected hints (the non-first user turns), but NOT
    the original problem statement (it is passed to the selector separately).
    """
    parts: list[str] = []
    seen_problem = False
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not content:
            continue
        if role == "assistant":
            parts.append(str(content))
        elif role == "user":
            if not seen_problem:
                seen_problem = True  # the problem statement itself -> skip
                continue
            parts.append(f"[hint given] {content}")
    trace = "\n\n".join(parts).strip()
    return trace or "(The student has not written any reasoning yet.)"


class HintSelector:
    """Async client for the frozen selector model on an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_urls,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 16000,
        timeout: float = 600.0,
        max_retries: int = 3,
    ):
        # base_urls: a list, or a comma-separated string -- one entry per
        # INDEPENDENT selector server. Client-side round-robin + failover across
        # them (see select) uses every selector node without any cross-node DP
        # coordination or a proxy single-point-of-failure.
        if isinstance(base_urls, str):
            base_urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        self.base_urls = list(base_urls) or ["http://localhost:30000/v1"]
        self.base_url = self.base_urls[0]  # back-compat (first endpoint)
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._clients: dict[str, Any] = {}  # base_url -> AsyncOpenAI (lazy)

    @classmethod
    def from_env(cls) -> "HintSelector":
        """Build from the SELECTOR_* env vars the run script already exports.

        ``SELECTOR_BASE_URLS`` (comma-separated, one per independent selector
        server) takes precedence over the single ``SELECTOR_BASE_URL``.
        """
        urls = os.environ.get("SELECTOR_BASE_URLS") or os.environ.get(
            "SELECTOR_BASE_URL", "http://localhost:30000/v1"
        )
        return cls(
            base_urls=urls,
            model=os.environ.get("SELECTOR_MODEL", "Qwen3.5-27B"),
            api_key=os.environ.get("SELECTOR_API_KEY", "EMPTY"),
            temperature=float(os.environ.get("SELECTOR_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("SELECTOR_TOP_P", "0.95")),
            max_tokens=int(os.environ.get("SELECTOR_MAX_TOKENS", "16000")),
            timeout=float(os.environ.get("SELECTOR_REQUEST_TIMEOUT_S", "600")),
            max_retries=int(os.environ.get("SELECTOR_MAX_RETRIES", "3")),
        )

    def _get_client(self, base_url: str):
        client = self._clients.get(base_url)
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
            self._clients[base_url] = client
        return client

    async def select(
        self, problem: str, trace: str, hints_str: str
    ) -> tuple[Optional[dict], Optional[str], Optional[str]]:
        """Pick a hint. Returns (selection_dict, raw_text, err). selection is None on failure.

        Spreads load across the independent selector servers (random start per
        call) and fails over to a different server on each retry -- so a slow or
        down server is skipped rather than failing the call. With >1 server keep
        ``max_retries`` >= 2 so a failover attempt is actually made.
        """
        prompt = selector_prompt(problem, trace, hints_str)
        n = len(self.base_urls)
        start = random.randrange(n)
        last_err = None
        for attempt in range(self.max_retries):
            base_url = self.base_urls[(start + attempt) % n]
            try:
                resp = await asyncio.wait_for(
                    self._get_client(base_url).chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout,
                )
                raw = resp.choices[0].message.content or ""
                selection, perr = parse_output(raw)
                if selection is not None:
                    return selection, raw, None
                last_err = f"parse failed: {perr}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{base_url}: {e}"
        return None, None, last_err
