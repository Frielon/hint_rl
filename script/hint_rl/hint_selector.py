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
import logging
import os
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Offline selector prompt + tolerant <output> parser, vendored locally in
# utils.py (same as hint_tool.py) so there is no cross-folder import dependency.
from utils import selector_prompt, hint_id_of, parse_output

# Multi-round Template F prompt + status renderer (auto-hint rollout). Defined
# LOCALLY in selector_multi.py (a self-contained copy; no cross-folder import).
from selector_multi import build_prompt_multi, render_hints_with_status


# Major-step exclusion mode for the selector prompt -- i.e. which already-revealed
# major steps are dropped from the candidate pool before the NEXT selector call.
# Only meaningful under the major_step strategy. Selected by the agent loop
# (data.hprl.step_exclude_mode / env HPRL_STEP_EXCLUDE_MODE).
#   STEP_EXCLUDE_APPLIED    -- drop only the steps actually revealed so far
#                              (``exclude_applied_steps``); the default.
#   STEP_EXCLUDE_CUMULATIVE -- drop EVERY step id <= the latest revealed step
#                              (``exclude_steps_through_latest``); forces the
#                              selector strictly forward (no re-offering an earlier step).
STEP_EXCLUDE_APPLIED = "applied"
STEP_EXCLUDE_CUMULATIVE = "cumulative"
_CUMULATIVE_ALIASES = {
    "cumulative", "prefix", "through_latest", "thru_latest", "upto", "up_to", "leq", "le",
}


def normalize_step_exclude_mode(mode) -> str:
    """Canonicalize a step-exclude mode to ``STEP_EXCLUDE_CUMULATIVE`` or, by default,
    ``STEP_EXCLUDE_APPLIED`` (anything not a known cumulative alias)."""
    s = str(mode or "").strip().lower()
    return STEP_EXCLUDE_CUMULATIVE if s in _CUMULATIVE_ALIASES else STEP_EXCLUDE_APPLIED


def _step_num(value) -> Optional[int]:
    """Best-effort integer value of a (major) step id, for ordering.

    Step ids are small integers: the pool's ``step_id`` and the string form of the
    selector's ``major_step_id`` recorded on the rollout state (e.g. ``"5"``). Mirrors
    ``step_id_of``: take the part before any dot and ``int()`` it. Returns None when
    the id can't be read as a number (such ids are treated as un-orderable -- the
    cumulative filter keeps them rather than guess).
    """
    if value is None:
        return None
    head = str(value).strip().split(".")[0]
    try:
        return int(head)
    except (TypeError, ValueError):
        return None


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


def exclude_steps_through_latest(hints_obj, applied: list[dict]) -> str:
    """Return the hint pool with every MAJOR STEP up to and INCLUDING the latest
    revealed one removed -- a *cumulative* (prefix) variant of ``exclude_applied_steps``.

    Where ``exclude_applied_steps`` drops only the steps that were actually revealed,
    this drops EVERY step whose ``step_id`` is <= the highest ``major_step_id`` revealed
    so far: once the selector has revealed step 5, steps 1..5 are all removed and only
    steps 6, 7, ... remain as candidates. The premise is monotonic progress -- a student
    given step 5 is past steps 1-5 -- so the selector is forced strictly forward instead
    of being free to re-offer an earlier step. (Because each successive pick is then
    necessarily a higher step, the "latest" revealed step is also the highest, so the
    threshold never moves backward.) Steps with an unparseable id are kept (they can't be
    ordered). On any parse problem the pool is passed through unfiltered.
    """
    threshold: Optional[int] = None
    for h in applied:
        n = _step_num(h.get("major_step_id")) if isinstance(h, dict) else None
        if n is not None and (threshold is None or n > threshold):
            threshold = n

    try:
        pool = json.loads(hints_obj) if isinstance(hints_obj, str) else hints_obj
    except Exception:  # noqa: BLE001 -- malformed pool: pass it through unfiltered
        return hints_obj if isinstance(hints_obj, str) else json.dumps(hints_obj, ensure_ascii=False)

    if threshold is None or not isinstance(pool, dict) or "steps" not in pool:
        return pool if isinstance(pool, str) else json.dumps(pool, ensure_ascii=False)

    new_steps = []
    for s in pool.get("steps", []):
        n = _step_num(s.get("step_id"))
        if n is None or n > threshold:  # keep un-orderable ids + strictly-later steps
            new_steps.append(s)
    return json.dumps({**pool, "steps": new_steps}, ensure_ascii=False)


def pool_is_exhausted(hints_str) -> bool:
    """True when the (post-exclusion) candidate pool has no hint left to offer.

    The selector prompt lists candidates as major steps each carrying a ``hints``
    list; once exclusion has removed every step (or emptied every step's hints)
    there is nothing for the selector to pick. Detecting this lets the agent loop
    skip a pointless -- and re-offer-prone -- selector round-trip and inject the
    "no hint available" turn directly. This is the common terminal state of the
    cumulative step-exclude mode (once the latest/highest step is revealed the pool
    is empty), but it also happens in any mode when the budget outlasts the pool.

    A pool that can't be PARSED is treated as NOT exhausted: that's a malformed/empty
    pool (a config issue), distinct from "ran out of hints" -- fall through to the
    normal selector path and its existing failure handling rather than masking it.
    """
    try:
        pool = json.loads(hints_str) if isinstance(hints_str, str) else hints_str
    except Exception:  # noqa: BLE001 -- unparseable pool: not the exhausted case
        return False
    if not isinstance(pool, dict) or "steps" not in pool:
        return False
    for s in pool.get("steps", []):
        if isinstance(s, dict) and s.get("hints"):
            return False  # a step with >=1 candidate hint remains -> not exhausted
    return True


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


# --- selector API mode ------------------------------------------------------
# Which backend serves the selector calls:
#   API_MODE_LOCAL  -- self-hosted vLLM endpoint(s) (gpt-oss-20b selector pods,
#                      launch_hprl_cluster.sh); the original and default mode.
#   API_MODE_OPENAI -- the REAL OpenAI API (api.openai.com): no selector pods at
#                      all (launch_hprl_cluster_openai.sh). Endpoint from
#                      SELECTOR_OPENAI_BASE_URL, key from OPENAI_API_KEY, and
#                      per-model request quirks handled below (reasoning models
#                      take max_completion_tokens + reasoning_effort and reject
#                      temperature/top_p -- same handling the offline eval's
#                      openai_sampler validated).
API_MODE_LOCAL = "local"
API_MODE_OPENAI = "openai"
_OPENAI_MODE_ALIASES = {"openai", "api", "openai_api", "openai-api"}
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Backoff between RETRIES in openai mode only (rate limits: an immediate re-send
# mostly burns the attempt). Exponential from base, capped, x[0.5, 1.5) jitter.
# Local mode keeps the original behavior: instant failover to the next server.
_RETRY_BACKOFF_BASE_S = 2.0
_RETRY_BACKOFF_CAP_S = 30.0
# Log cumulative token usage every N successful calls (openai mode; 0 disables).
# Per worker process -- sum across processes for the run total; the per-call
# selector dump (HPRL_SELECTOR_DUMP_DIR) remains the exact record.
_USAGE_LOG_EVERY = 200


def normalize_api_mode(mode) -> str:
    """Canonicalize an api mode to ``API_MODE_OPENAI`` or, by default, ``API_MODE_LOCAL``."""
    s = str(mode or "").strip().lower()
    return API_MODE_OPENAI if s in _OPENAI_MODE_ALIASES else API_MODE_LOCAL


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models (gpt-5*/o1*/o3*/o4*) hide their CoT and take
    max_completion_tokens / reasoning_effort instead of max_tokens, and reject
    temperature/top_p overrides (only the defaults are allowed). Same predicate
    as the offline eval's openai_sampler.is_reasoning_model."""
    m = str(model or "").lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


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
        api_mode: str = API_MODE_LOCAL,
        reasoning_effort: Optional[str] = None,
        max_concurrency: int = 0,
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
        self.api_mode = normalize_api_mode(api_mode)
        self.reasoning_effort = (reasoning_effort or "").strip() or None
        self.max_concurrency = int(max_concurrency)
        self._clients: dict[str, Any] = {}  # base_url -> AsyncOpenAI (lazy)
        # In-flight cap (openai mode only; see _concurrency_sem). The semaphore
        # is created lazily on the running event loop of the calling worker.
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_loop: Any = None
        # Cumulative API usage of THIS worker process (openai-mode cost visibility).
        self._n_calls = 0
        self._usage_prompt_tokens = 0
        self._usage_completion_tokens = 0

    @classmethod
    def from_env(cls) -> "HintSelector":
        """Build from the SELECTOR_* env vars the run script already exports.

        ``SELECTOR_API_MODE=openai`` switches the backend to the real OpenAI API:
        the endpoint then comes from ``SELECTOR_OPENAI_BASE_URL`` (NOT from the
        ``SELECTOR_BASE_URL(S)`` pair, whose localhost defaults are meaningless
        without selector pods) and the key from ``OPENAI_API_KEY`` (fallback:
        ``SELECTOR_API_KEY``). In the default local mode ``SELECTOR_BASE_URLS``
        (comma-separated, one per independent selector server) takes precedence
        over the single ``SELECTOR_BASE_URL``; the openai-only knobs are inert.
        """
        api_mode = normalize_api_mode(os.environ.get("SELECTOR_API_MODE", API_MODE_LOCAL))
        if api_mode == API_MODE_OPENAI:
            urls = os.environ.get("SELECTOR_OPENAI_BASE_URL") or OPENAI_DEFAULT_BASE_URL
            api_key = (
                (os.environ.get("OPENAI_API_KEY") or "").strip()
                or os.environ.get("SELECTOR_API_KEY", "EMPTY")
            )
            model = os.environ.get("SELECTOR_MODEL", "gpt-5-mini")
            if model.lower().startswith("gpt-oss"):
                # SELECTOR_MODEL kept its local-serving default; the OpenAI API has
                # no such model, so every call would 404 (and degrade to no-hint).
                logger.warning(
                    "HintSelector: SELECTOR_API_MODE=openai but SELECTOR_MODEL=%r looks "
                    "like a locally-served model -- set SELECTOR_MODEL to a real OpenAI "
                    "model (e.g. gpt-5-mini) or every hint call will fail.",
                    model,
                )
        else:
            urls = os.environ.get("SELECTOR_BASE_URLS") or os.environ.get(
                "SELECTOR_BASE_URL", "http://localhost:30000/v1"
            )
            api_key = os.environ.get("SELECTOR_API_KEY", "EMPTY")
            model = os.environ.get("SELECTOR_MODEL", "Qwen3.5-27B")
        selector = cls(
            base_urls=urls,
            model=model,
            api_key=api_key,
            temperature=float(os.environ.get("SELECTOR_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("SELECTOR_TOP_P", "0.95")),
            max_tokens=int(os.environ.get("SELECTOR_MAX_TOKENS", "16000")),
            timeout=float(os.environ.get("SELECTOR_REQUEST_TIMEOUT_S", "600")),
            max_retries=int(os.environ.get("SELECTOR_MAX_RETRIES", "3")),
            api_mode=api_mode,
            reasoning_effort=os.environ.get("SELECTOR_REASONING_EFFORT", "low"),
            max_concurrency=int(os.environ.get("SELECTOR_MAX_CONCURRENCY", "16")),
        )
        if selector.api_mode == API_MODE_OPENAI:
            logger.warning(
                "HintSelector: OPENAI API mode -- model=%s base=%s effort=%s "
                "max_concurrency=%d retries=%d timeout=%.0fs key=...%s",
                selector.model,
                ",".join(selector.base_urls),
                selector.reasoning_effort,
                selector.max_concurrency,
                selector.max_retries,
                selector.timeout,
                (api_key[-4:] if api_key and api_key != "EMPTY" else "UNSET"),
            )
        return selector

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
        """Pick a hint with the single-pick v4_cite prompt (utils.selector_prompt).

        Returns (selection_dict, raw_text, err). selection is None on failure.
        Used by the ``<hint_call/>`` rollout (hint_agent_loop).
        """
        return await self._complete(selector_prompt(problem, trace, hints_str))

    async def select_multi(
        self, problem: str, trace: str, pool, completed
    ) -> tuple[Optional[dict], Optional[str], Optional[str], str]:
        """Pick the next hint with the MULTI-ROUND Template F prompt (selector_multi).

        Renders the WHOLE pool with per-hint ``status`` (completed | pending) -- the
        auto-hint rollout's status-marking mechanism -- so the selector picks the
        next pending hint and also reports, in ``completed_hints``, any pending hint
        the student newly achieved this round (each with a verbatim ``quote``). Used
        by the auto-hint rollout (auto_hint_agent_loop). ``pool`` is the FULL hint
        pool (JSON str or dict); ``completed`` the ids already given/verified.

        Returns (selection_dict, raw_text, err, prompt); selection is None on
        failure. ``prompt`` is the EXACT text sent to the selector (so the rollout
        viewer can show it verbatim). The parsed dict carries at least ``hint_id`` /
        ``hint`` / ``completed_hints``.
        """
        hints_rendered = render_hints_with_status(pool, completed)
        prompt = build_prompt_multi(problem, trace, hints_rendered)
        selection, raw, err = await self._complete(prompt)
        return selection, raw, err, prompt

    def _request_kwargs(self, prompt: str) -> dict[str, Any]:
        """Chat-completion kwargs for one selector call, per backend/model quirks.

        Local mode (and openai-mode CHAT models like gpt-4.1-mini) sends the
        usual sampling params. Openai-mode REASONING models (gpt-5*/o*) instead
        take ``max_completion_tokens`` (which budgets the hidden reasoning
        tokens too) plus ``reasoning_effort``, and reject any non-default
        ``temperature`` / ``top_p`` -- so those are omitted entirely.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.api_mode == API_MODE_OPENAI and is_reasoning_model(self.model):
            kwargs["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def _concurrency_sem(self) -> Optional[asyncio.Semaphore]:
        """Per-process in-flight cap, OPENAI MODE ONLY (None = uncapped).

        The API enforces org-wide RPM/TPM limits, and one training step can fan
        out hundreds of concurrent hint calls across the agent-loop workers --
        uncapped, they'd trip 429s in bursts and burn the retry budget. Local
        vLLM WANTS the full fan-out (continuous batching), so the cap is never
        applied there. Lazily bound to the calling worker's running event loop.
        """
        if self.api_mode != API_MODE_OPENAI or self.max_concurrency <= 0:
            return None
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        return self._sem

    async def _one_request(self, base_url: str, prompt: str):
        """One chat completion against ``base_url`` (slot-capped in openai mode)."""
        kwargs = self._request_kwargs(prompt)

        async def _issue():
            return await asyncio.wait_for(
                self._get_client(base_url).chat.completions.create(**kwargs),
                timeout=self.timeout,
            )

        sem = self._concurrency_sem()
        if sem is None:
            return await _issue()
        async with sem:
            return await _issue()

    def _note_usage(self, resp) -> None:
        """Accumulate API token usage; periodically log it in openai mode.

        Per worker process. ``completion_tokens`` includes the hidden reasoning
        tokens of reasoning models (that's what's billed), so the log line is an
        honest cost proxy: cost ~ prompt_tokens x input price + completion_tokens
        x output price for SELECTOR_MODEL.
        """
        self._n_calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._usage_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._usage_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        if (
            self.api_mode == API_MODE_OPENAI
            and _USAGE_LOG_EVERY > 0
            and self._n_calls % _USAGE_LOG_EVERY == 0
        ):
            logger.warning(
                "HintSelector[%s]: %d calls this worker -- prompt_tokens=%d "
                "completion_tokens=%d (sum across workers for the run total)",
                self.model,
                self._n_calls,
                self._usage_prompt_tokens,
                self._usage_completion_tokens,
            )

    async def _complete(
        self, prompt: str
    ) -> tuple[Optional[dict], Optional[str], Optional[str]]:
        """Send one selector prompt and return (selection_dict, raw_text, err).

        Spreads load across the independent selector servers (random start per
        call) and fails over to a different server on each retry -- so a slow or
        down server is skipped rather than failing the call. With >1 server keep
        ``max_retries`` >= 2 so a failover attempt is actually made. Shared by
        ``select`` (single-pick prompt) and ``select_multi`` (multi-round prompt);
        the only difference between the two is which prompt template is sent.

        Openai mode adds an exponential backoff (+jitter) before each retry --
        there is one shared endpoint, so retry-after-a-beat is what recovers a
        rate-limit burst, not failover. Local mode keeps the original instant
        failover (no sleeps).
        """
        n = len(self.base_urls)
        start = random.randrange(n)
        last_err = None
        for attempt in range(self.max_retries):
            base_url = self.base_urls[(start + attempt) % n]
            if attempt > 0 and self.api_mode == API_MODE_OPENAI:
                delay = min(_RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)), _RETRY_BACKOFF_CAP_S)
                await asyncio.sleep(delay * (0.5 + random.random()))
            try:
                resp = await self._one_request(base_url, prompt)
                self._note_usage(resp)
                raw = resp.choices[0].message.content or ""
                selection, perr = parse_output(raw)
                if selection is not None:
                    return selection, raw, None
                last_err = f"parse failed: {perr}"
            except Exception as e:  # noqa: BLE001
                last_err = f"{base_url}: {e}"
        return None, None, last_err
