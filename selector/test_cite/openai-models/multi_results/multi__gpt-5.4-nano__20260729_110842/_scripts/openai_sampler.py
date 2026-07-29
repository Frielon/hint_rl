"""Shared OpenAI model-call layer for the closed-model evals.

Both drivers (single-round ``run_openai_selection.py`` and multi-round
``run_openai_multi_selection.py``) monkeypatch the reused gpt_oss_eval driver's
``one_sample`` with ``openai_one_sample`` from here, so the ONLY difference vs.
the open-weight run is the endpoint + per-model API-quirk handling:

  * reasoning models (``gpt-5*`` / ``o1*`` / ``o3*`` / ``o4*``) take
    ``max_completion_tokens`` (not ``max_tokens``) + optional ``reasoning_effort``
    and reject ``temperature`` / ``top_p`` overrides (only the default is allowed);
  * chat models (``gpt-4.1-*`` / ``gpt-4o-*``) take the usual sampling params.
"""
from __future__ import annotations

import os
import time
from typing import Any


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models hide their CoT and take max_completion_tokens /
    reasoning_effort instead of max_tokens, and reject temperature/top_p."""
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


# reasoning effort applied to reasoning models; set per-run via set_reasoning_effort().
_REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low") or None


def set_reasoning_effort(effort: str | None) -> None:
    global _REASONING_EFFORT
    _REASONING_EFFORT = effort or None


def openai_one_sample(client, model, prompt, temperature, top_p, max_tokens, retries=4):
    """Single OpenAI chat completion with light exponential backoff on error
    (rate limits / transient 5xx). Returns the dict shape the drivers expect.

    ``completion_tokens`` (from API usage) counts ALL generated tokens; for
    reasoning models that INCLUDES the hidden reasoning tokens, so it stays the
    clean cross-model length/cost metric. ``finish_reason == 'length'`` means the
    token budget was exhausted before the model finished (common on reasoning
    models when max_completion_tokens is too small)."""
    reasoning = is_reasoning_model(model)
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if reasoning:
                kwargs["max_completion_tokens"] = max_tokens
                if _REASONING_EFFORT:
                    kwargs["reasoning_effort"] = _REASONING_EFFORT
                # temperature / top_p intentionally omitted (only default allowed)
            else:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
                kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            choice = resp.choices[0]
            msg = choice.message
            usage = resp.usage
            return {
                "content": msg.content or "",
                "reasoning_content": getattr(msg, "reasoning_content", None),
                "finish_reason": choice.finish_reason,
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "latency_s": latency,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(min(2 ** attempt, 8))
    return {"error": last_err}
