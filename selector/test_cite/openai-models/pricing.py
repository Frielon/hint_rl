"""OpenAI price table + cost estimation for the closed-model evals.

Prices are USD per 1,000,000 tokens as ``(input, output)``. THEY CHANGE — verify
against https://openai.com/api/pricing/ before quoting, and edit ``PRICES`` below
or override at runtime with the ``OPENAI_PRICES_JSON`` env var, e.g.
``OPENAI_PRICES_JSON='{"gpt-4o-mini": [0.15, 0.60]}'``.

Cost is an ESTIMATE: it charges ``prompt_tokens`` at the input price (ignoring any
input-cache discount the API may have applied — so this is a slight OVER-estimate
when prompt caching kicks in) and ``completion_tokens`` at the output price. For
reasoning models (gpt-5*/o*) ``completion_tokens`` already includes the billed
hidden reasoning tokens, so they are correctly priced as output.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

# USD per 1M tokens: model -> (input, output).  As of 2026-07 — VERIFY before use.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-nano":   (0.05, 0.40),
    # TODO: fill in real prices for the models you're running now (cost shows
    # n/a until then). Uncomment + set from https://openai.com/api/pricing/ :
    # "gpt-5-mini":    (0.00, 0.00),
    # "gpt-5.4-mini":  (0.00, 0.00),
    # "gpt-5.4-nano":  (0.00, 0.00),
}

PRICES_AS_OF = "2026-07 (verify at openai.com/api/pricing)"


def _prices() -> dict[str, tuple[float, float]]:
    p = dict(PRICES)
    env = os.environ.get("OPENAI_PRICES_JSON")
    if env:
        try:
            p.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(env).items()})
        except Exception:  # noqa: BLE001
            pass
    return p


def price_for(model: str) -> Optional[tuple[float, float]]:
    """(input, output) USD/1M for a model, matching exact id then a prefix (so
    dated snapshots like ``gpt-4o-mini-2024-07-18`` resolve). None if unknown."""
    p = _prices()
    if model in p:
        return p[model]
    for k, v in sorted(p.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(k):
            return v
    return None


def estimate_cost(model: str, in_tok: int, out_tok: int) -> Optional[float]:
    pr = price_for(model)
    if pr is None:
        return None
    return round(in_tok / 1e6 * pr[0] + out_tok / 1e6 * pr[1], 4)


def usage_and_cost(records: list[dict], model: str) -> dict[str, Any]:
    """Sum prompt/completion tokens over every sample in ``records`` and price
    them. Returns a JSON-able ``usage`` block for the run summary."""
    in_tok = out_tok = n_calls = 0
    for r in records:
        for s in r.get("samples", []):
            pt, ct = s.get("prompt_tokens"), s.get("completion_tokens")
            if pt is not None:
                in_tok += pt
            if ct is not None:
                out_tok += ct
            if pt is not None or ct is not None:
                n_calls += 1
    pr = price_for(model)
    return {
        "model": model,
        "n_calls_with_usage": n_calls,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "price_per_1m_usd": ({"input": pr[0], "output": pr[1]} if pr else None),
        "prices_as_of": PRICES_AS_OF if pr else None,
        "input_cost_usd": (round(in_tok / 1e6 * pr[0], 4) if pr else None),
        "output_cost_usd": (round(out_tok / 1e6 * pr[1], 4) if pr else None),
        "est_cost_usd": estimate_cost(model, in_tok, out_tok),
        "note": (None if pr else f"no price for '{model}'; set OPENAI_PRICES_JSON or edit pricing.PRICES"),
    }
