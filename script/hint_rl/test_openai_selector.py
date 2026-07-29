# Tests for the OpenAI-API selector mode (SELECTOR_API_MODE=openai) in
# hint_selector.HintSelector -- the mode launch_hprl_cluster_openai{,_async}.sh
# turn on. Focus: from_env plumbing, per-model request kwargs, the per-process
# concurrency cap, retry backoff, and that the default LOCAL mode is unchanged.
#
# Run:  python test_openai_selector.py     (self-running, same as test_auto_hint.py)
from __future__ import annotations

import asyncio
import contextlib
import os
from types import SimpleNamespace

import hint_selector
from hint_selector import (
    API_MODE_LOCAL,
    API_MODE_OPENAI,
    HintSelector,
    OPENAI_DEFAULT_BASE_URL,
    is_reasoning_model,
    normalize_api_mode,
)

# All SELECTOR_*/OPENAI_* env the module reads; cleared inside _env() so the
# ambient shell (e.g. a sourced api_keys.sh) can't leak into assertions.
_ENV_KEYS = [
    "SELECTOR_API_MODE", "SELECTOR_OPENAI_BASE_URL", "SELECTOR_BASE_URL",
    "SELECTOR_BASE_URLS", "SELECTOR_MODEL", "SELECTOR_API_KEY",
    "SELECTOR_TEMPERATURE", "SELECTOR_TOP_P", "SELECTOR_MAX_TOKENS",
    "SELECTOR_REQUEST_TIMEOUT_S", "SELECTOR_MAX_RETRIES",
    "SELECTOR_REASONING_EFFORT", "SELECTOR_MAX_CONCURRENCY", "OPENAI_API_KEY",
]


@contextlib.contextmanager
def _env(**overrides):
    """Clear every selector env var, set ``overrides``, restore on exit."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _fast_backoff():
    """Shrink the openai-mode retry backoff so failure tests run instantly."""
    orig = hint_selector._RETRY_BACKOFF_BASE_S
    hint_selector._RETRY_BACKOFF_BASE_S = 0.001
    try:
        yield
    finally:
        hint_selector._RETRY_BACKOFF_BASE_S = orig


# --------------------------------------------------------------------------- #
# mode + model predicates
# --------------------------------------------------------------------------- #
def test_normalize_api_mode():
    assert normalize_api_mode("openai") == API_MODE_OPENAI
    assert normalize_api_mode(" OpenAI-API ") == API_MODE_OPENAI
    assert normalize_api_mode("api") == API_MODE_OPENAI
    assert normalize_api_mode("local") == API_MODE_LOCAL
    assert normalize_api_mode("") == API_MODE_LOCAL
    assert normalize_api_mode(None) == API_MODE_LOCAL
    assert normalize_api_mode("vllm") == API_MODE_LOCAL


def test_is_reasoning_model():
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("gpt-5.4-nano")
    assert is_reasoning_model("o3-mini")
    assert is_reasoning_model("o4-mini")
    assert not is_reasoning_model("gpt-4.1-mini")
    assert not is_reasoning_model("gpt-4o-mini")
    assert not is_reasoning_model("gpt-oss-20b")
    assert not is_reasoning_model(None)


# --------------------------------------------------------------------------- #
# from_env plumbing
# --------------------------------------------------------------------------- #
def test_from_env_openai_mode():
    with _env(
        SELECTOR_API_MODE="openai",
        OPENAI_API_KEY="sk-test-1234",
        # the localhost defaults the run scripts always export must be IGNORED:
        SELECTOR_BASE_URL="http://localhost:30000/v1",
        SELECTOR_BASE_URLS="http://10.20.0.1:30000/v1",
    ):
        s = HintSelector.from_env()
    assert s.api_mode == API_MODE_OPENAI
    assert s.base_urls == [OPENAI_DEFAULT_BASE_URL]
    assert s.api_key == "sk-test-1234"
    assert s.model == "gpt-5-mini"          # openai-mode default
    assert s.reasoning_effort == "low"      # default effort
    assert s.max_concurrency == 16          # default per-worker cap


def test_from_env_openai_overrides():
    with _env(
        SELECTOR_API_MODE="openai",
        SELECTOR_OPENAI_BASE_URL="https://gw.example/v1",
        SELECTOR_MODEL="gpt-4.1-mini",
        SELECTOR_REASONING_EFFORT="",
        SELECTOR_MAX_CONCURRENCY="4",
        SELECTOR_API_KEY="sk-fallback",     # no OPENAI_API_KEY -> fallback
    ):
        s = HintSelector.from_env()
    assert s.base_urls == ["https://gw.example/v1"]
    assert s.model == "gpt-4.1-mini"
    assert s.api_key == "sk-fallback"
    assert s.reasoning_effort is None
    assert s.max_concurrency == 4


def test_from_env_local_mode_unchanged():
    with _env(
        SELECTOR_BASE_URLS="http://a:30000/v1,http://b:30000/v1",
        SELECTOR_MODEL="gpt-oss-20b",
    ):
        s = HintSelector.from_env()   # no SELECTOR_API_MODE -> local
    assert s.api_mode == API_MODE_LOCAL
    assert s.base_urls == ["http://a:30000/v1", "http://b:30000/v1"]
    assert s.model == "gpt-oss-20b"
    assert s.api_key == "EMPTY"
    # openai-only knobs are inert in local mode (no loop needed: the local
    # branch returns None before ever touching asyncio):
    assert s._concurrency_sem() is None


# --------------------------------------------------------------------------- #
# per-model request kwargs
# --------------------------------------------------------------------------- #
def test_request_kwargs_openai_reasoning_model():
    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-5-mini", api_mode="openai",
                     reasoning_effort="low", max_tokens=16000)
    kw = s._request_kwargs("PROMPT")
    assert kw["model"] == "gpt-5-mini"
    assert kw["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert kw["max_completion_tokens"] == 16000
    assert kw["reasoning_effort"] == "low"
    # the API rejects non-default temperature/top_p on reasoning models:
    assert "temperature" not in kw and "top_p" not in kw and "max_tokens" not in kw


def test_request_kwargs_openai_chat_model():
    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-4.1-mini", api_mode="openai",
                     temperature=0.3, top_p=0.95, max_tokens=1000)
    kw = s._request_kwargs("P")
    assert kw["temperature"] == 0.3 and kw["top_p"] == 0.95 and kw["max_tokens"] == 1000
    assert "max_completion_tokens" not in kw and "reasoning_effort" not in kw


def test_request_kwargs_local_mode_unchanged():
    # local mode must send the ORIGINAL kwargs even for a gpt-5-named model
    # (a hypothetical locally-served checkpoint keeps vLLM semantics).
    s = HintSelector(["http://x:30000/v1"], "gpt-5-mini",
                     temperature=0.7, top_p=1.0, max_tokens=16000)
    kw = s._request_kwargs("P")
    assert kw["temperature"] == 0.7 and kw["top_p"] == 1.0 and kw["max_tokens"] == 16000
    assert "max_completion_tokens" not in kw and "reasoning_effort" not in kw


# --------------------------------------------------------------------------- #
# live-call plumbing against a FAKE client (no network)
# --------------------------------------------------------------------------- #
RAW_OK = '<output>{"hint_id": "2.1", "hint": "use the identity", "completed_hints": []}</output>'


def _fake_resp(content=RAW_OK, prompt_tokens=100, completion_tokens=10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    """Stands in for AsyncOpenAI via the selector's per-url client cache."""

    def __init__(self, behavior):
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0
        outer = self

        async def create(**kwargs):
            outer.calls.append(kwargs)
            outer.in_flight += 1
            outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
            try:
                await asyncio.sleep(0.01)
                return behavior(len(outer.calls))
            finally:
                outer.in_flight -= 1

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _wire(selector, client):
    for u in selector.base_urls:
        selector._clients[u] = client


def test_complete_openai_success_and_usage():
    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-5-mini", api_mode="openai",
                     reasoning_effort="high", max_retries=3)
    client = FakeClient(lambda n: _fake_resp())
    _wire(s, client)
    selection, raw, err = asyncio.run(s._complete("PROMPT"))
    assert err is None and raw == RAW_OK
    assert selection["hint_id"] == "2.1"
    assert client.calls[0]["reasoning_effort"] == "high"
    assert s._n_calls == 1
    assert s._usage_prompt_tokens == 100 and s._usage_completion_tokens == 10


def test_complete_openai_retry_with_backoff():
    def behavior(n):
        if n == 1:
            raise RuntimeError("rate limited (fake 429)")
        return _fake_resp()

    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-5-mini", api_mode="openai", max_retries=3)
    client = FakeClient(behavior)
    _wire(s, client)
    with _fast_backoff():
        selection, _raw, err = asyncio.run(s._complete("P"))
    assert err is None and selection is not None
    assert len(client.calls) == 2


def test_complete_openai_all_attempts_fail():
    def behavior(n):
        raise RuntimeError("boom")

    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-5-mini", api_mode="openai", max_retries=2)
    client = FakeClient(behavior)
    _wire(s, client)
    with _fast_backoff():
        selection, raw, err = asyncio.run(s._complete("P"))
    assert selection is None and raw is None
    assert "boom" in err
    assert len(client.calls) == 2


def test_concurrency_cap_openai_mode():
    s = HintSelector([OPENAI_DEFAULT_BASE_URL], "gpt-5-mini", api_mode="openai",
                     max_concurrency=2)
    client = FakeClient(lambda n: _fake_resp())
    _wire(s, client)

    async def fan_out():
        await asyncio.gather(*[s._complete("P") for _ in range(10)])

    asyncio.run(fan_out())
    assert len(client.calls) == 10
    assert client.max_in_flight <= 2


def test_no_cap_in_local_mode():
    s = HintSelector(["http://x:30000/v1"], "gpt-oss-20b", max_concurrency=16)
    client = FakeClient(lambda n: _fake_resp())
    _wire(s, client)

    async def fan_out():
        await asyncio.gather(*[s._complete("P") for _ in range(8)])

    asyncio.run(fan_out())
    # local mode never gates: with 8 concurrent calls and a 10ms fake latency,
    # the fan-out must actually overlap well beyond any would-be cap.
    assert client.max_in_flight > 4


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
