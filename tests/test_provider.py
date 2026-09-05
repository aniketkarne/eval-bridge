"""Tests for provider implementations."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from eval_bridge import (
    Fixture,
    FixtureProvider,
    OpenAICompatProvider,
    ProviderError,
    provider_from_config,
)
from eval_bridge.runner import RunnerConfig


# ---------------------------------------------------------------------------
# FixtureProvider
# ---------------------------------------------------------------------------

def test_fixture_provider_returns_canned_response():
    fx = Fixture(trace_id="t", prompt="hi", fixture_response="canned answer")
    p = FixtureProvider()
    r = p.complete(fx)
    assert r.text == "canned answer"
    assert p.calls == ["t"]


def test_fixture_provider_raises_when_no_canned_response():
    fx = Fixture(trace_id="t", prompt="hi")
    p = FixtureProvider()
    with pytest.raises(ProviderError):
        p.complete(fx)


def test_fixture_provider_default_response():
    fx = Fixture(trace_id="t", prompt="hi")
    p = FixtureProvider(default_response="fallback")
    assert p.complete(fx).text == "fallback"


# ---------------------------------------------------------------------------
# OpenAICompatProvider (mocked transport)
# ---------------------------------------------------------------------------

class _MockTransport(httpx.BaseTransport):
    def __init__(self, response_json: dict[str, Any] | None = None,
                 status: int = 200, raise_on: int | None = None) -> None:
        self.response_json = response_json or {
            "choices": [{"message": {"content": "hello"}}],
            "model": "gpt-test",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        self.status = status
        self.raise_on = raise_on
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.raise_on is not None:
            raise httpx.ConnectError("boom")
        return httpx.Response(
            self.status,
            json=self.response_json,
            request=request,
        )


def test_openai_provider_sends_auth_and_parses_response():
    transport = _MockTransport()
    client = httpx.Client(transport=transport)
    p = OpenAICompatProvider(
        base_url="https://example.com/v1",
        model="m",
        api_key="sk-test",
        client=client,
    )
    fx = Fixture(trace_id="t", prompt="hi")
    r = p.complete(fx)
    assert r.text == "hello"
    assert r.usage["prompt_tokens"] == 1
    # Auth header
    auth = transport.calls[0].headers.get("Authorization")
    assert auth == "Bearer sk-test"
    # URL is correct
    assert str(transport.calls[0].url) == "https://example.com/v1/chat/completions"
    p.close()


def test_openai_provider_uses_fixture_model_when_set():
    transport = _MockTransport()
    client = httpx.Client(transport=transport)
    p = OpenAICompatProvider(
        base_url="https://example.com/v1",
        model="default-model",
        api_key="sk-test",
        client=client,
    )
    fx = Fixture(trace_id="t", prompt="hi", model="fixture-model")
    p.complete(fx)
    body = json.loads(transport.calls[0].content)
    assert body["model"] == "fixture-model"


def test_openai_provider_no_api_key_raises():
    import os
    os.environ.pop("EVAL_BRIDGE_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    p = OpenAICompatProvider(api_key=None)
    with pytest.raises(ProviderError):
        p.complete(Fixture(trace_id="t", prompt="hi"))


def test_openai_provider_client_error_does_not_retry():
    transport = _MockTransport(status=400)
    client = httpx.Client(transport=transport)
    p = OpenAICompatProvider(api_key="sk", client=client, max_retries=3)
    with pytest.raises(ProviderError):
        p.complete(Fixture(trace_id="t", prompt="hi"))


def test_openai_provider_server_error_retries_then_succeeds():
    # First call returns 500, second returns 200. Use two transports... or just
    # use a single transport that always returns 200 (retries are only on 5xx
    # and connect errors; 200 means no retry). So this just exercises the
    # happy-path retry chain by raising once then succeeding.
    class Flaky(httpx.BaseTransport):
        def __init__(self):
            self.calls = 0

        def handle_request(self, request):
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(500, json={"error": "boom"}, request=request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                request=request,
            )

    t = Flaky()
    p = OpenAICompatProvider(api_key="sk", client=httpx.Client(transport=t),
                             max_retries=2)
    r = p.complete(Fixture(trace_id="t", prompt="hi"))
    assert r.text == "ok"
    assert t.calls == 2


def test_openai_provider_exhausted_retries():
    t = _MockTransport(status=500)
    p = OpenAICompatProvider(api_key="sk", client=httpx.Client(transport=t),
                             max_retries=1)
    with pytest.raises(ProviderError):
        p.complete(Fixture(trace_id="t", prompt="hi"))


# ---------------------------------------------------------------------------
# provider_from_config factory
# ---------------------------------------------------------------------------

def test_factory_returns_fixture_provider():
    cfg = RunnerConfig(provider="fixture")
    p = provider_from_config(cfg)
    assert isinstance(p, FixtureProvider)


def test_factory_returns_openai_provider():
    cfg = RunnerConfig(provider="openai_compat")
    p = provider_from_config(cfg)
    assert isinstance(p, OpenAICompatProvider)


def test_factory_unknown_raises():
    cfg = RunnerConfig(provider="imaginary")
    with pytest.raises(ProviderError):
        provider_from_config(cfg)
