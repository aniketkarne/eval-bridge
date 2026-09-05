"""LLM provider abstraction.

Two built-in implementations:

- ``FixtureProvider`` — offline, replays ``fixture.fixture_response`` (or
  raises if missing). Deterministic; intended for CI.
- ``OpenAICompatProvider`` — talks to any OpenAI-compatible chat completion
  endpoint (OpenAI, OpenRouter, Ollama with the OpenAI shim, vLLM, etc.).

Providers are pluggable: implement the ``Provider`` protocol and pass to
``Runner``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from .errors import ProviderError
from .fixture import Fixture


# ---------------------------------------------------------------------------
# Protocol + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompletionResult:
    """The provider's reply for one fixture."""

    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """Pluggable LLM provider."""

    name: str

    def complete(self, fixture: Fixture) -> CompletionResult:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Fixture (offline) provider
# ---------------------------------------------------------------------------

class FixtureProvider:
    """Offline provider that replays ``fixture.fixture_response``."""

    name = "fixture"

    def __init__(self, default_response: str = "") -> None:
        self._default = default_response
        self.calls: list[str] = []  # for tests/debugging

    def complete(self, fixture: Fixture) -> CompletionResult:
        self.calls.append(fixture.trace_id)
        if fixture.fixture_response is not None:
            return CompletionResult(
                text=fixture.fixture_response,
                raw={"fixture": True},
                model=fixture.model,
            )
        if self._default:
            return CompletionResult(
                text=self._default,
                raw={"fixture": True, "default": True},
                model=fixture.model,
            )
        raise ProviderError(
            "fixture has no fixture_response and no default provided",
            context={"trace_id": fixture.trace_id},
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    """OpenAI-compatible chat completion HTTP provider."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("EVAL_BRIDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout_s)

    # ------------------------------------------------------------------

    def complete(self, fixture: Fixture) -> CompletionResult:
        if not self.api_key:
            raise ProviderError(
                "OpenAI-compatible provider requires an API key "
                "(EVAL_BRIDGE_API_KEY or OPENAI_API_KEY)"
            )
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": fixture.model or self.model,
            "messages": fixture.chat_messages(),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_err = e
                continue
            if resp.status_code >= 500:
                last_err = ProviderError(
                    f"server error {resp.status_code}",
                    context={"body": resp.text[:500]},
                )
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    f"client error {resp.status_code}",
                    context={"body": resp.text[:500]},
                )
            payload = resp.json()
            text = _extract_text(payload)
            return CompletionResult(
                text=text,
                raw=payload,
                model=payload.get("model", body["model"]),
                usage=payload.get("usage", {}) or {},
            )
        raise ProviderError(
            "exhausted retries",
            context={"last_error": str(last_err) if last_err else "unknown"},
        )

    def close(self) -> None:
        self._client.close()


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    msg = first.get("message") or {}
    return str(msg.get("content", ""))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def provider_from_config(cfg: Any) -> Provider:
    """Build a provider from a Config or RunnerConfig dataclass."""
    name = getattr(cfg, "provider", "fixture")
    if name == "fixture":
        return FixtureProvider()
    if name in {"openai_compat", "openai"}:
        return OpenAICompatProvider(
            base_url=getattr(cfg, "base_url", "https://api.openai.com/v1"),
            model=getattr(cfg, "model", "gpt-4o-mini"),
            timeout_s=getattr(cfg, "timeout_s", 30.0),
            max_retries=getattr(cfg, "max_retries", 2),
        )
    raise ProviderError(f"unknown provider '{name}'")


__all__ = [
    "CompletionResult",
    "FixtureProvider",
    "OpenAICompatProvider",
    "Provider",
    "provider_from_config",
]
