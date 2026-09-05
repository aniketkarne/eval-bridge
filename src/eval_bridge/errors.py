"""Typed exceptions raised by eval-bridge."""

from __future__ import annotations

from typing import Any


class EvalBridgeError(Exception):
    """Base class for all eval-bridge errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        return f"{self.message} ({self.context})"


class ScrubberError(EvalBridgeError):
    """Raised when a scrubber configuration or pattern is invalid."""


class FixtureError(EvalBridgeError):
    """Raised when a fixture file cannot be parsed or validated."""


class ProviderError(EvalBridgeError):
    """Raised when an LLM provider cannot complete a request."""


class ConfigError(EvalBridgeError):
    """Raised when eval-bridge configuration cannot be loaded."""
