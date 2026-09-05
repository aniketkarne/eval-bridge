"""Configuration loader for eval-bridge.

Reads ``eval-bridge.toml`` from the current directory if present and merges it
with sensible defaults. We deliberately avoid a TOML parser dependency: we use
Python's built-in ``tomllib`` (3.11+) or ``tomli`` fallback, but the schema is
small enough that a minimal regex parser would also work. For simplicity, and
because ``tomllib`` ships with Python 3.11+, we require 3.11+ only for config
files; the rest of the package still supports 3.10. When no config is present
we return defaults - no exception is raised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .runner import RunnerConfig
from .scrubber import (
    Pattern,
    ScrubberConfig,
    build_extra_patterns,
)

CONFIG_FILENAME = "eval-bridge.toml"


@dataclass
class Config:
    """Top-level config: scrubber + runner."""

    scrubber: ScrubberConfig = field(default_factory=ScrubberConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as e:
            raise ConfigError(
                "Python 3.11+ required to read TOML, or install 'tomli'",
                context={"path": str(path)},
            ) from e
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(
    path: str | Path | None = None,
    *,
    search_cwd: bool = True,
) -> Config:
    """Load config from *path*, or ``eval-bridge.toml`` in cwd, or defaults."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if search_cwd:
        candidates.append(Path.cwd() / CONFIG_FILENAME)
        # Also try the parent of the fixtures dir (common project layout).
        candidates.append(Path.cwd().parent / CONFIG_FILENAME)

    raw: dict[str, Any] = {}
    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                raw = _read_toml(cand)
            except Exception as e:
                raise ConfigError(
                    "failed to parse config", context={"path": str(cand), "error": str(e)}
                ) from e
            break

    scrubber_raw = raw.get("scrubber", {}) or {}
    runner_raw = raw.get("runner", {}) or {}
    assertions_raw = runner_raw.get("assertions", {}) or {}

    extra_raw = scrubber_raw.get("extra_patterns", []) or []
    extras: list[Pattern] = []
    for item in extra_raw:
        try:
            extras.extend(build_extra_patterns([item]))
        except Exception as e:
            raise ConfigError(
                "invalid scrubber.extra_patterns entry",
                context={"entry": str(item), "error": str(e)},
            ) from e

    scrubber_cfg = ScrubberConfig(
        email=scrubber_raw.get("email", "{email}"),
        ip=scrubber_raw.get("ip", "{ip}"),
        jwt=scrubber_raw.get("jwt", "{jwt}"),
        api_token=scrubber_raw.get("api_token", "{api_token}"),
        bearer=scrubber_raw.get("bearer", "{api_token}"),
        credit_card=scrubber_raw.get("credit_card", "{credit_card}"),
        ssn=scrubber_raw.get("ssn", "{ssn}"),
        extra_patterns=tuple(extras),
    )

    runner_cfg = RunnerConfig(
        provider=runner_raw.get("provider", "fixture"),
        base_url=runner_raw.get("base_url", "https://api.openai.com/v1"),
        model=runner_raw.get("model", "gpt-4o-mini"),
        timeout_s=float(runner_raw.get("timeout_s", 30.0)),
        max_retries=int(runner_raw.get("max_retries", 2)),
        forbidden_substrings=list(assertions_raw.get("forbidden_substrings", []) or []),
        semantic_threshold=float(assertions_raw.get("semantic_threshold", 0.0)),
        residual_secret_scan=bool(assertions_raw.get("residual_secret_scan", True)),
        scorer_names=tuple(assertions_raw.get("scorer_names", []) or ()),
        judge_model=runner_raw.get("judge_model", "gpt-4o-mini"),
    )

    return Config(scrubber=scrubber_cfg, runner=runner_cfg)


def write_default_config(path: str | Path) -> Path:
    """Write a documented default ``eval-bridge.toml`` to *path*."""
    content = '''# eval-bridge configuration. All fields are optional.

[scrubber]
email       = "{email}"
ip          = "{ip}"
jwt         = "{jwt}"
api_token   = "{api_token}"
bearer      = "{api_token}"      # Bearer/Token/Basic auth headers
credit_card = "{credit_card}"
ssn         = "{ssn}"

# extra_patterns = [
#   { name = "internal_ticket", pattern = "TICKET-\\d{4,}", replacement = "{ticket}" },
# ]

[runner]
provider   = "fixture"            # "fixture" | "openai_compat"
base_url   = "https://api.openai.com/v1"
model      = "gpt-4o-mini"
timeout_s  = 30
max_retries = 2
judge_model = "gpt-4o-mini"      # separate from `model` — usually cheaper is fine

[runner.assertions]
forbidden_substrings    = ["BEGIN PRIVATE KEY"]
semantic_threshold      = 0.0
residual_secret_scan    = true
# scorer_names = ["hallucination", "faithfulness", "answer_relevance", "toxicity", "bias"]
'''
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


__all__ = [
    "CONFIG_FILENAME",
    "Config",
    "load_config",
    "write_default_config",
]
