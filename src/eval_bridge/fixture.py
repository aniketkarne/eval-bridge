"""Fixture model + loader.

A fixture is a single JSON or YAML document describing one captured LLM
interaction: the trace ID, the prompt (or full chat history), the model,
expected/forbidden substrings, optional JSON Schema, optional reference reply
for the semantic threshold, and any free-form metadata about the capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft7Validator

from .errors import FixtureError

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schemas" / "fixture.schema.json"
_VALIDATOR: Draft7Validator | None = None


def _validator() -> Draft7Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            _VALIDATOR = Draft7Validator(json.load(fh))
    return _VALIDATOR


@dataclass
class Fixture:
    """One captured LLM interaction."""

    trace_id: str
    prompt: str
    system: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    model: str | None = None
    expected_substrings: list[str] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    semantic_threshold: float | None = None
    reference_reply: str | None = None
    capture: dict[str, Any] = field(default_factory=dict)
    fixture_response: str | None = None
    context: str | None = None
    judge_criteria: str | None = None
    judge_scorers: list[str] = field(default_factory=list)
    message_assertions: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    category: str | None = None  # e.g. "pii-leak", "tool-misuse". Optional; defaults to trace_id stem.
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # 'extra' is merged in; preserve unknown keys at top level.
        d.pop("extra", None)
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise FixtureError("fixture must be a JSON object")
        try:
            trace_id = str(data["trace_id"])
            prompt = str(data["prompt"])
        except KeyError as e:
            raise FixtureError(
                "fixture missing required field",
                context={"missing": str(e)},
            ) from e

        return cls(
            trace_id=trace_id,
            category=data.get("category"),
            prompt=prompt,
            system=data.get("system"),
            messages=list(data.get("messages") or []),
            model=data.get("model"),
            expected_substrings=list(data.get("expected_substrings") or []),
            forbidden_substrings=list(data.get("forbidden_substrings") or []),
            schema=data.get("schema"),
            semantic_threshold=data.get("semantic_threshold"),
            reference_reply=data.get("reference_reply"),
            capture=dict(data.get("capture") or {}),
            fixture_response=data.get("fixture_response"),
            context=data.get("context"),
            judge_criteria=data.get("judge_criteria"),
            judge_scorers=list(data.get("judge_scorers") or []),
            message_assertions=list(data.get("message_assertions") or []),
            expected_tool_calls=list(data.get("expected_tool_calls") or []),
            extra={k: v for k, v in data.items()
                   if k not in cls._known_fields()},
        )

    @staticmethod
    def _known_fields() -> set[str]:
        return {
            "trace_id", "prompt", "system", "messages", "model",
            "expected_substrings", "forbidden_substrings", "schema",
            "semantic_threshold", "reference_reply", "capture",
            "fixture_response", "context", "judge_criteria", "judge_scorers",
            "message_assertions", "category", "expected_tool_calls",
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate against the JSON Schema; raise FixtureError on failure."""
        errors = sorted(_validator().iter_errors(self.to_dict()), key=lambda e: e.path)
        if errors:
            msg = "; ".join(f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors)
            raise FixtureError("fixture schema validation failed", context={"errors": msg})

    # ------------------------------------------------------------------
    # Chat construction
    # ------------------------------------------------------------------

    def chat_messages(self) -> list[dict[str, str]]:
        """Return the messages array, synthesizing from prompt/system if absent."""
        if self.messages:
            return list(self.messages)
        out: list[dict[str, str]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.append({"role": "user", "content": self.prompt})
        return out


@dataclass
class FixtureSet:
    """A bundle of fixtures + a name."""

    name: str
    fixtures: list[Fixture]

    def __iter__(self) -> Iterable[Fixture]:  # type: ignore[override]
        return iter(self.fixtures)

    def __len__(self) -> int:
        return len(self.fixtures)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_one(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(raw)
    else:
        loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise FixtureError(
            "expected a JSON object at the top level",
            context={"path": str(path)},
        )
    return loaded


def load_fixture(path: str | Path) -> Fixture:
    """Load a single fixture from *path*."""
    p = Path(path)
    if not p.exists():
        raise FixtureError("fixture not found", context={"path": str(p)})
    fx = Fixture.from_dict(_read_one(p))
    fx.validate()
    return fx


def load_fixture_dir(
    path: str | Path, *, pattern: str = "*.{json,yaml,yml}"
) -> FixtureSet:
    """Load every fixture in *path* (recursive)."""
    import fnmatch

    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FixtureError("fixture dir not found", context={"path": str(p)})

    fixtures: list[Fixture] = []
    candidates: list[Path] = []
    for ext in ("json", "yaml", "yml"):
        candidates.extend(p.rglob(f"*.{ext}"))

    # Apply the user-supplied glob pattern (e.g. "test_*.json") per file name.
    for cand in candidates:
        if any(fnmatch.fnmatch(cand.name, pat) for pat in (
            pattern, "*.json", "*.yaml", "*.yml"
        )):
            fx = Fixture.from_dict(_read_one(cand))
            fx.validate()
            fixtures.append(fx)

    fixtures.sort(key=lambda f: f.trace_id)
    return FixtureSet(name=p.name, fixtures=fixtures)


__all__ = [
    "Fixture",
    "FixtureSet",
    "load_fixture",
    "load_fixture_dir",
]
