"""Import production traces from observability platforms into eval-bridge fixtures.

Supported source formats (all are JSON or JSONL):

- **OpenTelemetry GenAI semantic conventions** — the cross-vendor standard
  emitted by Phoenix, Arize, OpenLLMetry, Traceloop, and most modern LLM
  observability stacks. See ``fixtures/import/otel-sample.jsonl`` for shape.
- **Langfuse trace export** — direct API export format, also JSONL.

The importer is intentionally strict-but-tolerant: it skips records that
don't look like LLM traces, surfaces a count of imported/skipped records
on stdout, and never raises on a single bad row (one bad trace should not
kill a 10k-line import).

Usage::

    eval-bridge import otel traces.jsonl --output tests/fixtures/
    eval-bridge import langfuse langfuse-export.jsonl --output tests/fixtures/

Each imported record becomes one fixture named ``{source}-{trace_id}.json``
inside the output directory. The importer runs the trace contents through
the scrubber so PII does not leak into the fixtures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import EvalBridgeError
from .fixture import Fixture
from .scrubber import Scrubber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one JSON object per non-empty line."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise EvalBridgeError(
                    "invalid JSON on line",
                    context={"path": str(path), "line": lineno, "error": str(e)},
                ) from e


def _first_str(d: Mapping[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return None


_SAFE_TRACE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(raw: str | None, fallback: str) -> str:
    """Make *raw* filesystem-safe; fall back if empty after cleaning."""
    if not raw:
        return fallback
    cleaned = _SAFE_TRACE_ID.sub("-", raw).strip("-")
    return cleaned or fallback


# ---------------------------------------------------------------------------
# OpenTelemetry (Phoenix, Arize, Traceloop, OpenLLMetry, ...)
# ---------------------------------------------------------------------------

def _otel_to_fixture(record: dict[str, Any], idx: int) -> Fixture | None:
    """Convert one OTel record into a Fixture, or None if it isn't an LLM trace.

    Supports the common subset of GenAI semantic conventions and the
    Phoenix-flavored span attributes (``gen_ai.*``, ``llm.*``).
    """
    attrs = record.get("attributes") or {}
    if not isinstance(attrs, dict):
        return None

    name = record.get("name") or ""
    is_llm = (
        name.startswith("llm.") or name.startswith("gen_ai.")
        or "gen_ai" in attrs or "llm" in attrs
    )
    if not is_llm:
        return None

    # Span kind + timings for the capture metadata
    capture: dict[str, Any] = {
        "source": "otel",
        "name": name,
        "span_kind": record.get("span_kind") or record.get("kind"),
    }
    start = record.get("start_time") or record.get("startTimeUnixNano") or record.get("startTime")
    end = record.get("end_time") or record.get("endTimeUnixNano") or record.get("endTime")
    if start:
        capture["start_time"] = start
    if end:
        capture["end_time"] = end

    # Pull out the prompt (input) and reply (output). Different vendors
    # use different keys; we accept any of the common ones.
    prompt = _first_str(attrs, "gen_ai.prompt", "gen_ai.input", "llm.prompt",
                         "llm.input_messages", "input", "prompt")
    reply = _first_str(attrs, "gen_ai.completion", "gen_ai.output", "llm.completion",
                        "llm.output_messages", "output", "completion", "response")
    model = _first_str(attrs, "gen_ai.response.model", "gen_ai.request.model",
                        "llm.model", "model")
    trace_id = _first_str(record, "trace_id", "traceId", "context.trace_id")
    span_id = _first_str(record, "span_id", "spanId", "context.span_id")
    if span_id and not trace_id:
        trace_id = span_id

    if not prompt or not reply:
        return None

    fixture_trace_id = _safe_id(trace_id, fallback=f"otel-{idx:06d}")
    return Fixture(
        trace_id=fixture_trace_id,
        prompt=prompt,
        fixture_response=reply,
        model=model,
        capture=capture,
    )


def import_otel(path: Path, output_dir: Path, scrubber: Scrubber | None = None) -> list[Path]:
    """Import OTel/LLM-trace JSONL records as eval-bridge fixtures.

    Returns the list of fixture paths written. Bad rows are skipped with
    a warning written to stderr. Raises ``EvalBridgeError`` only on
    unparseable JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sc = scrubber or Scrubber.from_default_config()
    written: list[Path] = []
    skipped = 0

    for idx, record in enumerate(_read_jsonl(path), start=1):
        try:
            fx = _otel_to_fixture(record, idx)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if fx is None:
            skipped += 1
            continue
        # Scrub the fixture's user-visible strings before writing.
        scrubbed = sc.scrub_mapping(fx.to_dict())
        clean_fx = Fixture.from_dict(scrubbed)
        clean_fx.capture = dict(fx.capture)  # preserve raw capture metadata
        out = output_dir / f"otel-{fx.trace_id}.json"
        out.write_text(json.dumps(clean_fx.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        written.append(out)

    if skipped:
        import sys
        print(f"[import:otel] skipped {skipped} non-LLM or malformed records",
              file=sys.stderr)
    return written


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------

def _langfuse_to_fixture(record: dict[str, Any], idx: int) -> Fixture | None:
    """Convert one Langfuse export record into a Fixture.

    Langfuse exports nested ``observations`` with the prompt and reply.
    We pick the first observation whose ``type`` is ``"GENERATION"``; if
    none exists, we fall back to the first observation.
    """
    obs = record.get("observations") or []
    if not isinstance(obs, list) or not obs:
        return None

    target: dict[str, Any] | None = None
    for o in obs:
        if isinstance(o, dict) and o.get("type") in ("GENERATION", "generation"):
            target = o
            break
    if target is None:
        for o in obs:
            if isinstance(o, dict):
                target = o
                break
    if target is None:
        return None

    # Langfuse puts the input prompt in `input` (str or list of messages)
    # and the reply in `output` (str or dict).
    raw_input = target.get("input")
    raw_output = target.get("output")
    prompt = _flatten_langfuse_text(raw_input)
    reply = _flatten_langfuse_text(raw_output)
    if not prompt or not reply:
        return None

    model = target.get("model") or target.get("modelParameters", {}).get("modelName")
    trace_id = record.get("id") or record.get("traceId") or target.get("traceId")
    fixture_trace_id = _safe_id(trace_id, fallback=f"langfuse-{idx:06d}")

    capture = {
        "source": "langfuse",
        "observation_id": target.get("id"),
        "observation_type": target.get("type"),
    }
    if target.get("startTime"):
        capture["start_time"] = target["startTime"]
    if target.get("endTime"):
        capture["end_time"] = target["endTime"]
    if target.get("usage"):
        capture["usage"] = target["usage"]
    if target.get("totalCost"):
        capture["total_cost"] = target["totalCost"]

    return Fixture(
        trace_id=fixture_trace_id,
        prompt=prompt,
        fixture_response=reply,
        model=model,
        capture=capture,
    )


def _flatten_langfuse_text(value: Any) -> str | None:
    """Pull a single string out of Langfuse's polymorphic input/output shape."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # OpenAI chat format: [{"role": ..., "content": ...}, ...]
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            parts.append(c["text"])
        if parts:
            return "\n".join(parts)
        return None
    if isinstance(value, dict):
        # Some Langfuse exports nest under {"text": "..."} or similar.
        for key in ("text", "content", "completion", "prompt", "output"):
            v = value.get(key)
            if isinstance(v, str):
                return v
        return None
    return str(value)


def import_langfuse(
    path: Path, output_dir: Path, scrubber: Scrubber | None = None
) -> list[Path]:
    """Import a Langfuse JSONL export as eval-bridge fixtures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sc = scrubber or Scrubber.from_default_config()
    written: list[Path] = []
    skipped = 0

    for idx, record in enumerate(_read_jsonl(path), start=1):
        try:
            fx = _langfuse_to_fixture(record, idx)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if fx is None:
            skipped += 1
            continue
        scrubbed = sc.scrub_mapping(fx.to_dict())
        clean_fx = Fixture.from_dict(scrubbed)
        clean_fx.capture = dict(fx.capture)
        out = output_dir / f"langfuse-{fx.trace_id}.json"
        out.write_text(json.dumps(clean_fx.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        written.append(out)

    if skipped:
        import sys
        print(f"[import:langfuse] skipped {skipped} non-LLM or malformed records",
              file=sys.stderr)
    return written


__all__ = [
    "import_langfuse",
    "import_otel",
]