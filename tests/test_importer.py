"""Tests for the OTel/Langfuse importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bridge import Fixture, Scrubber, import_langfuse, import_otel
from eval_bridge.errors import EvalBridgeError


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_import_otel_writes_one_fixture_per_llm_span(tmp_path: Path) -> None:
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [
        {
            "name": "llm.completion",
            "trace_id": "abc123",
            "attributes": {
                "gen_ai.prompt": "What is 2+2?",
                "gen_ai.completion": "4",
                "gen_ai.request.model": "gpt-4o-mini",
            },
            "start_time": "2026-01-01T00:00:00Z",
        },
        {
            "name": "http.request",  # NOT an LLM span -> skipped
            "attributes": {"url": "https://example.com"},
        },
        {
            "name": "gen_ai.invoke",
            "trace_id": "def456",
            "attributes": {
                "gen_ai.input": "Capital of France?",
                "gen_ai.output": "Paris",
            },
        },
    ])
    out = tmp_path / "fx"
    written = import_otel(src, out)
    assert len(written) == 2
    names = sorted(p.name for p in written)
    assert names == ["otel-abc123.json", "otel-def456.json"]
    # Fixture files exist on disk.
    for p in written:
        assert p.exists()
    payload = json.loads(written[0].read_text())
    assert payload["trace_id"] == "abc123"
    assert payload["fixture_response"] == "4"


def test_import_otel_scrubs_pii(tmp_path: Path) -> None:
    """Sensitive data in the source trace must not reach the fixture file."""
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        "trace_id": "pii-1",
        "attributes": {
            "gen_ai.prompt": "ping jane@example.com with card 4111 1111 1111 1111",
            "gen_ai.completion": "got it",
        },
    }])
    out = tmp_path / "fx"
    import_otel(src, out)
    payload = json.loads((out / "otel-pii-1.json").read_text())
    assert "{email}" in payload["prompt"]
    assert "{credit_card}" in payload["prompt"]
    assert "jane@example.com" not in payload["prompt"]
    assert "4111 1111 1111 1111" not in payload["prompt"]


def test_import_otel_preserves_capture_metadata(tmp_path: Path) -> None:
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        "trace_id": "meta-1",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {
            "gen_ai.prompt": "p",
            "gen_ai.completion": "r",
        },
    }])
    out = tmp_path / "fx"
    import_otel(src, out)
    payload = json.loads((out / "otel-meta-1.json").read_text())
    assert payload["capture"]["source"] == "otel"
    assert payload["capture"]["name"] == "llm.invoke"
    assert payload["capture"]["start_time"] == "2026-01-01T00:00:00Z"


def test_import_otel_skips_record_without_prompt_or_reply(tmp_path: Path) -> None:
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        "trace_id": "incomplete",
        "attributes": {"gen_ai.prompt": "p"},  # no completion
    }])
    out = tmp_path / "fx"
    written = import_otel(src, out)
    assert written == []


def test_import_otel_handles_malformed_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text("not json\n", encoding="utf-8")
    with pytest.raises(EvalBridgeError) as exc:
        import_otel(src, tmp_path / "out")
    assert "invalid JSON" in str(exc.value)


def test_import_otel_uses_trace_id_when_present(tmp_path: Path) -> None:
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        # Some vendors nest the trace_id under "context"; we should still
        # pick it up if the top-level key is missing.
        "traceId": "ctx-trace-7",
        "attributes": {
            "gen_ai.prompt": "p",
            "gen_ai.completion": "r",
        },
    }])
    out = tmp_path / "fx"
    written = import_otel(src, out)
    assert any("ctx-trace-7" in p.name for p in written)


def test_import_otel_sanitises_trace_ids(tmp_path: Path) -> None:
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        "trace_id": "../../etc/passwd",
        "attributes": {
            "gen_ai.prompt": "p",
            "gen_ai.completion": "r",
        },
    }])
    out = tmp_path / "fx"
    written = import_otel(src, out)
    # Path traversal must be neutralised.
    assert all(p.parent == out for p in written)
    # No slashes left in the filename.
    assert all("/" not in p.stem for p in written)


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------

def test_import_langfuse_writes_one_fixture_per_record(tmp_path: Path) -> None:
    src = tmp_path / "langfuse.jsonl"
    _write_jsonl(src, [{
        "id": "trace-1",
        "observations": [{
            "id": "obs-1",
            "type": "GENERATION",
            "model": "gpt-4o-mini",
            "input": "Hello?",
            "output": "World!",
        }],
    }, {
        "id": "trace-2",
        "observations": [{
            "type": "GENERATION",
            "input": "Q?",
            "output": "A.",
        }],
    }])
    out = tmp_path / "fx"
    written = import_langfuse(src, out)
    assert len(written) == 2
    payload = json.loads((out / "langfuse-trace-1.json").read_text())
    assert payload["prompt"] == "Hello?"
    assert payload["fixture_response"] == "World!"


def test_import_langfuse_unwraps_chat_format_input(tmp_path: Path) -> None:
    """Langfuse sometimes stores input as an OpenAI chat-format list."""
    src = tmp_path / "lf.jsonl"
    _write_jsonl(src, [{
        "id": "t-chat",
        "observations": [{
            "type": "GENERATION",
            "input": [
                {"role": "system", "content": "You translate to French."},
                {"role": "user", "content": "Hello world"},
            ],
            "output": "Bonjour le monde",
        }],
    }])
    out = tmp_path / "fx"
    import_langfuse(src, out)
    payload = json.loads((out / "langfuse-t-chat.json").read_text())
    # Both messages should be joined into the prompt.
    assert "You translate to French." in payload["prompt"]
    assert "Hello world" in payload["prompt"]


def test_import_langfuse_falls_back_to_first_observation(tmp_path: Path) -> None:
    """If no GENERATION observation exists, use the first one."""
    src = tmp_path / "lf.jsonl"
    _write_jsonl(src, [{
        "id": "t-event",
        "observations": [{
            "type": "EVENT",
            "input": "the prompt",
            "output": "the reply",
        }],
    }])
    out = tmp_path / "fx"
    written = import_langfuse(src, out)
    assert len(written) == 1


def test_import_langfuse_scrubs_pii(tmp_path: Path) -> None:
    src = tmp_path / "lf.jsonl"
    _write_jsonl(src, [{
        "id": "pii-lf",
        "observations": [{
            "type": "GENERATION",
            "input": "ping jane@example.com",
            "output": "ok",
        }],
    }])
    out = tmp_path / "fx"
    import_langfuse(src, out)
    payload = json.loads((out / "langfuse-pii-lf.json").read_text())
    assert "{email}" in payload["prompt"]
    assert "jane@example.com" not in payload["prompt"]


def test_import_langfuse_skips_record_without_observations(tmp_path: Path) -> None:
    src = tmp_path / "lf.jsonl"
    _write_jsonl(src, [{"id": "empty"}])
    out = tmp_path / "fx"
    assert import_langfuse(src, out) == []


def test_import_langfuse_preserves_capture_metadata(tmp_path: Path) -> None:
    src = tmp_path / "lf.jsonl"
    _write_jsonl(src, [{
        "id": "meta-lf",
        "observations": [{
            "id": "obs-meta",
            "type": "GENERATION",
            "input": "p",
            "output": "r",
            "model": "gpt-4o-mini",
            "usage": {"totalTokens": 42},
            "totalCost": 0.0012,
        }],
    }])
    out = tmp_path / "fx"
    import_langfuse(src, out)
    payload = json.loads((out / "langfuse-meta-lf.json").read_text())
    assert payload["capture"]["source"] == "langfuse"
    assert payload["capture"]["observation_id"] == "obs-meta"
    assert payload["capture"]["total_cost"] == 0.0012
    assert payload["capture"]["usage"]["totalTokens"] == 42


# ---------------------------------------------------------------------------
# Custom scrubber is honoured
# ---------------------------------------------------------------------------

def test_import_otel_accepts_custom_scrubber(tmp_path: Path) -> None:
    from eval_bridge import ScrubberConfig
    src = tmp_path / "traces.jsonl"
    _write_jsonl(src, [{
        "name": "llm.invoke",
        "trace_id": "sc-1",
        "attributes": {
            "gen_ai.prompt": "ping jane@example.com",
            "gen_ai.completion": "ok",
        },
    }])
    out = tmp_path / "fx"
    cfg = ScrubberConfig(email="[REDACTED_EMAIL]")
    import_otel(src, out, scrubber=Scrubber(cfg))
    payload = json.loads((out / "otel-sc-1.json").read_text())
    assert "[REDACTED_EMAIL]" in payload["prompt"]