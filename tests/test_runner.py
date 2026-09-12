"""Tests for the eval runner, assertions, and providers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from eval_bridge import (
    Fixture,
    FixtureProvider,
    OpenAICompatProvider,
    ProviderError,
    Runner,
    RunnerConfig,
    Scrubber,
    jaccard,
)
from eval_bridge.runner import AssertionReport, RunnerReport, TestCaseResult


# ---------------------------------------------------------------------------
# jaccard baseline
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    assert jaccard("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard("hello world", "foo bar") == 0.0


def test_jaccard_partial():
    s = jaccard("the quick brown fox", "the slow brown dog")
    assert 0.0 < s < 1.0


def test_jaccard_both_empty():
    assert jaccard("", "") == 1.0


def test_jaccard_one_empty():
    assert jaccard("", "x") == 0.0


# ---------------------------------------------------------------------------
# Runner execution with FixtureProvider
# ---------------------------------------------------------------------------

def _fx(trace_id="t1", prompt="hi", **kw) -> Fixture:
    return Fixture(trace_id=trace_id, prompt=prompt, **kw)


def _run_one(
    fixture: Fixture,
    *,
    provider_response: str | None = None,
    simulated_latency_ms: float | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> RunnerReport:
    """Run a single fixture through the offline FixtureProvider.

    If *provider_response* is given, the fixture is rebuilt with that as
    its ``fixture_response`` (so the test can vary the canned reply
    without mutating the input fixture).

    If *simulated_latency_ms* is given, ``time.perf_counter`` is
    monkey-patched to advance by that many ms (so the runner observes a
    known duration). Requires *monkeypatch*.
    """
    if provider_response is not None:
        data = fixture.to_dict()
        data["fixture_response"] = provider_response
        fixture = Fixture.from_dict(data)
    if simulated_latency_ms is not None:
        if monkeypatch is None:
            raise ValueError("simulated_latency_ms requires a monkeypatch fixture")
        delay_s = simulated_latency_ms / 1000.0
        base = {"t": 0.0}

        def fake_perf_counter() -> float:
            # The runner takes a snapshot at start and reads it after the
            # provider call; advance once on the second read.
            base["t"] += delay_s
            return base["t"]

        monkeypatch.setattr("eval_bridge.runner.time.perf_counter", fake_perf_counter)
    return Runner(provider=FixtureProvider()).run([fixture])


def test_runner_passing_fixture():
    fx = _fx(expected_substrings=["ok"], fixture_response="ok thanks")
    report = Runner().run([fx])
    assert report.total == 1
    assert report.passed == 1
    assert report.failed == 0


def test_runner_failing_expected_substring():
    fx = _fx(expected_substrings=["nope"], fixture_response="ok thanks")
    report = Runner().run([fx])
    assert report.failed == 1
    failing = [a for a in report.results[0].assertions if not a.passed]
    assert any(a.name == "expected_substring" for a in failing)


def test_runner_failing_forbidden_substring():
    fx = _fx(forbidden_substrings=["leak"], fixture_response="this is a leak")
    report = Runner().run([fx])
    assert report.failed == 1


def test_runner_forbidden_substring_global_config():
    fx = _fx(fixture_response="this is a leak")
    cfg = RunnerConfig(forbidden_substrings=["leak"])
    report = Runner(config=cfg).run([fx])
    assert report.failed == 1


def test_runner_schema_validates_json_response():
    fx = _fx(
        fixture_response=json.dumps({"answer": 42}),
        schema={
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    report = Runner().run([fx])
    assert report.passed == 1


def test_runner_schema_fails_on_bad_json():
    fx = _fx(fixture_response="not json")
    fx.schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    report = Runner().run([fx])
    failing = [a for a in report.results[0].assertions if not a.passed]
    assert any(a.name == "schema" for a in failing)


def test_runner_semantic_threshold_pass():
    fx = _fx(
        fixture_response="the quick brown fox jumps",
        reference_reply="the quick brown fox leaps",
        semantic_threshold=0.5,
    )
    report = Runner().run([fx])
    assert report.passed == 1


def test_runner_semantic_threshold_fail():
    fx = _fx(
        fixture_response="alpha",
        reference_reply="omega beta gamma delta",
        semantic_threshold=0.5,
    )
    report = Runner().run([fx])
    failing = [a for a in report.results[0].assertions if not a.passed]
    assert any(a.name == "semantic_threshold" for a in failing)


def test_runner_residual_secret_scan_fails():
    fx = _fx(fixture_response="contact jane@example.com")
    report = Runner().run([fx])
    failing = [a for a in report.results[0].assertions if not a.passed]
    assert any(a.name == "residual_secret_scan" for a in failing)


def test_runner_residual_secret_scan_disabled():
    fx = _fx(fixture_response="contact jane@example.com")
    cfg = RunnerConfig(residual_secret_scan=False)
    report = Runner(config=cfg).run([fx])
    assert report.passed == 1


def test_runner_provider_error_marks_failed():
    fx = _fx()  # no fixture_response -> raises ProviderError
    report = Runner(provider=FixtureProvider()).run([fx])
    assert report.failed == 1
    assert report.results[0].error is not None


def test_runner_run_dir(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({
        "trace_id": "a",
        "prompt": "hi",
        "fixture_response": "ok",
    }))
    (tmp_path / "b.json").write_text(json.dumps({
        "trace_id": "b",
        "prompt": "hi",
        "fixture_response": "ok",
    }))
    report = Runner().run_dir(tmp_path)
    assert report.total == 2
    assert report.passed == 2


def test_runner_report_to_dict_shape():
    fx = _fx(fixture_response="ok")
    report = Runner().run([fx])
    d = report.to_dict()
    assert d["total"] == 1
    assert d["passed"] == 1
    assert d["failed"] == 0
    assert "results" in d
    assert "duration_s" in d


# ---------------------------------------------------------------------------
# tool_call assertion kind (Task 3)
# ---------------------------------------------------------------------------

def test_tool_call_assertion_passes_when_correct():
    fixture = Fixture(
        trace_id="tool-001",
        prompt="What's the weather in Paris?",
        fixture_response='{"result":"sunny"}',
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Paris"}},
        ],
    )
    report = _run_one(fixture, provider_response='{"name":"get_weather","arguments":{"city":"Paris"}}')
    assert report.passed, report.results[0].assertions


def test_tool_call_assertion_fails_on_wrong_tool():
    fixture = Fixture(
        trace_id="tool-002",
        prompt="...",
        fixture_response=None,
        expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
    )
    report = _run_one(fixture, provider_response='{"name":"send_email","arguments":{"to":"x@y"}}')
    assert not report.passed


# ---------------------------------------------------------------------------
# latency_ms_max budget assertion (Task 4)
# ---------------------------------------------------------------------------

def test_latency_budget_passes_under(monkeypatch: pytest.MonkeyPatch):
    fixture = Fixture(trace_id="lat-1", prompt="x", fixture_response="ok", latency_ms_max=1000)
    report = _run_one(fixture, simulated_latency_ms=50, monkeypatch=monkeypatch)
    assert report.passed
    latency_assertions = [
        a for a in report.results[0].assertions if a.name == "latency_ms"
    ]
    assert latency_assertions and latency_assertions[0].passed


def test_latency_budget_fails_over(monkeypatch: pytest.MonkeyPatch):
    fixture = Fixture(trace_id="lat-2", prompt="x", fixture_response="ok", latency_ms_max=10)
    report = _run_one(fixture, simulated_latency_ms=500, monkeypatch=monkeypatch)
    assert not report.passed
    failing = [a for a in report.results[0].assertions if not a.passed]
    assert any("latency_ms" in a.name for a in failing)
