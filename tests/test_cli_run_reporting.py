"""CLI-level accounting and reporting guarantees.

- ``capture``: the fixture written to disk contains no scrubbed-away PII (keys
  included) and its ``scrubber_counts`` describe the replacements actually
  made to that payload - not a second scrub over a JSON serialization.
- ``run``: a hard baseline regression (``regressed``/``removed``) must not
  destroy the run's report - the JUnit artifact and summary print before the
  exit-3, because that is exactly the CI run that needs them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from eval_bridge.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# capture: scrubber_counts truth
# ---------------------------------------------------------------------------

def test_capture_scrubs_pii_keys_and_counts_the_replacement(runner, tmp_path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "t-key",
        "prompt": "clean prompt",
        "meta": {"jane@example.com": "value"},
    }))
    out = tmp_path / "fx.json"
    result = runner.invoke(
        main, ["capture", str(incident), "--output", str(out), "--write"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    blob = json.dumps(payload)
    # The PII key must not survive into the written fixture...
    assert "jane@example.com" not in blob
    assert payload["meta"] == {"{email}": "value"}
    # ...and the counts must describe exactly what was replaced there.
    counts = dict(payload["scrubber_counts"])
    assert counts.get("email") == 1


def test_capture_counts_cover_replacements_missed_by_serialization_scrub(runner, tmp_path):
    # A config extra pattern that matches only per-leaf: json.dumps escapes
    # the newline, so scrubbing the serialization counts zero replacements
    # even though the written payload changed.
    cfg = tmp_path / "eval-bridge.toml"
    cfg.write_text(
        "[scrubber]\n"
        'extra_patterns = [\n'
        '  { name = "pair", pattern = "line1\\\\nline2", replacement = "{pair}" },\n'
        "]\n"
    )
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "t-nl",
        "prompt": "x\nline1\nline2\ny",
    }))
    out = tmp_path / "fx.json"
    result = runner.invoke(
        main,
        ["capture", str(incident), "--output", str(out), "--write", "-c", str(cfg)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["prompt"] == "x\n{pair}\ny"
    counts = dict(payload["scrubber_counts"])
    assert counts.get("pair") == 1


def test_capture_variant_counts_describe_the_written_variant(runner, tmp_path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "t-var",
        "prompt": "contact jane@example.com",
        "meta": {"jane@example.com": "value"},
    }))
    out = tmp_path / "fx.json"
    result = runner.invoke(
        main,
        ["capture", str(incident), "--output", str(out), "--write", "--mutate", "2"],
    )
    assert result.exit_code == 0, result.output
    # No variant may carry scrubbed-away PII - not even in dict keys - and
    # whatever scrubber_counts each variant reports must be an account of
    # replacements in that variant's own written payload.
    for n in (1, 2):
        vp = json.loads((tmp_path / f"fx.m{n}.json").read_text())
        blob = json.dumps(vp)
        assert "jane@example.com" not in blob
        counts = dict(vp["scrubber_counts"])
        assert isinstance(vp["scrubber_counts"], list)
        for cnt in counts.values():
            assert cnt >= 1


# ---------------------------------------------------------------------------
# run: reports survive the exit-3
# ---------------------------------------------------------------------------

def _write_fixture(fixtures: Path, trace_id: str, response: str, expected=None):
    fx = {"trace_id": trace_id, "prompt": "hi", "fixture_response": response}
    if expected is not None:
        fx["expected_substrings"] = expected
    (fixtures / f"{trace_id}.json").write_text(json.dumps(fx))


def test_run_baseline_regression_writes_junit_before_exit_3(runner, tmp_path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    _write_fixture(fixtures, "a", "ok", expected=["ok"])
    baseline = tmp_path / "baseline.json"
    r1 = runner.invoke(
        main, ["run", str(fixtures), "--write-baseline", "--baseline", str(baseline)]
    )
    assert r1.exit_code == 0, r1.output
    assert baseline.exists()

    # The fixture now fails its assertion: passed True -> False = "regressed".
    _write_fixture(fixtures, "a", "ok", expected=["nope"])
    junit = tmp_path / "junit.xml"
    r2 = runner.invoke(
        main, ["run", str(fixtures), "--baseline", str(baseline), "--junit", str(junit)]
    )
    assert r2.exit_code == 3, r2.output
    # The CI artifact must exist even though the run failed hard.
    assert junit.exists()
    assert "regressed" in r2.stdout
    assert "total=1" in r2.stdout


def test_run_removed_trace_writes_junit_before_exit_3(runner, tmp_path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    _write_fixture(fixtures, "a", "ok")
    _write_fixture(fixtures, "b", "ok")
    baseline = tmp_path / "baseline.json"
    r1 = runner.invoke(
        main, ["run", str(fixtures), "--write-baseline", "--baseline", str(baseline)]
    )
    assert r1.exit_code == 0, r1.output

    (fixtures / "b.json").unlink()
    junit = tmp_path / "junit.xml"
    r2 = runner.invoke(
        main, ["run", str(fixtures), "--baseline", str(baseline), "--junit", str(junit)]
    )
    assert r2.exit_code == 3, r2.output
    assert junit.exists()
    assert "removed" in r2.stdout


def test_run_silent_regression_reported_and_exits_zero(runner, tmp_path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    _write_fixture(fixtures, "a", "ok")
    baseline = tmp_path / "baseline.json"
    r1 = runner.invoke(
        main, ["run", str(fixtures), "--write-baseline", "--baseline", str(baseline)]
    )
    assert r1.exit_code == 0, r1.output

    # Output changed but all assertions still pass.
    _write_fixture(fixtures, "a", "ok!")
    junit = tmp_path / "junit.xml"
    r2 = runner.invoke(
        main, ["run", str(fixtures), "--baseline", str(baseline), "--junit", str(junit)]
    )
    assert r2.exit_code == 0, r2.output
    assert "silent_regression" in r2.stdout
    assert junit.exists()


def test_run_skips_baseline_write_on_hard_regression(runner, tmp_path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    _write_fixture(fixtures, "a", "ok", expected=["ok"])
    baseline = tmp_path / "baseline.json"
    r1 = runner.invoke(
        main, ["run", str(fixtures), "--write-baseline", "--baseline", str(baseline)]
    )
    assert r1.exit_code == 0, r1.output
    original = baseline.read_text()

    _write_fixture(fixtures, "a", "ok", expected=["nope"])
    r2 = runner.invoke(
        main,
        [
            "run", str(fixtures),
            "--baseline", str(baseline),
            "--write-baseline",
            "--junit", str(tmp_path / "junit.xml"),
        ],
    )
    assert r2.exit_code == 3, r2.output
    # A failing run must not silently accept the regressed output as baseline.
    assert baseline.read_text() == original
    assert "baseline not written" in r2.output
