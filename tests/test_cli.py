"""Tests for the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from eval_bridge.cli import main


@pytest.fixture
def runner() -> CliRunner:
    # click 8.2+ removed `mix_stderr`; stdout/stderr are captured separately.
    return CliRunner()


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

def test_capture_dry_run_prints_scrubbed_json(runner: CliRunner, tmp_path: Path, capsys):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "tr-1",
        "prompt": "contact jane@example.com",
    }))
    result = runner.invoke(main, ["capture", str(incident)])
    assert result.exit_code == 0, result.stdout
    # dry-run writes the fixture to stdout, not a file
    assert "{email}" in result.stdout
    # confirm it did NOT create a file on disk under tests/fixtures
    assert not (Path.cwd() / "tests" / "fixtures" / "tr-1.json").exists()


def test_capture_writes_file(runner: CliRunner, tmp_path: Path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "tr-1",
        "prompt": "ping jane@example.com",
    }))
    out = tmp_path / "fx.json"
    result = runner.invoke(main, ["capture", str(incident), "--output", str(out), "--write"])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["trace_id"] == "tr-1"
    assert "{email}" in payload["prompt"]


def test_capture_writes_yaml_when_extension_is_yaml(runner: CliRunner, tmp_path: Path):
    import yaml
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "tr-1",
        "prompt": "ping jane@example.com",
    }))
    out = tmp_path / "fx.yaml"
    result = runner.invoke(main, ["capture", str(incident), "--output", str(out), "--write"])
    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(out.read_text())
    assert payload["trace_id"] == "tr-1"


def test_capture_with_mutate_writes_variants(runner: CliRunner, tmp_path: Path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "tr-mut",
        "prompt": "Summarize the Q3 sales report for Jane Doe on 2025-09-12.",
    }))
    out = tmp_path / "fx.json"
    result = runner.invoke(main, [
        "capture", str(incident),
        "--output", str(out), "--write", "--mutate", "3",
    ])
    assert result.exit_code == 0, result.output
    # Base fixture present.
    assert out.exists()
    # Three variants present with the .mN suffix.
    for n in (1, 2, 3):
        variant = tmp_path / f"fx.m{n}.json"
        assert variant.exists(), f"missing variant {variant}"
        payload = json.loads(variant.read_text())
        assert payload["trace_id"] == f"tr-mut.m{n}"
        # Variant trace_ids carry the mN suffix; the base prompt content
        # is replaced by the mutator.
        assert payload["prompt"]
    # stdout must mention each variant file.
    for n in (1, 2, 3):
        assert f"fx.m{n}.json" in result.stdout


def test_capture_with_mutate_dry_run_does_not_write(runner: CliRunner, tmp_path: Path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "tr-mut-dry",
        "prompt": "Summarize the report.",
    }))
    out = tmp_path / "fx.json"
    # Default mode is --dry-run.
    result = runner.invoke(main, [
        "capture", str(incident),
        "--output", str(out), "--mutate", "2",
    ])
    assert result.exit_code == 0, result.output
    # No files written.
    assert not out.exists()
    assert not (tmp_path / "fx.m1.json").exists()
    # stdout shows the variants.
    assert "tr-mut-dry.m1" in result.stdout
    assert "tr-mut-dry.m2" in result.stdout


def test_capture_with_category_uses_next_incident_id(
    runner: CliRunner, tmp_path: Path
):
    fx_dir = tmp_path / "fixtures"
    fx_dir.mkdir()
    (tmp_path / "incident.json").write_text(json.dumps({
        "trace_id": "ignored",
        "prompt": "leak my email jane@example.com",
        "forbidden_substrings": ["jane@example.com"],
    }))
    counter = fx_dir / ".eval-bridge-counter"
    counter.write_text('{"pii-leak": 7}\n')

    result = runner.invoke(
        main,
        [
            "capture", str(tmp_path / "incident.json"),
            "--output-dir", str(fx_dir),
            "--category", "pii-leak",
            "--write",
        ],
    )
    assert result.exit_code == 0, result.output
    written = list(fx_dir.glob("pii-leak-*.json"))
    assert len(written) == 1
    fixture = json.loads(written[0].read_text())
    assert fixture["trace_id"].startswith("pii-leak-")
    counter_after = json.loads(counter.read_text())
    assert counter_after["pii-leak"] == 8


def test_capture_category_requires_output_dir(runner: CliRunner, tmp_path: Path):
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"trace_id": "x", "prompt": "y"}))
    result = runner.invoke(
        main,
        ["capture", str(incident), "--category", "pii-leak", "--write"],
    )
    assert result.exit_code != 0
    assert "--category requires --output-dir" in result.output


def test_capture_output_dir_requires_category(runner: CliRunner, tmp_path: Path):
    fx_dir = tmp_path / "fixtures"
    fx_dir.mkdir()
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"trace_id": "x", "prompt": "y"}))
    result = runner.invoke(
        main,
        ["capture", str(incident), "--output-dir", str(fx_dir), "--write"],
    )
    assert result.exit_code != 0
    assert "--output-dir requires --category" in result.output


def test_capture_output_and_output_dir_are_mutually_exclusive(
    runner: CliRunner, tmp_path: Path
):
    fx_dir = tmp_path / "fixtures"
    fx_dir.mkdir()
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({"trace_id": "x", "prompt": "y"}))
    out = tmp_path / "fx.json"
    result = runner.invoke(
        main,
        [
            "capture", str(incident),
            "--output", str(out),
            "--output-dir", str(fx_dir),
            "--category", "pii-leak",
            "--write",
        ],
    )
    assert result.exit_code != 0
    assert "use either --output or --output-dir, not both" in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_emits_junit_and_summary(runner: CliRunner, tmp_path: Path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    (fixtures / "a.json").write_text(json.dumps({
        "trace_id": "a", "prompt": "hi", "fixture_response": "ok",
    }))
    junit = tmp_path / "junit.xml"
    result = runner.invoke(main, [
        "run", str(fixtures), "--junit", str(junit), "--no-exit-on-fail",
    ])
    assert result.exit_code == 0, result.stdout
    assert junit.exists()
    assert "passed=1" in result.stdout


def test_run_exits_nonzero_on_failure(runner: CliRunner, tmp_path: Path):
    fixtures = tmp_path / "fx"
    fixtures.mkdir()
    (fixtures / "bad.json").write_text(json.dumps({
        "trace_id": "bad",
        "prompt": "hi",
        "fixture_response": "x",
        "expected_substrings": ["y"],
    }))
    junit = tmp_path / "junit.xml"
    result = runner.invoke(main, ["run", str(fixtures), "--junit", str(junit)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# scrub
# ---------------------------------------------------------------------------

def test_scrub_prints_to_stdout(runner: CliRunner, tmp_path: Path):
    f = tmp_path / "in.txt"
    f.write_text("contact jane@example.com")
    result = runner.invoke(main, ["scrub", str(f), "--no-residual-scan"])
    assert result.exit_code == 0, result.stdout
    assert "{email}" in result.stdout


def test_scrub_writes_to_file(runner: CliRunner, tmp_path: Path):
    f = tmp_path / "in.txt"
    f.write_text("contact jane@example.com")
    out = tmp_path / "clean.txt"
    result = runner.invoke(main, [
        "scrub", str(f), "--output", str(out), "--no-residual-scan"
    ])
    assert result.exit_code == 0, result.stdout
    assert "{email}" in out.read_text()


def test_scrub_residual_scan_fails_on_leak(runner: CliRunner, tmp_path: Path):
    # The scrubber's default config scrubs emails, but a contrived input that
    # the scrubber is configured NOT to scrub (via config) should still trigger
    # the residual scan. We use a custom config that disables email scrubbing
    # by setting the replacement to an empty string.
    cfg = tmp_path / "eval-bridge.toml"
    cfg.write_text('[scrubber]\nemail = ""\n')
    f = tmp_path / "in.txt"
    f.write_text("contact jane@example.com")
    result = runner.invoke(main, ["scrub", str(f), "-c", str(cfg)])
    assert result.exit_code == 2
    assert "residual" in result.stderr.lower()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def test_doctor_runs(runner: CliRunner):
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "scrubber config" in result.stdout
    assert "OK" in result.stdout


def test_version(runner: CliRunner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.4.0" in result.stdout
