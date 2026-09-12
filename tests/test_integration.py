"""End-to-end integration: capture -> run -> JUnit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_end_to_end_capture_run_and_junit(tmp_path: Path, monkeypatch):
    # Run in an isolated cwd so capture doesn't pollute the repo.
    monkeypatch.chdir(tmp_path)

    # 1. Capture a fixture from an "incident" (dry-run mode = default).
    incident = tmp_path / "incident.json"
    incident.write_text(json.dumps({
        "trace_id": "trace-x",
        "prompt": "leak: jane@example.com with token ghp_" + "a" * 40,
        "expected_substrings": ["ok"],
    }))

    # Pass the project's src/ on PYTHONPATH so the subprocess can resolve
    # `eval_bridge` even when pytest's venv doesn't have it on sys.path.
    import os as _os
    src_path = Path(__file__).resolve().parent.parent / "src"
    sub_env = _os.environ.copy()
    sub_env["PYTHONPATH"] = str(src_path) + _os.pathsep + sub_env.get("PYTHONPATH", "")

    res = subprocess.run(
        [sys.executable, "-m", "eval_bridge.cli",
         "capture", str(incident),
         "--output", "tests/fixtures/trace-x.json",
         "--write"],
        capture_output=True, text=True,
        env=sub_env,
    )
    assert res.returncode == 0, res.stdout + res.stderr

    fixture_path = tmp_path / "tests" / "fixtures" / "trace-x.json"
    assert fixture_path.exists()
    fixture = json.loads(fixture_path.read_text())
    assert fixture["prompt"] == "leak: {email} with token {api_token}"
    assert fixture["fixture_response"] is None

    # 2. Write a config that supplies fixture_response + assertions
    cfg = tmp_path / "eval-bridge.toml"
    cfg.write_text(
        '[scrubber]\n'
        'email = "{email}"\n'
        'api_token = "{api_token}"\n'
        '\n'
        '[runner]\n'
        'provider = "fixture"\n'
    )

    # Manually set the fixture_response on disk (simulating what the user does
    # after editing) so the offline provider can replay it.
    fixture["fixture_response"] = "everything ok"
    fixture["expected_substrings"] = ["ok"]
    fixture_path.write_text(json.dumps(fixture, indent=2))

    # 3. Run with JUnit output.
    junit = tmp_path / "junit.xml"
    res = subprocess.run(
        [sys.executable, "-m", "eval_bridge.cli",
         "run", "tests/fixtures",
         "--junit", str(junit),
         "--no-exit-on-fail"],
        capture_output=True, text=True,
        env=sub_env,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert junit.exists()
    assert 'tests="1"' in junit.read_text()
