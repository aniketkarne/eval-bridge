"""Tests for baseline diff (silent-regression detection)."""

from __future__ import annotations

from pathlib import Path

from eval_bridge.baseline import (
    BASELINE_FILE,
    diff_baseline,
    load_baseline,
    write_baseline,
)


def test_no_diff_when_identical():
    fx = {"trace_id": "t-001", "actual_output": "hello", "passed": True}
    diffs = diff_baseline({"t-001": fx}, {"t-001": fx})
    assert diffs == []


def test_silent_regression_detected():
    before = {"t-001": {"trace_id": "t-001", "actual_output": "hello", "passed": True}}
    after = {"t-001": {"trace_id": "t-001", "actual_output": "hello!", "passed": True}}
    diffs = diff_baseline(before, after)
    assert len(diffs) == 1
    assert diffs[0]["kind"] == "silent_regression"
    assert diffs[0]["trace_id"] == "t-001"


def test_write_and_load_roundtrip(tmp_path: Path):
    data = {"t-001": {"trace_id": "t-001", "actual_output": "x", "passed": True}}
    write_baseline(tmp_path / "b.json", data)
    assert load_baseline(tmp_path / "b.json") == data
