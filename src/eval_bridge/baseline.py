"""Baseline diff: detect outputs that change even when all assertions still pass."""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_FILE = ".baseline.json"


def write_baseline(path: Path, snapshot: dict[str, dict]) -> None:
    """Persist the current run's outputs as the new baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "snapshot": snapshot,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> dict[str, dict] | None:
    """Load a previously-written baseline; return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    snap = payload.get("snapshot")
    if not isinstance(snap, dict):
        return None
    return snap


def diff_baseline(before: dict[str, dict], after: dict[str, dict]) -> list[dict]:
    """Return list of {trace_id, kind, before, after}.

    Kinds:
      - "silent_regression": passed=True before AND after, but actual_output differs.
      - "fixed": passed=False before, passed=True after.
      - "regressed": passed=True before, passed=False after.
      - "removed": in before but not in after.
    """
    diffs = []
    for tid in sorted(set(before) | set(after)):
        if tid not in after:
            diffs.append({"trace_id": tid, "kind": "removed", "before": before[tid]})
            continue
        if tid not in before:
            continue
        a, b = before[tid], after[tid]
        a_passed, b_passed = a.get("passed"), b.get("passed")
        a_out, b_out = a.get("actual_output"), b.get("actual_output")
        if a_passed and not b_passed:
            diffs.append({"trace_id": tid, "kind": "regressed", "before": a, "after": b})
        elif not a_passed and b_passed:
            diffs.append({"trace_id": tid, "kind": "fixed", "before": a, "after": b})
        elif a_passed and b_passed and a_out != b_out:
            diffs.append({
                "trace_id": tid, "kind": "silent_regression",
                "before": a, "after": b,
            })
    return diffs


__all__ = ["BASELINE_FILE", "diff_baseline", "load_baseline", "write_baseline"]
