"""Stable, human-scannable IDs for incident fixtures."""

from __future__ import annotations

import json
import re
from pathlib import Path

COUNTER_FILE = Path(".eval-bridge-counter")

# Hyphen-separated lowercase words; length 2..32. Examples: pii-leak,
# tool-misuse, hallucination, wrong-tool, format-error, generic.
_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")


def _load_counter(fixtures_dir: Path) -> dict[str, int]:
    """Read the on-disk counter; missing or malformed -> empty."""
    counter_path = fixtures_dir / COUNTER_FILE
    if not counter_path.exists():
        return {}
    try:
        data = json.loads(counter_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, int) and v >= 0:
            out[k] = v
    return out


def _save_counter(fixtures_dir: Path, counter: dict[str, int]) -> None:
    counter_path = fixtures_dir / COUNTER_FILE
    counter_path.write_text(
        json.dumps(counter, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def next_incident_id(fixtures_dir: Path, category: str) -> str:
    """Increment and persist the counter for *category*; return the new ID.

    The counter file `.eval-bridge-counter` lives in the fixtures directory
    and is committed to git so IDs are stable across machines.
    """
    if not _CATEGORY_RE.match(category):
        raise ValueError(
            f"invalid category {category!r}: must match {_CATEGORY_RE.pattern}"
        )
    counter = _load_counter(fixtures_dir)
    n = counter.get(category, 0) + 1
    counter[category] = n
    _save_counter(fixtures_dir, counter)
    return f"{category}-{n:03d}"


__all__ = ["COUNTER_FILE", "next_incident_id"]