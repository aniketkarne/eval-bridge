"""The fixture schema must ship inside the package.

Regression: schemas/fixture.schema.json used to live at the repo root while
package-data expected it under eval_bridge/schemas/, so wheels shipped without
it and every `load_fixture` / `capture` / `run` call crashed with
FileNotFoundError once the package was pip-installed (non-editable).
"""

from __future__ import annotations

import json
from pathlib import Path

import eval_bridge
from eval_bridge import Fixture
from eval_bridge.fixture import SCHEMA_PATH, _validator


def test_schema_lives_inside_the_package() -> None:
    package_dir = Path(eval_bridge.__file__).resolve().parent
    assert SCHEMA_PATH.is_relative_to(package_dir), (
        f"schema at {SCHEMA_PATH} is outside the installed package; "
        "it must live under eval_bridge/schemas/ to ship in the wheel"
    )
    assert SCHEMA_PATH.exists(), f"schema missing at {SCHEMA_PATH}"


def test_schema_loads_as_valid_json_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(schema, dict)
    assert "$schema" in schema
    # Cached validator must build from the packaged file without touching
    # any repo-root path.
    assert _validator() is not None


def test_validate_works_from_packaged_schema() -> None:
    fx = Fixture(trace_id="pkg-001", prompt="hello")
    fx.validate()  # must not raise
