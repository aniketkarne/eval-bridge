"""Tests for fixture model + loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bridge import Fixture, FixtureSet, load_fixture, load_fixture_dir
from eval_bridge.errors import FixtureError


def test_from_dict_minimal():
    fx = Fixture.from_dict({"trace_id": "t1", "prompt": "hello"})
    assert fx.trace_id == "t1"
    assert fx.prompt == "hello"
    assert fx.expected_substrings == []


def test_from_dict_round_trip():
    fx = Fixture.from_dict({
        "trace_id": "t1",
        "prompt": "hello",
        "expected_substrings": ["hi"],
        "forbidden_substrings": ["bye"],
        "model": "gpt-4o-mini",
    })
    d = fx.to_dict()
    assert d["trace_id"] == "t1"
    assert d["expected_substrings"] == ["hi"]
    assert d["model"] == "gpt-4o-mini"


def test_validate_ok():
    fx = Fixture.from_dict({
        "trace_id": "t1",
        "prompt": "hello",
        "expected_substrings": ["hi"],
    })
    fx.validate()  # should not raise


def test_chat_messages_synthetic():
    fx = Fixture.from_dict({"trace_id": "t1", "prompt": "hi", "system": "be brief"})
    msgs = fx.chat_messages()
    assert msgs == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_chat_messages_explicit_passthrough():
    fx = Fixture.from_dict({
        "trace_id": "t1",
        "prompt": "ignored",
        "messages": [{"role": "user", "content": "explicit"}],
    })
    assert fx.chat_messages() == [{"role": "user", "content": "explicit"}]


def test_missing_required_field_raises():
    with pytest.raises(FixtureError):
        Fixture.from_dict({"prompt": "no id"})


def test_load_fixture_file(tmp_path: Path):
    p = tmp_path / "fx.json"
    p.write_text(json.dumps({"trace_id": "abc", "prompt": "hi"}))
    fx = load_fixture(p)
    assert fx.trace_id == "abc"


def test_load_fixture_dir_yaml(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"trace_id": "a", "prompt": "x"}))
    (tmp_path / "b.yaml").write_text("trace_id: b\nprompt: y\n")
    fs = load_fixture_dir(tmp_path)
    assert isinstance(fs, FixtureSet)
    assert {f.trace_id for f in fs.fixtures} == {"a", "b"}


def test_load_fixture_dir_recursive(tmp_path: Path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.json").write_text(json.dumps({"trace_id": "deep", "prompt": "x"}))
    fs = load_fixture_dir(tmp_path)
    assert {f.trace_id for f in fs.fixtures} == {"deep"}


def test_load_fixture_dir_missing(tmp_path: Path):
    with pytest.raises(FixtureError):
        load_fixture_dir(tmp_path / "nope")


def test_load_fixture_missing(tmp_path: Path):
    with pytest.raises(FixtureError):
        load_fixture(tmp_path / "nope.json")


def test_extra_field_preserved():
    fx = Fixture.from_dict({
        "trace_id": "t1",
        "prompt": "hi",
        "custom_field": "preserved",
    })
    assert fx.extra["custom_field"] == "preserved"
    assert "custom_field" in fx.to_dict()
