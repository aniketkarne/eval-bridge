"""Tests for per-message multi-turn assertions."""

from __future__ import annotations

import pytest

from eval_bridge import Fixture, Runner


def _fx(messages: list[dict], message_assertions: list[dict] | None = None) -> Fixture:
    """Build a multi-turn fixture with a fixture_response that replays through
    FixtureProvider. The reply here only matters for any 'final' assertions —
    message-level assertions look at the messages themselves."""
    return Fixture(
        trace_id="mt-1",
        prompt="ignored",
        messages=messages,
        fixture_response="ok",
        message_assertions=message_assertions or [],
    )


# ---------------------------------------------------------------------------
# Shape + roundtrip
# ---------------------------------------------------------------------------

def test_fixture_roundtrips_message_assertions() -> None:
    fx = Fixture(
        trace_id="rt",
        prompt="ignored",
        messages=[{"role": "user", "content": "hi"}],
        message_assertions=[
            {"role": "assistant", "must_contain": ["x"]},
            {"role": "any", "must_not_contain": ["y"]},
        ],
    )
    payload = fx.to_dict()
    assert payload["message_assertions"] == [
        {"role": "assistant", "must_contain": ["x"]},
        {"role": "any", "must_not_contain": ["y"]},
    ]
    rebuilt = Fixture.from_dict(payload)
    assert rebuilt.message_assertions == fx.message_assertions


def test_fixture_schema_accepts_message_assertions() -> None:
    fx = Fixture(
        trace_id="v",
        prompt="ignored",
        messages=[{"role": "user", "content": "hi"}],
        message_assertions=[{"role": "any", "must_contain": ["ok"]}],
    )
    fx.validate()  # raises on schema violation


def test_fixture_schema_rejects_unknown_role() -> None:
    fx = Fixture(
        trace_id="v",
        prompt="ignored",
        messages=[{"role": "user", "content": "hi"}],
        message_assertions=[{"role": "tool", "must_contain": ["x"]}],
    )
    with pytest.raises(Exception):
        fx.validate()


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

def test_runner_message_must_contain_passes_when_present() -> None:
    fx = _fx(
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Hello"},
        ],
        message_assertions=[
            {"role": "system", "must_contain": ["translator"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    assert result.passed is True


def test_runner_message_must_contain_fails_when_missing() -> None:
    fx = _fx(
        messages=[{"role": "user", "content": "Hello"}],
        message_assertions=[
            {"role": "user", "must_contain": ["missing needle"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    assert result.passed is False
    failing = [a for a in result.assertions if not a.passed]
    assert any(a.name == "message.0.must_contain" for a in failing)


def test_runner_message_must_not_contain_passes_when_absent() -> None:
    fx = _fx(
        messages=[{"role": "assistant", "content": "Here you go."}],
        message_assertions=[
            {"role": "assistant", "must_not_contain": ["PASSWORD", "SECRET"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    assert result.passed is True


def test_runner_message_must_not_contain_fails_when_present() -> None:
    fx = _fx(
        messages=[{"role": "assistant", "content": "The SECRET is abc"}],
        message_assertions=[
            {"role": "assistant", "must_not_contain": ["SECRET"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    assert result.passed is False
    assert any(a.name == "message.0.must_not_contain" for a in result.assertions if not a.passed)


def test_runner_message_role_filter_skips_non_matching() -> None:
    """A rule with role='assistant' must not match user messages."""
    fx = _fx(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        message_assertions=[
            {"role": "assistant", "must_contain": ["hello"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    # Only one assertion should fire (assistant message matches).
    msg_assertions = [a for a in result.assertions if a.name.startswith("message.")]
    assert len(msg_assertions) == 1
    assert msg_assertions[0].passed


def test_runner_message_any_role_matches_every_message() -> None:
    fx = _fx(
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
            {"role": "assistant", "content": "ASSISTANT"},
        ],
        message_assertions=[
            {"role": "any", "must_contain": ["SYS"]},  # only matches system
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    msg_assertions = [a for a in result.assertions if a.name.startswith("message.")]
    # System message matches and passes; user and assistant messages fail
    # because they don't contain "SYS".
    assert len(msg_assertions) == 3
    assert msg_assertions[0].passed is True   # system contains SYS
    assert msg_assertions[1].passed is False  # user doesn't
    assert msg_assertions[2].passed is False  # assistant doesn't


def test_runner_no_message_assertions_emits_no_message_reports() -> None:
    fx = _fx(messages=[{"role": "user", "content": "hi"}])
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    assert not any(a.name.startswith("message.") for a in result.assertions)


def test_runner_default_role_is_any() -> None:
    """Omitting role from a rule is equivalent to role='any'."""
    fx = _fx(
        messages=[{"role": "user", "content": "hello world"}],
        message_assertions=[{"must_contain": ["hello"]}],  # no role
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    msg = [a for a in result.assertions if a.name.startswith("message.")]
    assert len(msg) == 1
    assert msg[0].passed


def test_runner_message_assertion_uses_index_for_named_report() -> None:
    """The assertion name encodes which message index triggered it."""
    fx = _fx(
        messages=[
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ],
        message_assertions=[
            {"role": "any", "must_contain": ["nonexistent"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    failing = [a.name for a in result.assertions if not a.passed]
    assert "message.0.must_contain" in failing
    assert "message.1.must_contain" in failing
    assert "message.2.must_contain" in failing


def test_runner_message_assertions_ignore_non_string_content() -> None:
    """Tool messages with structured content (lists/dicts) are skipped."""
    fx = _fx(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": [{"type": "text", "text": "data"}]},
        ],
        message_assertions=[
            {"role": "any", "must_contain": ["hi"]},
        ],
    )
    runner = Runner(provider=None)
    result = runner.run_one(fx)
    msg = [a for a in result.assertions if a.name.startswith("message.")]
    # Only the user message fires; tool message is skipped (non-string).
    assert len(msg) == 1
    assert msg[0].name == "message.0.must_contain"