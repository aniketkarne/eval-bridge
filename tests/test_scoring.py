"""Tests for the LLM-as-judge scoring layer."""

from __future__ import annotations

import json

import httpx
import pytest

# ``httpx`` is already a runtime dep. We mock at the transport layer so we
# don't need any extra test-only packages.

from eval_bridge import (
    ANSWER_RELEVANCE,
    BUILTIN_SCORERS,
    BIAS,
    FAITHFULNESS,
    Fixture,
    HALLUCINATION,
    LLMJudge,
    OfflineJudge,
    Runner,
    RunnerConfig,
    TOXICITY,
    default_scorer_set,
    g_eval,
)


def _fx(prompt: str = "What is the capital of France?",
        reply_holder: list[str] | None = None) -> Fixture:
    """A minimal fixture. ``reply_holder`` lets tests inject the canned reply."""
    return Fixture(
        trace_id="f-1",
        prompt=prompt,
        fixture_response="Paris",  # offline provider will replay this
        judge_scorers=["hallucination", "answer_relevance"],
    )


# ---------------------------------------------------------------------------
# Scorer / ScoreResult basics
# ---------------------------------------------------------------------------

def test_builtin_scorers_have_unique_names() -> None:
    names = [s.name for s in BUILTIN_SCORERS]
    assert len(names) == len(set(names))
    assert set(names) == {"hallucination", "faithfulness", "answer_relevance",
                          "toxicity", "bias"}


def test_g_eval_returns_a_scorer_with_custom_criteria() -> None:
    s = g_eval("the reply mentions a number")
    assert s.name == "g_eval"
    assert s.threshold == 0.7


def test_g_eval_threshold_is_configurable() -> None:
    s = g_eval("strict", threshold=0.95)
    assert s.threshold == 0.95


def test_default_scorer_set_returns_all_builtins() -> None:
    assert {s.name for s in default_scorer_set()} == {s.name for s in BUILTIN_SCORERS}


def test_default_scorer_set_filters_by_name() -> None:
    selected = default_scorer_set(names=("hallucination", "bias"))
    assert {s.name for s in selected} == {"hallucination", "bias"}


def test_default_scorer_set_silently_drops_unknown_names() -> None:
    selected = default_scorer_set(names=("hallucination", "nonexistent"))
    assert [s.name for s in selected] == ["hallucination"]


# ---------------------------------------------------------------------------
# OfflineJudge behaviour
# ---------------------------------------------------------------------------

def test_offline_judge_returns_one_when_no_canned_verdict() -> None:
    j = OfflineJudge()
    res = j.score("hallucination", "r", "p", "Paris", trace_id="unknown")
    assert res.score == 1.0
    assert res.reason == "no_canned_verdict"


def test_offline_judge_returns_canned_verdict_when_set() -> None:
    j = OfflineJudge()
    j.set("hallucination", "f-1", 0.4, "contains an unsourced claim")
    res = j.score("hallucination", "r", "p", "Paris", trace_id="f-1")
    assert res.score == 0.4
    assert res.reason == "contains an unsourced claim"


def test_offline_judge_isolates_per_fixture() -> None:
    j = OfflineJudge()
    j.set("hallucination", "f-1", 0.2, "f-1 reason")
    res_other = j.score("hallucination", "r", "p", "Paris", trace_id="f-2")
    assert res_other.score == 1.0
    assert res_other.reason == "no_canned_verdict"


def test_offline_judge_clamps_score_to_unit_interval() -> None:
    j = OfflineJudge()
    j.set("hallucination", "f-1", 1.7)
    res = j.score("hallucination", "r", "p", "Paris", trace_id="f-1")
    assert res.score == 1.0  # clamped
    j.set("hallucination", "f-1", -0.3)
    res = j.score("hallucination", "r", "p", "Paris", trace_id="f-1")
    assert res.score == 0.0  # clamped


def test_offline_judge_handles_malformed_canned_verdict() -> None:
    j = OfflineJudge({("hallucination", "f-1"): {"score": "not-a-number"}})
    res = j.score("hallucination", "r", "p", "Paris", trace_id="f-1")
    assert res.score == 0.0
    assert res.reason == "malformed_canned_verdict"


# ---------------------------------------------------------------------------
# Scorer.evaluate against an OfflineJudge
# ---------------------------------------------------------------------------

def test_scorer_evaluate_passes_when_score_meets_threshold() -> None:
    j = OfflineJudge()
    j.set("hallucination", "f-1", 0.9, "grounded")
    result = HALLUCINATION.evaluate(judge=j, fixture=_fx(), reply="Paris")
    assert result.passed is True
    assert result.score == 0.9
    assert result.threshold == 0.7


def test_scorer_evaluate_fails_when_score_below_threshold() -> None:
    j = OfflineJudge()
    j.set("hallucination", "f-1", 0.3, "unsupported claims")
    result = HALLUCINATION.evaluate(judge=j, fixture=_fx(), reply="Paris")
    assert result.passed is False


def test_scorer_evaluate_passes_on_not_applicable() -> None:
    """A 'not_applicable' verdict must NOT fail — it lets scorers abstain."""
    j = OfflineJudge()
    j.set("faithfulness", "f-1", 0.0, "not_applicable")
    result = FAITHFULNESS.evaluate(judge=j, fixture=_fx(), reply="Paris",
                                   context=None)
    assert result.passed is True
    assert result.is_na is True


def test_g_eval_uses_fixture_judge_criteria_when_provided() -> None:
    fx = Fixture(
        trace_id="f-ge",
        prompt="How many planets?",
        fixture_response="Eight",
        judge_criteria="the reply names a number",
        judge_scorers=["g_eval"],
    )
    j = OfflineJudge()
    # No canned verdict -> default 1.0 passes.
    result = next(s for s in [g_eval(fx.judge_criteria)])  # type: ignore[arg-type]
    res = result.evaluate(judge=j, fixture=fx, reply="Eight planets")
    assert res.passed is True


# ---------------------------------------------------------------------------
# LLMJudge (network mock via httpx.MockTransport)
# ---------------------------------------------------------------------------

def _mock_transport(handler):
    """Build an httpx.Client whose requests are routed to *handler*."""
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_llm_judge_calls_endpoint_and_parses_score() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(
                {"score": 0.82, "reason": "grounded in context"}
            )}}]
        })
    client = _mock_transport(handler)
    j = LLMJudge(base_url="https://api.example.com/v1", model="judge-mini",
                 client=client)
    res = j.score("hallucination", "rubric", "prompt", "reply",
                  context="ctx", trace_id="f-1")
    assert res.score == 0.82
    assert res.reason == "grounded in context"
    assert res.raw.get("score") == 0.82
    client.close()


def test_llm_judge_strips_code_fences_from_response() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": (
                '```json\n{"score": 0.5, "reason": "ok"}\n```'
            )}}]
        })
    client = _mock_transport(handler)
    j = LLMJudge(base_url="https://api.example.com/v1", client=client)
    res = j.score("hallucination", "r", "p", "rep", trace_id="f-1")
    assert res.score == 0.5
    client.close()


def test_llm_judge_raises_on_unparseable_response() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not json at all"}}]
        })
    client = _mock_transport(handler)
    j = LLMJudge(base_url="https://api.example.com/v1", client=client)
    with pytest.raises(Exception) as exc_info:
        j.score("hallucination", "r", "p", "rep", trace_id="f-1")
    assert "unparseable" in str(exc_info.value).lower()
    client.close()


def test_llm_judge_raises_on_http_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"})
    client = _mock_transport(handler)
    j = LLMJudge(base_url="https://api.example.com/v1", client=client)
    with pytest.raises(Exception) as exc_info:
        j.score("hallucination", "r", "p", "rep", trace_id="f-1")
    assert "429" in str(exc_info.value)
    client.close()


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

def test_runner_with_offline_judge_adds_scorer_assertions() -> None:
    """When a fixture lists judge_scorers and the runner uses OfflineJudge,
    each scorer becomes a distinct assertion in the report."""
    fx = Fixture(
        trace_id="r-1",
        prompt="Q",
        fixture_response="Paris",
        judge_scorers=["hallucination", "answer_relevance"],
    )
    runner = Runner(provider=None, judge=OfflineJudge())
    result = runner.run_one(fx)
    scorer_names = {a.name for a in result.assertions
                    if a.name.startswith("scorer.")}
    assert "scorer.hallucination" in scorer_names
    assert "scorer.answer_relevance" in scorer_names


def test_runner_offline_judge_passes_when_verdicts_meet_threshold() -> None:
    fx = Fixture(
        trace_id="r-2",
        prompt="Q",
        fixture_response="Paris",
        judge_scorers=["hallucination"],
    )
    judge = OfflineJudge()
    judge.set("hallucination", "r-2", 0.95, "grounded")
    runner = Runner(provider=None, judge=judge)
    result = runner.run_one(fx)
    assert result.passed is True
    assert any(a.name == "scorer.hallucination" and a.passed for a in result.assertions)


def test_runner_offline_judge_fails_when_verdict_below_threshold() -> None:
    fx = Fixture(
        trace_id="r-3",
        prompt="Q",
        fixture_response="Paris",
        judge_scorers=["toxicity"],
    )
    judge = OfflineJudge()
    judge.set("toxicity", "r-3", 0.1, "harmful content")
    runner = Runner(provider=None, judge=judge)
    result = runner.run_one(fx)
    assert result.passed is False
    failing = next(a for a in result.assertions
                   if a.name == "scorer.toxicity")
    assert failing.passed is False
    assert "0.100" in failing.detail


def test_runner_judge_exception_is_reported_as_failing_assertion() -> None:
    """A judge that raises must not crash the runner; it becomes a failing
    scorer assertion so the rest of the report still emits."""
    class ExplodingJudge(OfflineJudge):
        def score(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("judge exploded")

    fx = Fixture(
        trace_id="r-4",
        prompt="Q",
        fixture_response="Paris",
        judge_scorers=["hallucination"],
    )
    runner = Runner(provider=None, judge=ExplodingJudge())
    result = runner.run_one(fx)
    assert result.passed is False
    failing = next(a for a in result.assertions
                   if a.name == "scorer.hallucination")
    assert "judge exploded" in failing.detail


def test_runner_without_judge_scorers_does_not_emit_scorer_assertions() -> None:
    fx = Fixture(trace_id="r-5", prompt="Q", fixture_response="Paris")
    runner = Runner(provider=None, judge=OfflineJudge())
    result = runner.run_one(fx)
    assert not any(a.name.startswith("scorer.") for a in result.assertions)


def test_runner_uses_runner_config_scorer_set_when_fixture_has_none() -> None:
    """RunnerConfig.scorer_names activates scorers even on fixtures that
    do not list their own judge_scorers."""
    fx = Fixture(trace_id="r-6", prompt="Q", fixture_response="Paris")
    cfg = RunnerConfig(scorer_names=("bias",))
    runner = Runner(provider=None, judge=OfflineJudge(), config=cfg)
    result = runner.run_one(fx)
    assert any(a.name == "scorer.bias" for a in result.assertions)


def test_runner_g_eval_with_criteria_emits_assertion() -> None:
    fx = Fixture(
        trace_id="r-7",
        prompt="Q",
        fixture_response="42",
        judge_criteria="the reply is a number",
        judge_scorers=["g_eval"],
    )
    runner = Runner(provider=None, judge=OfflineJudge())
    result = runner.run_one(fx)
    assert any(a.name == "scorer.g_eval" for a in result.assertions)


def test_runner_g_eval_without_criteria_is_silently_skipped() -> None:
    """G-Eval needs criteria — without it, it must not emit a failing
    assertion (otherwise every fixture with judge_scorers=['g_eval']
    fails)."""
    fx = Fixture(
        trace_id="r-8",
        prompt="Q",
        fixture_response="42",
        judge_scorers=["g_eval"],  # no judge_criteria
    )
    runner = Runner(provider=None, judge=OfflineJudge())
    result = runner.run_one(fx)
    assert not any(a.name == "scorer.g_eval" for a in result.assertions)


def test_fixture_roundtrips_through_dict() -> None:
    """Schema additions must roundtrip through to_dict/from_dict."""
    fx = Fixture(
        trace_id="rt-1",
        prompt="Q",
        fixture_response="A",
        context="ctx",
        judge_criteria="criterion",
        judge_scorers=["hallucination", "g_eval"],
    )
    payload = fx.to_dict()
    assert payload["context"] == "ctx"
    assert payload["judge_criteria"] == "criterion"
    assert payload["judge_scorers"] == ["hallucination", "g_eval"]
    rebuilt = Fixture.from_dict(payload)
    assert rebuilt.context == "ctx"
    assert rebuilt.judge_criteria == "criterion"
    assert rebuilt.judge_scorers == ["hallucination", "g_eval"]


def test_fixture_validates_against_schema_with_new_fields() -> None:
    """The updated fixture schema must accept the new fields."""
    fx = Fixture(
        trace_id="v-1",
        prompt="Q",
        fixture_response="A",
        context="ctx",
        judge_criteria="criterion",
        judge_scorers=["hallucination", "answer_relevance", "toxicity"],
    )
    fx.validate()  # raises FixtureError on schema violation


def test_fixture_schema_rejects_unknown_scorer_name() -> None:
    """The enum constraint on judge_scorers should reject typos."""
    fx = Fixture(
        trace_id="v-2",
        prompt="Q",
        fixture_response="A",
        judge_scorers=["hallicination"],  # typo
    )
    with pytest.raises(Exception) as exc_info:
        fx.validate()
    assert "judge_scorers" in str(exc_info.value) or "enum" in str(exc_info.value).lower()