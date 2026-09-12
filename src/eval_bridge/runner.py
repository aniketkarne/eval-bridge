"""Eval runner: executes fixtures, runs assertions, emits JUnit XML.

Assertion layers (each independently togglable):

1. **Forbidden substrings** - hard safety gate.
2. **Expected substrings** - regression check.
3. **JSON Schema** - structural check (when ``fixture.schema`` is set).
4. **Semantic threshold** - deterministic lexical baseline (Jaccard token
   similarity) against ``fixture.reference_reply``. We intentionally avoid an
   embedding model: it would be non-deterministic and slow. The lexical
   baseline is good enough for regression detection.
5. **Residual secret scan** - final safety gate using
   ``scrubber.find_residual_secrets``.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator, ValidationError

from .errors import ProviderError
from .fixture import Fixture, FixtureSet
from .provider import CompletionResult, FixtureProvider, Provider, provider_from_config
from .scrubber import Scrubber, find_residual_secrets
from .scoring import (
    BUILTIN_SCORERS,
    Judge,
    OfflineJudge,
    Scorer,
    default_scorer_set,
    g_eval,
)

# ---------------------------------------------------------------------------
# Token-level Jaccard
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def jaccard(a: str, b: str) -> float:
    """Symmetric lexical similarity over word tokens.

    Returns a value in [0.0, 1.0]. Two empty inputs -> 1.0 (vacuously similar).
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AssertionReport:
    """Outcome of a single assertion layer."""

    name: str
    passed: bool
    detail: str = ""
    expected: Any = None
    actual: Any = None


@dataclass
class TestCaseResult:
    """Result for one fixture execution."""

    # Tell pytest not to collect this class as a test.
    __test__ = False

    trace_id: str
    classname: str
    name: str
    passed: bool
    duration_s: float
    assertions: list[AssertionReport] = field(default_factory=list)
    response_text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "classname": self.classname,
            "name": self.name,
            "passed": self.passed,
            "duration_s": self.duration_s,
            "response": self.response_text,
            "error": self.error,
            "assertions": [
                {
                    "name": a.name,
                    "passed": a.passed,
                    "detail": a.detail,
                    "expected": a.expected,
                    "actual": a.actual,
                }
                for a in self.assertions
            ],
        }


@dataclass
class RunnerReport:
    """Aggregate runner output."""

    results: list[TestCaseResult] = field(default_factory=list)

    # Properties used by CLI / CI
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def duration_s(self) -> float:
        return sum(r.duration_s for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "duration_s": self.duration_s,
            "results": [r.to_dict() for r in self.results],
        }

    # ------------------------------------------------------------------
    # JUnit XML
    # ------------------------------------------------------------------

    def write_junit(self, path: str | Path) -> Path:
        """Write a JUnit XML report. Returns the path written."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        suite = ET.Element("testsuite", attrib={
            "name": "eval-bridge",
            "tests": str(self.total),
            "failures": str(self.failed),
            "errors": str(sum(1 for r in self.results if r.error)),
            "skipped": "0",
            "time": f"{self.duration_s:.3f}",
            "timestamp": ts,
            "id": str(uuid.uuid4()),
        })

        for r in self.results:
            tc = ET.SubElement(suite, "testcase", attrib={
                "classname": r.classname,
                "name": r.name,
                "time": f"{r.duration_s:.3f}",
            })
            if r.error:
                ET.SubElement(tc, "error", attrib={
                    "message": r.error,
                    "type": "ProviderError",
                }).text = r.response_text or r.error
            elif not r.passed:
                for a in r.assertions:
                    if not a.passed:
                        ET.SubElement(tc, "failure", attrib={
                            "message": a.detail or a.name,
                            "type": a.name,
                        }).text = (
                            f"assertion={a.name}\n"
                            f"expected={a.expected!r}\n"
                            f"actual={a.actual!r}\n"
                            f"response={r.response_text!r}"
                        )
            ET.SubElement(tc, "system-out").text = r.response_text

        tree = ET.ElementTree(suite)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # ET writes bytes; ensure declaration + utf-8 for CI compatibility.
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunnerConfig:
    """Knobs for the runner."""

    provider: str = "fixture"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_s: float = 30.0
    max_retries: int = 2
    forbidden_substrings: list[str] = field(default_factory=list)
    semantic_threshold: float = 0.0
    residual_secret_scan: bool = True
    # Scoring layer — opt-in. Names from BUILTIN_SCORERS + "g_eval" (custom).
    scorer_names: tuple[str, ...] = field(default_factory=tuple)
    judge_model: str = "gpt-4o-mini"


class Runner:
    """Executes fixtures, runs assertions, accumulates a report."""

    def __init__(
        self,
        provider: Provider | None = None,
        scrubber: Scrubber | None = None,
        config: RunnerConfig | None = None,
        judge: Judge | None = None,
        scorer_set: list[Scorer] | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.provider = provider or provider_from_config(self.config)
        self.scrubber = scrubber or Scrubber.from_default_config()
        # Judge defaults to OfflineJudge (deterministic, safe for CI).
        self.judge = judge if judge is not None else OfflineJudge()
        # Scorer set: explicit > config > empty.
        if scorer_set is not None:
            self.scorer_set = scorer_set
        elif self.config.scorer_names:
            builtins = {s.name: s for s in BUILTIN_SCORERS}
            resolved: list[Scorer] = []
            for n in self.config.scorer_names:
                if n in builtins:
                    resolved.append(builtins[n])
                elif n == "g_eval":
                    # G-Eval needs per-fixture criteria; instantiate lazily
                    # in _assert. We carry a placeholder here.
                    resolved.append(Scorer(name="g_eval", rubric="", threshold=0.7))
            self.scorer_set = resolved
        else:
            self.scorer_set = []

    @classmethod
    def from_config(cls, config: RunnerConfig | None = None) -> "Runner":
        return cls(config=config)

    # ------------------------------------------------------------------
    # Single fixture execution
    # ------------------------------------------------------------------

    def run_one(self, fixture: Fixture) -> TestCaseResult:
        start = time.perf_counter()
        classname = f"eval-bridge.{fixture.trace_id}"
        name = fixture.trace_id

        try:
            response = self.provider.complete(fixture)
            text = response.text
            error: str | None = None
        except ProviderError as e:
            duration = time.perf_counter() - start
            return TestCaseResult(
                trace_id=fixture.trace_id,
                classname=classname,
                name=name,
                passed=False,
                duration_s=duration,
                assertions=[
                    AssertionReport(
                        name="provider",
                        passed=False,
                        detail=str(e),
                    )
                ],
                response_text="",
                error=str(e),
            )

        assertions = self._assert(fixture, text, response)
        # Latency budget assertion (v0.5.0). Evaluate AFTER duration is
        # known so we can compare the actual call latency against the
        # budget. Only emitted when fixture.latency_ms_max is set.
        if fixture.latency_ms_max is not None:
            duration = time.perf_counter() - start
            assertions.append(
                self._assert_latency_budget(duration * 1000.0, fixture.latency_ms_max)
            )
        passed = all(a.passed for a in assertions)
        duration = time.perf_counter() - start
        return TestCaseResult(
            trace_id=fixture.trace_id,
            classname=classname,
            name=name,
            passed=passed,
            duration_s=duration,
            assertions=assertions,
            response_text=text,
            error=error,
        )

    @staticmethod
    def _assert_latency_budget(actual_ms: float, max_ms: int) -> AssertionReport:
        """Assert the actual call latency does not exceed the budget.

        Concise name ``latency_ms`` is grep-friendly; consumers that want
        to surface slow calls can filter ``"latency_ms" in a.name``.
        """
        if actual_ms <= max_ms:
            return AssertionReport(
                name="latency_ms",
                passed=True,
                detail=f"{actual_ms:.0f}ms <= {max_ms}ms",
                expected=f"<= {max_ms}ms",
                actual=actual_ms,
            )
        return AssertionReport(
            name="latency_ms",
            passed=False,
            detail=f"{actual_ms:.0f}ms > budget {max_ms}ms",
            expected=f"<= {max_ms}ms",
            actual=actual_ms,
        )

    def _assert(self, fixture: Fixture, text: str, response: CompletionResult | None = None) -> list[AssertionReport]:
        reports: list[AssertionReport] = []

        # Global + per-fixture forbidden substrings
        forbidden = list(self.config.forbidden_substrings) + list(fixture.forbidden_substrings)
        for needle in forbidden:
            ok = needle not in text
            reports.append(AssertionReport(
                name="forbidden_substring",
                passed=ok,
                detail=f"forbidden substring not present: {needle!r}" if ok
                       else f"forbidden substring present: {needle!r}",
                expected=f"NOT CONTAIN {needle!r}",
                actual=text,
            ))

        # Expected substrings
        for needle in fixture.expected_substrings:
            ok = needle in text
            reports.append(AssertionReport(
                name="expected_substring",
                passed=ok,
                detail=f"contains {needle!r}" if ok
                       else f"missing {needle!r}",
                expected=f"CONTAIN {needle!r}",
                actual=text,
            ))

        # JSON Schema
        if fixture.schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                reports.append(AssertionReport(
                    name="schema",
                    passed=False,
                    detail=f"response is not valid JSON: {e}",
                    expected=fixture.schema,
                    actual=text,
                ))
            else:
                validator = Draft7Validator(fixture.schema)
                errors = sorted(validator.iter_errors(parsed), key=lambda e: e.path)
                if errors:
                    msg = "; ".join(
                        f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors
                    )
                    reports.append(AssertionReport(
                        name="schema",
                        passed=False,
                        detail=msg,
                        expected=fixture.schema,
                        actual=parsed,
                    ))
                else:
                    reports.append(AssertionReport(
                        name="schema",
                        passed=True,
                        detail="response matches schema",
                        expected=fixture.schema,
                        actual=parsed,
                    ))

        # Semantic threshold (lexical Jaccard baseline)
        threshold = fixture.semantic_threshold if fixture.semantic_threshold is not None \
                    else self.config.semantic_threshold
        if threshold > 0.0 and fixture.reference_reply is not None:
            sim = jaccard(text, fixture.reference_reply)
            reports.append(AssertionReport(
                name="semantic_threshold",
                passed=sim >= threshold,
                detail=f"jaccard={sim:.3f} >= {threshold:.3f}" if sim >= threshold
                       else f"jaccard={sim:.3f} < {threshold:.3f}",
                expected=f">= {threshold:.3f}",
                actual=sim,
            ))

        # Residual secret scan (final safety gate)
        if self.config.residual_secret_scan:
            leaks = find_residual_secrets(text)
            reports.append(AssertionReport(
                name="residual_secret_scan",
                passed=not leaks,
                detail="no residual secrets" if not leaks
                       else f"{len(leaks)} residual secret(s): "
                            + ", ".join(l["name"] for l in leaks),
                expected="no secrets",
                actual=leaks,
            ))

        # LLM-as-judge scoring (opt-in via RunnerConfig.scorer_names or
        # per-fixture judge_scorers). CI uses the offline judge by default
        # so verdicts are deterministic.
        active_scorers = self._active_scorers(fixture)
        for scorer in active_scorers:
            try:
                result = scorer.evaluate(
                    judge=self.judge,
                    fixture=fixture,
                    reply=text,
                    context=fixture.context,
                )
                reports.append(AssertionReport(
                    name=f"scorer.{scorer.name}",
                    passed=result.passed,
                    detail=(
                        f"score={result.score:.3f} threshold={scorer.threshold:.3f} "
                        f"reason={result.reason!r}"
                    ),
                    expected=f">= {scorer.threshold:.3f}",
                    actual=result.score,
                ))
            except Exception as e:  # noqa: BLE001 - judge failure must not abort the run
                reports.append(AssertionReport(
                    name=f"scorer.{scorer.name}",
                    passed=False,
                    detail=f"judge raised: {e}",
                    expected="no exception",
                    actual=str(e),
                ))

        # Per-message assertions (multi-turn fixtures). Each entry in
        # fixture.message_assertions is checked against every chat message
        # whose role matches (or "any"). Emit one AssertionReport per
        # matched message.
        for i, msg in enumerate(fixture.chat_messages()):
            role = (msg.get("role") or "").lower()
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            for rule in fixture.message_assertions:
                rule_role = (rule.get("role") or "any").lower()
                if rule_role != "any" and rule_role != role:
                    continue
                for needle in rule.get("must_contain") or []:
                    ok = needle in content
                    reports.append(AssertionReport(
                        name=f"message.{i}.must_contain",
                        passed=ok,
                        detail=f"message {i} (role={role}) contains {needle!r}" if ok
                               else f"message {i} (role={role}) missing {needle!r}",
                        expected=f"CONTAIN {needle!r}",
                        actual=content,
                    ))
                for needle in rule.get("must_not_contain") or []:
                    ok = needle not in content
                    reports.append(AssertionReport(
                        name=f"message.{i}.must_not_contain",
                        passed=ok,
                        detail=f"message {i} (role={role}) does not contain {needle!r}" if ok
                               else f"message {i} (role={role}) contains forbidden {needle!r}",
                        expected=f"NOT CONTAIN {needle!r}",
                        actual=content,
                    ))

        # Tool-call correctness (v0.5.0). Compare the model's response to
        # fixture.expected_tool_calls. The response text is parsed as JSON;
        # if it has a top-level "tool_calls" array, use it directly; if it
        # looks like a single tool call (has "name"), treat it as a one-
        # element list; otherwise no tool calls were made and the
        # assertion fails by design.
        if fixture.expected_tool_calls:
            actual_calls = self._extract_tool_calls(text, response)
            reports.append(self._assert_tool_calls(actual_calls, fixture.expected_tool_calls))

        return reports

    @staticmethod
    def _normalise_tool_call(call: dict[str, Any]) -> dict[str, Any]:
        """Sort argument keys so deep-equal is order-independent."""
        if "arguments" in call and isinstance(call["arguments"], dict):
            call = {**call, "arguments": {k: call["arguments"][k] for k in sorted(call["arguments"])}}
        return call

    @staticmethod
    def _extract_tool_calls(text: str, response: CompletionResult | None) -> list[dict[str, Any]]:
        """Best-effort extraction of tool calls from a provider response.

        Order of preference:
        1. ``response.raw["tool_calls"]`` if the provider returned it.
        2. ``response.raw["choices"][0]["message"]["tool_calls"]`` (OpenAI shape).
        3. Parse *text* as JSON: a list of calls, a dict with "tool_calls",
           or a single call (has "name").
        """
        if response is not None:
            raw = response.raw or {}
            if isinstance(raw, dict):
                if isinstance(raw.get("tool_calls"), list):
                    return [c for c in raw["tool_calls"] if isinstance(c, dict)]
                choices = raw.get("choices") or []
                if choices:
                    msg = (choices[0] or {}).get("message") or {}
                    tcs = msg.get("tool_calls")
                    if isinstance(tcs, list):
                        return [c for c in tcs if isinstance(c, dict)]
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict)]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("tool_calls"), list):
                return [c for c in parsed["tool_calls"] if isinstance(c, dict)]
            # A single tool call: a dict with a "name" key.
            if "name" in parsed:
                return [parsed]
        return []

    def _assert_tool_calls(
        self,
        actual_calls: list[dict[str, Any]],
        expected: list[dict[str, Any]],
    ) -> AssertionReport:
        """Verify the model's response tool-call matches the expected sequence.

        Compares name, and arguments deep-equal (modulo key order). Order of
        tool calls in the response must match the expected order.
        """
        actual_norm = [self._normalise_tool_call(c) for c in actual_calls]
        expected_norm = [self._normalise_tool_call(c) for c in expected]
        if actual_norm == expected_norm:
            return AssertionReport(
                name="tool_calls",
                passed=True,
                detail=f"{len(expected_norm)} tool call(s) match",
                expected=expected_norm,
                actual=actual_norm,
            )
        return AssertionReport(
            name="tool_calls",
            passed=False,
            detail=f"expected {expected_norm!r} got {actual_norm!r}",
            expected=expected_norm,
            actual=actual_norm,
        )

    def _active_scorers(self, fixture: Fixture) -> list[Scorer]:
        """Combine runner-wide scorer set with per-fixture overrides.

        Per-fixture ``judge_scorers`` is additive: it does not replace the
        runner set. ``"g_eval"`` in the per-fixture list triggers a custom
        G-Eval scorer using the fixture's ``judge_criteria`` (or a no-op
        if criteria is empty).
        """
        wants_g_eval = False
        names = set(s.name for s in self.scorer_set)
        for n in fixture.judge_scorers:
            if n == "g_eval":
                # Only emit a g_eval scorer if the fixture supplied
                # criteria; otherwise silently skip (no failing assertion
                # for missing-criteria).
                if fixture.judge_criteria:
                    wants_g_eval = True
            elif n in {"hallucination", "faithfulness", "answer_relevance",
                       "toxicity", "bias"}:
                names.add(n)

        builtins = {s.name: s for s in BUILTIN_SCORERS}
        out: list[Scorer] = []
        for n in sorted(names):
            if n in builtins:
                out.append(builtins[n])
        if wants_g_eval and fixture.judge_criteria:
            out.append(g_eval(fixture.judge_criteria, threshold=0.7))
        return out

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run(self, fixtures: Iterable[Fixture]) -> RunnerReport:
        report = RunnerReport()
        for fx in fixtures:
            report.results.append(self.run_one(fx))
        return report

    def run_dir(self, path: str | Path, pattern: str = "*.{json,yaml,yml}") -> RunnerReport:
        from .fixture import load_fixture_dir
        return self.run(load_fixture_dir(path, pattern=pattern).fixtures)


__all__ = [
    "AssertionReport",
    "Runner",
    "RunnerConfig",
    "RunnerReport",
    "TestCaseResult",
    "jaccard",
]
