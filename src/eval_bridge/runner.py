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


class Runner:
    """Executes fixtures, runs assertions, accumulates a report."""

    def __init__(
        self,
        provider: Provider | None = None,
        scrubber: Scrubber | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.provider = provider or provider_from_config(self.config)
        self.scrubber = scrubber or Scrubber.from_default_config()

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

        assertions = self._assert(fixture, text)
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

    def _assert(self, fixture: Fixture, text: str) -> list[AssertionReport]:
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

        return reports

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
