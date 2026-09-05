"""Tests for JUnit XML output."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from eval_bridge import Fixture, Runner, RunnerConfig


def _read_junit(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def test_junit_passing_case(tmp_path: Path):
    fx = Fixture(trace_id="ok", prompt="hi", fixture_response="ok")
    report = Runner().run([fx])
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    root = _read_junit(p)
    assert root.tag == "testsuite"
    assert root.attrib["tests"] == "1"
    assert root.attrib["failures"] == "0"
    assert root.attrib["errors"] == "0"
    cases = root.findall("testcase")
    assert len(cases) == 1
    assert cases[0].attrib["name"] == "ok"
    assert cases[0].find("failure") is None


def test_junit_failing_case_has_failure_element(tmp_path: Path):
    fx = Fixture(trace_id="bad", prompt="hi", fixture_response="x",
                 forbidden_substrings=["x"])
    report = Runner().run([fx])
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    root = _read_junit(p)
    assert root.attrib["failures"] == "1"
    case = root.find("testcase")
    failure = case.find("failure")
    assert failure is not None
    assert failure.attrib["type"] == "forbidden_substring"


def test_junit_provider_error_marks_error_element(tmp_path: Path):
    fx = Fixture(trace_id="boom", prompt="hi")  # no fixture_response
    report = Runner().run([fx])
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    root = _read_junit(p)
    assert root.attrib["errors"] == "1"
    case = root.find("testcase")
    err = case.find("error")
    assert err is not None
    assert err.attrib["type"] == "ProviderError"


def test_junit_mixed_results(tmp_path: Path):
    fxs = [
        Fixture(trace_id="p1", prompt="a", fixture_response="ok"),
        Fixture(trace_id="p2", prompt="b", fixture_response="no",
                expected_substrings=["yes"]),
    ]
    report = Runner().run(fxs)
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    root = _read_junit(p)
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    cases = root.findall("testcase")
    assert {c.attrib["name"] for c in cases} == {"p1", "p2"}


def test_junit_creates_parent_dir(tmp_path: Path):
    fx = Fixture(trace_id="ok", prompt="hi", fixture_response="ok")
    report = Runner().run([fx])
    out = tmp_path / "deep" / "nested" / "junit.xml"
    report.write_junit(out)
    assert out.exists()


def test_junit_xml_declaration_present(tmp_path: Path):
    fx = Fixture(trace_id="ok", prompt="hi", fixture_response="ok")
    report = Runner().run([fx])
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    raw = p.read_bytes()
    assert raw.startswith(b"<?xml")
    assert b"encoding='utf-8'" in raw or b'encoding="utf-8"' in raw


def test_junit_duration_sums_correctly(tmp_path: Path):
    fxs = [Fixture(trace_id=f"t{i}", prompt="x", fixture_response="ok")
           for i in range(3)]
    report = Runner().run(fxs)
    p = tmp_path / "junit.xml"
    report.write_junit(p)
    root = _read_junit(p)
    suite_time = float(root.attrib["time"])
    case_times = sum(float(c.attrib["time"]) for c in root.findall("testcase"))
    assert abs(suite_time - case_times) < 0.05
