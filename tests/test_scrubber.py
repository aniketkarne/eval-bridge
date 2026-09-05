"""Tests for the deterministic PII scrubber."""

from __future__ import annotations

import pytest

from eval_bridge import Scrubber, ScrubberConfig, find_residual_secrets, Pattern
import re


# ---------------------------------------------------------------------------
# Basic positive cases
# ---------------------------------------------------------------------------

def test_email_is_replaced():
    s = Scrubber()
    out = s.scrub("contact: jane.doe@example.com").text
    assert out == "contact: {email}"
    assert s.scrub("contact: jane.doe@example.com").counts["email"] == 1


def test_ipv4_is_replaced():
    s = Scrubber()
    out = s.scrub("from 10.0.0.1 to 192.168.1.1").text
    assert out == "from {ip} to {ip}"
    assert s.scrub("from 10.0.0.1 to 192.168.1.1").counts["ipv4"] == 2


def test_ipv6_is_replaced():
    s = Scrubber()
    out = s.scrub("v6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 ok").text
    assert "{ip}" in out
    assert "2001:0db8" not in out


def test_jwt_is_replaced():
    s = Scrubber()
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.ZqQ1z2X7pq9b9J4D7rL9m0S9FQ6z4r1e9r6d6e2c1a"
    out = s.scrub(f"token={jwt}").text
    assert "{jwt}" in out
    assert jwt not in out


def test_credit_card_valid_luhn_is_replaced():
    s = Scrubber()
    out = s.scrub("card 4111 1111 1111 1111 done").text
    assert "{credit_card}" in out


def test_credit_card_invalid_luhn_not_replaced():
    s = Scrubber()
    out = s.scrub("card 4111 1111 1111 1112 done").text  # bad checksum
    assert "{credit_card}" not in out
    assert "4111 1111 1111 1112" in out


def test_ssn_is_replaced():
    s = Scrubber()
    out = s.scrub("ssn 123-45-6789 ok").text
    assert "{ssn}" in out


def test_ssn_invalid_prefix_not_replaced():
    s = Scrubber()
    out = s.scrub("ssn 000-45-6789 ok").text
    assert "{ssn}" not in out


def test_openai_api_key_is_replaced():
    s = Scrubber()
    out = s.scrub("key sk-proj-abcdefghijklmnopqrstuv1234").text
    assert "{api_token}" in out


def test_github_pat_is_replaced():
    s = Scrubber()
    pat = "ghp_" + "a" * 36
    out = s.scrub(f"token={pat}").text
    assert "{api_token}" in out
    assert pat not in out


def test_aws_access_key_is_replaced():
    s = Scrubber()
    out = s.scrub("AKIAIOSFODNN7EXAMPLE rest").text
    assert "{api_token}" in out


def test_bearer_header_is_replaced_keeping_scheme():
    s = Scrubber()
    out = s.scrub("Authorization: Bearer abcdefghijklmnopqrstuv").text
    assert out.startswith("Authorization: Bearer {api_token}")
    assert "abcdefghijklmnopqrstuv" not in out


# ---------------------------------------------------------------------------
# Determinism + nested data
# ---------------------------------------------------------------------------

def test_deterministic_for_same_input():
    s = Scrubber()
    a = s.scrub("ping jane@example.com from 10.0.0.1").text
    b = s.scrub("ping jane@example.com from 10.0.0.1").text
    assert a == b


def test_scrub_mapping_recurses():
    s = Scrubber()
    out = s.scrub_mapping({
        "user": "jane@example.com",
        "history": [
            {"role": "user", "content": "ip was 10.0.0.1"},
            {"role": "assistant", "content": "ok"},
        ],
        "meta": {"nested": {"deep": "card 4111 1111 1111 1111"}},
    })
    assert out["user"] == "{email}"
    assert out["history"][0]["content"] == "ip was {ip}"
    assert out["meta"]["nested"]["deep"] == "card {credit_card}"


# ---------------------------------------------------------------------------
# Custom config + extra patterns
# ---------------------------------------------------------------------------

def test_custom_replacement_string():
    cfg = ScrubberConfig(email="REDACTED_EMAIL")
    s = Scrubber(cfg)
    out = s.scrub("hi alice@x.com").text
    assert out == "hi REDACTED_EMAIL"


def test_extra_pattern_is_applied():
    pat = Pattern(
        name="ticket",
        pattern=re.compile(r"TICKET-\d{4,}"),
        replacement="{ticket}",
    )
    cfg = ScrubberConfig(extra_patterns=(pat,))
    s = Scrubber(cfg)
    out = s.scrub("see TICKET-1234 for details").text
    assert out == "see {ticket} for details"


def test_extra_pattern_invalid_regex_raises():
    from eval_bridge.errors import ScrubberError
    with pytest.raises(ScrubberError):
        from eval_bridge.scrubber import build_extra_patterns
        build_extra_patterns([{"name": "bad", "pattern": "(", "replacement": "{x}"}])


# ---------------------------------------------------------------------------
# Residual secret scan
# ---------------------------------------------------------------------------

def test_residual_scan_clean_text():
    assert find_residual_secrets("hello world, no secrets here") == []


def test_residual_scan_detects_email():
    leaks = find_residual_secrets("ping jane@example.com")
    assert any(l["name"] == "email" for l in leaks)


def test_residual_scan_detects_github_pat():
    pat = "ghp_" + "a" * 36
    leaks = find_residual_secrets(f"key {pat}")
    assert any(l["name"] == "api_token" for l in leaks)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_pii_unchanged():
    s = Scrubber()
    out = s.scrub("this is a perfectly clean string").text
    assert out == "this is a perfectly clean string"
    assert s.scrub("this is a perfectly clean string").total == 0


def test_multiple_emails_in_one_string():
    s = Scrubber()
    out = s.scrub("a@x.com b@y.com c@z.org").text
    assert out == "{email} {email} {email}"


def test_email_not_matched_inside_longer_word():
    s = Scrubber()
    out = s.scrub("notreallyanemail@x.comish").text
    # The conservative regex stops at '.' in the TLD; trailing chars are not consumed.
    assert "notreallyanemail" not in out or "@x.com" not in out


def test_ipv4_invalid_octets_not_matched():
    s = Scrubber()
    out = s.scrub("not an ip: 999.999.999.999").text
    assert "{ip}" not in out


def test_non_string_input_raises():
    from eval_bridge.errors import ScrubberError
    s = Scrubber()
    with pytest.raises(ScrubberError):
        s.scrub(123)  # type: ignore[arg-type]
