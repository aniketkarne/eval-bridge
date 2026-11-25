"""Scrub accounting: mapping scrubbing must count what it actually replaces.

Two failure modes are pinned here:

1. ``capture`` reported ``scrubber_counts`` computed over ``json.dumps(data)``
   (the JSON serialization of the raw input) instead of the scrubbing that
   produced the written fixture. The serialization is not the content:
   matches in keys were counted but never replaced, and patterns that only
   match per-leaf (e.g. anything crossing an escaped newline) were replaced
   but never counted.
2. Dict keys were never scrubbed at all, so PII-bearing keys landed verbatim
   in captured fixtures while the counts claimed they were handled.
"""

from __future__ import annotations

import json
import re

from eval_bridge import Scrubber, ScrubberConfig, Pattern


def test_scrub_mapping_with_counts_counts_every_replacement():
    s = Scrubber()
    out, counts = s.scrub_mapping_with_counts({
        "user": "jane@example.com",
        "history": [
            {"role": "user", "content": "ip was 10.0.0.1 and jane@example.com"},
            {"role": "assistant", "content": "ok"},
        ],
    })
    assert out["user"] == "{email}"
    assert out["history"][0]["content"] == "ip was {ip} and {email}"
    assert counts == {"email": 2, "ipv4": 1}


def test_counts_are_replacements_not_serialization_matches():
    # A pattern that matches only in the real leaf, never in its JSON
    # serialization: json.dumps escapes the newline to a literal "\n".
    pat = Pattern(name="pair", pattern=re.compile(r"line1\nline2"), replacement="{pair}")
    s = Scrubber(ScrubberConfig(extra_patterns=(pat,)))
    data = {"k": "x\nline1\nline2\ny"}
    out, counts = s.scrub_mapping_with_counts(data)
    assert out["k"] == "x\n{pair}\ny"
    assert counts == {"pair": 1}
    # The old accounting (scrub the serialization) sees nothing to replace
    # while the fixture content actually changed - counts and content must
    # never disagree like that again.
    serial_counts = s.scrub(json.dumps(data)).counts
    assert serial_counts.get("pair", 0) == 0


def test_scrub_mapping_scrubs_pii_keys():
    s = Scrubber()
    out = s.scrub_mapping({"meta": {"jane@example.com": "value"}})
    assert out == {"meta": {"{email}": "value"}}
    assert "jane@example.com" not in json.dumps(out)


def test_scrub_mapping_key_collisions_lose_no_entries():
    s = Scrubber()
    out, counts = s.scrub_mapping_with_counts({
        "jane@example.com": 1,
        "bob@example.com": 2,
    })
    # Both keys scrub to "{email}" - deterministic suffixes keep both entries.
    assert out == {"{email}": 1, "{email}-2": 2}
    assert counts["email"] == 2


def test_scrub_mapping_key_rename_avoids_existing_original_key():
    s = Scrubber()
    out, _ = s.scrub_mapping_with_counts({
        "jane@example.com": "a",
        "{email}": "b",
    })
    # The renamed key must not clobber a literal "{email}" key from the input.
    assert out["{email}"] == "b"
    assert out["{email}-2"] == "a"


def test_scrub_mapping_counts_match_scrubbed_output():
    # Invariant: sum(counts) == number of placeholder substitutions actually
    # present in the returned copy (keys + values).
    s = Scrubber()
    data = {
        "trace_id": "t-1",
        "prompt": "contact jane@example.com, card 4111 1111 1111 1111",
        "nested": {"deep": ["ssn 123-45-6789", {"k": "from 10.0.0.1"}]},
    }
    out, counts = s.scrub_mapping_with_counts(data)
    blob = json.dumps(out)
    substitutions = (
        blob.count("{email}") + blob.count("{credit_card}")
        + blob.count("{ssn}") + blob.count("{ip}")
    )
    assert sum(counts.values()) == substitutions
    assert counts == {"email": 1, "credit_card": 1, "ssn": 1, "ipv4": 1}


def test_scrub_mapping_with_counts_agrees_with_scrub_mapping():
    s = Scrubber()
    data = {"a": "jane@example.com", "b": {"c": ["10.0.0.1"]}}
    plain = s.scrub_mapping(data)
    with_counts, _ = s.scrub_mapping_with_counts(data)
    assert plain == with_counts
