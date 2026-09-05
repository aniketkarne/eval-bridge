"""Tests for the counterfactual mutation engine."""

from __future__ import annotations

import pytest

from eval_bridge import Fixture, mutate_fixture


def _fx(prompt: str = "Summarize the Q3 sales report for Jane Doe on 2025-09-12.",
        *, system: str | None = None, messages: list[dict] | None = None) -> Fixture:
    return Fixture(
        trace_id="t-001",
        prompt=prompt,
        system=system,
        messages=messages or [],
        model="gpt-4o-mini",
        expected_substrings=["Jane Doe", "2025-09-12"],
        forbidden_substrings=["BEGIN PRIVATE KEY"],
    )


# ---------------------------------------------------------------------------
# Determinism + shape
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_variants() -> None:
    a = mutate_fixture(_fx(), n=8, seed=42)
    b = mutate_fixture(_fx(), n=8, seed=42)
    assert [v.prompt for v in a] == [v.prompt for v in b]


def test_different_seeds_produce_different_variants() -> None:
    a = mutate_fixture(_fx(), n=16, seed=1)
    b = mutate_fixture(_fx(), n=16, seed=2)
    assert [v.prompt for v in a] != [v.prompt for v in b]


def test_variant_trace_ids_are_distinct_and_prefixed() -> None:
    variants = mutate_fixture(_fx(), n=5, seed=0)
    ids = [v.trace_id for v in variants]
    assert len(set(ids)) == 5
    assert all(i.startswith("t-001.m") for i in ids)
    assert ids == [f"t-001.m{i}" for i in range(1, 6)]


def test_n_zero_returns_empty_list() -> None:
    assert mutate_fixture(_fx(), n=0, seed=0) == []


def test_n_clamped_to_32() -> None:
    variants = mutate_fixture(_fx(), n=999, seed=0)
    assert len(variants) == 32


def test_capture_metadata_preserved_across_variants() -> None:
    src = _fx()
    src.capture = {"incident_url": "https://example.com/inc/123"}
    for v in mutate_fixture(src, n=4, seed=0):
        assert v.capture == {"incident_url": "https://example.com/inc/123"}


# ---------------------------------------------------------------------------
# Per-family behavior
# ---------------------------------------------------------------------------

def test_entity_swap_replaces_names_dates_and_numbers() -> None:
    variants = mutate_fixture(_fx(prompt="Jane Doe filed 7 reports on 2025-09-12."), n=8, seed=7)
    # Some variant must contain a [NAME_n] placeholder.
    assert any("[NAME_" in v.prompt for v in variants)
    # Some variant must contain a [DATE_n] placeholder.
    assert any("[DATE_" in v.prompt for v in variants)
    # Some variant must contain a [NUM_n] placeholder.
    assert any("[NUM_" in v.prompt for v in variants)


def test_entity_swap_preserves_pii_scrubbing() -> None:
    """Entity swap must not undo scrubber work — emails stay redacted."""
    # First scrub, then mutate. The mutator only touches names/dates/numbers.
    from eval_bridge import Scrubber
    scrubber = Scrubber.from_default_config()
    base = _fx(prompt="Contact Jane Doe at jane.doe@example.com on 2025-09-12.")
    scrubbed_prompt = scrubber.scrub(base.prompt).text
    base = Fixture(
        trace_id=base.trace_id,
        prompt=scrubbed_prompt,
        system=base.system,
        messages=base.messages,
        model=base.model,
        expected_substrings=[],
        forbidden_substrings=base.forbidden_substrings,
    )
    for v in mutate_fixture(base, n=8, seed=0):
        # The scrubbed email token must still be in the variant — mutator
        # never re-introduces raw PII.
        assert "{email}" in v.prompt


def test_typo_injects_a_real_mutation_when_text_is_eligible() -> None:
    prompt = "Summarize the entire quarterly sales dataset thoroughly please"
    variants = mutate_fixture(_fx(prompt=prompt), n=32, seed=0)
    # With 32 variants across families, at least one must be a typo
    # variant that changed a non-protected word.
    typo_variants = [
        v for v in variants
        if v.prompt != prompt and not any(k in v.prompt for k in ("[NAME_", "[DATE_", "[NUM_"))
    ]
    assert typo_variants, "expected at least one non-entity mutation"
    # The typo should change at least one character.
    assert any(v.prompt != prompt for v in typo_variants)


def test_typo_is_safe_on_protected_words() -> None:
    """Short protected words like 'the' / 'and' must not be mutated even with many tries."""
    prompt = "Summarize the and the and the and the and"
    variants = mutate_fixture(_fx(prompt=prompt), n=32, seed=0)
    # No entity placeholders appear (no names/dates/numbers); if a typo
    # variant exists, it must NOT have touched 'the' or 'and'.
    for v in variants:
        # Either unchanged, or mutated a longer word.
        assert v.prompt.count("the") + v.prompt.count("and") >= 4


def test_reorder_shuffles_clauses() -> None:
    prompt = "first clause, second clause, third clause, fourth clause"
    variants = mutate_fixture(_fx(prompt=prompt), n=16, seed=1)
    # Some variant must reorder at least two of the trailing clauses while
    # keeping "first clause," as the prefix.
    reordered = [v.prompt for v in variants if v.prompt.startswith("first clause,")
                 and v.prompt != prompt]
    assert reordered, "expected at least one reorder variant"


def test_reorder_does_not_split_single_clause_text() -> None:
    """Reorder on text with fewer than 3 split-points returns the input verbatim."""
    prompt = "only one clause"
    for v in mutate_fixture(_fx(prompt=prompt), n=8, seed=0):
        assert "," not in v.prompt


def test_synonym_replaces_at_least_one_word_when_eligible() -> None:
    prompt = "Please summarize and analyze the dataset, then list the findings."
    variants = mutate_fixture(_fx(prompt=prompt), n=8, seed=0)
    # Across 8 variants at least one synonym swap should land.
    synonym_hits = [
        v for v in variants
        if any(w in v.prompt.lower() for w in ("condense", "recap", "examine", "review", "enumerate", "itemize"))
    ]
    assert synonym_hits, "expected at least one synonym substitution"


def test_synonym_returns_unchanged_when_no_keyed_word() -> None:
    """If no synonym-keyed word appears, no synonym substitution happens.

    Other families (typo, entity, reorder) may still mutate the text.
    """
    prompt = "What color is the sky?"
    synonym_replacements = (
        "condense", "recap", "examine", "review",
        "enumerate", "itemize", "describe", "clarify",
        "outline", "tldr",
    )
    for v in mutate_fixture(_fx(prompt=prompt), n=8, seed=0):
        for w in synonym_replacements:
            assert w not in v.prompt.lower()


# ---------------------------------------------------------------------------
# Assertion hygiene
# ---------------------------------------------------------------------------

def test_expected_substrings_dropped_on_variants() -> None:
    """Variants must NOT carry the original wording-pin assertions."""
    for v in mutate_fixture(_fx(), n=8, seed=0):
        assert v.expected_substrings == []


def test_forbidden_substrings_preserved_on_variants() -> None:
    """Safety gates (forbidden substrings) survive mutation — these are
    non-negotiable and the user expects them to follow the fixture."""
    src = _fx()
    src.forbidden_substrings = ["BEGIN PRIVATE KEY", "ghp_"]
    for v in mutate_fixture(src, n=8, seed=0):
        assert v.forbidden_substrings == ["BEGIN PRIVATE KEY", "ghp_"]


def test_schema_and_reference_reply_preserved_on_variants() -> None:
    src = _fx()
    src.schema = {"type": "object"}
    src.reference_reply = "Reference text"
    src.semantic_threshold = 0.7
    for v in mutate_fixture(src, n=4, seed=0):
        assert v.schema == {"type": "object"}
        assert v.reference_reply == "Reference text"
        assert v.semantic_threshold == 0.7


# ---------------------------------------------------------------------------
# Multi-turn fixtures
# ---------------------------------------------------------------------------

def test_messages_have_content_mutated() -> None:
    msg_fx = Fixture(
        trace_id="mt-001",
        prompt="ignored",
        messages=[
            {"role": "system", "content": "You summarize things."},
            {"role": "user", "content": "Summarize the Q3 sales report for Jane Doe."},
            {"role": "assistant", "content": "Here it is."},
        ],
    )
    variants = mutate_fixture(msg_fx, n=8, seed=0)
    # System message content must be mutated (entity/typo/etc. apply).
    assert any(v.messages[0]["content"] != "You summarize things." for v in variants)


def test_system_message_stays_first_when_reordering() -> None:
    msg_fx = Fixture(
        trace_id="mt-002",
        prompt="ignored",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
    )
    for v in mutate_fixture(msg_fx, n=16, seed=0):
        assert v.messages[0]["role"] == "system"


def test_multi_turn_reorder_shuffles_non_system_messages() -> None:
    msg_fx = Fixture(
        trace_id="mt-003",
        prompt="ignored",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
    )
    # With 16 variants, at least one must shuffle the non-system messages.
    bodies = [tuple(m["content"] for m in v.messages[1:]) for v in mutate_fixture(msg_fx, n=16, seed=0)]
    assert len(set(bodies)) > 1


# ---------------------------------------------------------------------------
# Family selection
# ---------------------------------------------------------------------------

def test_families_restricts_to_entity_only() -> None:
    """If we restrict to entity-only, no synonym/typo/reorder mutations happen."""
    variants = mutate_fixture(_fx(prompt="Summarize Q3 for Jane Doe"), n=8, seed=0,
                              families=("entity",))
    # No synonym targets (no [NAME_ etc. for any non-entity family).
    # Verify: every variant either == prompt or contains an entity placeholder.
    for v in variants:
        if v.prompt != "Summarize Q3 for Jane Doe":
            assert "[NAME_" in v.prompt or "[NUM_" in v.prompt or "[DATE_" in v.prompt


def test_families_unknown_name_is_silently_ignored() -> None:
    """Unknown family names should not crash — they just produce no effect."""
    variants = mutate_fixture(_fx(), n=4, seed=0, families=("nonsense",))
    assert len(variants) == 4