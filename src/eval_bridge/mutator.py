"""Counterfactual mutation engine.

Given a single captured failure, generate N adversarial variations of the
fixture. The goal is to detect prompt overfitting: if a developer patches
a prompt to pass *exactly* the captured wording, a synonym or reordered
variant should still fail until the underlying principle is fixed.

All operations are deterministic given a seed, so re-running with the same
seed produces the same variants. Zero external dependencies.

Four perturbation families:

- **Entity swap**: replace names, dates, and numbers with stable
  placeholders (``[NAME_1]``, ``[DATE_1]``, ``[NUM_1]``) so the structure
  is preserved but the specific values are noised.
- **Typo & casing**: adjacent-character transposition, dropped punctuation,
  lowercase collapse. Models the mobile-keyboard failure mode.
- **Reorder**: shuffle mid-sentence clauses (commas/semicolons as split
  points) and optionally shuffle ``messages`` order for multi-turn fixtures.
- **Synonym**: small built-in lookup table (``summarize`` ↔ ``condense``,
  ``explain`` ↔ ``describe``, ...). Cheap, catches overfit on specific
  verbs.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Any

from .fixture import Fixture


# ---------------------------------------------------------------------------
# Synonym table. Keep tiny on purpose — bigger tables drift out of date and
# noisier mutations hurt more than they help.
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "summarize": ("condense", "recap", "tldr"),
    "condense": ("summarize", "recap"),
    "explain": ("describe", "clarify", "outline"),
    "describe": ("explain", "detail"),
    "list": ("enumerate", "itemize"),
    "translate": ("convert", "render"),
    "fix": ("repair", "correct", "patch"),
    "create": ("make", "build", "generate"),
    "delete": ("remove", "drop", "erase"),
    "find": ("locate", "search", "look up"),
    "compare": ("contrast", "differ"),
    "analyze": ("examine", "review", "study"),
    "rewrite": ("rephrase", "reword"),
    "extract": ("pull", "grab", "fetch"),
    "format": ("style", "structure"),
    "validate": ("check", "verify"),
    "convert": ("transform", "translate"),
    "should": ("must", "needs to"),
    "must": ("should", "needs to"),
    "please": ("kindly", ""),  # empty string = deletion
}


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Conservative entity patterns. We never mutate PII (the scrubber handles
# that); we mutate ordinary names, dates, and numbers that the model
# shouldn't be overfitting on.
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})\b"
)
_NUM_RE = re.compile(r"\b\d+\b")

_CLAUSE_SPLIT_RE = re.compile(r"(?<=,)\s+|(?<=;)\s+|(?<=\.)\s+")

# Common contractions and short words we won't typo. Typos on these add
# noise without testing anything useful.
_PROTECTED_TYPO = frozenset({
    "the", "and", "you", "are", "for", "with", "this", "that", "from",
    "have", "has", "was", "were", "but", "not", "can", "will", "would",
    "should", "could", "into", "than", "then", "them", "they",
})

_WORD_RE = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mutate_fixture(
    fixture: Fixture,
    *,
    n: int = 5,
    seed: int = 0,
    families: tuple[str, ...] | None = None,
) -> list[Fixture]:
    """Return *n* deterministic mutations of *fixture*.

    Parameters
    ----------
    fixture
        Source fixture. Returned as the first element is NOT preserved —
        callers that want the original should keep their own reference.
    n
        Number of variants to generate. Clamped to [1, 32].
    seed
        RNG seed for deterministic output.
    families
        Restrict to these perturbation families. Default is all four:
        ``"entity"``, ``"typo"``, ``"reorder"``, ``"synonym"``.

    Each variant preserves ``trace_id`` and ``capture`` so they replay
    against the same production event. Variants get distinct
    ``trace_id`` suffixes so the runner can keep them separate.
    """
    if n < 1:
        return []
    n = min(n, 32)
    rng = random.Random(seed)
    if families is None:
        families = ("entity", "typo", "reorder", "synonym")

    ops: list[tuple[str, Any]] = []
    if "entity" in families:
        ops.append(("entity", _entity_swap))
    if "typo" in families:
        ops.append(("typo", _typo))
    if "reorder" in families:
        ops.append(("reorder", _reorder_clauses))
    if "synonym" in families:
        ops.append(("synonym", _synonym))
    if not ops:
        # No recognized families — return the source unchanged so the
        # caller can still iterate n variants. Better than crashing.
        ops = [("identity", _identity)]

    variants: list[Fixture] = []
    for i in range(n):
        # Round-robin pick the op, then add a touch of randomness so the
        # same family still varies across indices.
        op_name, op_fn = ops[i % len(ops)]
        mutated_prompt = op_fn(fixture.prompt, rng)
        mutated_system = (
            op_fn(fixture.system, rng) if fixture.system else fixture.system
        )
        mutated_messages = (
            [_msg_with_swapped_content(m, op_fn, rng) for m in fixture.messages]
            if fixture.messages
            else list(fixture.messages)
        )
        if op_name == "reorder" and len(mutated_messages) > 2:
            mutated_messages = _reorder_messages(mutated_messages, rng)

        # Strip any assertions that pinned the original wording — they
        # would defeat the purpose of mutation. Forbidden substrings stay
        # (they're usually safety-critical). Schema / reference_reply /
        # model / capture metadata are preserved verbatim.
        cleaned = replace(
            fixture,
            trace_id=f"{fixture.trace_id}.m{i + 1}",
            prompt=mutated_prompt,
            system=mutated_system,
            messages=mutated_messages,
            expected_substrings=[],
        )
        variants.append(cleaned)
    return variants


# ---------------------------------------------------------------------------
# Operations. Each takes (text, rng) -> text.
# ---------------------------------------------------------------------------

def _entity_swap(text: str, rng: random.Random) -> str:
    """Replace names/dates/numbers with stable placeholders.

    Replacement happens in reverse order so span indices don't shift mid-
    pass. The counters are RNG-seeded so the same seed produces the same
    placeholder identities.
    """
    counters = {"name": 0, "date": 0, "num": 0}

    def pick_label(kind: str) -> str:
        counters[kind] += 1
        # Mix the counter with rng for variety, but stay deterministic
        # across calls in this pass.
        return f"[{kind.upper()}_{counters[kind]}]"

    edits: list[tuple[int, int, str]] = []

    # Names (whole-word capitalized tokens)
    for m in _NAME_RE.finditer(text):
        edits.append((m.start(), m.end(), pick_label("name")))
    # Dates
    for m in _DATE_RE.finditer(text):
        edits.append((m.start(), m.end(), pick_label("date")))
    # Numbers
    for m in _NUM_RE.finditer(text):
        edits.append((m.start(), m.end(), pick_label("num")))

    if not edits:
        return text
    # Resolve overlaps: keep the earliest-starting match.
    edits.sort(key=lambda e: (e[0], -e[1]))
    deduped: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, repl in edits:
        if start < last_end:
            continue
        deduped.append((start, end, repl))
        last_end = end

    out_parts: list[str] = []
    cursor = 0
    for start, end, repl in deduped:
        out_parts.append(text[cursor:start])
        out_parts.append(repl)
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _typo(text: str, rng: random.Random) -> str:
    """Inject one transposition or punctuation drop per call.

    Transposition is preferred (closer to mobile-keyboard errors). If the
    text is too short or has no eligible word, returns the input unchanged.
    """
    words = list(_WORD_RE.finditer(text))
    eligible = [m for m in words if m.group(0).lower() not in _PROTECTED_TYPO and len(m.group(0)) >= 4]
    if not eligible:
        return text
    target = rng.choice(eligible)
    word = target.group(0)

    # 70% transposition, 30% dropped letter
    if rng.random() < 0.7:
        # Swap two adjacent characters (avoid first/last which would just
        # be a casing change).
        i = rng.randrange(1, len(word) - 2)
        mutated = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    else:
        i = rng.randrange(1, len(word) - 1)
        mutated = word[:i] + word[i + 1:]

    return text[: target.start()] + mutated + text[target.end():]


def _reorder_clauses(text: str, rng: random.Random) -> str:
    """Shuffle comma/semicolon-separated clauses.

    Only operates on text with >=2 split points — single-clause text is
    returned unchanged. Preserves the opening clause (so the prompt still
    starts with the same instruction).
    """
    parts = _CLAUSE_SPLIT_RE.split(text)
    if len(parts) < 3:
        return text
    # parts[0] is the leading text before the first comma; preserve it.
    head = parts[0]
    body = parts[1:]
    # Fisher-Yates shuffle, but only if it'll actually move something.
    rng.shuffle(body)
    if body == parts[1:]:
        # Force at least one swap so callers get a real mutation.
        body[0], body[-1] = body[-1], body[0]
    return head + ", " + ", ".join(body)


def _synonym(text: str, rng: random.Random) -> str:
    """Replace the first eligible word found in *text* with a synonym.

    Returns the input unchanged if no synonym-keyed word appears.
    """
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in _SYNONYMS:
            replacement = rng.choice(_SYNONYMS[w])
            if not replacement:
                # Empty = word deletion. Don't delete the only word.
                continue
            # Preserve the original casing of the first character.
            if m.group(0)[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            return text[: m.start()] + replacement + text[m.end():]
    return text


def _identity(text: str, rng: random.Random) -> str:
    """No-op op used when the caller requested only unknown families."""
    return text


# ---------------------------------------------------------------------------
# Message-level helpers (multi-turn fixtures)
# ---------------------------------------------------------------------------

def _msg_with_swapped_content(msg: dict[str, Any], op_fn: Any, rng: random.Random) -> dict[str, Any]:
    """Apply a text op to a single chat message's content."""
    content = msg.get("content")
    if not isinstance(content, str):
        return dict(msg)
    return {**msg, "content": op_fn(content, rng)}


def _reorder_messages(messages: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Shuffle non-system messages. System message stays first."""
    if not messages:
        return list(messages)
    head: list[dict[str, Any]] = []
    body: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            head.append(m)
        else:
            body.append(m)
    rng.shuffle(body)
    return head + body


__all__ = ["mutate_fixture"]