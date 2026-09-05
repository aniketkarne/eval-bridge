"""Deterministic PII scrubber.

Built-in patterns: emails, IPv4/IPv6, JWT, API keys (sk-*, ghp_*, xoxb-*, AKIA*,
generic bearer tokens), credit cards (Luhn-validated), US SSNs.

Replacements are configurable. The output is **deterministic** - the same input
always produces the same output - which is essential for replayable fixtures.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from .errors import ScrubberError

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Email: pragmatic RFC-5322 subset. Anchored on whitespace/punctuation.
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}"
    r"(?![A-Za-z0-9._%+-])"
)

# IPv4 (0-255 per octet) and IPv6 (compressed, including ::).
_IPV4_RE = re.compile(
    r"(?<![0-9.])"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}"
    r"(?![0-9.])"
)
_IPV6_RE = re.compile(
    r"(?<![A-Fa-f0-9:])"
    r"(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}"
    r"(?::[A-Fa-f0-9]{1,4}){0,6}"
    r"|::(?:[A-Fa-f0-9]{1,4}:){0,6}[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,7}:"
    r"(?![A-Fa-f0-9:])"
)

# JWT: three base64url segments separated by dots. Require header to start with
# eyJ (the base64 of '{"').
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")

# Common API token prefixes. These are high-precision; if your org uses
# something else add it via extra_patterns.
_API_TOKEN_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),                       # OpenAI
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{16,}"),                 # OpenAI project
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),                        # GitHub PAT
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),                # GitHub fine-grained
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),               # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}"),                            # AWS access key
    re.compile(r"\bASIA[0-9A-Z]{16}"),                            # AWS session
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),                    # Google API
    re.compile(r"\bya29\.[0-9A-Za-z_\-]{20,}"),                   # Google OAuth
)

# Generic "Bearer xxx" tokens (and "Token xxx", "Basic xxx"). We capture the
# header but only replace the value (group 2).
_BEARER_RE = re.compile(
    r"(?i)\b(Bearer|Token|Basic)\s+([A-Za-z0-9._\-+/=]{12,})"
)

# Credit cards: 13-19 digits, optionally separated by spaces or dashes.
# Validated with Luhn before being treated as a card.
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

# US SSN: AAA-GG-SSSS, must not start with 000, 666, or 9xx in area.
_SSN_RE = re.compile(
    r"(?<!\d)(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"
)


@dataclass(frozen=True)
class Pattern:
    """A named regex pattern + replacement."""

    name: str
    pattern: re.Pattern[str]
    replacement: str
    validator: "callable | None" = None  # type: ignore[type-arg]


@dataclass(frozen=True)
class ScrubberConfig:
    """User-facing scrubber configuration.

    Every field corresponds to a built-in pattern. Replacement strings may
    contain ``{placeholder}`` literals (e.g. ``"{email}"``) or any other
    fixed string. Extra patterns can be added for org-specific identifiers.
    """

    email: str = "{email}"
    ip: str = "{ip}"
    jwt: str = "{jwt}"
    api_token: str = "{api_token}"
    bearer: str = "{api_token}"  # Bearer/Token/Basic headers share this replacement
    credit_card: str = "{credit_card}"
    ssn: str = "{ssn}"

    extra_patterns: tuple[Pattern, ...] = field(default_factory=tuple)

    def replace(self, **kwargs: Any) -> "ScrubberConfig":
        """Return a copy with the given fields overwritten."""
        return replace(self, **kwargs)


def _luhn_ok(digits: str) -> bool:
    total = 0
    alt = False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _strip_card(raw: str) -> str:
    return re.sub(r"[ -]", "", raw)


def _validate_card(raw: str) -> bool:
    digits = _strip_card(raw)
    return 13 <= len(digits) <= 19 and digits.isdigit() and _luhn_ok(digits)


def default_patterns(cfg: ScrubberConfig) -> list[Pattern]:
    """Build the ordered pattern list for *cfg*."""
    patterns: list[Pattern] = []
    if cfg.email:
        patterns.append(Pattern("email", _EMAIL_RE, cfg.email))
    if cfg.ip:
        patterns.append(Pattern("ipv4", _IPV4_RE, cfg.ip))
        patterns.append(Pattern("ipv6", _IPV6_RE, cfg.ip))
    if cfg.ssn:
        patterns.append(Pattern("ssn", _SSN_RE, cfg.ssn, validator=lambda m: True))
    if cfg.jwt:
        patterns.append(Pattern("jwt", _JWT_RE, cfg.jwt))
    if cfg.bearer:
        patterns.append(
            Pattern("bearer", _BEARER_RE, cfg.bearer,
                    validator=lambda m: True),  # group 2 replaced, group 1 preserved
        )
    if cfg.credit_card:
        patterns.append(
            Pattern("credit_card", _CARD_RE, cfg.credit_card, validator=_validate_card),
        )
    if cfg.api_token:
        for pat in _API_TOKEN_RES:
            patterns.append(Pattern("api_token", pat, cfg.api_token))
    patterns.extend(cfg.extra_patterns)
    return patterns


# Public alias used by tests / SDK.
DEFAULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    _EMAIL_RE,
    _IPV4_RE,
    _IPV6_RE,
    _SSN_RE,
    _JWT_RE,
    _BEARER_RE,
    _CARD_RE,
    *_API_TOKEN_RES,
)


# ---------------------------------------------------------------------------
# Result + Scrubber
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScrubResult:
    """Outcome of a scrub operation."""

    text: str
    matches: tuple[dict[str, Any], ...]
    # A simple per-name counter for quick assertions.
    counts: Mapping[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Scrubber:
    """Deterministic scrubber.

    >>> s = Scrubber.from_default_config()
    >>> s.scrub("ping jane@example.com from 10.0.0.1").text
    'ping {email} from {ip}'
    """

    __slots__ = ("_config", "_patterns")

    def __init__(self, config: ScrubberConfig | None = None) -> None:
        self._config = config or ScrubberConfig()
        self._patterns = default_patterns(self._config)

    @classmethod
    def from_default_config(cls) -> "Scrubber":
        return cls(ScrubberConfig())

    @property
    def config(self) -> ScrubberConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub(self, text: str) -> ScrubResult:
        """Scrub *text*, returning the redacted string + match metadata."""
        if not isinstance(text, str):  # pragma: no cover - defensive
            raise ScrubberError(
                "scrub() expects str input",
                context={"type": type(text).__name__},
            )

        matches: list[dict[str, Any]] = []
        working = text

        # Apply each pattern in order, on the latest working string.
        for pat in self._patterns:
            new_parts: list[str] = []
            cursor = 0
            for m in pat.pattern.finditer(working):
                raw = m.group(0)
                if pat.validator is not None and not pat.validator(raw):
                    continue

                # Capture group handling: for Bearer we replace group 2 only,
                # preserving the literal "Bearer " prefix.
                if pat.name == "bearer":
                    scheme = m.group(1)
                    scheme_case = scheme[:1] + scheme[1:].lower()
                    if scheme.isupper():
                        scheme_case = scheme.upper()
                    start = m.start(2)
                    end = m.end(2)
                    new_parts.append(working[cursor:start])
                    new_parts.append(pat.replacement)
                    cursor = end
                    matches.append(
                        {
                            "name": pat.name,
                            "value": raw,
                            "span": [m.start(), m.end()],
                            "replacement": pat.replacement,
                            "scheme": scheme,
                        }
                    )
                else:
                    new_parts.append(working[cursor:m.start()])
                    new_parts.append(pat.replacement)
                    cursor = m.end()
                    matches.append(
                        {
                            "name": pat.name,
                            "value": raw,
                            "span": [m.start(), m.end()],
                            "replacement": pat.replacement,
                        }
                    )
            new_parts.append(working[cursor:])
            working = "".join(new_parts)

        counts: dict[str, int] = {}
        for m in matches:
            counts[m["name"]] = counts.get(m["name"], 0) + 1

        return ScrubResult(
            text=working,
            matches=tuple(matches),
            counts=counts,
        )

    def scrub_mapping(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Recursively scrub every string leaf in *data*."""
        out: dict[str, Any] = {}
        for k, v in data.items():
            out[k] = self._scrub_value(v)
        return out

    def _scrub_value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.scrub(v).text
        if isinstance(v, Mapping):
            return {k: self._scrub_value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            scrubbed = [self._scrub_value(x) for x in v]
            return type(v)(scrubbed) if isinstance(v, tuple) else scrubbed
        return v


# ---------------------------------------------------------------------------
# Residual secret scan
# ---------------------------------------------------------------------------

# Anything matching these after scrubbing means the scrubber missed something.
_RESIDUAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL_RE),
    ("ipv4", _IPV4_RE),
    ("jwt", _JWT_RE),
    ("credit_card", _CARD_RE),
    ("ssn", _SSN_RE),
    ("api_token", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("api_token", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("api_token", re.compile(r"\bAKIA[0-9A-Z]{16}")),
)


def find_residual_secrets(text: str) -> list[dict[str, Any]]:
    """Return any remaining secret-looking spans after scrubbing.

    Used by the runner as a final safety gate. Each returned dict has the same
    shape as ``ScrubResult.matches`` entries.
    """
    out: list[dict[str, Any]] = []
    for name, pat in _RESIDUAL_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            if name == "credit_card" and not _validate_card(raw):
                continue
            out.append(
                {
                    "name": name,
                    "value": raw,
                    "span": [m.start(), m.end()],
                }
            )
    return out


__all__ = [
    "DEFAULT_PATTERNS",
    "Pattern",
    "ScrubResult",
    "Scrubber",
    "ScrubberConfig",
    "default_patterns",
    "find_residual_secrets",
]


def _self_test() -> None:  # pragma: no cover - dev sanity
    s = Scrubber.from_default_config()
    sample = (
        "ping jane@example.com from 10.0.0.1 "
        "card 4111 1111 1111 1111 "
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature "
        "ssn 123-45-6789 token ghp_abcdef0123456789abcdef"
    )
    r = s.scrub(sample)
    assert "{email}" in r.text and "{ip}" in r.text, r.text
    assert "{credit_card}" in r.text, r.text
    assert "{jwt}" in r.text, r.text
    assert "{ssn}" in r.text, r.text
    assert "{api_token}" in r.text, r.text


_ASCII_SAFE = string.ascii_letters + string.digits + "_{}"


def assert_safe_identifier(name: str) -> None:
    """Raise if *name* cannot be used as a deterministic placeholder key."""
    if not name or any(c not in _ASCII_SAFE for c in name):
        raise ScrubberError(
            "invalid identifier", context={"name": name}
        )


def build_extra_patterns(items: Iterable[Mapping[str, str]]) -> list[Pattern]:
    """Build ``Pattern`` objects from a mapping iterable of user config."""
    out: list[Pattern] = []
    for item in items:
        try:
            name = item["name"]
            pattern = item["pattern"]
            replacement = item.get("replacement", "{" + name + "}")
        except KeyError as e:  # pragma: no cover - validated upstream
            raise ScrubberError(
                "extra_patterns requires name, pattern, optional replacement",
                context={"missing": str(e)},
            ) from e
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ScrubberError(
                "invalid regex", context={"pattern": pattern, "error": str(e)}
            ) from e
        out.append(Pattern(name=name, pattern=compiled, replacement=replacement))
    return out
