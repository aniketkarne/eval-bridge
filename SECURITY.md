# Security Policy

## Reporting a vulnerability

Report privately via GitHub security advisories on this repository, or email
`aniket@karne.ai`. Expect an ack within a few days. Please don't open a public
issue for anything that could leak captured data.

## What this tool touches

eval-bridge is local-first. It reads and writes files you point it at and
commits nothing on its own. The only network surface is opt-in:

- `--judge llm` and `provider = "openai_compat"` send fixture prompts and
  responses to the configured endpoint (`runner.base_url`).
- Nothing else — capture, scrub, import, and the offline judge never open a
  connection.

If your fixtures contain data you can't send to a model provider, run with the
default offline judge.

## Known limitations

These are properties of the design, not open bugs. Read them before trusting
the tool with regulated data:

- **The scrubber is a pattern library, not a guarantee.** Built-in patterns
  cover emails, IPs, JWTs, common API token prefixes, Luhn-valid cards, and
  US SSNs. Organization-specific identifiers (employee IDs, ticket numbers,
  internal hostnames) are invisible to it — add them via
  `scrubber.extra_patterns` in `eval-bridge.toml`.
- **The residual secret scan is a net, not proof of absence.** It re-scans the
  scrubbed output with a subset of the patterns. Passing means "no known
  pattern remains", not "no PII remains".
- **`scrubber_counts` is an accounting of replacements made, not a
  completeness claim.** A count of zero can mean clean input or unmodeled PII.
- **Patterns can't be exhaustive for free-form text.** A name in prose, an
  address without an SSN-shaped number, a secret without a known prefix —
  none of these match anything. Review captured fixtures before committing
  them; they are meant to live in your repo long-term.

## Supported versions

Security fixes land on the latest release only.
