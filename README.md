# eval-bridge

<img width="1169" height="507" alt="eval-bridge: production-incident regression tests for LLM applications" src="https://github.com/user-attachments/assets/20e18d30-38a7-49e6-ab53-254e030d8bcc" />

> **Production-incident regression tests for LLM applications.**

```
production trace               ->  fixture (PII-scrubbed, deterministic)
       |                                |
       |  eval-bridge capture          |  committed to tests/fixtures/
       v                                v
your LLM app                   eval-bridge run on every PR
       |                                |
       |  failed in prod                |  PASS = bug can never silently return
       +-------------------------------->
```

A customer reports your support agent leaked a credit-card number. You:

```bash
eval-bridge capture incident.json \
    --output-dir tests/fixtures/ \
    --category pii-leak --write
# wrote tests/fixtures/pii-leak-042.json
# .eval-bridge-counter updated: pii-leak -> 42
```

A week later a teammate refactors the prompt and opens a PR. CI runs:

```bash
eval-bridge run tests/fixtures \
    --baseline tests/fixtures/.baseline.json \
    --junit junit.xml
```

If the agent ever leaks that pattern again, CI exits non-zero and a PR comment posts the failure verbatim:

```
pii-leak-042  FAIL  forbidden_substrings
  expected {credit_card} not in reply; got 4111 1111 1111 1111
```

**That is the product.** Everything else — assertions, mutation, judges — is plumbing.

---

## What eval-bridge is NOT

- **Not an LLM evaluation framework.** Promptfoo, DeepEval, Braintrust cover that space; we don't compete there.
- **Not an observability platform.** Use Langfuse, Phoenix, or Helicone for tracing. `eval-bridge import` reads their exports but doesn't replace them.
- **Not a red-team generator.** `--mutate N` catches overfit on the captured wording; pair with a red-team tool for systematic exploits.
- **Not a metrics dashboard.** `eval-bridge run` produces a JUnit XML and a baseline diff. It does not score your app on a 0-100 scale.

If you find yourself wishing eval-bridge did any of those four things, you probably want a different tool. That's a feature: it keeps the surface small enough to actually finish the integration.

---

## The canonical use case

A support agent called `get_user_by_id` with `{user_id: 1234}` when it should have called `refund_user` with `{order_id: 9876}`. Production trace captured:

```json
{
  "trace_id": "trace-bad-tool",
  "prompt": "Customer wants a refund for order 9876.",
  "messages": [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "Please refund order 9876."}
  ],
  "expected_tool_calls": [
    {"name": "refund_user", "arguments": {"order_id": "9876"}}
  ],
  "latency_ms_max": 2000
}
```

```bash
# 1. Capture it as a fixture (PII gets scrubbed at write-time).
eval-bridge capture incident.json \
    --output-dir tests/fixtures/ \
    --category tool-misuse --write
# wrote tests/fixtures/tool-misuse-001.json

# 2. Commit it. That file IS the regression test.

# 3. In CI:
eval-bridge run tests/fixtures \
    --baseline tests/fixtures/.baseline.json \
    --junit junit.xml
```

Future changes to the agent that re-introduce the same buggy tool-call — or that regress the latency budget — get caught before merge.

---

## Quickstart

### 1. Install

```bash
pip install eval-bridge
# or, from a clone:
pip install -e ".[dev]"
```

### 2. Promote a production failure into a fixture

Create `incident.json`:

```json
{
  "trace_id": "ignored-by-cli",
  "prompt": "My email is jane.doe@example.com and my card is 4111 1111 1111 1111",
  "forbidden_substrings": ["@example.com", "4111"]
}
```

```bash
eval-bridge capture incident.json \
    --output-dir tests/fixtures/ \
    --category pii-leak --write
# wrote tests/fixtures/pii-leak-001.json
# .eval-bridge-counter updated: pii-leak -> 1
```

The scrubber replaces `jane.doe@example.com` → `{email}` and the card → `{credit_card}` *before* writing. Set `forbidden_substrings` to assert the reply must not contain those patterns. Every future PR is checked against this exact fixture.

### 3. Run the regression suite in CI

```bash
eval-bridge run tests/fixtures --junit junit.xml
cat junit.xml   # <testsuite tests="1" failures="0"/>
```

With a baseline diff (recommended for any fleet larger than 1 fixture):

```bash
# First run: write the baseline.
eval-bridge run tests/fixtures \
    --write-baseline \
    --baseline tests/fixtures/.baseline.json

# Every subsequent run: compare.
eval-bridge run tests/fixtures \
    --baseline tests/fixtures/.baseline.json \
    --junit junit.xml
```

Exits with code `1` on hard regressions, `3` on silent regressions (output changed but assertions still pass), `0` on clean.

---

## Assertions

Every fixture can declare zero or more assertions. They run after the model call. Multiple assertions per fixture are normal.

| Kind | Field on fixture | What it checks |
|------|------------------|----------------|
| `forbidden_substrings` | `["4111", "@example.com"]` | Reply does not contain any of these substrings. |
| `expected_substrings` | `["Redacted"]` | Reply contains all of these substrings. |
| `json_schema` | JSON Schema object | Reply JSON validates against the schema. |
| `tool_calls` | `expected_tool_calls` | Model called the named tool(s) with matching arguments (deep-equal, key-order independent). |
| `latency_ms` | `latency_ms_max` | Wall-clock latency did not exceed the budget. |
| `semantic_threshold` | `0.0..1.0` + `reference_reply` | Lexical Jaccard similarity between reply and reference. |
| `llm_judge` (optional) | `judge_scorers: ["hallucination", "answer_relevance"]` | Subjective LLM-as-judge scoring. See below. |

The first six are deterministic and free. `llm_judge` is the only one that costs money; treat it as a last resort when the assertion cannot be written as a substring/JSON/latency check.

Example fixture exercising multiple kinds:

```json
{
  "trace_id": "support-bot-001",
  "messages": [
    {"role": "system", "content": "You are a customer-support agent."},
    {"role": "user", "content": "Refund order 9876"}
  ],
  "forbidden_substrings": ["PASSWORD", "API_KEY"],
  "expected_substrings": ["30 days", "refund"],
  "expected_tool_calls": [
    {"name": "refund_user", "arguments": {"order_id": "9876"}}
  ],
  "latency_ms_max": 2000
}
```

---

## Counterfactual mutation

`--mutate N` on `capture` generates N adversarial variants of the captured fixture. Goal: catch **prompt overfitting** — a developer who patches the system prompt to pass the exact captured wording will still fail on a synonym or reordered variant until the underlying principle is fixed.

Four deterministic, dependency-free perturbation families:

- **Entity swap** — names, dates, numbers → `[NAME_1]`, `[DATE_1]`, `[NUM_1]`.
- **Typo & casing** — adjacent-character transposition or dropped letter.
- **Reorder** — shuffles comma/semicolon-separated clauses.
- **Synonym** — small built-in lookup (`summarize` ↔ `condense`, ...).

Variants drop `expected_substrings` (those pinned the exact wording) but keep `forbidden_substrings`, `schema`, `reference_reply`, and `capture` metadata.

```bash
eval-bridge capture incident.json \
    --output-dir tests/fixtures/ \
    --category pii-leak --write --mutate 4
# wrote pii-leak-042.json
# wrote pii-leak-042.m1.json (entity swap)
# wrote pii-leak-042.m2.json (typo)
# wrote pii-leak-042.m3.json (reorder)
# wrote pii-leak-042.m4.json (synonym)
```

Mutation is a "did I overfit?" check, not a security suite. For adversarial coverage use a dedicated red-team tool — `eval-bridge` will replay their outputs against your fixtures.

---

## Capturing from observability platforms

Once `tests/fixtures/` exists, seed it from existing traces:

```bash
# OpenTelemetry / Phoenix / Arize / Traceloop JSONL
eval-bridge import otel traces.jsonl --output tests/fixtures/

# Langfuse export
eval-bridge import langfuse langfuse-export.jsonl --output tests/fixtures/
```

Non-LLM spans and malformed records are skipped (count printed to stderr). Every fixture is scrubbed before disk — no raw trace data ever lands in the repo.

Programmatic:

```python
from eval_bridge import import_otel, import_langfuse, Scrubber
written = import_otel("traces.jsonl", Path("tests/fixtures/"),
                       scrubber=Scrubber(my_config))
```

---

## Optional: LLM-as-judge

Most incident regression tests can be written as deterministic checks. Reach for LLM-as-judge only when the assertion genuinely needs subjective evaluation (faithfulness to retrieved context, toxicity, custom rubrics).

```toml
# eval-bridge.toml
[runner.assertions]
scorer_names = ["faithfulness", "hallucination"]
```

```json
{
  "trace_id": "rag-001",
  "prompt": "What does the doc say about refunds?",
  "context": "...retrieved context...",
  "judge_scorers": ["faithfulness", "hallucination"]
}
```

Built-in scorers:

| Scorer | Measures |
|--------|----------|
| `hallucination` | Reply contains unsourced factual claims. |
| `faithfulness` | Reply is grounded in supplied context (RAG). |
| `answer_relevance` | Reply addresses the user's question. |
| `toxicity` | Reply contains harmful or unsafe content. |
| `bias` | Reply contains demographic stereotyping. |
| `g_eval` | Custom criteria string, per fixture. |

Two judge backends ship:

- **`OfflineJudge`** (CI default) — deterministic. Looks up canned verdicts from a `(scorer_name, trace_id)` map; falls back to 1.0 (pass). CI stays hermetic and free.
- **`LLMJudge`** — calls any OpenAI-compatible chat-completion endpoint. Set `OPENAI_API_KEY` or `EVAL_BRIDGE_JUDGE_API_KEY`.

```bash
eval-bridge run tests/fixtures --judge llm --judge-model gpt-4o-mini
```

### Known limitations of LLM-as-judge

- **Verbosity bias** — judges favour longer replies. Calibrate per use case.
- **Position bias** — option ordering in rubrics sways verdicts. Judge prompt is fixed and short, but not zero.
- **Cost** — each live judge call is an extra LLM request. Use a cheaper model than the one under test; cache verdicts where you can.

---

## Multi-turn message assertions

For multi-turn fixtures, attach per-message rules that run against every message in the chat history:

```json
{
  "trace_id": "support-bot-001",
  "messages": [
    {"role": "system", "content": "You are a customer-support agent."},
    {"role": "user", "content": "What's the refund policy?"},
    {"role": "assistant", "content": "30 days, full refund, no questions asked."}
  ],
  "message_assertions": [
    {"role": "system", "must_contain": ["customer-support"]},
    {"role": "assistant", "must_contain": ["30 days"], "must_not_contain": ["PASSWORD", "API_KEY"]},
    {"role": "any", "must_not_contain": ["BEGIN PRIVATE KEY"]}
  ]
}
```

Each rule emits one assertion per matching message.

---

## Demo

Real output from this repo's own smoke fixtures. Every command below exits 0 and writes the artifacts shown.

```bash
eval-bridge doctor
```

```
scrubber config
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃             ┃               ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ email       │ {email}       │
│ ip          │ {ip}          │
│ jwt         │ {jwt}         │
│ api_token   │ {api_token}   │
│ bearer      │ {api_token}   │
│ credit_card │ {credit_card} │
│ ssn         │ {ssn}         │
└─────────────┴───────────────┘
scrubber sample output:
ping {email} from {ip}, card {credit_card}, jwt {jwt}, ssn {ssn}, token
{api_token}, auth: Bearer {api_token}
counts: {'email': 1, 'ipv4': 1, 'ssn': 1, 'jwt': 1, 'bearer': 2, 'credit_card': 1}
residual secret scan: OK
```

```bash
eval-bridge run tests/fixtures --junit junit.xml --no-exit-on-fail
```

```
                    eval-bridge report
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ trace_id       ┃ status ┃ duration ┃ failing assertion ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ smoke-greeting │ PASS   │   0.000s │                   │
│ smoke-judge    │ PASS   │   0.000s │                   │
│ smoke-pii      │ PASS   │   0.000s │                   │
│ smoke-schema   │ PASS   │   0.000s │                   │
└────────────────┴────────┴──────────┴───────────────────┘
total=4 passed=4 failed=0 duration=0.000s
junit: junit.xml
```

Capturing the canonical PII-leak incident:

```bash
$ echo '{"trace_id":"x","prompt":"Contact jane.doe@example.com, card 4111 1111 1111 1111","forbidden_substrings":["example.com","4111"]}' > /tmp/incident.json
$ eval-bridge capture /tmp/incident.json --output-dir /tmp/fxdemo --category pii-leak --write
wrote /tmp/fxdemo/pii-leak-001.json  (scrubber matches: [('email', 1), ('credit_card', 1)])

$ cat /tmp/fxdemo/pii-leak-001.json
{
  "trace_id": "pii-leak-001",
  "prompt": "Contact {email}, card {credit_card}",
  "forbidden_substrings": ["example.com", "4111"],
  "category": "pii-leak",
  "expected_tool_calls": [],
  "latency_ms_max": null,
  "scrubber_counts": [["email", 1], ["credit_card", 1]]
}

$ cat /tmp/fxdemo/.eval-bridge-counter
{
  "pii-leak": 1
}
```

`jane.doe@example.com` → `{email}`. The card → `{credit_card}`. **No PII on disk.** A second capture in the same directory produces `pii-leak-002.json` and increments the counter to 2.

---

## Configuration

`eval-bridge` reads `eval-bridge.toml` (optional) from the current directory:

```toml
[scrubber]
email       = "{email}"
ip          = "{ip}"
jwt         = "{jwt}"
api_token   = "{api_token}"
credit_card = "{credit_card}"
ssn         = "{ssn}"

# Extra regexes. Each pattern is applied in order.
extra_patterns = [
  { name = "internal_ticket", pattern = "TICKET-\\d{4,}", replacement = "{ticket}" },
]

[runner]
provider    = "fixture"           # "fixture" | "openai_compat"
base_url    = "https://api.openai.com/v1"
model       = "gpt-4o-mini"
timeout_s   = 30
max_retries = 2

[runner.assertions]
schema_path          = ""           # optional JSON Schema for responses
forbidden_substrings = ["BEGIN PRIVATE KEY"]
semantic_threshold   = 0.0          # 0..1 Jaccard baseline
```

---

## CLI reference

```
eval-bridge capture <input>
        --output PATH                    # explicit fixture path
        --output-dir DIR --category CAT  # auto-name + counter
        --write | --dry-run              # default is --dry-run
        --mutate N                       # N counterfactual variants
        --config PATH

eval-bridge run <fixtures_dir>
        --junit PATH
        --provider fixture | openai_compat
        --baseline PATH                  # diff current run against baseline
        --write-baseline                 # save current run as new baseline
        --judge offline | llm            # judge backend (default: offline)
        --judge-model NAME
        --exit-on-fail | --no-exit-on-fail
        --config PATH

eval-bridge import otel   <traces.jsonl>     --output-dir DIR
eval-bridge import langfuse <export.jsonl>   --output-dir DIR

eval-bridge scrub <file>                     --output PATH
eval-bridge doctor                           # show config + scrubber coverage
```

Exit codes:

| Code | Meaning |
|------|---------|
| `0` | All fixtures pass, no regressions vs baseline |
| `1` | One or more assertions failed |
| `2` | Residual PII / secret scan failed (refuses to write) |
| `3` | Baseline diff detected regressions — previously-fixed production failures have returned |

---

## CI / PR-comment action

A composite GitHub Action at `.github/actions/eval-bridge-pr-comment/` runs after your `eval-bridge run` step on PRs and posts a summary:

```yaml
# .github/workflows/ci.yml
- run: eval-bridge run tests/fixtures --baseline tests/fixtures/.baseline.json --junit junit.xml
- uses: ./.github/actions/eval-bridge-pr-comment
  if: github.event_name == 'pull_request'
  with:
    junit-path: junit.xml
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

The action also surfaces baseline-diff events (regressions, silent regressions) directly in the PR comment.

---

## Programmatic SDK

```python
from eval_bridge import Scrubber, Fixture, Runner

# Scrub PII
scrubber = Scrubber.from_default_config()
clean = scrubber.scrub("ping jane@example.com from 10.0.0.1")
# -> "ping {email} from {ip}"

# Run offline
runner = Runner.from_config()
report = runner.run_dir("tests/fixtures")
report.write_junit("junit.xml")

# Stable incident IDs
from eval_bridge import next_incident_id
tid = next_incident_id(Path("tests/fixtures"), "pii-leak")
# -> "pii-leak-001"

# Bring your own scrubber to ingest
from eval_bridge import import_otel
import_otel("traces.jsonl", Path("tests/fixtures/"),
           scrubber=Scrubber(my_config))
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The bundled `tests/fixtures/` is self-contained: PII patterns, capture → run → JUnit pipeline, mutation engine, importers, multi-turn assertions, judge scoring, baseline diff. `pytest -q` runs the suite offline (`OfflineJudge`), no network required.

---

## Limitations

- **Lexical Jaccard, not embeddings.** `semantic_threshold` is a word-token Jaccard against `reference_reply`. Fast and deterministic but insensitive to paraphrase and word-order changes. If you need paraphrase robustness, plug in an embedding similarity at the call site and treat `eval-bridge`'s threshold as a cheap pre-filter.
- **Default PII patterns cover common cases.** Built-ins target email, IPv4/IPv6, JWTs, common API-key prefixes (OpenAI, GitHub, Slack, AWS, Google), Bearer auth headers, Luhn-valid credit cards, and US SSNs. Org-specific identifiers (internal ticket IDs, account numbers, internal hostnames) must be added via `extra_patterns` in `eval-bridge.toml`. Set a category's replacement to `""` to disable that class entirely.
- **OpenAI-compatible provider retries on 5xx only.** Network errors and 5xx responses retry up to `max_retries` with no backoff; tune if your endpoint rate-limits.
- **Deterministic fixtures surface non-determinism as flaky CI.** If your model is non-deterministic, `eval-bridge` shows that as flakes — the fixture doesn't change. Use a baseline diff to spot drifts even when assertions pass.
- **Mutations are local heuristics, not adversarial ML.** Four-family rule set catches overfit on the captured wording but is not a substitute for paraphrasing models or red-team generation. Pair with a red-team tool if you need broad adversarial coverage.
- **Tool-call argument matching is exact (deep-equal after key sort).** If your model returns numbers as strings, configure fixtures to match (`"42"` vs `42`).

---

## License

MIT © Aniket Karne
