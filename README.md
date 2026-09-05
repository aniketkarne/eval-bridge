# eval-bridge

> Capture LLM failures, scrub PII deterministically, and replay them offline as
> reproducible eval fixtures with JUnit XML output.

```
+-----------------------------+
|   Your LLM App / Production |
+-----------------------------+
              |
              |  failures, bad answers, user complaints
              v
+-----------------------------+
|  eval-bridge capture        |
|  - ingest JSON / YAML       |
|  - scrub PII (deterministic)|
|  - assert safety invariants |
+-----------------------------+
              |
              v
+-----------------------------+
|  Fixtures (JSON / YAML)     |
|  tests/fixtures/*.json      |
+-----------------------------+
              |
              v
+-----------------------------+
|  eval-bridge runner         |
|  - offline fixture mode     |
|  - OpenAI-compatible HTTP   |
|  - semantic/lexical asserts |
|  - JUnit XML report         |
+-----------------------------+
```

## Why

- **Capture once, replay forever.** A failed production trace becomes a fixture
  that runs in CI with no live LLM calls.
- **No secrets in version control.** Every capture passes through a deterministic
  PII scrubber (emails, IPs, JWTs, API tokens, credit cards, SSNs, custom regexes)
  before it lands on disk.
- **Pluggable providers.** Use offline fixtures for hermetic CI, or any
  OpenAI-compatible HTTP endpoint for live regression sweeps.
- **CI-friendly.** Emits JUnit XML that GitHub Actions, GitLab, and Jenkins consume
  natively, plus a residual-secret scan that fails the run if anything sensitive
  leaks through.

## 3-step quickstart

### 1. Install

```bash
pip install eval-bridge
# or, from a clone:
pip install -e ".[dev]"
```

### 2. Capture a failure and write a fixture

Create `incident.json`:

```json
{
  "trace_id": "tr-001",
  "prompt": "My email is jane.doe@example.com and my card is 4111 1111 1111 1111",
  "expected_substrings": ["@example.com"],
  "capture": {
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You redact PII."},
      {"role": "user", "content": "My email is jane.doe@example.com and my card is 4111 1111 1111 1111"}
    ]
  }
}
```

```bash
eval-bridge capture incident.json --output tests/fixtures/tr-001.json --dry-run
# dry-run prints what would be written; drop --dry-run to write
```

### 3. Run the eval

```bash
eval-bridge run tests/fixtures --junit junit.xml
cat junit.xml   # <testsuite tests="1" failures="0"/>
```

### 4. (optional) Generate adversarial variants

```bash
eval-bridge capture incident.json --output tests/fixtures/tr-001.json --write --mutate 5
# writes tr-001.json plus tr-001.m1.json .. tr-001.m5.json
# each variant mutates names/dates/numbers, injects typos, reorders clauses,
# or swaps synonyms — catching prompt overfit on the exact captured wording
```

## Demo

Real output from this repo's own smoke fixtures. Every command below exits 0
and writes the artifacts shown.

`eval-bridge doctor` — inspects the active scrubber config and runs a sample
through it to confirm coverage:

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
counts: {'email': 1, 'ipv4': 1, 'ssn': 1, 'jwt': 1, 'bearer': 2, 'credit_card':
1}
residual secret scan: OK
```

`eval-bridge run tests/fixtures --junit junit.xml --no-exit-on-fail` — runs the
three bundled smoke fixtures against the offline provider and emits a JUnit
report:

```
eval-bridge report                    
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ trace_id       ┃ status ┃ duration ┃ failing assertion ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ smoke-greeting │ PASS   │   0.000s │                   │
│ smoke-pii      │ PASS   │   0.000s │                   │
│ smoke-schema   │ PASS   │   0.000s │                   │
└────────────────┴────────┴──────────┴───────────────────┘
total=3 passed=3 failed=0 duration=0.000s
junit: junit.xml
```

First lines of the generated `junit.xml`:

```xml
<?xml version='1.0' encoding='utf-8'?>
<testsuite name="eval-bridge" tests="3" failures="0" errors="0" skipped="0" time="0.000" timestamp="2026-09-05T14:30:40" id="5c44783e-b3db-4e9a-be96-72af1400af95"><testcase classname="eval-bridge.smoke-greeting" name="smoke-greeting" time="0.000"><system-out>Hello!</system-out></testcase><testcase classname="eval-bridge.smoke-pii" name="smoke-pii" time="0.000"><system-out>Redacted: {email}, token {api_token}</system-out></testcase><testcase classname="eval-bridge.smoke-schema" name="smoke-schema" time="0.000"><system-out>{"answer": 42}</system-out></testcase></testsuite>
```

`eval-bridge capture` — feeds a realistic incident (email + IPv4 + Visa + JWT)
through the scrubber and writes a fixture. Note that `jane.doe@example.com`,
`10.0.0.1`, `4111 1111 1111 1111`, and the JWT are all replaced with their
respective `{placeholder}` tokens:

```bash
$ eval-bridge capture /tmp/incident.json --output /tmp/demo-pii.json --write
wrote /tmp/demo-pii.json  (scrubber matches: [('email', 1), ('ipv4', 1), ('jwt', 1), ('credit_card', 1)])
```

The scrubbed fixture on disk:

```json
{
  "trace_id": "demo-pii",
  "prompt": "Contact {email} from {ip}, card {credit_card}, JWT {jwt}",
  "model": "gpt-4o-mini",
  "fixture_response": null,
  "scrubber_counts": [["email", 1], ["ipv4", 1], ["jwt", 1], ["credit_card", 1]]
}
```

## Configuration

`eval-bridge` reads `eval-bridge.toml` (optional) from the current directory:

```toml
[scrubber]
# Replacement tokens for each detected class. Use {placeholder} tokens when you
# want a stable, non-reversible label; any literal string is also accepted.
email       = "{email}"
ip          = "{ip}"
jwt         = "{jwt}"
api_token   = "{api_token}"
credit_card = "{credit_card}"
ssn         = "{ssn}"

# Extra regexes you want scrubbed. Each pattern is applied in order.
extra_patterns = [
  { name = "internal_ticket", pattern = "TICKET-\\d{4,}", replacement = "{ticket}" },
]

[runner]
provider   = "fixture"            # "fixture" | "openai_compat"
base_url   = "https://api.openai.com/v1"
model      = "gpt-4o-mini"
timeout_s  = 30
max_retries = 2

[runner.assertions]
schema_path         = ""           # optional JSON Schema for responses
forbidden_substrings = ["BEGIN PRIVATE KEY"]
semantic_threshold  = 0.0          # 0..1 Jaccard baseline
```

## CLI

```
eval-bridge capture <input>      [--output PATH] [--dry-run] [--mutate N]
eval-bridge run <fixtures_dir>   [--junit PATH] [--provider fixture|openai_compat]
eval-bridge scrub <input>        [--output PATH]    # scrub without writing fixture
eval-bridge doctor               # show config + detect scrubber coverage
```

## Counterfactual mutation

Passing `--mutate N` to `capture` generates N adversarial variants of the
captured fixture. The goal is to catch **prompt overfitting**: a developer
who patches the system prompt to pass the exact captured wording will still
fail on a synonym or reordered variant until the underlying principle is
fixed.

Four deterministic, dependency-free perturbation families (round-robin across
the N slots, all reproducible via `--seed` on the Python API):

- **Entity swap** — names, dates, and numbers become `[NAME_1]`, `[DATE_1]`,
  `[NUM_1]` placeholders. Preserves sentence structure.
- **Typo & casing** — adjacent-character transposition or dropped letter.
  Models mobile-keyboard errors. Protected against mutating short common
  words (`the`, `and`, `you`, ...).
- **Reorder** — shuffles comma/semicolon-separated clauses. Preserves the
  opening clause. For multi-turn fixtures, also shuffles non-system
  messages.
- **Synonym** — swaps a small built-in lookup (`summarize` ↔ `condense`,
  `explain` ↔ `describe`, etc.). Tiny table on purpose: bigger tables drift
  out of date and noisier mutations hurt more than they help.

Variants drop the original `expected_substrings` (those pinned the exact
wording) but keep `forbidden_substrings`, `schema`, `reference_reply`, and
`capture` metadata.

Real output from `eval-bridge capture incident.json --write --mutate 4`:

```
wrote base.json       (scrubber matches: [])
wrote base.m1.json    (variant demo-mut.m1)
wrote base.m2.json    (variant demo-mut.m2)
wrote base.m3.json    (variant demo-mut.m3)
wrote base.m4.json    (variant demo-mut.m4)
```

Inspecting the four variants on a single input
(`Summarize the Q3 sales report for Jane Doe filed on 2025-09-12.`):

| Variant   | Family   | Prompt                                                              |
|-----------|----------|---------------------------------------------------------------------|
| `base.m1` | entity   | `[NAME_1] the Q3 sales report for [NAME_2] filed on [DATE_1].`       |
| `base.m2` | typo     | `Summarize the Q3 sales report for Jae Doe filed on 2025-09-12.`    |
| `base.m3` | reorder  | `Summarize the Q3 sales report for Jane Doe filed on 2025-09-12.`   |
| `base.m4` | synonym  | `Condense the Q3 sales report for Jane Doe filed on 2025-09-12.`    |

Programmatic API:

```python
from eval_bridge import Fixture, mutate_fixture

fx = Fixture(trace_id="t-1", prompt="Summarize Q3 for Jane Doe.")
variants = mutate_fixture(fx, n=8, seed=42)  # reproducible
# or restrict to one family:
variants = mutate_fixture(fx, n=4, families=("entity", "synonym"))
```

## LLM-as-judge scoring

Six built-in scorers evaluate a reply against a rubric via an LLM judge:

| Scorer             | What it measures                                  |
|--------------------|---------------------------------------------------|
| `hallucination`    | Reply contains unsourced factual claims.          |
| `faithfulness`     | Reply is grounded in the supplied context (RAG).  |
| `answer_relevance` | Reply addresses the user's actual question.        |
| `toxicity`         | Reply contains harmful or unsafe content.         |
| `bias`             | Reply contains demographic stereotyping or unfair generalisation. |
| `g_eval`           | Custom criteria string supplied per fixture.      |

Each scorer returns a score 0.0–1.0 and passes when the score meets its
threshold (default 0.7). A score of 0.0 with reason `not_applicable` is
treated as N/A and passes silently — this lets scorers abstain when their
rubric is irrelevant (e.g. faithfulness when no context is supplied).

Two judge backends ship:

- **`OfflineJudge`** (CI default) — deterministic. Looks up canned verdicts
  from a `(scorer_name, trace_id)` map; falls back to 1.0 (pass) when no
  verdict is registered. CI builds stay hermetic and free.
- **`LLMJudge`** — calls any OpenAI-compatible chat-completion endpoint
  with the judge model. Set `OPENAI_API_KEY` or `EVAL_BRIDGE_JUDGE_API_KEY`.

```bash
# CI mode (default — offline judge, free)
eval-bridge run tests/fixtures --junit junit.xml

# Live mode (calls the judge model against your endpoint)
eval-bridge run tests/fixtures --junit junit.xml --judge llm --judge-model gpt-4o-mini
```

Enable scorers globally via `eval-bridge.toml`:

```toml
[runner]
judge_model = "gpt-4o-mini"   # usually cheaper than the model under test

[runner.assertions]
scorer_names = ["hallucination", "answer_relevance", "toxicity"]
```

Or per-fixture:

```json
{
  "trace_id": "rag-001",
  "prompt": "What does the doc say about refunds?",
  "context": "...retrieved context...",
  "judge_scorers": ["faithfulness", "hallucination"],
  "fixture_response": "Refunds are processed within 30 days."
}
```

For G-Eval (custom criteria), include the rubric in the fixture:

```json
{
  "trace_id": "custom-001",
  "prompt": "Translate to French",
  "fixture_response": "Bonjour le monde",
  "judge_criteria": "The reply is in French and ends with a period.",
  "judge_scorers": ["g_eval"]
}
```

Programmatic API:

```python
from eval_bridge import Runner, RunnerConfig, Fixture
from eval_bridge.scoring import OfflineJudge, LLMJudge

# CI: hermetic, deterministic.
runner = Runner(judge=OfflineJudge())

# Dev: live LLM judge against an OpenAI-compatible endpoint.
judge = LLMJudge(base_url="https://api.openai.com/v1", model="gpt-4o-mini")
runner = Runner(judge=judge)
judge.close()

# Bring your own scorer.
from eval_bridge.scoring import Scorer
my_scorer = Scorer(name="brand_voice", rubric="...", threshold=0.8)
runner = Runner(judge=OfflineJudge(), scorer_set=[my_scorer])
```

### Known limitations of LLM-as-judge

- **Verbosity bias** — judges tend to favour longer replies. Calibrate
  thresholds per use case, not across the board.
- **Position bias** — order of options in the rubric can sway verdicts. The
  judge prompt is fixed and short to minimise this, but it is not zero.
- **Cost** — each live judge call is one extra LLM request. Use a cheaper
  model than the one under test; cache verdicts where appropriate.

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
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Limitations

- **Lexical Jaccard, not embeddings.** The semantic-threshold assertion is a
  word-token Jaccard similarity against `reference_reply`. It is fast and
  deterministic, but insensitive to paraphrase and word-order changes. If you
  need paraphrase robustness, swap in an embedding similarity at the call site
  and treat eval-bridge's threshold as a cheap pre-filter.
- **Default PII patterns cover common cases.** Built-ins target email, IPv4/IPv6,
  JWTs, common API-key prefixes (OpenAI, GitHub, Slack, AWS, Google), Bearer
  auth headers, Luhn-valid credit cards, and US SSNs. Org-specific identifiers
  (internal ticket IDs, account numbers, internal hostnames) must be added via
  `extra_patterns` in `eval-bridge.toml`. Set a category's replacement to `""`
  in TOML to disable that class entirely.
- **OpenAI-compatible provider retries on 5xx only.** 4xx errors fail fast —
  there is no point retrying client errors. Network errors and 5xx responses
  are retried up to `max_retries` times with no backoff; tune `max_retries`
  in config if your endpoint rate-limits aggressively.
- **Deterministic fixtures capture deterministic behavior.** The whole point of
  eval-bridge is replayability — fixtures are scrubbed and pinned. If your
  model is non-deterministic, the harness surfaces that as flaky CI; the
  fixture itself does not change.
- **Mutations are local heuristics, not adversarial ML.** The mutator uses a
  four-family rule set (entity swap, typo, reorder, synonym). It catches
  overfit on the specific captured wording but is not a substitute for
  paraphrasing models or red-team generation. Pair with an LLM-driven
  adversarial generator if you need coverage of less obvious rewordings.

## License

MIT © Aniket Karne
