"""Command-line interface for eval-bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from . import __version__
from .baseline import BASELINE_FILE, diff_baseline, load_baseline, write_baseline
from .config import load_config
from .fixture import Fixture, load_fixture, load_fixture_dir
from .incident import next_incident_id
from .scoring import (
    LLMJudge,
    OfflineJudge,
    default_scorer_set,
)
from .mutator import mutate_fixture
from .runner import Runner
from .scrubber import Scrubber, find_residual_secrets
from .importer import import_langfuse, import_otel

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_any(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(raw)
    else:
        loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise click.ClickException(f"{path}: expected a JSON object at top level")
    return loaded


def _write_any(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="eval-bridge")
def main() -> None:
    """Capture LLM failures, deterministic PII scrubbing, offline eval runner."""


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Where to write the fixture (JSON or YAML). "
                   "Mutually exclusive with --output-dir.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Directory to write the fixture into. When combined "
                   "with --category, the filename is auto-generated as "
                   "<category>-NNN.json and the counter is updated.")
@click.option("--category", default=None,
              help="Incident category for stable ID generation (e.g. pii-leak, "
                   "tool-misuse, hallucination). Requires --output-dir.")
@click.option("--dry-run/--write", default=True,
              help="Dry-run prints the scrubbed fixture to stdout; --write saves it.")
@click.option("--mutate", "mutate_n", type=int, default=0,
              help="Generate N adversarial variants.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
def capture(
    input_path: Path,
    output: Path | None,
    output_dir: Path | None,
    category: str | None,
    dry_run: bool,
    mutate_n: int,
    config: Path | None,
) -> None:
    """Capture an incident and emit a scrubbed eval fixture."""
    if output is not None and output_dir is not None:
        raise click.UsageError("use either --output or --output-dir, not both")
    if category is not None and output_dir is None:
        raise click.UsageError("--category requires --output-dir")
    if output_dir is not None and category is None:
        raise click.UsageError("--output-dir requires --category")

    cfg = load_config(config) if config else load_config()
    scrubber = Scrubber(cfg.scrubber)
    data = _read_any(input_path)

    scrubbed = scrubber.scrub_mapping(data)
    fixture = Fixture.from_dict(scrubbed)
    fixture.validate()

    if output_dir is not None and category is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        new_id = next_incident_id(output_dir, category)
        fixture.trace_id = new_id
        fixture.category = category
        out = output_dir / f"{new_id}.json"
    elif output is not None:
        out = output
    else:
        out = Path("tests/fixtures") / f"{fixture.trace_id}.json"

    payload = fixture.to_dict()
    payload["scrubber_counts"] = list(scrubber.scrub(json.dumps(data)).counts.items())

    if dry_run:
        click.echo(f"# dry-run: would write scrubbed fixture to {out}")
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        if mutate_n > 0:
            for v in mutate_fixture(fixture, n=mutate_n, seed=0):
                vp = v.to_dict()
                vp["scrubber_counts"] = list(
                    scrubber.scrub(json.dumps(v.to_dict())).counts.items()
                )
                click.echo()
                click.echo(f"# variant {v.trace_id}:")
                click.echo(json.dumps(vp, indent=2, ensure_ascii=False))
        return

    _write_any(out, payload)
    click.echo(f"wrote {out}  (scrubber matches: {payload['scrubber_counts']})")

    if mutate_n > 0:
        for v in mutate_fixture(fixture, n=mutate_n, seed=0):
            vp = v.to_dict()
            vp["scrubber_counts"] = list(
                scrubber.scrub(json.dumps(v.to_dict())).counts.items()
            )
            v_out = out.with_name(f"{out.stem}.{v.trace_id.split('.')[-1]}{out.suffix}")
            _write_any(v_out, vp)
            click.echo(f"wrote {v_out}  (variant {v.trace_id})")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.argument("fixtures_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--junit", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write a JUnit XML report to this path.")
@click.option("--provider", type=click.Choice(["fixture", "openai_compat"]),
              default=None, help="Override provider from config.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
@click.option("--exit-on-fail/--no-exit-on-fail", default=True,
              help="Exit with non-zero status if any case fails.")
@click.option("--judge", type=click.Choice(["offline", "llm"]), default="offline",
              help="LLM-as-judge mode. 'offline' (default) is deterministic and "
                   "uses canned verdicts keyed by (scorer, trace_id); 'llm' calls "
                   "the judge_model against an OpenAI-compatible endpoint.")
@click.option("--judge-model", default=None,
              help="Override the judge model (default: from runner.judge_model).")
@click.option("--baseline", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Compare against this baseline file. Reports silent regressions, "
                   "fixes, and removals; prints them but does not change exit code "
                   "for silent_regression events. Exit code 3 is added on top of "
                   "failure for 'regressed' or 'removed' events.")
@click.option("--write-baseline/--no-write-baseline", "write_baseline_flag", default=False,
              help="Write the current run's snapshot as the new baseline. If "
                   "--baseline is provided, writes to that path; otherwise writes "
                   "to <fixtures_dir>/.baseline.json.")
def run(
    fixtures_dir: Path,
    junit: Path | None,
    provider: str | None,
    config: Path | None,
    exit_on_fail: bool,
    judge: str,
    judge_model: str | None,
    baseline: Path | None,
    write_baseline_flag: bool,
) -> None:
    """Run fixtures and emit a JUnit XML report."""
    cfg = load_config(config) if config else load_config()
    if provider is not None:
        cfg.runner.provider = provider
    if judge_model is not None:
        cfg.runner.judge_model = judge_model

    judge_obj: OfflineJudge | LLMJudge
    if judge == "llm":
        judge_obj = LLMJudge(
            base_url=cfg.runner.base_url,
            model=cfg.runner.judge_model,
            api_key=__import__("os").environ.get("EVAL_BRIDGE_JUDGE_API_KEY")
                   or __import__("os").environ.get("OPENAI_API_KEY"),
        )
    else:
        judge_obj = OfflineJudge()

    runner = Runner(
        config=cfg.runner,
        scrubber=Scrubber(cfg.scrubber),
        judge=judge_obj,
    )
    report = runner.run_dir(fixtures_dir)
    if judge == "llm":
        judge_obj.close()  # type: ignore[attr-defined]

    # Build a snapshot from the current run for baseline diffing.
    snapshot: dict[str, dict] = {
        r.trace_id: {
            "trace_id": r.trace_id,
            "passed": r.passed,
            "actual_output": r.response_text,
        }
        for r in report.results
    }

    prior = load_baseline(baseline) if baseline is not None else None
    if prior is not None:
        diffs = diff_baseline(prior, snapshot)
        if diffs:
            dtable = Table(title="baseline diff")
            dtable.add_column("trace_id", style="bold")
            dtable.add_column("kind")
            for d in diffs:
                dtable.add_row(d["trace_id"], d["kind"])
            console.print(dtable)
            # Only hard regressions/removals flip exit code; silent_regression
            # is reported loudly in stdout but does not fail the build.
            if any(d["kind"] in ("regressed", "removed") for d in diffs):
                # Defer to the exit-on-fail block below if --no-exit-on-fail;
                # otherwise propagate exit code 3.
                if exit_on_fail:
                    sys.exit(3)

    if write_baseline_flag:
        target = baseline if baseline is not None else (fixtures_dir / BASELINE_FILE)
        write_baseline(target, snapshot)
        console.print(f"baseline written: {target}")

    # Pretty stdout summary
    table = Table(title="eval-bridge report")
    table.add_column("trace_id", style="bold")
    table.add_column("status")
    table.add_column("duration", justify="right")
    table.add_column("failing assertion")
    for r in report.results:
        failing = next((a.name for a in r.assertions if not a.passed), "")
        table.add_row(
            r.trace_id,
            "PASS" if r.passed else "FAIL",
            f"{r.duration_s:.3f}s",
            failing,
        )
    console.print(table)
    console.print(
        f"[bold]total={report.total} passed={report.passed} "
        f"failed={report.failed} duration={report.duration_s:.3f}s[/bold]"
    )

    if junit is not None:
        path = report.write_junit(junit)
        console.print(f"junit: {path}")

    if exit_on_fail and report.failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# scrub
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Write scrubbed content here (defaults to stdout).")
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
@click.option("--residual-scan/--no-residual-scan", default=True,
              help="After scrubbing, run a residual-secret scan and fail if anything leaks.")
def scrub(
    input_path: Path,
    output: Path | None,
    config: Path | None,
    residual_scan: bool,
) -> None:
    """Scrub PII from a file (YAML/JSON/Text) without turning it into a fixture."""
    cfg = load_config(config) if config else load_config()
    scrubber = Scrubber(cfg.scrubber)
    raw = input_path.read_text(encoding="utf-8")
    result = scrubber.scrub(raw)

    if output is None:
        click.echo(result.text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.text, encoding="utf-8")
        click.echo(f"wrote {output}  (matches: {dict(result.counts)})")

    if residual_scan:
        leaks = find_residual_secrets(result.text)
        if leaks:
            err_console.print(
                f"[red]residual secret scan failed: {len(leaks)} leak(s)[/red]"
            )
            for leak in leaks:
                err_console.print(f"  {leak['name']} -> {leak['value']!r}")
            sys.exit(2)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
def doctor(config: Path | None) -> None:
    """Show config + detect scrubber coverage on a synthetic sample."""
    cfg = load_config(config) if config else load_config()

    table = Table(title="scrubber config")
    for field_name in ("email", "ip", "jwt", "api_token", "bearer", "credit_card", "ssn"):
        table.add_row(field_name, getattr(cfg.scrubber, field_name))
    console.print(table)

    scrubber = Scrubber(cfg.scrubber)
    sample = (
        "ping jane.doe@example.com from 10.0.0.1, "
        "card 4111 1111 1111 1111, "
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature, "
        "ssn 123-45-6789, "
        "token ghp_abcdef0123456789abcdef, "
        "auth: Bearer abcdefghijklmnopqrstuvwxyz012345"
    )
    res = scrubber.scrub(sample)
    console.print("[bold]scrubber sample output:[/bold]")
    console.print(res.text)
    console.print(f"[bold]counts:[/bold] {dict(res.counts)}")

    leaks = find_residual_secrets(res.text)
    if leaks:
        err_console.print(f"[red]residual leaks: {leaks}[/red]")
        sys.exit(2)
    console.print("[green]residual secret scan: OK[/green]")


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

@main.group(name="import")
def import_() -> None:
    """Import production traces from observability platforms."""


@import_.command(name="otel")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(file_okay=False, path_type=Path),
              required=True, help="Directory to write scrubbed fixtures into.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
def import_otel_cmd(input_path: Path, output: Path, config: Path | None) -> None:
    """Import OpenTelemetry / Phoenix / Arize JSONL traces as fixtures.

    Reads one JSON object per line. Skips non-LLM spans and prints the
    skip count to stderr. Each imported trace becomes one fixture, scrubbed
    before write so PII never lands in the repo.
    """
    cfg = load_config(config) if config else load_config()
    scrubber = Scrubber(cfg.scrubber)
    written = import_otel(input_path, output, scrubber=scrubber)
    console.print(f"[bold]imported {len(written)} fixtures -> {output}[/bold]")


@import_.command(name="langfuse")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(file_okay=False, path_type=Path),
              required=True, help="Directory to write scrubbed fixtures into.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Path to eval-bridge.toml.")
def import_langfuse_cmd(input_path: Path, output: Path, config: Path | None) -> None:
    """Import a Langfuse JSONL export as fixtures."""
    cfg = load_config(config) if config else load_config()
    scrubber = Scrubber(cfg.scrubber)
    written = import_langfuse(input_path, output, scrubber=scrubber)
    console.print(f"[bold]imported {len(written)} fixtures -> {output}[/bold]")


if __name__ == "__main__":  # pragma: no cover
    main()
