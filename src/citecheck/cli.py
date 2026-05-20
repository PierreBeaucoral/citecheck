"""Typer CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from citecheck import __version__, budget, metrics
from citecheck.extraction.grobid_client import is_alive
from citecheck.models import (
    CheckedReference,
    HallucinationVerdict,
    Reference,
    ResolutionStatus,
    RetractionStatus,
)
from citecheck.pipeline.check import run as run_check
from citecheck.pipeline.extract import run as run_extract

# Load .env from the current working directory if present. Idempotent; safe in tests.
load_dotenv()

app = typer.Typer(
    name="citecheck",
    help="Citation integrity checker for academic papers.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Forces command-group behavior so subcommands work even when only one is registered."""


@app.command()
def version() -> None:
    """Print the installed citecheck version."""
    typer.echo(__version__)


@app.command()
def stats() -> None:
    """Show OpenAlex / Crossref API consumption from the local metrics DB."""
    rows = metrics.summary(7)
    if not rows:
        typer.echo("No API calls recorded yet.")
        return
    table = Table(title="API consumption (rolling 7 days)")
    table.add_column("Source")
    table.add_column("Endpoint")
    table.add_column("Today", justify="right")
    table.add_column("7-day total", justify="right")
    table.add_column("Cache hit %", justify="right")
    for r in rows:
        table.add_row(
            r["source"],
            r["endpoint"],
            str(r["today"]),
            str(r["window"]),
            f"{r['cache_hit_pct']:.0%}",
        )
    console.print(table)

    until = budget.exhausted_until()
    if until is not None:
        console.print(
            f"\n[yellow]OpenAlex circuit is OPEN.[/yellow] "
            f"Budget exhausted until [bold]{until.isoformat()}[/bold]. "
            "Tool is running in Crossref-only fallback."
        )


_STATUS_STYLES = {
    ResolutionStatus.RESOLVED: "green",
    ResolutionStatus.AMBIGUOUS: "yellow",
    ResolutionStatus.UNRESOLVED: "dim",
    ResolutionStatus.ERROR: "red",
}


def _truncate(text: str | None, width: int) -> str:
    if not text:
        return ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _render_table(refs: list[Reference]) -> Table:
    table = Table(title="Extracted references", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", overflow="fold", max_width=60)
    table.add_column("Year", justify="right")
    table.add_column("Status")
    table.add_column("DOI", overflow="fold", max_width=40)
    table.add_column("Score", justify="right")

    for i, ref in enumerate(refs, 1):
        style = _STATUS_STYLES[ref.status]
        title = ref.raw.title or ref.raw.raw_text
        year = str(ref.raw.year) if ref.raw.year else "—"
        doi = ref.resolved_doi or ref.raw.doi or ""
        score = f"{ref.resolution_score:.2f}" if ref.resolution_score is not None else "—"
        table.add_row(
            str(i),
            _truncate(title, 60),
            year,
            f"[{style}]{ref.status.value}[/{style}]",
            doi,
            score,
        )
    return table


def _render_summary(refs: list[Reference]) -> str:
    total = len(refs)
    counts = {s: 0 for s in ResolutionStatus}
    for r in refs:
        counts[r.status] += 1
    pct = lambda n: f"{n}/{total} ({n / total:.0%})" if total else "0"  # noqa: E731
    return (
        f"[bold]Total:[/bold] {total}  "
        f"[green]resolved:[/green] {pct(counts[ResolutionStatus.RESOLVED])}  "
        f"[yellow]ambiguous:[/yellow] {pct(counts[ResolutionStatus.AMBIGUOUS])}  "
        f"[dim]unresolved:[/dim] {pct(counts[ResolutionStatus.UNRESOLVED])}  "
        f"[red]errors:[/red] {pct(counts[ResolutionStatus.ERROR])}"
    )


_RETRACTION_STYLES = {
    RetractionStatus.RETRACTED: "bold red",
    RetractionStatus.EXPRESSION_OF_CONCERN: "bold yellow",
    RetractionStatus.CORRECTION: "yellow",
    RetractionStatus.CLEAN: "green",
    RetractionStatus.UNCHECKED: "dim",
    RetractionStatus.ERROR: "red",
}

_HALLUCINATION_STYLES = {
    HallucinationVerdict.LIKELY_HALLUCINATED: "bold red",
    HallucinationVerdict.SUSPICIOUS: "yellow",
    HallucinationVerdict.REAL_LOW_CONFIDENCE: "cyan",
    HallucinationVerdict.REAL_HIGH_CONFIDENCE: "green",
    HallucinationVerdict.UNCHECKED: "dim",
}

# Compact labels keep the table readable when both Retraction and Hallucination
# columns are shown.
_HALL_LABELS = {
    HallucinationVerdict.LIKELY_HALLUCINATED: "hallucinated",
    HallucinationVerdict.SUSPICIOUS: "suspicious",
    HallucinationVerdict.REAL_LOW_CONFIDENCE: "real(low)",
    HallucinationVerdict.REAL_HIGH_CONFIDENCE: "real(high)",
    HallucinationVerdict.UNCHECKED: "unchecked",
}


def _render_check_table(checked: list[CheckedReference]) -> Table:
    table = Table(title="Reference check report", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", overflow="fold", max_width=48)
    table.add_column("Year", justify="right")
    table.add_column("Resolution")
    table.add_column("Retraction")
    table.add_column("Hallucination")
    table.add_column("DOI", overflow="fold", max_width=32)

    for i, item in enumerate(checked, 1):
        ref = item.reference
        retr = item.retraction
        hall = item.hallucination
        res_style = _STATUS_STYLES[ref.status]
        retr_style = _RETRACTION_STYLES[retr.status]
        hall_style = _HALLUCINATION_STYLES[hall.verdict]
        title = ref.raw.title or ref.raw.raw_text
        year = str(ref.raw.year) if ref.raw.year else "—"
        doi = ref.resolved_doi or ref.raw.doi or ""
        table.add_row(
            str(i),
            _truncate(title, 48),
            year,
            f"[{res_style}]{ref.status.value}[/{res_style}]",
            f"[{retr_style}]{retr.status.value}[/{retr_style}]",
            f"[{hall_style}]{_HALL_LABELS[hall.verdict]}[/{hall_style}]",
            doi,
        )
    return table


def _render_check_summary(checked: list[CheckedReference]) -> str:
    total = len(checked)
    if not total:
        return "[dim]No references found.[/dim]"
    retr_counts = dict.fromkeys(RetractionStatus, 0)
    hall_counts = dict.fromkeys(HallucinationVerdict, 0)
    for c in checked:
        retr_counts[c.retraction.status] += 1
        hall_counts[c.hallucination.verdict] += 1
    pct = lambda n: f"{n}/{total} ({n / total:.0%})"  # noqa: E731
    return (
        "[bold]Retraction:[/bold]  "
        f"[green]clean:[/green] {pct(retr_counts[RetractionStatus.CLEAN])}  "
        f"[bold red]retracted:[/bold red] {pct(retr_counts[RetractionStatus.RETRACTED])}  "
        f"[bold yellow]EOC:[/bold yellow] {pct(retr_counts[RetractionStatus.EXPRESSION_OF_CONCERN])}  "
        f"[yellow]correction:[/yellow] {pct(retr_counts[RetractionStatus.CORRECTION])}  "
        f"[dim]unchecked:[/dim] {pct(retr_counts[RetractionStatus.UNCHECKED])}  "
        f"[red]err:[/red] {pct(retr_counts[RetractionStatus.ERROR])}\n"
        "[bold]Hallucination:[/bold]  "
        f"[green]real(high):[/green] {pct(hall_counts[HallucinationVerdict.REAL_HIGH_CONFIDENCE])}  "
        f"[cyan]real(low):[/cyan] {pct(hall_counts[HallucinationVerdict.REAL_LOW_CONFIDENCE])}  "
        f"[yellow]suspicious:[/yellow] {pct(hall_counts[HallucinationVerdict.SUSPICIOUS])}  "
        f"[bold red]hallucinated:[/bold red] {pct(hall_counts[HallucinationVerdict.LIKELY_HALLUCINATED])}  "
        f"[dim]unchecked:[/dim] {pct(hall_counts[HallucinationVerdict.UNCHECKED])}"
    )


@app.command()
def check(
    pdf_path: Annotated[Path, typer.Argument(help="Path to the PDF to scan.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON to stdout instead of a table.")
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Bypass the on-disk cache (~/.citecheck/cache.db). Forces fresh API calls.",
        ),
    ] = False,
    only_hallucination: Annotated[
        bool,
        typer.Option(
            "--only-hallucination",
            help="Skip the retraction check; run only the hallucination detector.",
        ),
    ] = False,
    skip_hallucination: Annotated[
        bool,
        typer.Option(
            "--skip-hallucination",
            help="Skip the hallucination detector (faster — no OpenAlex author/venue calls).",
        ),
    ] = False,
    skip_liveness: Annotated[
        bool, typer.Option("--skip-liveness", help="Skip the GROBID liveness probe.")
    ] = False,
) -> None:
    """Extract references, resolve them, and run retraction + hallucination checks."""
    if only_hallucination and skip_hallucination:
        typer.secho(
            "--only-hallucination and --skip-hallucination are mutually exclusive.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    if not pdf_path.is_file():
        typer.secho(f"File not found: {pdf_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if not skip_liveness and not is_alive():
        typer.secho(
            "GROBID is not reachable at http://localhost:8070. "
            "Start it with: docker compose up -d grobid",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    checked = run_check(
        pdf_path,
        use_cache=not no_cache,
        skip_retraction=only_hallucination,
        skip_hallucination=skip_hallucination,
    )

    # Warn once if OpenAlex was unavailable for any check. Streaming this banner
    # before the report keeps it visible even in long outputs.
    if any(
        "openalex" in n.lower() and "exhausted" in n.lower()
        for c in checked
        for n in (c.retraction.notes + c.hallucination.caveats + [s.reasoning for s in c.hallucination.signals])
    ):
        until = budget.exhausted_until()
        when = until.isoformat() if until else "later today"
        typer.secho(
            f"⚠  OpenAlex daily budget exhausted; ran in Crossref-only fallback "
            f"(retraction is_retracted check and hallucination L3/L4/L5 layers "
            f"degraded). Resumes at {when}.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    if as_json:
        sys.stdout.write(
            json.dumps(
                [c.model_dump(mode="json") for c in checked],
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.stdout.write("\n")
        return

    console.print(_render_check_table(checked))
    console.print()
    console.print(_render_check_summary(checked))


@app.command()
def extract(
    pdf_path: Annotated[Path, typer.Argument(help="Path to the PDF to scan.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON to stdout instead of a table.")
    ] = False,
    skip_liveness: Annotated[
        bool,
        typer.Option(
            "--skip-liveness", help="Skip the GROBID liveness probe (useful in CI/tests)."
        ),
    ] = False,
) -> None:
    """Extract references from a PDF and resolve them against Crossref."""
    if not pdf_path.is_file():
        # typer.secho respects click's stdout/stderr capture; rich.Console(stderr=True)
        # writes directly to sys.stderr and bypasses test runners.
        typer.secho(f"File not found: {pdf_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if not skip_liveness and not is_alive():
        typer.secho(
            "GROBID is not reachable at http://localhost:8070. "
            "Start it with: docker compose up -d grobid",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    refs = run_extract(pdf_path)

    if as_json:
        sys.stdout.write(
            json.dumps([r.model_dump(mode="json") for r in refs], ensure_ascii=False, indent=2)
        )
        sys.stdout.write("\n")
        return

    console.print(_render_table(refs))
    console.print()
    console.print(_render_summary(refs))


if __name__ == "__main__":
    app()
