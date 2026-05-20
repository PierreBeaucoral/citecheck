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

from citecheck import __version__
from citecheck.extraction.grobid_client import is_alive
from citecheck.models import Reference, ResolutionStatus
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
