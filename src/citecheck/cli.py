"""Typer CLI entry point. Commands are registered as phases come online."""

import typer

app = typer.Typer(
    name="citecheck",
    help="Citation integrity checker for academic papers.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Forces command-group behavior so subcommands work even when only one is registered."""


@app.command()
def version() -> None:
    """Print the installed citecheck version."""
    from citecheck import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
