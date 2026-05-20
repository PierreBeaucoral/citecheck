"""Smoke tests — verify the package imports and the CLI is wired."""

from typer.testing import CliRunner

import citecheck
from citecheck.cli import app


def test_package_imports() -> None:
    assert citecheck.__version__


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert citecheck.__version__ in result.stdout
