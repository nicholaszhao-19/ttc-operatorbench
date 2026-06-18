"""CLI behavior tests."""

from typer.testing import CliRunner

from ttc_operatorbench.cli import app


def test_cli_version_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_cli_help_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "TTC OperatorBench experimental harness." in result.stdout


def test_cli_without_args_shows_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "TTC OperatorBench experimental harness." in result.stdout
