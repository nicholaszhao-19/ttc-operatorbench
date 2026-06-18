"""Command-line interface for TTC OperatorBench."""

import typer

from ttc_operatorbench import __version__

app = typer.Typer(
    add_completion=False,
    help="TTC OperatorBench experimental harness.",
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show the package version."),
) -> None:
    """Run TTC OperatorBench commands."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
