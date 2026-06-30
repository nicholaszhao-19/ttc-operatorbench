"""Command-line interface for TTC OperatorBench."""

from pathlib import Path
from typing import Annotated

import typer

from ttc_operatorbench import __version__
from ttc_operatorbench.evals.experiment import (
    ExperimentArtifacts,
    load_experiment_config,
    run_experiment,
)
from ttc_operatorbench.evals.portfolio import (
    PortfolioRun,
    load_portfolio_runs,
    write_portfolio_report,
)

DEFAULT_PORTFOLIO_RUN_IDS = (
    "hf_qwen3_06b_smoke",
    "hf_qwen25_coder_05b_curated",
    "hf_qwen25_coder_15b_probe",
)
DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/experiments/toy_protocol.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")
DEFAULT_REPORT_ROOT = Path("reports/runs")
DEFAULT_PORTFOLIO_OUTPUT_PATH = Path("reports/portfolio_report.md")
DEFAULT_PORTFOLIO_RUNS_OPTION = ",".join(DEFAULT_PORTFOLIO_RUN_IDS)

app = typer.Typer(
    add_completion=False,
    help="TTC OperatorBench experimental harness.",
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the package version."),
    ] = False,
) -> None:
    """Run TTC OperatorBench commands."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def run_experiment_from_options(
    *,
    config_path: Path,
    run_id: str | None,
    output_root: Path | None,
    report_root: Path | None,
) -> ExperimentArtifacts:
    """Run an experiment from CLI-compatible options."""
    config = load_experiment_config(config_path)
    updates = {}
    if output_root is not None:
        updates["output_root"] = output_root
    if report_root is not None:
        updates["report_root"] = report_root
    if updates:
        config = config.model_copy(update=updates)
    return run_experiment(config, run_id=run_id)


def write_portfolio_from_options(
    *,
    run_ids: tuple[str, ...],
    output_root: Path,
    report_root: Path,
    output: Path,
) -> tuple[Path, tuple[PortfolioRun, ...]]:
    """Write a portfolio report from CLI-compatible options."""
    runs = load_portfolio_runs(
        run_ids=run_ids,
        output_root=output_root,
        report_root=report_root,
    )
    return write_portfolio_report(output, runs), runs


@app.command("run-experiment")
def run_experiment_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="Experiment config path."),
    ] = DEFAULT_EXPERIMENT_CONFIG_PATH,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Override run identifier."),
    ] = None,
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Override output root."),
    ] = None,
    report_root: Annotated[
        Path | None,
        typer.Option("--report-root", help="Override report root."),
    ] = None,
) -> None:
    """Run a config-driven TTC OperatorBench experiment."""
    artifacts = run_experiment_from_options(
        config_path=config,
        run_id=run_id,
        output_root=output_root,
        report_root=report_root,
    )
    typer.echo(f"wrote attempts to {artifacts.attempts_path}")
    typer.echo(f"wrote search results to {artifacts.search_results_path}")
    typer.echo(f"wrote summary to {artifacts.summary_path}")
    typer.echo(f"wrote run manifest to {artifacts.run_manifest_path}")
    typer.echo(f"wrote failure taxonomy to {artifacts.failure_taxonomy_path}")
    typer.echo(f"wrote decision log to {artifacts.decision_log_path}")
    typer.echo(f"wrote state-action analysis to {artifacts.state_action_analysis_path}")
    typer.echo(f"wrote state-action analysis CSV to {artifacts.state_action_analysis_csv_path}")
    typer.echo(f"wrote decision to {artifacts.decision_path}")
    typer.echo(f"wrote report to {artifacts.report_path}")
    if artifacts.skipped_models:
        typer.echo(f"skipped {len(artifacts.skipped_models)} model(s)")


@app.command("portfolio-report")
def portfolio_report_command(
    runs: Annotated[
        str,
        typer.Option("--runs", help="Comma-separated run IDs to include."),
    ] = DEFAULT_PORTFOLIO_RUNS_OPTION,
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Experiment output root."),
    ] = DEFAULT_OUTPUT_ROOT,
    report_root: Annotated[
        Path,
        typer.Option("--report-root", help="Experiment report root."),
    ] = DEFAULT_REPORT_ROOT,
    output: Annotated[
        Path,
        typer.Option("--output", help="Portfolio report path."),
    ] = DEFAULT_PORTFOLIO_OUTPUT_PATH,
) -> None:
    """Create a portfolio report from completed run artifacts."""
    run_ids = tuple(run.strip() for run in runs.split(",") if run.strip())
    report_path, _ = write_portfolio_from_options(
        run_ids=run_ids,
        output_root=output_root,
        report_root=report_root,
        output=output,
    )
    typer.echo(f"wrote portfolio report to {report_path}")
