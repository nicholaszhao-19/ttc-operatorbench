"""Command-line interface for TTC OperatorBench."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from ttc_operatorbench import __version__

if TYPE_CHECKING:
    from ttc_operatorbench.evals.experiment import ExperimentArtifacts

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
    help="Run and audit TTC OperatorBench experiments.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    context: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = False,
) -> None:
    """Run TTC OperatorBench commands."""
    del version
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())


@app.command("run")
def run_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="JSON-compatible YAML experiment config.",
        ),
    ] = Path("configs/experiments/toy_protocol.yaml"),
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Override the generated run directory name."),
    ] = None,
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Override the configured output root."),
    ] = None,
    report_root: Annotated[
        Path | None,
        typer.Option("--report-root", help="Override the configured report root."),
    ] = None,
) -> None:
    """Run a config-driven local or gated real-model experiment."""
    artifacts = _run_experiment_from_config(
        config,
        run_id=run_id,
        output_root=output_root,
        report_root=report_root,
    )
    _echo_artifacts(artifacts)


@app.command()
def doctor(
    evalplus: Annotated[
        bool,
        typer.Option(
            "--evalplus",
            help="Also require the pinned EvalPlus package and a running Docker daemon.",
        ),
    ] = False,
) -> None:
    """Check local prerequisites without running model inference."""
    checks: list[tuple[str, bool, str]] = []
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        (
            "Python",
            supported_python,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    uv_path = shutil.which("uv")
    checks.append(("uv", uv_path is not None, uv_path or "not found"))

    if evalplus:
        evalplus_available = importlib.util.find_spec("evalplus") is not None
        checks.append(
            (
                "EvalPlus",
                evalplus_available,
                "installed" if evalplus_available else "not installed",
            )
        )
        docker_path = shutil.which("docker")
        docker_ok, docker_detail = _docker_status(docker_path)
        checks.append(("Docker", docker_ok, docker_detail))

    for label, passed, detail in checks:
        status = "ok" if passed else "missing"
        typer.echo(f"[{status}] {label}: {detail}")
    if not all(passed for _, passed, _ in checks):
        raise typer.Exit(code=1)


@app.command("verify-results")
def verify_results(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Committed result-bundle manifest.",
        ),
    ] = Path("artifacts/results/stop_then_escalate_v1/manifest.json"),
) -> None:
    """Verify committed research summaries and task observations."""
    from ttc_operatorbench.evals.result_bundle import verify_result_bundle

    result = verify_result_bundle(manifest)
    typer.echo(
        f"verified {result.bundle_id}: {result.artifact_count} artifacts, "
        f"{result.record_count} records, {result.total_bytes} bytes"
    )


@app.command("validate-trajectories")
def validate_trajectories(
    trajectory_dir: Annotated[
        list[Path],
        typer.Option(
            "--trajectory-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Public-only trajectory directory; pass once per matched policy.",
        ),
    ],
) -> None:
    """Validate matched public trajectories before joining hidden labels."""
    from ttc_operatorbench.core.trajectory import read_trajectory_pool
    from ttc_operatorbench.evals.trajectory_analysis import (
        validate_shared_sample_roots,
    )

    if len(trajectory_dir) < 2:
        raise typer.BadParameter("pass at least two --trajectory-dir values")
    result = validate_shared_sample_roots(
        tuple(read_trajectory_pool(path) for path in trajectory_dir)
    )
    typer.echo(
        f"validated {len(result.pool_ids)} trajectories: "
        f"{result.compared_root_count} shared-root comparisons, "
        f"SHA-256 {result.canonical_sha256}"
    )


def _docker_status(docker_path: str | None) -> tuple[bool, str]:
    if docker_path is None:
        return False, "client not found"
    try:
        result = subprocess.run(
            [docker_path, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if result.returncode != 0:
        detail = result.stderr.strip() or "daemon unavailable"
        return False, detail
    return True, f"server {result.stdout.strip()}"


def _echo_artifacts(artifacts: ExperimentArtifacts) -> None:
    typer.echo(f"wrote attempts to {artifacts.attempts_path}")
    typer.echo(f"wrote search results to {artifacts.search_results_path}")
    typer.echo(f"wrote summary to {artifacts.summary_path}")
    typer.echo(f"wrote decision to {artifacts.decision_path}")
    typer.echo(f"wrote report to {artifacts.report_path}")
    if artifacts.skipped_models:
        typer.echo(f"skipped {len(artifacts.skipped_models)} model(s)")


def _run_experiment_from_config(
    config: Path,
    *,
    run_id: str | None,
    output_root: Path | None,
    report_root: Path | None,
) -> ExperimentArtifacts:
    from ttc_operatorbench.evals.experiment import run_experiment_from_config

    return run_experiment_from_config(
        config,
        run_id=run_id,
        output_root=output_root,
        report_root=report_root,
    )
