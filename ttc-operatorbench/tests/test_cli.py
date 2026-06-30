"""CLI behavior tests."""

import json
from pathlib import Path

import yaml
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


def test_cli_run_experiment_command_writes_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text(
        """
experiment_id: cli_unit_protocol
task_ids: [is_even]
policies: [greedy]
models:
  - name: dummy
    provider: dummy
    model_id: dummy
budgets:
  - name: one_call
    max_attempts: 1
    max_verifier_calls: 1
decision_policy: greedy
baseline_policies: [greedy]
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run-experiment",
            "--config",
            str(config_path),
            "--run-id",
            "cli_run",
            "--output-root",
            str(tmp_path / "outputs"),
            "--report-root",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "wrote attempts" in result.stdout
    assert (tmp_path / "outputs" / "cli_run" / "search_results.jsonl").exists()
    assert (tmp_path / "reports" / "cli_run" / "report.md").exists()


def test_cli_portfolio_report_command_writes_report(tmp_path: Path) -> None:
    runner = CliRunner()
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    run_dir = output_root / "run_one"
    run_dir.mkdir(parents=True)
    (report_root / "run_one").mkdir(parents=True)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "portfolio_cli",
                "task_suite": "toy_code",
                "task_ids": ["is_even"],
                "policies": ["greedy"],
                "budgets": [{"name": "one_call"}],
                "models": [{"model_id": "dummy", "model_tier": "structural_control"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "decision.json").write_text(
        json.dumps({"verdict": "inconclusive"}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text("[]", encoding="utf-8")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "search_results.jsonl").write_text("", encoding="utf-8")
    output_path = tmp_path / "portfolio.md"

    result = runner.invoke(
        app,
        [
            "portfolio-report",
            "--runs",
            "run_one",
            "--output-root",
            str(output_root),
            "--report-root",
            str(report_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "wrote portfolio report" in result.stdout
    assert output_path.exists()
