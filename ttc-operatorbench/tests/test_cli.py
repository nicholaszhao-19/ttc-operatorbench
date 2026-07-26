"""Tests for the supported package command surface."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from ttc_operatorbench import __version__
from ttc_operatorbench.cli import app


def test_version_option_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_no_arguments_shows_commands() -> None:
    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "run" in result.stdout


def test_run_command_uses_shared_experiment_entrypoint(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config: Path, **kwargs: object) -> SimpleNamespace:
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace(
            attempts_path=Path("outputs/attempts.jsonl"),
            search_results_path=Path("outputs/search_results.jsonl"),
            summary_path=Path("outputs/summary.json"),
            decision_path=Path("outputs/decision.json"),
            report_path=Path("reports/report.md"),
            skipped_models=(),
        )

    monkeypatch.setattr("ttc_operatorbench.cli._run_experiment_from_config", fake_run)
    result = CliRunner().invoke(
        app,
        ["run", "--config", str(config_path), "--run-id", "cli-smoke"],
    )

    assert result.exit_code == 0
    assert captured == {
        "config": config_path,
        "run_id": "cli-smoke",
        "output_root": None,
        "report_root": None,
    }
    assert "wrote report to reports/report.md" in result.stdout


def test_doctor_reports_required_tools(monkeypatch: Any) -> None:
    monkeypatch.setattr("ttc_operatorbench.cli.shutil.which", lambda _: "/usr/bin/uv")

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "[ok] Python" in result.stdout
    assert "[ok] uv: /usr/bin/uv" in result.stdout
