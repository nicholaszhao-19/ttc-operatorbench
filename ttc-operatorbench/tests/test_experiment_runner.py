"""Tests for the config-driven experiment protocol runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ttc_operatorbench.evals.experiment import (
    BudgetProfile,
    ExperimentConfig,
    ExperimentModel,
    build_decision,
    load_experiment_config,
    run_experiment,
)
from ttc_operatorbench.logging.writer import read_search_results_jsonl


def small_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="unit_toy_protocol",
        description="Small deterministic protocol for unit tests.",
        task_ids=("is_even", "factorial"),
        policies=("greedy", "best_of_n_2", "repair_only", "operator_bandit"),
        models=(
            ExperimentModel(
                name="dummy_control",
                provider="dummy",
                model_id="dummy-control",
            ),
        ),
        budgets=(
            BudgetProfile(name="one_call", max_attempts=1, max_verifier_calls=1, max_tokens=500),
            BudgetProfile(name="two_call", max_attempts=2, max_verifier_calls=2, max_tokens=1000),
        ),
        seeds=(0,),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_protocol_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/toy_protocol.yaml"))

    assert config.experiment_id == "toy_protocol"
    assert "operator_bandit" in config.policies
    assert len(config.budgets) >= 2
    assert config.models[0].provider == "dummy"


def test_hf_smoke_protocol_config_loads_and_is_gated() -> None:
    config = load_experiment_config(Path("configs/experiments/hf_smoke_protocol.yaml"))

    assert config.experiment_id == "hf_smoke_protocol"
    assert config.models[0].provider == "huggingface"
    assert config.models[0].requires_real_model_gate is True
    assert config.task_ids == ("is_even",)


def test_protocol_rejects_duplicate_budget_names(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="bad_protocol",
            task_ids=("is_even",),
            policies=("greedy",),
            models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
            budgets=(
                BudgetProfile(name="same", max_attempts=1),
                BudgetProfile(name="same", max_attempts=2),
            ),
            output_root=tmp_path / "outputs",
            report_root=tmp_path / "reports",
        )


def test_run_experiment_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = small_config(tmp_path)

    artifacts = run_experiment(config)

    assert artifacts.attempts_path.exists()
    assert artifacts.search_results_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.summary_csv_path.exists()
    assert artifacts.config_snapshot_path.exists()
    assert artifacts.decision_path.exists()
    assert artifacts.report_path.exists()
    assert artifacts.token_plot_path is not None
    assert artifacts.token_plot_path.exists()
    assert artifacts.verifier_plot_path is not None
    assert artifacts.verifier_plot_path.exists()

    expected_results = (
        len(config.models)
        * len(config.seeds)
        * len(config.budgets)
        * len(config.policies)
        * len(config.task_ids)
    )
    assert len(artifacts.results) == expected_results

    reloaded_results = read_search_results_jsonl(artifacts.search_results_path)
    assert len(reloaded_results) == expected_results
    assert read_json(artifacts.config_snapshot_path)["experiment_id"] == config.experiment_id

    attempt_rows = [
        json.loads(line)
        for line in artifacts.attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert attempt_rows
    assert all(row["metadata"]["experiment_id"] == config.experiment_id for row in attempt_rows)
    assert all("result_metadata" in row for row in attempt_rows)


def test_budget_sweep_is_reflected_in_results_and_summary(tmp_path: Path) -> None:
    config = small_config(tmp_path)

    artifacts = run_experiment(config)

    budget_names = {result.metadata["budget_name"] for result in artifacts.results}
    assert budget_names == {"one_call", "two_call"}
    for result in artifacts.results:
        assert result.budget.max_tokens is not None
        assert result.budget.max_verifier_calls is not None
        for attempt in result.attempts:
            assert attempt.cumulative_tokens <= result.budget.max_tokens
            assert attempt.cumulative_verifier_calls <= result.budget.max_verifier_calls

    summary_budget_names = {row["budget_name"] for row in artifacts.summary}
    assert summary_budget_names == budget_names


def test_decision_report_compares_operator_bandit_to_baselines(tmp_path: Path) -> None:
    config = small_config(tmp_path)
    artifacts = run_experiment(config)

    decision = build_decision(artifacts.results, config)

    assert decision["verdict"] in {"promising", "needs_analysis", "inconclusive"}
    assert decision["decision_policy"] == "operator_bandit"
    assert decision["best_baseline_policy"] in config.baseline_policies
    assert "decision_policy_metrics" in decision
    assert "best_baseline_metrics" in decision


def test_decision_is_inconclusive_when_no_policy_solves(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="no_solve_protocol",
        task_ids=("is_even",),
        policies=("greedy", "operator_bandit"),
        models=(
            ExperimentModel(
                name="dummy_wrong",
                provider="dummy",
                model_id="dummy-wrong",
                script="always_wrong",
            ),
        ),
        budgets=(BudgetProfile(name="two_call", max_attempts=2, max_verifier_calls=2),),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
        baseline_policies=("greedy",),
    )

    artifacts = run_experiment(config)

    assert artifacts.decision["verdict"] == "inconclusive"
    assert "No compared policy solved any task" in artifacts.decision["rationale"]


def test_huggingface_models_are_skipped_without_real_model_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUN_REAL_MODEL_TESTS", raising=False)
    config = ExperimentConfig(
        experiment_id="hf_gated",
        task_ids=("is_even",),
        policies=("greedy",),
        models=(
            ExperimentModel(
                name="qwen_tiny",
                provider="huggingface",
                model_id="Qwen/Qwen3-0.6B",
            ),
        ),
        budgets=(BudgetProfile(name="one_call", max_attempts=1, max_verifier_calls=1),),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
    )

    artifacts = run_experiment(config)

    assert artifacts.results == ()
    assert artifacts.skipped_models == (
        {"model_name": "qwen_tiny", "reason": "set RUN_REAL_MODEL_TESTS=1 to run real models"},
    )
    assert read_json(artifacts.decision_path)["verdict"] == "insufficient_data"


def test_run_experiment_script_uses_config_and_overrides_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text(
        json.dumps(
            small_config(tmp_path).model_dump(
                mode="json",
                exclude={"output_root", "report_root"},
            )
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("RUN_REAL_MODEL_TESTS", None)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--run-id",
            "cli_run",
            "--output-root",
            str(tmp_path / "cli_outputs"),
            "--report-root",
            str(tmp_path / "cli_reports"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "wrote attempts" in completed.stdout
    assert (tmp_path / "cli_outputs" / "cli_run" / "search_results.jsonl").exists()
    assert (tmp_path / "cli_reports" / "cli_run" / "report.md").exists()
