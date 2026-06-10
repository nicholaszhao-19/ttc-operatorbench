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

from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    Generation,
    SamplingConfig,
    SearchResult,
    Task,
)
from ttc_operatorbench.evals.experiment import (
    BudgetProfile,
    ExperimentConfig,
    ExperimentModel,
    build_decision,
    load_experiment_config,
    make_experiment_policy,
    run_experiment,
)
from ttc_operatorbench.logging.writer import read_search_results_jsonl
from ttc_operatorbench.search.operator_bandit import (
    FixedOperatorOrderScheduler,
    OperatorBanditScheduler,
)

HF_CURATED_CONFIG_PATHS = (
    Path("configs/experiments/hf_curated_qwen25_coder_05b_protocol.yaml"),
    Path("configs/experiments/hf_curated_qwen25_coder_15b_probe_protocol.yaml"),
    Path("configs/experiments/hf_curated_qwen25_coder_15b_protocol.yaml"),
    Path("configs/experiments/hf_curated_qwen25_coder_7b_probe_protocol.yaml"),
    Path("configs/experiments/hf_curated_qwen25_coder_7b_protocol.yaml"),
)


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
                model_tier="structural_control",
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


def synthetic_result(
    *,
    policy_name: str,
    budget_name: str,
    success: bool,
    total_tokens: int,
) -> SearchResult:
    attempt = AttemptLog(
        attempt_id=f"{policy_name}:{budget_name}:attempt",
        task_id="is_even",
        model_id="synthetic",
        operator_name=policy_name,
        prompt="Write a Python function is_even(n).",
        generation_text="def is_even(n):\n    return n % 2 == 0",
        input_tokens=5,
        output_tokens=total_tokens - 5,
        total_tokens=total_tokens,
        latency_seconds=0.0,
        verification_passed=success,
        verification_score=1.0 if success else 0.0,
        cumulative_tokens=total_tokens,
        cumulative_verifier_calls=1,
        cumulative_seconds=0.0,
        selected=success,
        policy_name=policy_name,
    )
    return SearchResult(
        task_id="is_even",
        policy_name=policy_name,
        budget=Budget(max_attempts=1),
        attempts=(attempt,),
        selected_attempt_id=attempt.attempt_id if success else None,
        success=success,
        total_tokens=total_tokens,
        total_verifier_calls=1,
        total_seconds=0.0,
        metadata={"budget_name": budget_name},
    )


def test_default_protocol_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/toy_protocol.yaml"))

    assert config.experiment_id == "toy_protocol"
    assert config.task_suite == "toy_code"
    assert "operator_bandit" in config.policies
    assert len(config.budgets) >= 2
    assert config.models[0].provider == "dummy"


def test_hf_smoke_protocol_config_loads_and_is_gated() -> None:
    config = load_experiment_config(Path("configs/experiments/hf_smoke_protocol.yaml"))

    assert config.experiment_id == "hf_smoke_protocol"
    assert config.task_suite == "toy_code"
    assert config.models[0].provider == "huggingface"
    assert config.models[0].model_tier == "smoke"
    assert config.models[0].requires_real_model_gate is True
    assert config.task_ids == ("is_even",)


def test_curated_protocol_configs_load() -> None:
    curated = load_experiment_config(Path("configs/experiments/curated_protocol.yaml"))
    ablation = load_experiment_config(Path("configs/experiments/curated_ablation_protocol.yaml"))

    assert curated.task_suite == "curated_code"
    assert len(curated.task_ids) == 20
    assert "operator_bandit" in curated.policies
    assert "eight_call" in {budget.name for budget in curated.budgets}
    assert ablation.task_suite == "curated_code"
    assert {
        "operator_bandit_no_error_bonus",
        "operator_bandit_unit_cost",
        "fixed_operator_order",
    }.issubset(ablation.policies)


def test_hf_curated_protocol_configs_load_and_are_gated() -> None:
    expected = {
        "hf_curated_qwen25_coder_05b_protocol": (
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "small_coder_sanity",
            5,
            {"one_call", "two_call", "four_call"},
        ),
        "hf_curated_qwen25_coder_15b_protocol": (
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "small_coder",
            20,
            {"one_call", "two_call", "four_call", "eight_call"},
        ),
        "hf_curated_qwen25_coder_15b_probe_protocol": (
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "small_coder",
            5,
            {"one_call", "two_call", "four_call"},
        ),
        "hf_curated_qwen25_coder_7b_protocol": (
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "strong_local_candidate",
            10,
            {"one_call", "two_call", "four_call"},
        ),
        "hf_curated_qwen25_coder_7b_probe_protocol": (
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "strong_local_candidate",
            1,
            {"one_call", "two_call"},
        ),
    }

    for path in HF_CURATED_CONFIG_PATHS:
        config = load_experiment_config(path)
        model_id, model_tier, task_count, budget_names = expected[config.experiment_id]
        assert config.task_suite == "curated_code"
        assert len(config.task_ids) == task_count
        assert config.models[0].provider == "huggingface"
        assert config.models[0].model_id == model_id
        assert config.models[0].model_tier == model_tier
        assert config.models[0].requires_real_model_gate is True
        assert {budget.name for budget in config.budgets} == budget_names
        assert config.decision_policy == "operator_bandit"


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


def test_budget_profile_requires_at_least_one_limit() -> None:
    with pytest.raises(ValidationError):
        BudgetProfile(name="empty")


def test_protocol_rejects_unsupported_policy(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="bad_policy_protocol",
            task_ids=("is_even",),
            policies=("greedy", "mystery_policy"),
            models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
            budgets=(BudgetProfile(name="one_call", max_attempts=1),),
            output_root=tmp_path / "outputs",
            report_root=tmp_path / "reports",
        )


def test_protocol_rejects_task_ids_from_wrong_suite(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="bad_task_protocol",
            task_suite="curated_code",
            task_ids=("is_even",),
            policies=("greedy",),
            models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
            budgets=(BudgetProfile(name="one_call", max_attempts=1),),
            output_root=tmp_path / "outputs",
            report_root=tmp_path / "reports",
        )


def test_make_experiment_policy_supports_ablation_variants() -> None:
    no_bonus = make_experiment_policy("operator_bandit_no_error_bonus")
    unit_cost = make_experiment_policy("operator_bandit_unit_cost")
    fixed_order = make_experiment_policy("fixed_operator_order")

    assert isinstance(no_bonus, OperatorBanditScheduler)
    assert no_bonus.policy_name == "operator_bandit_no_error_bonus"
    assert no_bonus.error_type_bonuses == {}
    assert isinstance(unit_cost, OperatorBanditScheduler)
    assert unit_cost.policy_name == "operator_bandit_unit_cost"
    assert unit_cost.cost_metric == "unit"
    assert isinstance(fixed_order, FixedOperatorOrderScheduler)


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
    assert read_json(artifacts.config_snapshot_path)["task_suite"] == "toy_code"
    assert read_json(artifacts.config_snapshot_path)["models"][0]["model_tier"] == (
        "structural_control"
    )

    attempt_rows = [
        json.loads(line)
        for line in artifacts.attempts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert attempt_rows
    assert all(row["metadata"]["experiment_id"] == config.experiment_id for row in attempt_rows)
    assert all("result_metadata" in row for row in attempt_rows)
    first_attempt = attempt_rows[0]
    assert first_attempt["model_id"] == "dummy-control"
    assert first_attempt["policy_name"]
    assert first_attempt["operator_name"]
    assert first_attempt["input_tokens"] >= 0
    assert first_attempt["output_tokens"] >= 0
    assert first_attempt["total_tokens"] == (
        first_attempt["input_tokens"] + first_attempt["output_tokens"]
    )
    assert "verification_passed" in first_attempt
    assert first_attempt["metadata"]["budget_name"] in {"one_call", "two_call"}
    assert first_attempt["metadata"]["model_tier"] == "structural_control"
    assert artifacts.summary[0]["model_tier"] == "structural_control"
    assert "dummy-control[structural_control]" in artifacts.report_path.read_text(
        encoding="utf-8"
    )


def test_budget_sweep_is_reflected_in_results_and_summary(tmp_path: Path) -> None:
    config = small_config(tmp_path)

    artifacts = run_experiment(config)

    budget_names = {result.metadata["budget_name"] for result in artifacts.results}
    assert budget_names == {"one_call", "two_call"}
    for result in artifacts.results:
        assert result.metadata["task_suite"] == "toy_code"
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

    assert decision["verdict"] == "needs_analysis"
    assert decision["decision_policy"] == "operator_bandit"
    assert decision["best_baseline_policy"] in config.baseline_policies
    assert "decision_policy_metrics" in decision
    assert "best_baseline_metrics" in decision
    budget_comparisons = decision["budget_comparisons"]
    assert any(
        comparison["budget_name"] == "one_call"
        and comparison["status"] == "needs_analysis"
        for comparison in budget_comparisons
    )


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


def test_inconclusive_budget_prevents_overall_promising_verdict(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="mixed_budget_protocol",
        task_ids=("is_even",),
        policies=("greedy", "operator_bandit"),
        models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
        budgets=(
            BudgetProfile(name="one_call", max_attempts=1),
            BudgetProfile(name="two_call", max_attempts=1),
        ),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
        baseline_policies=("greedy",),
    )
    results = (
        synthetic_result(
            policy_name="greedy",
            budget_name="one_call",
            success=False,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="operator_bandit",
            budget_name="one_call",
            success=False,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="greedy",
            budget_name="two_call",
            success=False,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="operator_bandit",
            budget_name="two_call",
            success=True,
            total_tokens=10,
        ),
    )

    decision = build_decision(results, config)

    assert decision["verdict"] == "needs_analysis"
    assert {comparison["status"] for comparison in decision["budget_comparisons"]} == {
        "inconclusive",
        "promising",
    }


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
        decision_policy="greedy",
    )

    artifacts = run_experiment(config)

    assert artifacts.results == ()
    assert artifacts.skipped_models == (
        {"model_name": "qwen_tiny", "reason": "set RUN_REAL_MODEL_TESTS=1 to run real models"},
    )
    assert read_json(artifacts.decision_path)["verdict"] == "insufficient_data"


def test_hf_curated_protocols_skip_without_real_model_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUN_REAL_MODEL_TESTS", raising=False)

    for path in HF_CURATED_CONFIG_PATHS:
        config = load_experiment_config(path).model_copy(
            update={"output_root": tmp_path / "outputs", "report_root": tmp_path / "reports"}
        )
        artifacts = run_experiment(config)

        assert artifacts.results == ()
        assert artifacts.skipped_models == (
            {
                "model_name": config.models[0].name,
                "reason": "set RUN_REAL_MODEL_TESTS=1 to run real models",
            },
        )
        assert read_json(artifacts.decision_path)["verdict"] == "insufficient_data"


def test_huggingface_provider_is_reused_across_protocol_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0
    generations = 0

    class FakeHFProvider:
        def __init__(self, **kwargs: Any):
            nonlocal constructions
            constructions += 1
            self.model_id = kwargs["model_id"]

        def generate(
            self,
            task: Task,
            sampling: SamplingConfig | None = None,
        ) -> Generation:
            nonlocal generations
            del sampling
            generations += 1
            entrypoint = task.metadata["entrypoint"]
            text = f"def {entrypoint}(*args):\n    return None"
            return Generation(
                prompt=task.prompt,
                generation_text=text,
                input_tokens=len(task.prompt.split()),
                output_tokens=len(text.split()),
                total_tokens=len(task.prompt.split()) + len(text.split()),
                latency_seconds=0.0,
                model_name=self.model_id,
                provider_name="fake-hf",
            )

    monkeypatch.setenv("RUN_REAL_MODEL_TESTS", "1")
    monkeypatch.setattr(
        "ttc_operatorbench.evals.experiment.HuggingFaceModelProvider",
        FakeHFProvider,
    )
    config = ExperimentConfig(
        experiment_id="hf_reuse",
        task_ids=("is_even", "factorial"),
        policies=("greedy", "operator_bandit"),
        models=(
            ExperimentModel(
                name="fake_hf",
                provider="huggingface",
                model_id="fake/hf",
                model_tier="test_hf",
            ),
        ),
        budgets=(
            BudgetProfile(name="one_call", max_attempts=1, max_verifier_calls=1),
            BudgetProfile(name="two_call", max_attempts=1, max_verifier_calls=1),
        ),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
        baseline_policies=("greedy",),
    )

    artifacts = run_experiment(config)

    assert constructions == 1
    assert generations == len(artifacts.results)
    assert len(artifacts.results) == 8
    assert all(result.metadata["model_tier"] == "test_hf" for result in artifacts.results)


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
