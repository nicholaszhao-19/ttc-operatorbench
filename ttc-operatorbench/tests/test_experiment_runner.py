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
    VerificationResult,
)
from ttc_operatorbench.evals import experiment as experiment_module
from ttc_operatorbench.evals.experiment import (
    BudgetProfile,
    ExperimentConfig,
    ExperimentModel,
    attach_hidden_verifications,
    build_decision,
    dummy_sequence_for,
    load_experiment_config,
    make_experiment_policy,
    policy_visible_task,
    run_experiment,
    write_policy_success_plot,
)
from ttc_operatorbench.logging.writer import read_search_results_jsonl
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.baselines import BestOfNPolicy, GreedyPolicy, MonkeySampleNPolicy
from ttc_operatorbench.search.differential_selection import (
    BottleneckAwareControllerPolicy,
    DifferentialSelectionPolicy,
)
from ttc_operatorbench.search.operator_bandit import (
    FixedOperatorOrderScheduler,
    OperatorBanditScheduler,
)
from ttc_operatorbench.tasks.toy_code import get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

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


def test_policy_visible_task_strips_hidden_evaluation_data() -> None:
    task = get_toy_task("is_even")

    visible_task = policy_visible_task(task)

    assert task.hidden_tests
    assert "hidden_tests" in task.allowed_verifier_inputs
    assert visible_task.hidden_tests == ()
    assert "hidden_tests" not in visible_task.allowed_verifier_inputs
    assert visible_task.public_tests == task.public_tests
    assert visible_task.prompt == task.prompt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_result(
    *,
    policy_name: str,
    budget_name: str,
    success: bool,
    total_tokens: int,
    hidden_success: bool | None = None,
) -> SearchResult:
    public_verification = VerificationResult(
        verification_passed=success,
        verification_score=1.0 if success else 0.0,
        scope="public",
    )
    hidden_verification = (
        VerificationResult(
            verification_passed=hidden_success,
            verification_score=1.0 if hidden_success else 0.0,
            scope="hidden",
            error_type=None if hidden_success else "test_failure",
        )
        if hidden_success is not None
        else None
    )
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
        public_verification=public_verification,
        hidden_verification=hidden_verification,
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


def synthetic_solved_result_with_final_tokens(
    *,
    policy_name: str,
    budget_name: str,
    solution_tokens: int,
    final_tokens: int,
) -> SearchResult:
    solution_attempt = AttemptLog(
        attempt_id=f"{policy_name}:{budget_name}:solution",
        task_id="is_even",
        model_id="synthetic",
        operator_name=policy_name,
        prompt="Write a Python function is_even(n).",
        generation_text="def is_even(n):\n    return n % 2 == 0",
        input_tokens=5,
        output_tokens=solution_tokens - 5,
        total_tokens=solution_tokens,
        latency_seconds=0.0,
        verification_passed=True,
        verification_score=1.0,
        cumulative_tokens=solution_tokens,
        cumulative_verifier_calls=1,
        cumulative_seconds=0.0,
        selected=True,
        policy_name=policy_name,
    )
    attempts = [solution_attempt]
    if final_tokens > solution_tokens:
        attempts.append(
            AttemptLog(
                attempt_id=f"{policy_name}:{budget_name}:extra",
                task_id="is_even",
                model_id="synthetic",
                operator_name=policy_name,
                prompt="Write a Python function is_even(n).",
                generation_text="def is_even(n):\n    return False",
                input_tokens=5,
                output_tokens=final_tokens - solution_tokens - 5,
                total_tokens=final_tokens - solution_tokens,
                latency_seconds=0.0,
                verification_passed=False,
                verification_score=0.0,
                cumulative_tokens=final_tokens,
                cumulative_verifier_calls=2,
                cumulative_seconds=0.0,
                selected=False,
                policy_name=policy_name,
            )
        )
    return SearchResult(
        task_id="is_even",
        policy_name=policy_name,
        budget=Budget(max_attempts=2),
        attempts=tuple(attempts),
        selected_attempt_id=solution_attempt.attempt_id,
        success=True,
        total_tokens=final_tokens,
        total_verifier_calls=len(attempts),
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
    strong_baselines = load_experiment_config(
        Path("configs/experiments/curated_strong_baselines_protocol.yaml")
    )

    assert curated.task_suite == "curated_code"
    assert len(curated.task_ids) == 50
    assert "operator_bandit" in curated.policies
    assert "eight_call" in {budget.name for budget in curated.budgets}
    assert ablation.task_suite == "curated_code"
    assert {
        "operator_bandit_no_error_bonus",
        "operator_bandit_unit_cost",
        "fixed_operator_order",
    }.issubset(ablation.policies)
    assert strong_baselines.task_suite == "curated_code"
    assert {"best_of_n_8", "best_of_n_16"}.issubset(strong_baselines.policies)
    assert "sixteen_call" in {budget.name for budget in strong_baselines.budgets}


def test_differential_toy_protocol_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/differential_toy_protocol.yaml"))

    assert config.experiment_id == "differential_toy_protocol"
    assert config.decision_policy == "bottleneck_controller"
    assert {"diffcodegen_select", "bottleneck_controller"}.issubset(config.policies)
    assert "diffcodegen_select" in config.baseline_policies


def test_monkey_toy_protocol_config_loads() -> None:
    config = load_experiment_config(Path("configs/experiments/monkey_toy_protocol.yaml"))

    assert config.experiment_id == "monkey_toy_protocol"
    assert config.decision_policy == "best_of_n_4"
    assert config.policies == ("greedy", "best_of_n_4", "monkey_sample_8")
    assert config.models[0].script == "sampling_control"
    assert config.budgets[0].max_attempts == 8


def test_sampling_control_uses_the_same_candidate_stream_for_every_policy() -> None:
    task = get_toy_task("is_even")

    greedy = dummy_sequence_for("sampling_control", "greedy", task)
    best_of_n = dummy_sequence_for("sampling_control", "best_of_n_4", task)
    monkey = dummy_sequence_for("sampling_control", "monkey_sample_8", task)

    assert greedy == best_of_n == monkey
    assert len(monkey) >= 8


def test_monkey_protocol_reports_pass_at_k_end_to_end(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/experiments/monkey_toy_protocol.yaml"))
    config = config.model_copy(
        update={
            "task_ids": ("is_even",),
            "output_root": tmp_path / "outputs",
            "report_root": tmp_path / "reports",
        }
    )

    artifacts = run_experiment(config)

    row = next(row for row in artifacts.summary if row["policy_name"] == "monkey_sample_8")
    assert row["fixed_sample_pass_at_1"] == pytest.approx(0.5)
    assert row["fixed_sample_pass_at_2"] == pytest.approx(11 / 14)
    assert row["fixed_sample_pass_at_4"] == pytest.approx(69 / 70)
    assert row["fixed_sample_pass_at_8"] == pytest.approx(1.0)
    assert row["total_attempts"] == 8
    assert "pass@4=0.986" in artifacts.report_path.read_text(encoding="utf-8")


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
            50,
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
        assert config.policy_state_scope == "per_run"
        assert {budget.name for budget in config.budgets} == budget_names
        assert config.decision_policy == "operator_bandit"
        if "best_of_n_2" in config.policies:
            assert config.models[0].do_sample is True
            assert config.models[0].temperature == pytest.approx(0.7)
            assert config.models[0].top_p == pytest.approx(0.95)


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
    best_of_n_16 = make_experiment_policy("best_of_n_16")
    monkey_sample_18 = make_experiment_policy("monkey_sample_18")
    diffcodegen = make_experiment_policy("diffcodegen_select")
    bottleneck = make_experiment_policy("bottleneck_controller")
    no_bonus = make_experiment_policy("operator_bandit_no_error_bonus")
    unit_cost = make_experiment_policy("operator_bandit_unit_cost")
    fixed_order = make_experiment_policy("fixed_operator_order")

    assert isinstance(best_of_n_16, BestOfNPolicy)
    assert best_of_n_16.name == "best_of_n_16"
    assert isinstance(monkey_sample_18, MonkeySampleNPolicy)
    assert monkey_sample_18.name == "monkey_sample_18"
    assert monkey_sample_18.n == 18
    assert isinstance(diffcodegen, DifferentialSelectionPolicy)
    assert diffcodegen.name == "diffcodegen_select"
    assert isinstance(bottleneck, BottleneckAwareControllerPolicy)
    assert bottleneck.name == "bottleneck_controller"
    assert isinstance(no_bonus, OperatorBanditScheduler)
    assert no_bonus.policy_name == "operator_bandit_no_error_bonus"
    assert no_bonus.error_type_bonuses == {}
    assert isinstance(unit_cost, OperatorBanditScheduler)
    assert unit_cost.policy_name == "operator_bandit_unit_cost"
    assert unit_cost.cost_metric == "unit"
    assert isinstance(fixed_order, FixedOperatorOrderScheduler)


def test_make_experiment_policy_records_requested_state_scope() -> None:
    policy = make_experiment_policy("operator_bandit", policy_state_scope="per_run")

    assert isinstance(policy, OperatorBanditScheduler)
    assert policy.policy_state_scope == "per_run"


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
    assert first_attempt["public_verification"]["scope"] == "public"
    assert first_attempt["hidden_verification"]["scope"] == "hidden"
    assert first_attempt["result_metadata"]["hidden_grading_policy_visible"] is False
    assert first_attempt["metadata"]["budget_name"] in {"one_call", "two_call"}
    assert first_attempt["metadata"]["model_tier"] == "structural_control"
    assert artifacts.summary[0]["model_tier"] == "structural_control"
    assert "hidden_solve_rate" in artifacts.summary[0]
    assert "oracle_hidden_solve_rate" in artifacts.summary[0]
    assert "policy_state_scope" in artifacts.summary[0]
    assert "overfit_rate" in artifacts.summary[0]
    assert artifacts.decision["metric_scope"] == "hidden"
    report_text = artifacts.report_path.read_text(encoding="utf-8")
    assert "dummy-control[structural_control]" in report_text
    assert "policy_state_scope: per_task" in report_text
    assert "oracle_hidden_solve_rate=" in report_text


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


def test_policy_state_scope_controls_bandit_stat_reuse(tmp_path: Path) -> None:
    one_budget = (BudgetProfile(name="two_call", max_attempts=2, max_verifier_calls=2),)
    base_config = small_config(tmp_path).model_copy(
        update={
            "policies": ("greedy", "operator_bandit"),
            "budgets": one_budget,
        }
    )

    per_task_artifacts = run_experiment(
        base_config.model_copy(update={"experiment_id": "per_task_scope"})
    )
    per_run_artifacts = run_experiment(
        base_config.model_copy(
            update={"experiment_id": "per_run_scope", "policy_state_scope": "per_run"}
        )
    )

    per_task_counts = [
        result.metadata["operator_decision_count_before"]
        for result in per_task_artifacts.results
        if result.policy_name == "operator_bandit"
    ]
    per_run_counts = [
        result.metadata["operator_decision_count_before"]
        for result in per_run_artifacts.results
        if result.policy_name == "operator_bandit"
    ]

    assert per_task_counts == [0, 0]
    assert per_run_counts[0] == 0
    assert per_run_counts[1] > 0


def test_hidden_grading_is_attached_after_policy_execution() -> None:
    task = get_toy_task("is_even")
    public_only_candidate = "def is_even(n):\n    return n in {0, -4}"
    provider = DummyModelProvider({task.task_id: (public_only_candidate,)})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    raw_result = GreedyPolicy().run(
        task,
        provider,
        verifier,
        Budget(max_attempts=1, max_verifier_calls=1, max_tokens=1_000),
    )
    graded_result = attach_hidden_verifications(raw_result, task, verifier)

    assert raw_result.success is True
    assert raw_result.attempts[0].public_verification is not None
    assert raw_result.attempts[0].hidden_verification is None
    assert graded_result.attempts[0].hidden_verification is not None
    assert graded_result.attempts[0].hidden_verification.verification_passed is False
    assert graded_result.metadata["hidden_grading_policy_visible"] is False


def test_hidden_grading_is_skipped_when_task_has_no_hidden_tests() -> None:
    task = Task(
        task_id="public-only",
        prompt="Write a function f() that returns 1.",
        public_tests=("assert f() == 1",),
        metadata={"entrypoint": "f"},
        allowed_verifier_inputs={"entrypoint": "f", "public_tests": ("assert f() == 1",)},
    )
    provider = DummyModelProvider({task.task_id: ("def f():\n    return 1",)})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    raw_result = GreedyPolicy().run(
        task,
        provider,
        verifier,
        Budget(max_attempts=1, max_verifier_calls=1, max_tokens=1_000),
    )
    graded_result = attach_hidden_verifications(raw_result, task, verifier)

    assert graded_result.metadata["hidden_tests_available"] is False
    assert graded_result.attempts[0].hidden_verification is None


def test_policy_success_plot_uses_hidden_curves_when_hidden_grading_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_plot_success_curve(
        curves: dict[str, dict[int, float]],
        path: Path,
        *,
        xlabel: str,
        title: str,
    ) -> Path:
        captured["curves"] = curves
        captured["xlabel"] = xlabel
        captured["title"] = title
        path.write_text("fake plot", encoding="utf-8")
        return path

    monkeypatch.setattr(experiment_module, "plot_success_curve", fake_plot_success_curve)
    result = synthetic_result(
        policy_name="operator_bandit",
        budget_name="one_call",
        success=True,
        hidden_success=False,
        total_tokens=10,
    )

    plot_path = write_policy_success_plot(
        tmp_path / "success_vs_tokens.png",
        (result,),
        metric="tokens",
    )

    assert plot_path == tmp_path / "success_vs_tokens.png"
    assert captured["title"] == "Hidden success by token budget"
    assert max(captured["curves"]["operator_bandit"].values()) == 0.0


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
    assert {comparison["relationship"] for comparison in decision["budget_comparisons"]} == {
        "inconclusive",
        "win",
    }


def test_decision_uses_common_auc_budget_grid(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="common_auc_grid_protocol",
        task_ids=("is_even",),
        policies=("greedy", "operator_bandit"),
        models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
        budgets=(BudgetProfile(name="one_call", max_attempts=2),),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
        baseline_policies=("greedy",),
    )
    results = (
        synthetic_solved_result_with_final_tokens(
            policy_name="greedy",
            budget_name="one_call",
            solution_tokens=10,
            final_tokens=10,
        ),
        synthetic_solved_result_with_final_tokens(
            policy_name="operator_bandit",
            budget_name="one_call",
            solution_tokens=10,
            final_tokens=100,
        ),
    )

    decision = build_decision(results, config)
    comparison = decision["budget_comparisons"][0]

    assert decision["verdict"] == "matches_baseline"
    assert comparison["status"] == "matches_baseline"
    assert comparison["relationship"] == "tie"
    assert comparison["decision_policy_metrics"]["token_auc"] == (
        comparison["best_baseline_metrics"]["token_auc"]
    )


def test_decision_prefers_hidden_metrics_when_hidden_grading_exists(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="hidden_metric_protocol",
        task_ids=("is_even",),
        policies=("greedy", "operator_bandit"),
        models=(ExperimentModel(name="dummy", provider="dummy", model_id="dummy"),),
        budgets=(BudgetProfile(name="one_call", max_attempts=1),),
        output_root=tmp_path / "outputs",
        report_root=tmp_path / "reports",
        baseline_policies=("greedy",),
    )
    results = (
        synthetic_result(
            policy_name="greedy",
            budget_name="one_call",
            success=True,
            hidden_success=True,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="operator_bandit",
            budget_name="one_call",
            success=True,
            hidden_success=False,
            total_tokens=10,
        ),
    )

    decision = build_decision(results, config)
    comparison = decision["budget_comparisons"][0]

    assert decision["metric_scope"] == "hidden"
    assert decision["verdict"] == "needs_analysis"
    assert comparison["metric_scope"] == "hidden"
    assert comparison["relationship"] == "loss"
    assert comparison["decision_policy_metrics"]["public_solve_rate"] == 1.0
    assert comparison["decision_policy_metrics"]["hidden_solve_rate"] == 0.0


def test_all_tie_budgets_are_not_called_promising(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="all_tie_protocol",
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
            success=True,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="operator_bandit",
            budget_name="one_call",
            success=True,
            total_tokens=10,
        ),
        synthetic_result(
            policy_name="greedy",
            budget_name="two_call",
            success=True,
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

    assert decision["verdict"] == "matches_baseline"
    assert {comparison["status"] for comparison in decision["budget_comparisons"]} == {
        "matches_baseline"
    }
    assert {comparison["relationship"] for comparison in decision["budget_comparisons"]} == {
        "tie"
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
