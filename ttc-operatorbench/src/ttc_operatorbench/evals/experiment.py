"""Config-driven experiment protocol runner."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    Generation,
    SearchResult,
    Task,
    VerificationResult,
)
from ttc_operatorbench.evals.metrics import (
    area_under_success_curve,
    group_results_by_policy,
    hidden_solve_rate,
    hidden_success_curve_by_token_budget,
    hidden_success_curve_by_verifier_budget,
    mean_fixed_sample_pass_at_k,
    median_tokens_to_hidden_solution,
    median_tokens_to_solution,
    oracle_hidden_solve_rate,
    oracle_hidden_success,
    overfit_rate,
    public_hidden_gap,
    solve_rate,
    success_curve_by_token_budget,
    success_curve_by_verifier_budget,
    tokens_to_first_hidden_solution,
    tokens_to_first_solution,
    verifier_calls_to_first_hidden_solution,
    verifier_calls_to_first_solution,
)
from ttc_operatorbench.evals.plots import plot_success_curve
from ttc_operatorbench.logging.writer import write_search_results_jsonl
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider
from ttc_operatorbench.search.baselines import (
    BaselinePolicy,
    BestOfNPolicy,
    GreedyPolicy,
    LocalRevisionBasicPolicy,
    ModelProvider,
    MonkeySampleNPolicy,
    PlanThenCodePolicy,
    RepairOnlyPolicy,
)
from ttc_operatorbench.search.differential_selection import (
    BottleneckAwareControllerPolicy,
    DifferentialSelectionPolicy,
)
from ttc_operatorbench.search.operator_bandit import (
    FixedOperatorOrderScheduler,
    OperatorBanditScheduler,
)
from ttc_operatorbench.tasks.curated_code import CURATED_REFERENCE_CANDIDATES
from ttc_operatorbench.tasks.registry import TaskSuite, get_task, validate_task_ids
from ttc_operatorbench.tasks.toy_code import HIDDEN_TESTS_KEY
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"

ProviderKind = Literal["dummy", "huggingface"]
DummyScriptKind = Literal["toy_control", "sampling_control", "always_wrong"]
MetricScope = Literal["public", "hidden"]
PolicyStateScope = Literal["per_task", "per_run"]
PASS_AT_K_POINTS = (1, 2, 4, 8, 16, 18)
SUPPORTED_POLICIES = (
    "greedy",
    "best_of_n_2",
    "best_of_n_4",
    "best_of_n_8",
    "best_of_n_16",
    "monkey_sample_1",
    "monkey_sample_2",
    "monkey_sample_4",
    "monkey_sample_8",
    "monkey_sample_16",
    "monkey_sample_18",
    "repair_only",
    "plan_then_code",
    "local_revision_basic",
    "diffcodegen_select",
    "bottleneck_controller",
    "operator_bandit",
    "operator_bandit_no_error_bonus",
    "operator_bandit_unit_cost",
    "fixed_operator_order",
)

CORRECT_CANDIDATES: dict[str, str] = {
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "factorial": (
        "def factorial(n):\n"
        "    result = 1\n"
        "    for value in range(2, n + 1):\n"
        "        result *= value\n"
        "    return result"
    ),
    "reverse_string": "def reverse_string(s):\n    return s[::-1]",
    "is_prime": (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for divisor in range(2, int(n ** 0.5) + 1):\n"
        "        if n % divisor == 0:\n"
        "            return False\n"
        "    return True"
    ),
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a"
    ),
    "gcd": (
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return abs(a)"
    ),
    "palindrome": "def palindrome(s):\n    return s == s[::-1]",
    **CURATED_REFERENCE_CANDIDATES,
}
GREEDY_CONTROL_SOLVES = {
    "is_even",
    "reverse_string",
    "gcd",
    "count_vowels",
    "sum_squares",
    "flatten_once",
    "binary_search",
    "clamp",
    "count_words",
}


class ExperimentModel(BaseModel):
    """One model/provider entry in an experiment protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    provider: ProviderKind
    model_id: str = Field(min_length=1)
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_tier: str = "unspecified"
    enabled: bool = True
    script: DummyScriptKind = "toy_control"
    device: str = "cpu"
    dtype: str = "auto"
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    seed: int | None = Field(default=None, ge=0)
    trust_remote_code: bool = False
    requires_real_model_gate: bool = True


class BudgetProfile(BaseModel):
    """Named budget point for cost-sweep experiments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    max_attempts: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_verifier_calls: int | None = Field(default=None, gt=0)
    max_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_has_limit(self) -> Self:
        if all(
            limit is None
            for limit in (
                self.max_attempts,
                self.max_tokens,
                self.max_verifier_calls,
                self.max_seconds,
            )
        ):
            raise ValueError("at least one budget limit must be set")
        return self

    def to_budget(self) -> Budget:
        """Convert the profile into the core budget schema."""
        return Budget(
            max_attempts=self.max_attempts,
            max_tokens=self.max_tokens,
            max_verifier_calls=self.max_verifier_calls,
            max_seconds=self.max_seconds,
        )


class ExperimentConfig(BaseModel):
    """Reproducible experiment protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    description: str = ""
    task_suite: TaskSuite = "toy_code"
    task_ids: tuple[str, ...]
    policies: tuple[str, ...]
    models: tuple[ExperimentModel, ...]
    budgets: tuple[BudgetProfile, ...]
    seeds: tuple[int, ...] = (0,)
    output_root: Path = Path("outputs/runs")
    report_root: Path = Path("reports/runs")
    decision_policy: str = "operator_bandit"
    policy_state_scope: PolicyStateScope = "per_task"
    baseline_policies: tuple[str, ...] = (
        "greedy",
        "best_of_n_2",
        "best_of_n_4",
        "best_of_n_8",
        "best_of_n_16",
        "monkey_sample_1",
        "monkey_sample_2",
        "monkey_sample_4",
        "monkey_sample_8",
        "monkey_sample_16",
        "monkey_sample_18",
        "repair_only",
        "plan_then_code",
        "local_revision_basic",
    )

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        _require_nonempty(self.task_ids, "task_ids")
        _require_nonempty(self.policies, "policies")
        _require_nonempty(self.models, "models")
        _require_nonempty(self.budgets, "budgets")
        _require_nonempty(self.seeds, "seeds")
        _reject_duplicates(self.policies, "policies")
        _reject_duplicates(tuple(model.name for model in self.models), "models")
        _reject_duplicates(tuple(budget.name for budget in self.budgets), "budgets")
        _reject_duplicates(self.task_ids, "task_ids")
        validate_task_ids(self.task_suite, self.task_ids)
        unsupported = sorted(set(self.policies) - set(SUPPORTED_POLICIES))
        if unsupported:
            raise ValueError(f"unsupported policies: {unsupported}")
        if self.decision_policy not in self.policies:
            raise ValueError("decision_policy must be included in policies")
        if not any(policy in self.policies for policy in self.baseline_policies):
            raise ValueError("at least one baseline policy must be included in policies")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        return self


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Files and in-memory data produced by one experiment run."""

    output_dir: Path
    report_dir: Path
    attempts_path: Path
    search_results_path: Path
    summary_path: Path
    summary_csv_path: Path
    config_snapshot_path: Path
    token_plot_path: Path | None
    verifier_plot_path: Path | None
    decision_path: Path
    report_path: Path
    results: tuple[SearchResult, ...]
    summary: tuple[dict[str, Any], ...]
    decision: dict[str, Any]
    skipped_models: tuple[dict[str, str], ...]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a JSON-compatible YAML experiment config."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def run_experiment(config: ExperimentConfig, *, run_id: str | None = None) -> ExperimentArtifacts:
    """Run a full protocol over models, seeds, budgets, policies, and tasks."""
    safe_run_id = _path_component(run_id or config.experiment_id)
    output_dir = config.output_root / safe_run_id
    report_dir = config.report_root / safe_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    verifier = PythonUnitTestVerifier(timeout_seconds=2.0)
    tasks = tuple(get_task(config.task_suite, task_id) for task_id in config.task_ids)
    results: list[SearchResult] = []
    skipped_models: list[dict[str, str]] = []

    for model in config.models:
        if not model.enabled:
            skipped_models.append({"model_name": model.name, "reason": "disabled"})
            continue
        if _real_model_gate_blocks(model):
            skipped_models.append(
                {
                    "model_name": model.name,
                    "reason": f"set {REAL_MODEL_TESTS_ENV}=1 to run real models",
                }
            )
            continue
        for seed in config.seeds:
            reusable_provider = make_reusable_provider(model, seed)
            for budget_profile in config.budgets:
                budget = budget_profile.to_budget()
                for policy_name in config.policies:
                    reusable_policy = (
                        make_experiment_policy(
                            policy_name,
                            policy_state_scope=config.policy_state_scope,
                        )
                        if config.policy_state_scope == "per_run"
                        else None
                    )
                    for task in tasks:
                        policy_task = policy_visible_task(task)
                        provider = reusable_provider or make_provider(
                            model,
                            policy_name,
                            policy_task,
                            seed,
                        )
                        policy = reusable_policy or make_experiment_policy(
                            policy_name,
                            policy_state_scope=config.policy_state_scope,
                        )
                        raw_result = policy.run(
                            policy_task,
                            provider,
                            verifier,
                            budget,
                            run_id=(
                                f"{config.experiment_id}:{model.name}:seed_{seed}:"
                                f"{budget_profile.name}"
                            ),
                        )
                        hidden_graded_result = attach_hidden_verifications(
                            raw_result,
                            task,
                            verifier,
                        )
                        results.append(
                            annotate_result(
                                hidden_graded_result,
                                experiment_id=config.experiment_id,
                                task_suite=config.task_suite,
                                model=model,
                                seed=seed,
                                budget_profile=budget_profile,
                                policy_state_scope=config.policy_state_scope,
                            )
                        )

    result_tuple = tuple(results)
    attempts_path = write_attempts_jsonl(output_dir / "attempts.jsonl", result_tuple)
    search_results_path = write_search_results_jsonl(
        output_dir / "search_results.jsonl",
        result_tuple,
    )
    summary = summarize_experiment_results(result_tuple)
    summary_path = write_json(output_dir / "summary.json", list(summary))
    summary_csv_path = write_summary_csv(output_dir / "summary.csv", summary)
    config_snapshot_path = write_json(
        output_dir / "config_snapshot.yaml",
        config.model_dump(mode="json"),
    )
    token_plot_path = write_policy_success_plot(
        report_dir / "success_vs_tokens.png",
        result_tuple,
        metric="tokens",
    )
    verifier_plot_path = write_policy_success_plot(
        report_dir / "success_vs_verifier_calls.png",
        result_tuple,
        metric="verifier_calls",
    )
    decision = build_decision(result_tuple, config)
    decision_path = write_json(output_dir / "decision.json", decision)
    report_path = write_report_markdown(
        report_dir / "report.md",
        config=config,
        results=result_tuple,
        summary=summary,
        decision=decision,
        skipped_models=tuple(skipped_models),
    )

    return ExperimentArtifacts(
        output_dir=output_dir,
        report_dir=report_dir,
        attempts_path=attempts_path,
        search_results_path=search_results_path,
        summary_path=summary_path,
        summary_csv_path=summary_csv_path,
        config_snapshot_path=config_snapshot_path,
        token_plot_path=token_plot_path,
        verifier_plot_path=verifier_plot_path,
        decision_path=decision_path,
        report_path=report_path,
        results=result_tuple,
        summary=summary,
        decision=decision,
        skipped_models=tuple(skipped_models),
    )


def policy_visible_task(task: Task) -> Task:
    """Return the task view allowed to reach model providers and search policies."""
    allowed_verifier_inputs = {
        key: value
        for key, value in task.allowed_verifier_inputs.items()
        if key != HIDDEN_TESTS_KEY
    }
    return task.model_copy(
        update={
            "hidden_tests": (),
            "allowed_verifier_inputs": allowed_verifier_inputs,
        }
    )


def make_experiment_policy(
    policy_name: str,
    *,
    policy_state_scope: PolicyStateScope = "per_task",
) -> (
    BaselinePolicy
    | DifferentialSelectionPolicy
    | BottleneckAwareControllerPolicy
    | OperatorBanditScheduler
    | FixedOperatorOrderScheduler
):
    """Create one policy from its protocol name."""
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy_name}")
    if policy_name == "greedy":
        return GreedyPolicy()
    if policy_name.startswith("best_of_n_"):
        n = int(policy_name.removeprefix("best_of_n_"))
        best_of_n_policy = BestOfNPolicy(n=n)
        best_of_n_policy.name = policy_name
        return best_of_n_policy
    if policy_name.startswith("monkey_sample_"):
        n = int(policy_name.removeprefix("monkey_sample_"))
        monkey_policy = MonkeySampleNPolicy(n=n)
        monkey_policy.name = policy_name
        return monkey_policy
    if policy_name == "repair_only":
        return RepairOnlyPolicy(max_repairs=1)
    if policy_name == "plan_then_code":
        return PlanThenCodePolicy()
    if policy_name == "local_revision_basic":
        return LocalRevisionBasicPolicy(max_revisions=1)
    if policy_name == "diffcodegen_select":
        return DifferentialSelectionPolicy(n=4)
    if policy_name == "bottleneck_controller":
        return BottleneckAwareControllerPolicy(min_samples=2, max_samples=4)
    if policy_name == "operator_bandit":
        return OperatorBanditScheduler(
            exploration_weight=1.0,
            policy_state_scope=policy_state_scope,
        )
    if policy_name == "operator_bandit_no_error_bonus":
        return OperatorBanditScheduler(
            exploration_weight=1.0,
            error_type_bonuses={},
            policy_name="operator_bandit_no_error_bonus",
            policy_state_scope=policy_state_scope,
        )
    if policy_name == "operator_bandit_unit_cost":
        return OperatorBanditScheduler(
            exploration_weight=1.0,
            cost_metric="unit",
            policy_name="operator_bandit_unit_cost",
            policy_state_scope=policy_state_scope,
        )
    if policy_name == "fixed_operator_order":
        return FixedOperatorOrderScheduler()
    raise ValueError(f"unsupported policy: {policy_name}")


def make_reusable_provider(model: ExperimentModel, seed: int) -> ModelProvider | None:
    """Create a provider reusable across task/policy/budget runs when safe."""
    if model.provider != "huggingface":
        return None
    return make_provider(model, policy_name="", task=None, seed=seed)


def make_provider(
    model: ExperimentModel,
    policy_name: str,
    task: Task | None,
    seed: int,
) -> ModelProvider:
    """Create a model provider for one task/policy run."""
    if model.provider == "dummy":
        if task is None:
            raise ValueError("dummy providers require a task")
        return DummyModelProvider(
            {task.task_id: dummy_sequence_for(model.script, policy_name, task)},
            provider_name="dummy",
            model_name=model.model_id,
        )
    return HuggingFaceModelProvider(
        model_id=model.model_id,
        model_revision=model.model_revision,
        tokenizer_revision=model.tokenizer_revision,
        device=model.device,
        dtype=model.dtype,
        max_new_tokens=model.max_new_tokens,
        temperature=model.temperature,
        top_p=model.top_p,
        do_sample=model.do_sample,
        seed=model.seed if model.seed is not None else seed,
        trust_remote_code=model.trust_remote_code,
    )


def dummy_sequence_for(
    script: DummyScriptKind,
    policy_name: str,
    task: Task,
) -> tuple[str, ...]:
    """Return deterministic dummy generations for a controlled protocol run."""
    wrong = wrong_candidate(task)
    if script == "always_wrong":
        return (wrong, wrong, wrong, wrong)

    correct = CORRECT_CANDIDATES[task.task_id]
    if script == "sampling_control":
        return tuple(correct if index % 2 else wrong for index in range(18))
    if policy_name == "greedy":
        return (correct,) if task.task_id in GREEDY_CONTROL_SOLVES else (wrong,)
    if policy_name.startswith("best_of_n_"):
        return (wrong, correct, wrong, correct)
    if policy_name.startswith("monkey_sample_"):
        return (wrong, correct, wrong, correct)
    if policy_name in {
        "repair_only",
        "local_revision_basic",
        "operator_bandit",
        "operator_bandit_no_error_bonus",
        "operator_bandit_unit_cost",
        "fixed_operator_order",
        "diffcodegen_select",
        "bottleneck_controller",
    }:
        return (wrong, correct, "Use the requested function.", correct)
    if policy_name == "plan_then_code":
        return ("Use the requested function signature and return expression.", correct)
    return (wrong,)


def wrong_candidate(task: Task) -> str:
    """Return a simple public-test-failing candidate for a task."""
    entrypoint = task.metadata["entrypoint"]
    return f"def {entrypoint}(*args):\n    return None"


def annotate_result(
    result: SearchResult,
    *,
    experiment_id: str,
    task_suite: TaskSuite,
    model: ExperimentModel,
    seed: int,
    budget_profile: BudgetProfile,
    policy_state_scope: PolicyStateScope,
) -> SearchResult:
    """Attach experiment metadata to a result and every attempt."""
    metadata = {
        **result.metadata,
        "experiment_id": experiment_id,
        "model_name": model.name,
        "model_id": model.model_id,
        "model_tier": model.model_tier,
        "model_provider": model.provider,
        "task_suite": task_suite,
        "seed": seed,
        "budget_name": budget_profile.name,
        "policy_state_scope": policy_state_scope,
    }
    attempts = tuple(
        attempt.model_copy(
            update={
                "metadata": {
                    **attempt.metadata,
                    "experiment_id": experiment_id,
                    "model_name": model.name,
                    "model_tier": model.model_tier,
                    "task_suite": task_suite,
                    "budget_name": budget_profile.name,
                    "seed": seed,
                    "policy_state_scope": policy_state_scope,
                }
            }
        )
        for attempt in result.attempts
    )
    return result.model_copy(update={"attempts": attempts, "metadata": metadata})


def attach_hidden_verifications(
    result: SearchResult,
    task: Task,
    verifier: PythonUnitTestVerifier,
) -> SearchResult:
    """Attach hidden verifier results after policy execution has finished."""
    hidden_tests_available = _has_hidden_tests(task)
    attempts = tuple(
        _attempt_with_hidden_verification(
            attempt,
            task,
            verifier,
            hidden_tests_available=hidden_tests_available,
        )
        for attempt in result.attempts
    )
    metadata = {
        **result.metadata,
        "hidden_tests_available": hidden_tests_available,
        "hidden_grading_policy_visible": False,
    }
    return result.model_copy(update={"attempts": attempts, "metadata": metadata})


def _attempt_with_hidden_verification(
    attempt: AttemptLog,
    task: Task,
    verifier: PythonUnitTestVerifier,
    *,
    hidden_tests_available: bool,
) -> AttemptLog:
    public_verification = attempt.public_verification or VerificationResult(
        verification_passed=attempt.verification_passed,
        verification_score=attempt.verification_score,
        scope="public",
        verifier_name="python_unit_tests",
        stdout=attempt.stdout,
        stderr=attempt.stderr,
        error_type=attempt.error_type,
    )
    hidden_verification = attempt.hidden_verification
    if hidden_tests_available and hidden_verification is None and _should_grade_hidden(attempt):
        generation = Generation(
            prompt=attempt.prompt,
            generation_text=attempt.generation_text,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            total_tokens=attempt.total_tokens,
            latency_seconds=attempt.latency_seconds,
            model_name=attempt.model_id,
            provider_name=attempt.provider_name,
            metadata=attempt.metadata,
        )
        hidden_verification = verifier.verify_hidden_generation(task, generation)
    return attempt.model_copy(
        update={
            "public_verification": public_verification,
            "hidden_verification": hidden_verification,
        }
    )


def _should_grade_hidden(attempt: AttemptLog) -> bool:
    if attempt.error_type == "not_verified_plan":
        return False
    return not str(attempt.operator_name).endswith("/plan")


def _has_hidden_tests(task: Task) -> bool:
    if task.hidden_tests:
        return True
    hidden_inputs = task.allowed_verifier_inputs.get(HIDDEN_TESTS_KEY)
    return bool(hidden_inputs)


def write_attempts_jsonl(path: Path, results: Sequence[SearchResult]) -> Path:
    """Write one JSONL row per attempt across all search results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            for attempt in result.attempts:
                row = attempt.model_dump()
                row["result_metadata"] = result.metadata
                file.write(json.dumps(row, sort_keys=True))
                file.write("\n")
    return path


def summarize_experiment_results(
    results: Sequence[SearchResult],
) -> tuple[dict[str, Any], ...]:
    """Summarize results by model, policy, and budget profile."""
    grouped: dict[tuple[str, str, str, str, str], list[SearchResult]] = {}
    comparison_grouped: dict[tuple[str, str, str, str], list[SearchResult]] = {}
    for result in results:
        key = (
            str(result.metadata.get("model_name", "unknown")),
            str(result.metadata.get("model_id", "unknown")),
            str(result.metadata.get("model_tier", "unknown")),
            result.policy_name,
            str(result.metadata.get("budget_name", "unknown")),
        )
        grouped.setdefault(key, []).append(result)
        comparison_key = (key[0], key[1], key[2], key[4])
        comparison_grouped.setdefault(comparison_key, []).append(result)

    rows: list[dict[str, Any]] = []
    for (
        model_name,
        model_id,
        model_tier,
        policy_name,
        budget_name,
    ), group in sorted(grouped.items()):
        comparison_group = tuple(
            comparison_grouped[(model_name, model_id, model_tier, budget_name)]
        )
        public_token_curve = success_curve_by_token_budget(
            tuple(group),
            _token_budget_grid(comparison_group),
        )
        public_verifier_curve = success_curve_by_verifier_budget(
            tuple(group),
            _verifier_budget_grid(comparison_group),
        )
        hidden_token_curve = hidden_success_curve_by_token_budget(
            tuple(group),
            _token_budget_grid(comparison_group),
        )
        hidden_verifier_curve = hidden_success_curve_by_verifier_budget(
            tuple(group),
            _verifier_budget_grid(comparison_group),
        )
        rows.append(
            {
                "model_name": model_name,
                "model_id": model_id,
                "model_tier": model_tier,
                "policy_name": policy_name,
                "budget_name": budget_name,
                "number_of_results": len(group),
                "number_of_tasks": len({result.task_id for result in group}),
                "number_of_seeds": len({result.metadata.get("seed") for result in group}),
                "policy_state_scope": _single_metadata_value(group, "policy_state_scope"),
                "solved_count": sum(1 for result in group if result.success),
                "solve_rate": solve_rate(tuple(group)),
                "public_solve_rate": solve_rate(tuple(group)),
                "hidden_solved_count": sum(
                    1 for result in group if tokens_to_first_hidden_solution(result) is not None
                ),
                "hidden_solve_rate": hidden_solve_rate(tuple(group)),
                "oracle_hidden_solved_count": sum(
                    1 for result in group if oracle_hidden_success(result)
                ),
                "oracle_hidden_solve_rate": oracle_hidden_solve_rate(tuple(group)),
                **{
                    f"fixed_sample_pass_at_{k}": mean_fixed_sample_pass_at_k(tuple(group), k)
                    for k in PASS_AT_K_POINTS
                },
                "public_hidden_gap": public_hidden_gap(tuple(group)),
                "overfit_rate": overfit_rate(tuple(group)),
                "median_tokens_to_solution": median_tokens_to_solution(tuple(group)),
                "median_tokens_to_hidden_solution": median_tokens_to_hidden_solution(
                    tuple(group)
                ),
                "median_verifier_calls_to_solution": _median_or_none(
                    [
                        calls
                        for result in group
                        if (calls := verifier_calls_to_first_solution(result)) is not None
                    ]
                ),
                "median_verifier_calls_to_hidden_solution": _median_or_none(
                    [
                        calls
                        for result in group
                        if (
                            calls := verifier_calls_to_first_hidden_solution(result)
                        )
                        is not None
                    ]
                ),
                "token_auc": area_under_success_curve(public_token_curve),
                "verifier_call_auc": area_under_success_curve(public_verifier_curve),
                "hidden_token_auc": area_under_success_curve(hidden_token_curve),
                "hidden_verifier_call_auc": area_under_success_curve(hidden_verifier_curve),
                "total_attempts": sum(len(result.attempts) for result in group),
                "total_tokens": sum(result.total_tokens for result in group),
                "total_verifier_calls": sum(result.total_verifier_calls for result in group),
            }
        )
    return tuple(rows)


def write_summary_csv(path: Path, summary: Sequence[Mapping[str, Any]]) -> Path:
    """Write summary rows to CSV for spreadsheet/report workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = tuple(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return path


def _single_metadata_value(results: Sequence[SearchResult], key: str) -> str:
    values = {str(result.metadata.get(key, "unknown")) for result in results}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def write_policy_success_plot(
    path: Path,
    results: Sequence[SearchResult],
    *,
    metric: Literal["tokens", "verifier_calls"],
) -> Path | None:
    """Write a policy-level success curve plot."""
    if not results:
        return None
    grouped = group_results_by_policy(results)
    metric_scope = _decision_metric_scope(results)
    if metric == "tokens":
        curves = _policy_success_curves(grouped, metric="tokens", metric_scope=metric_scope)
        title_prefix = "Hidden" if metric_scope == "hidden" else "Public"
        return plot_success_curve(
            curves,
            path,
            xlabel="Token budget",
            title=f"{title_prefix} success by token budget",
        )
    curves = _policy_success_curves(
        grouped,
        metric="verifier_calls",
        metric_scope=metric_scope,
    )
    title_prefix = "Hidden" if metric_scope == "hidden" else "Public"
    return plot_success_curve(
        curves,
        path,
        xlabel="Verifier-call budget",
        title=f"{title_prefix} success by verifier-call budget",
    )


def _policy_success_curves(
    grouped: Mapping[str, Sequence[SearchResult]],
    *,
    metric: Literal["tokens", "verifier_calls"],
    metric_scope: MetricScope,
) -> dict[str, dict[int, float]]:
    if metric == "tokens":
        if metric_scope == "hidden":
            return {
                policy_name: hidden_success_curve_by_token_budget(policy_results)
                for policy_name, policy_results in grouped.items()
            }
        return {
            policy_name: success_curve_by_token_budget(policy_results)
            for policy_name, policy_results in grouped.items()
        }
    if metric_scope == "hidden":
        return {
            policy_name: hidden_success_curve_by_verifier_budget(policy_results)
            for policy_name, policy_results in grouped.items()
        }
    return {
        policy_name: success_curve_by_verifier_budget(policy_results)
        for policy_name, policy_results in grouped.items()
    }


def build_decision(results: Sequence[SearchResult], config: ExperimentConfig) -> dict[str, Any]:
    """Compare the configured decision policy against configured baselines."""
    grouped = group_results_by_policy(results)
    decision_policy_results = grouped.get(config.decision_policy)
    baseline_groups = {
        policy: grouped[policy]
        for policy in config.baseline_policies
        if policy in grouped and policy != config.decision_policy
    }
    if decision_policy_results is None or not baseline_groups:
        return {
            "verdict": "insufficient_data",
            "decision_policy": config.decision_policy,
            "metric_scope": _decision_metric_scope(results),
            "rationale": "Decision policy or baseline policies were not present in results.",
        }

    compared_results = _flatten_policy_groups(
        {"decision": decision_policy_results, **baseline_groups}
    )
    metric_scope = _decision_metric_scope(compared_results)
    token_budgets = _token_budget_grid(compared_results)
    verifier_budgets = _verifier_budget_grid(compared_results)
    decision_score = _policy_score(
        decision_policy_results,
        token_budgets=token_budgets,
        verifier_budgets=verifier_budgets,
        metric_scope=metric_scope,
    )
    best_baseline_policy, best_baseline_score = _best_baseline_score(
        baseline_groups,
        token_budgets=token_budgets,
        verifier_budgets=verifier_budgets,
        metric_scope=metric_scope,
    )
    budget_comparisons = _budget_comparisons(
        decision_policy_results,
        baseline_groups,
        config,
    )
    decision_solve_rate = float(decision_score["solve_rate"] or 0.0)
    decision_token_auc = float(decision_score["token_auc"] or 0.0)
    baseline_solve_rate = float(best_baseline_score["solve_rate"] or 0.0)
    baseline_token_auc = float(best_baseline_score["token_auc"] or 0.0)
    if decision_solve_rate == 0.0 and baseline_solve_rate == 0.0:
        return {
            "verdict": "inconclusive",
            "decision_policy": config.decision_policy,
            "best_baseline_policy": best_baseline_policy,
            "metric_scope": metric_scope,
            "decision_policy_metrics": decision_score,
            "best_baseline_metrics": best_baseline_score,
            "budget_comparisons": budget_comparisons,
            "rationale": "No compared policy solved any task; treat this as a structural run only.",
        }

    overall_promising = (
        decision_solve_rate > 0.0
        and decision_solve_rate >= baseline_solve_rate
        and decision_token_auc >= baseline_token_auc
    )
    budget_statuses = {comparison["status"] for comparison in budget_comparisons}
    budget_relationships = {
        comparison.get("relationship", "unknown") for comparison in budget_comparisons
    }
    has_budget_loss = "needs_analysis" in budget_statuses
    has_unresolved_budget = bool({"inconclusive", "insufficient_data"} & budget_statuses)
    has_budget_win = "win" in budget_relationships
    promising = (
        overall_promising
        and has_budget_win
        and not has_budget_loss
        and not has_unresolved_budget
    )
    if promising:
        verdict = "promising"
        rationale = (
            "Decision policy matches or exceeds the strongest baseline at every "
            "compared budget."
        )
    elif overall_promising and not has_budget_loss and not has_unresolved_budget:
        verdict = "matches_baseline"
        rationale = (
            "Decision policy matches the strongest configured baseline, but no "
            "budget shows a clear win."
        )
    else:
        verdict = "needs_analysis"
        rationale = (
            "Decision policy does not dominate the strongest configured baseline "
            "across all budgets, including inconclusive budget points."
        )
    return {
        "verdict": verdict,
        "decision_policy": config.decision_policy,
        "best_baseline_policy": best_baseline_policy,
        "metric_scope": metric_scope,
        "decision_policy_metrics": decision_score,
        "best_baseline_metrics": best_baseline_score,
        "budget_comparisons": budget_comparisons,
        "rationale": rationale,
    }


def _best_baseline_score(
    baseline_groups: Mapping[str, Sequence[SearchResult]],
    *,
    token_budgets: Sequence[int] | None = None,
    verifier_budgets: Sequence[int] | None = None,
    metric_scope: MetricScope = "public",
) -> tuple[str, dict[str, float | int | str | None]]:
    baseline_scores = {
        policy_name: _policy_score(
            policy_results,
            token_budgets=token_budgets,
            verifier_budgets=verifier_budgets,
            metric_scope=metric_scope,
        )
        for policy_name, policy_results in baseline_groups.items()
    }
    return max(
        baseline_scores.items(),
        key=lambda item: (item[1]["solve_rate"], item[1]["token_auc"]),
    )


def _budget_comparisons(
    decision_policy_results: Sequence[SearchResult],
    baseline_groups: Mapping[str, Sequence[SearchResult]],
    config: ExperimentConfig,
) -> tuple[dict[str, Any], ...]:
    comparisons: list[dict[str, Any]] = []
    for budget in config.budgets:
        decision_results = tuple(
            result
            for result in decision_policy_results
            if result.metadata.get("budget_name") == budget.name
        )
        budget_baselines = {
            policy_name: tuple(
                result
                for result in policy_results
                if result.metadata.get("budget_name") == budget.name
            )
            for policy_name, policy_results in baseline_groups.items()
        }
        budget_baselines = {
            policy_name: policy_results
            for policy_name, policy_results in budget_baselines.items()
            if policy_results
        }
        if not decision_results or not budget_baselines:
            comparisons.append(
                {
                    "budget_name": budget.name,
                    "status": "insufficient_data",
                    "metric_scope": _decision_metric_scope(decision_results),
                    "rationale": "Decision policy or baselines are missing for this budget.",
                }
            )
            continue

        compared_results = _flatten_policy_groups(
            {"decision": decision_results, **budget_baselines}
        )
        metric_scope = _decision_metric_scope(compared_results)
        token_budgets = _token_budget_grid(compared_results)
        verifier_budgets = _verifier_budget_grid(compared_results)
        decision_score = _policy_score(
            decision_results,
            token_budgets=token_budgets,
            verifier_budgets=verifier_budgets,
            metric_scope=metric_scope,
        )
        best_policy, best_score = _best_baseline_score(
            budget_baselines,
            token_budgets=token_budgets,
            verifier_budgets=verifier_budgets,
            metric_scope=metric_scope,
        )
        decision_solve_rate = float(decision_score["solve_rate"] or 0.0)
        decision_token_auc = float(decision_score["token_auc"] or 0.0)
        baseline_solve_rate = float(best_score["solve_rate"] or 0.0)
        baseline_token_auc = float(best_score["token_auc"] or 0.0)

        if decision_solve_rate == 0.0 and baseline_solve_rate == 0.0:
            status = "inconclusive"
            relationship = "inconclusive"
            rationale = "No compared policy solved any task at this budget."
        elif (
            decision_solve_rate > 0.0
            and decision_solve_rate >= baseline_solve_rate
            and decision_token_auc >= baseline_token_auc
        ):
            relationship = (
                "win"
                if (
                    decision_solve_rate > baseline_solve_rate
                    or decision_token_auc > baseline_token_auc
                )
                else "tie"
            )
            if relationship == "win":
                status = "promising"
                rationale = "Decision policy exceeds the strongest baseline here."
            else:
                status = "matches_baseline"
                rationale = "Decision policy matches the strongest baseline here."
        else:
            status = "needs_analysis"
            relationship = "loss"
            rationale = "A baseline is stronger at this budget."

        comparisons.append(
            {
                "budget_name": budget.name,
                "status": status,
                "relationship": relationship,
                "metric_scope": metric_scope,
                "decision_policy_metrics": decision_score,
                "best_baseline_policy": best_policy,
                "best_baseline_metrics": best_score,
                "rationale": rationale,
            }
        )
    return tuple(comparisons)


def write_report_markdown(
    path: Path,
    *,
    config: ExperimentConfig,
    results: Sequence[SearchResult],
    summary: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    skipped_models: Sequence[Mapping[str, str]],
) -> Path:
    """Write a compact human-readable experiment report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {config.experiment_id}",
        "",
        config.description or "No description provided.",
        "",
        f"Verdict: {decision['verdict']}",
        "",
        f"Decision metric scope: {decision.get('metric_scope', 'public')}",
        "",
        f"Rationale: {decision['rationale']}",
        "",
        "## Protocol",
        "",
        f"- task_suite: {config.task_suite}",
        f"- task_count: {len(config.task_ids)}",
        f"- task_ids: {', '.join(config.task_ids)}",
        f"- models: {', '.join(_model_label(model) for model in config.models)}",
        f"- policies: {', '.join(config.policies)}",
        f"- policy_state_scope: {config.policy_state_scope}",
        f"- budgets: {', '.join(budget.name for budget in config.budgets)}",
        f"- seeds: {', '.join(str(seed) for seed in config.seeds)}",
        "",
        "## Summary Rows",
        "",
    ]
    for row in summary:
        line = (
            "- "
            f"{row['model_name']} / {row['model_tier']} / "
            f"{row['policy_name']} / {row['budget_name']}: "
            f"public_solve_rate={row['public_solve_rate']:.3f}, "
            f"hidden_solve_rate={row['hidden_solve_rate']:.3f}, "
            f"oracle_hidden_solve_rate={row['oracle_hidden_solve_rate']:.3f}, "
            f"overfit_rate={row['overfit_rate']:.3f}, "
            f"token_auc={row['token_auc']:.3f}, "
            f"hidden_token_auc={row['hidden_token_auc']:.3f}, "
            f"attempts={row['total_attempts']}"
        )
        pass_at_values = [
            f"pass@{k}={value:.3f}"
            for k in PASS_AT_K_POINTS
            if isinstance((value := row.get(f"fixed_sample_pass_at_{k}")), float)
        ]
        if pass_at_values:
            line += ", " + ", ".join(pass_at_values)
        lines.append(line)
    budget_comparisons = decision.get("budget_comparisons")
    if isinstance(budget_comparisons, (list, tuple)):
        lines.extend(["", "## Budget Comparisons", ""])
        for comparison in budget_comparisons:
            if not isinstance(comparison, Mapping):
                continue
            lines.append(
                "- "
                f"{comparison.get('budget_name', 'unknown')}: "
                f"{comparison.get('status', 'unknown')} - "
                f"scope={comparison.get('metric_scope', 'public')} - "
                f"best_baseline={comparison.get('best_baseline_policy', 'unknown')} - "
                f"{comparison.get('rationale', '')}"
            )
    failure_examples = _failure_examples(results)
    if failure_examples:
        lines.extend(["", "## Failure Examples", ""])
        for example in failure_examples:
            lines.append(
                "- "
                f"{example['task_id']} / {example['policy_name']} / {example['budget_name']}: "
                f"last_error={example['last_error_type']}, attempts={example['attempts']}"
            )
    if skipped_models:
        lines.extend(["", "## Skipped Models", ""])
        for skipped in skipped_models:
            lines.append(f"- {skipped['model_name']}: {skipped['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _model_label(model: ExperimentModel) -> str:
    return f"{model.name}:{model.model_id}[{model.model_tier}]"


def _failure_examples(
    results: Sequence[SearchResult],
    limit: int = 3,
) -> tuple[dict[str, Any], ...]:
    examples: list[dict[str, Any]] = []
    for result in results:
        if result.success:
            continue
        last_attempt = result.attempts[-1] if result.attempts else None
        examples.append(
            {
                "task_id": result.task_id,
                "policy_name": result.policy_name,
                "budget_name": result.metadata.get("budget_name", "unknown"),
                "attempts": len(result.attempts),
                "last_error_type": last_attempt.error_type if last_attempt is not None else None,
            }
        )
        if len(examples) >= limit:
            break
    return tuple(examples)


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _policy_score(
    results: Sequence[SearchResult],
    *,
    token_budgets: Sequence[int] | None = None,
    verifier_budgets: Sequence[int] | None = None,
    metric_scope: MetricScope = "public",
) -> dict[str, float | int | str | None]:
    return _policy_score_for_scope(
        results,
        token_budgets=token_budgets,
        verifier_budgets=verifier_budgets,
        metric_scope=metric_scope,
    )


def _policy_score_for_scope(
    results: Sequence[SearchResult],
    *,
    token_budgets: Sequence[int] | None = None,
    verifier_budgets: Sequence[int] | None = None,
    metric_scope: MetricScope,
) -> dict[str, float | int | str | None]:
    public_token_curve = success_curve_by_token_budget(results, token_budgets)
    public_verifier_curve = success_curve_by_verifier_budget(results, verifier_budgets)
    hidden_token_curve = hidden_success_curve_by_token_budget(results, token_budgets)
    hidden_verifier_curve = hidden_success_curve_by_verifier_budget(results, verifier_budgets)
    first_solution_tokens = [
        tokens for result in results if (tokens := tokens_to_first_solution(result)) is not None
    ]
    first_hidden_solution_tokens = [
        tokens
        for result in results
        if (tokens := tokens_to_first_hidden_solution(result)) is not None
    ]
    if metric_scope == "hidden":
        primary_solve_rate = hidden_solve_rate(results)
        primary_token_auc = area_under_success_curve(hidden_token_curve)
        primary_verifier_auc = area_under_success_curve(hidden_verifier_curve)
        primary_median_tokens = _median_or_none(first_hidden_solution_tokens)
    else:
        primary_solve_rate = solve_rate(results)
        primary_token_auc = area_under_success_curve(public_token_curve)
        primary_verifier_auc = area_under_success_curve(public_verifier_curve)
        primary_median_tokens = _median_or_none(first_solution_tokens)
    return {
        "metric_scope": metric_scope,
        "solve_rate": primary_solve_rate,
        "token_auc": primary_token_auc,
        "verifier_call_auc": primary_verifier_auc,
        "median_tokens_to_solution": primary_median_tokens,
        "public_solve_rate": solve_rate(results),
        "hidden_solve_rate": hidden_solve_rate(results),
        "oracle_hidden_solve_rate": oracle_hidden_solve_rate(results),
        "public_hidden_gap": public_hidden_gap(results),
        "overfit_rate": overfit_rate(results),
        "public_token_auc": area_under_success_curve(public_token_curve),
        "hidden_token_auc": area_under_success_curve(hidden_token_curve),
        "public_verifier_call_auc": area_under_success_curve(public_verifier_curve),
        "hidden_verifier_call_auc": area_under_success_curve(hidden_verifier_curve),
        "median_tokens_to_public_solution": _median_or_none(first_solution_tokens),
        "median_tokens_to_hidden_solution": _median_or_none(first_hidden_solution_tokens),
        "number_of_results": len(results),
    }


def _flatten_policy_groups(
    policy_groups: Mapping[str, Sequence[SearchResult]],
) -> tuple[SearchResult, ...]:
    return tuple(result for results in policy_groups.values() for result in results)


def _decision_metric_scope(results: Sequence[SearchResult]) -> MetricScope:
    return "hidden" if _has_hidden_verifications(results) else "public"


def _has_hidden_verifications(results: Sequence[SearchResult]) -> bool:
    return any(
        attempt.hidden_verification is not None
        for result in results
        for attempt in result.attempts
    )


def _token_budget_grid(results: Sequence[SearchResult]) -> tuple[int, ...]:
    budgets = {
        0,
        *(attempt.cumulative_tokens for result in results for attempt in result.attempts),
    }
    return tuple(sorted(budget for budget in budgets if budget >= 0))


def _verifier_budget_grid(results: Sequence[SearchResult]) -> tuple[int, ...]:
    budgets = {
        0,
        *(attempt.cumulative_verifier_calls for result in results for attempt in result.attempts),
    }
    return tuple(sorted(budget for budget in budgets if budget >= 0))


def _median_or_none(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _real_model_gate_blocks(model: ExperimentModel) -> bool:
    return (
        model.provider == "huggingface"
        and model.requires_real_model_gate
        and os.environ.get(REAL_MODEL_TESTS_ENV) != "1"
    )


def _require_nonempty(values: Sequence[Any], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")


def _reject_duplicates(values: Sequence[str], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


def _path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "unknown"


__all__ = [
    "BudgetProfile",
    "DummyScriptKind",
    "ExperimentArtifacts",
    "ExperimentConfig",
    "ExperimentModel",
    "ProviderKind",
    "REAL_MODEL_TESTS_ENV",
    "TaskSuite",
    "annotate_result",
    "attach_hidden_verifications",
    "build_decision",
    "dummy_sequence_for",
    "load_experiment_config",
    "make_experiment_policy",
    "make_provider",
    "policy_visible_task",
    "run_experiment",
    "summarize_experiment_results",
    "write_attempts_jsonl",
    "write_policy_success_plot",
    "write_report_markdown",
    "write_summary_csv",
]
