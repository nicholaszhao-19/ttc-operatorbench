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

from ttc_operatorbench.core.schema import Budget, SearchResult, Task
from ttc_operatorbench.evals.metrics import (
    area_under_success_curve,
    group_results_by_policy,
    median_tokens_to_solution,
    solve_rate,
    success_curve_by_token_budget,
    success_curve_by_verifier_budget,
    tokens_to_first_solution,
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
    PlanThenCodePolicy,
    RepairOnlyPolicy,
)
from ttc_operatorbench.search.operator_bandit import OperatorBanditScheduler
from ttc_operatorbench.tasks.toy_code import ToyTaskId, get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"

ProviderKind = Literal["dummy", "huggingface"]
DummyScriptKind = Literal["toy_control", "always_wrong"]

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
}
GREEDY_CONTROL_SOLVES = {"is_even", "reverse_string", "gcd"}


class ExperimentModel(BaseModel):
    """One model/provider entry in an experiment protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    provider: ProviderKind
    model_id: str = Field(min_length=1)
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
    max_cost: float | None = Field(default=None, gt=0.0)

    def to_budget(self) -> Budget:
        """Convert the profile into the core budget schema."""
        return Budget(
            max_attempts=self.max_attempts,
            max_tokens=self.max_tokens,
            max_verifier_calls=self.max_verifier_calls,
            max_seconds=self.max_seconds,
            max_cost=self.max_cost,
        )


class ExperimentConfig(BaseModel):
    """Reproducible experiment protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    description: str = ""
    task_ids: tuple[ToyTaskId, ...]
    policies: tuple[str, ...]
    models: tuple[ExperimentModel, ...]
    budgets: tuple[BudgetProfile, ...]
    seeds: tuple[int, ...] = (0,)
    output_root: Path = Path("outputs/runs")
    report_root: Path = Path("reports/runs")
    decision_policy: str = "operator_bandit"
    baseline_policies: tuple[str, ...] = (
        "greedy",
        "best_of_n_2",
        "best_of_n_4",
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
    tasks = tuple(get_toy_task(task_id) for task_id in config.task_ids)
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
            for budget_profile in config.budgets:
                budget = budget_profile.to_budget()
                for policy_name in config.policies:
                    for task in tasks:
                        provider = make_provider(model, policy_name, task, seed)
                        policy = make_experiment_policy(policy_name)
                        raw_result = policy.run(
                            task,
                            provider,
                            verifier,
                            budget,
                            run_id=(
                                f"{config.experiment_id}:{model.name}:seed_{seed}:"
                                f"{budget_profile.name}"
                            ),
                        )
                        results.append(
                            annotate_result(
                                raw_result,
                                experiment_id=config.experiment_id,
                                model=model,
                                seed=seed,
                                budget_profile=budget_profile,
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


def make_experiment_policy(policy_name: str) -> BaselinePolicy | OperatorBanditScheduler:
    """Create one policy from its protocol name."""
    if policy_name == "greedy":
        return GreedyPolicy()
    if policy_name.startswith("best_of_n_"):
        n = int(policy_name.removeprefix("best_of_n_"))
        policy = BestOfNPolicy(n=n)
        policy.name = policy_name
        return policy
    if policy_name == "repair_only":
        return RepairOnlyPolicy(max_repairs=1)
    if policy_name == "plan_then_code":
        return PlanThenCodePolicy()
    if policy_name == "local_revision_basic":
        return LocalRevisionBasicPolicy(max_revisions=1)
    if policy_name == "operator_bandit":
        return OperatorBanditScheduler(exploration_weight=1.0)
    raise ValueError(f"unsupported policy: {policy_name}")


def make_provider(
    model: ExperimentModel,
    policy_name: str,
    task: Task,
    seed: int,
) -> ModelProvider:
    """Create a model provider for one task/policy run."""
    if model.provider == "dummy":
        return DummyModelProvider(
            {task.task_id: dummy_sequence_for(model.script, policy_name, task)},
            provider_name="dummy",
            model_name=model.model_id,
        )
    return HuggingFaceModelProvider(
        model_id=model.model_id,
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
    if policy_name == "greedy":
        return (correct,) if task.task_id in GREEDY_CONTROL_SOLVES else (wrong,)
    if policy_name.startswith("best_of_n_"):
        return (wrong, correct, wrong, correct)
    if policy_name in {"repair_only", "local_revision_basic", "operator_bandit"}:
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
    model: ExperimentModel,
    seed: int,
    budget_profile: BudgetProfile,
) -> SearchResult:
    """Attach experiment metadata to a result and every attempt."""
    metadata = {
        **result.metadata,
        "experiment_id": experiment_id,
        "model_name": model.name,
        "model_id": model.model_id,
        "model_provider": model.provider,
        "seed": seed,
        "budget_name": budget_profile.name,
    }
    attempts = tuple(
        attempt.model_copy(
            update={
                "metadata": {
                    **attempt.metadata,
                    "experiment_id": experiment_id,
                    "model_name": model.name,
                    "budget_name": budget_profile.name,
                    "seed": seed,
                }
            }
        )
        for attempt in result.attempts
    )
    return result.model_copy(update={"attempts": attempts, "metadata": metadata})


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
    grouped: dict[tuple[str, str, str, str], list[SearchResult]] = {}
    for result in results:
        key = (
            str(result.metadata.get("model_name", "unknown")),
            str(result.metadata.get("model_id", "unknown")),
            result.policy_name,
            str(result.metadata.get("budget_name", "unknown")),
        )
        grouped.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for (model_name, model_id, policy_name, budget_name), group in sorted(grouped.items()):
        token_curve = success_curve_by_token_budget(tuple(group))
        verifier_curve = success_curve_by_verifier_budget(tuple(group))
        rows.append(
            {
                "model_name": model_name,
                "model_id": model_id,
                "policy_name": policy_name,
                "budget_name": budget_name,
                "number_of_results": len(group),
                "number_of_tasks": len({result.task_id for result in group}),
                "number_of_seeds": len({result.metadata.get("seed") for result in group}),
                "solved_count": sum(1 for result in group if result.success),
                "solve_rate": solve_rate(tuple(group)),
                "median_tokens_to_solution": median_tokens_to_solution(tuple(group)),
                "median_verifier_calls_to_solution": _median_or_none(
                    [
                        calls
                        for result in group
                        if (calls := verifier_calls_to_first_solution(result)) is not None
                    ]
                ),
                "token_auc": area_under_success_curve(token_curve),
                "verifier_call_auc": area_under_success_curve(verifier_curve),
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
    if metric == "tokens":
        curves = {
            policy_name: success_curve_by_token_budget(policy_results)
            for policy_name, policy_results in grouped.items()
        }
        return plot_success_curve(
            curves,
            path,
            xlabel="Token budget",
            title="Success by token budget",
        )
    curves = {
        policy_name: success_curve_by_verifier_budget(policy_results)
        for policy_name, policy_results in grouped.items()
    }
    return plot_success_curve(
        curves,
        path,
        xlabel="Verifier-call budget",
        title="Success by verifier-call budget",
    )


def build_decision(results: Sequence[SearchResult], config: ExperimentConfig) -> dict[str, Any]:
    """Compare the adaptive policy against configured baselines."""
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
            "rationale": "Decision policy or baseline policies were not present in results.",
        }

    decision_score = _policy_score(decision_policy_results)
    baseline_scores = {
        policy_name: _policy_score(policy_results)
        for policy_name, policy_results in baseline_groups.items()
    }
    best_baseline_policy, best_baseline_score = max(
        baseline_scores.items(),
        key=lambda item: (item[1]["solve_rate"], item[1]["token_auc"]),
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
            "decision_policy_metrics": decision_score,
            "best_baseline_metrics": best_baseline_score,
            "rationale": "No compared policy solved any task; treat this as a structural run only.",
        }

    promising = (
        decision_solve_rate > 0.0
        and decision_solve_rate >= baseline_solve_rate
        and decision_token_auc >= baseline_token_auc
    )
    return {
        "verdict": "promising" if promising else "needs_analysis",
        "decision_policy": config.decision_policy,
        "best_baseline_policy": best_baseline_policy,
        "decision_policy_metrics": decision_score,
        "best_baseline_metrics": best_baseline_score,
        "rationale": (
            "Adaptive policy matches or exceeds the strongest configured baseline."
            if promising
            else "Adaptive policy does not yet dominate the strongest configured baseline."
        ),
    }


def write_report_markdown(
    path: Path,
    *,
    config: ExperimentConfig,
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
        f"Rationale: {decision['rationale']}",
        "",
        "## Summary Rows",
        "",
    ]
    for row in summary:
        lines.append(
            "- "
            f"{row['model_name']} / {row['policy_name']} / {row['budget_name']}: "
            f"solve_rate={row['solve_rate']:.3f}, "
            f"token_auc={row['token_auc']:.3f}, "
            f"attempts={row['total_attempts']}"
        )
    if skipped_models:
        lines.extend(["", "## Skipped Models", ""])
        for skipped in skipped_models:
            lines.append(f"- {skipped['model_name']}: {skipped['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _policy_score(results: Sequence[SearchResult]) -> dict[str, float | int | None]:
    token_curve = success_curve_by_token_budget(results)
    verifier_curve = success_curve_by_verifier_budget(results)
    first_solution_tokens = [
        tokens for result in results if (tokens := tokens_to_first_solution(result)) is not None
    ]
    return {
        "solve_rate": solve_rate(results),
        "token_auc": area_under_success_curve(token_curve),
        "verifier_call_auc": area_under_success_curve(verifier_curve),
        "median_tokens_to_solution": _median_or_none(first_solution_tokens),
        "number_of_results": len(results),
    }


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
    "annotate_result",
    "build_decision",
    "dummy_sequence_for",
    "load_experiment_config",
    "make_experiment_policy",
    "make_provider",
    "run_experiment",
    "summarize_experiment_results",
    "write_attempts_jsonl",
    "write_policy_success_plot",
    "write_report_markdown",
    "write_summary_csv",
]
