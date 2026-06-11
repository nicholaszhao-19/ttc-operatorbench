"""Tiny HF-compatible toy evaluation runner."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from ttc_operatorbench.core.schema import Budget, SearchResult, Task
from ttc_operatorbench.evals.metrics import (
    assert_monotone_nondecreasing,
    solve_rate,
    success_curve_by_token_budget,
    tokens_to_first_solution,
    verifier_calls_to_first_solution,
)
from ttc_operatorbench.evals.plots import plot_success_curve_by_token_budget
from ttc_operatorbench.logging.writer import write_search_results_jsonl
from ttc_operatorbench.models.hf_provider import DEFAULT_HF_SMOKE_MODEL_ID
from ttc_operatorbench.search.baselines import (
    BaselinePolicy,
    BestOfNPolicy,
    GreedyPolicy,
    ModelProvider,
    RepairOnlyPolicy,
)
from ttc_operatorbench.search.operator_bandit import OperatorBanditScheduler
from ttc_operatorbench.tasks.toy_code import ToyTaskId, get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

DEFAULT_HF_TOY_TASK_IDS: tuple[ToyTaskId, ...] = (
    "is_even",
    "factorial",
    "reverse_string",
    "gcd",
    "palindrome",
)
DEFAULT_HF_TOY_POLICIES = ("greedy", "best_of_n_2", "repair_only")
DEFAULT_HF_TOY_OUTPUT_ROOT = Path("outputs/hf_toy_eval")

ProviderFactory = Callable[[str, Task], ModelProvider]


@dataclass(frozen=True)
class HFToyEvalConfig:
    """Configuration for a tiny real-model-compatible toy evaluation."""

    model_id: str = DEFAULT_HF_SMOKE_MODEL_ID
    output_dir: Path = DEFAULT_HF_TOY_OUTPUT_ROOT
    max_tasks: int = 5
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 0
    policies: tuple[str, ...] = DEFAULT_HF_TOY_POLICIES
    device: str = "cpu"
    dtype: str = "auto"
    task_ids: tuple[ToyTaskId, ...] = DEFAULT_HF_TOY_TASK_IDS


@dataclass(frozen=True)
class HFToyEvalArtifacts:
    """Paths and in-memory objects produced by a toy evaluation run."""

    output_dir: Path
    attempts_path: Path
    search_results_path: Path
    summary_path: Path
    plot_path: Path | None
    results: tuple[SearchResult, ...]
    summary: tuple[dict[str, Any], ...]


def make_policy(policy_name: str) -> BaselinePolicy | OperatorBanditScheduler:
    """Create a supported baseline policy by name."""
    if policy_name == "greedy":
        return GreedyPolicy()
    if policy_name == "best_of_n_2":
        policy = BestOfNPolicy(n=2)
        policy.name = "best_of_n_2"
        return policy
    if policy_name == "best_of_n_4":
        policy = BestOfNPolicy(n=4)
        policy.name = "best_of_n_4"
        return policy
    if policy_name == "repair_only":
        return RepairOnlyPolicy(max_repairs=1)
    if policy_name == "operator_bandit":
        return OperatorBanditScheduler(exploration_weight=1.0)
    raise ValueError(f"unsupported policy: {policy_name}")


def select_tasks(config: HFToyEvalConfig) -> tuple[Task, ...]:
    """Select the configured prefix of tiny toy-code tasks."""
    if config.max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    return tuple(get_toy_task(task_id) for task_id in config.task_ids[: config.max_tasks])


def default_budget_for(policy_name: str) -> Budget:
    """Return a small budget that lets the chosen policy run but keeps smoke runs tiny."""
    max_attempts_by_policy = {
        "greedy": 1,
        "best_of_n_2": 2,
        "best_of_n_4": 4,
        "repair_only": 2,
        "operator_bandit": 4,
    }
    max_attempts = max_attempts_by_policy[policy_name]
    return Budget(
        max_attempts=max_attempts,
        max_verifier_calls=max_attempts,
        max_tokens=4_000,
    )


def default_output_dir_for_run(
    model_id: str,
    policies: Sequence[str],
    *,
    root: Path = DEFAULT_HF_TOY_OUTPUT_ROOT,
) -> Path:
    """Return a stable, non-clobbering default output directory for a CLI run."""
    policy_slug = "__".join(_path_component(policy) for policy in policies) or "no_policy"
    return root / _path_component(model_id) / policy_slug


def run_hf_toy_eval(
    config: HFToyEvalConfig,
    provider_factory: ProviderFactory,
) -> HFToyEvalArtifacts:
    """Run selected policies and toy tasks through the existing verifier pipeline."""
    tasks = select_tasks(config)
    verifier = PythonUnitTestVerifier(timeout_seconds=2.0)
    results: list[SearchResult] = []

    for policy_name in config.policies:
        policy = make_policy(policy_name)
        budget = default_budget_for(policy_name)
        for task in tasks:
            provider = provider_factory(policy_name, task)
            results.append(
                policy.run(
                    task,
                    provider,
                    verifier,
                    budget,
                    run_id=f"hf-toy-eval:{config.model_id}",
                )
            )

    result_tuple = tuple(results)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = write_attempts_jsonl(
        output_dir / "attempts.jsonl",
        result_tuple,
        config.model_id,
    )
    search_results_path = write_search_results_jsonl(
        output_dir / "search_results.jsonl",
        result_tuple,
    )
    summary = summarize_results(result_tuple, default_model_id=config.model_id)
    summary_path = write_summary_json(output_dir / "summary.json", summary)
    plot_path = write_success_plot(output_dir / "success_vs_tokens.png", result_tuple)

    return HFToyEvalArtifacts(
        output_dir=output_dir,
        attempts_path=attempts_path,
        search_results_path=search_results_path,
        summary_path=summary_path,
        plot_path=plot_path,
        results=result_tuple,
        summary=summary,
    )


def write_attempts_jsonl(
    path: Path,
    results: Sequence[SearchResult],
    default_model_id: str,
) -> Path:
    """Write one JSONL row per generated attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            for attempt in result.attempts:
                row = attempt.model_dump()
                row["model_id"] = row.get("model_id") or default_model_id
                file.write(json.dumps(row, sort_keys=True))
                file.write("\n")
    return path


def summarize_results(
    results: Sequence[SearchResult],
    *,
    default_model_id: str,
) -> tuple[dict[str, Any], ...]:
    """Compute summary metrics grouped by policy and model ID."""
    grouped: dict[tuple[str, str], list[SearchResult]] = {}
    for result in results:
        model_id = model_id_for_result(result) or default_model_id
        grouped.setdefault((result.policy_name, model_id), []).append(result)

    rows: list[dict[str, Any]] = []
    for (policy_name, model_id), group in sorted(grouped.items()):
        solved = sum(1 for result in group if result.success)
        tokens_to_solution = [
            tokens
            for result in group
            if (tokens := tokens_to_first_solution(result)) is not None
        ]
        verifier_calls_to_solution = [
            calls
            for result in group
            if (calls := verifier_calls_to_first_solution(result)) is not None
        ]
        rows.append(
            {
                "policy_name": policy_name,
                "model_id": model_id,
                "number_of_tasks": len(group),
                "solved_count": solved,
                "solve_rate": solve_rate(tuple(group)),
                "median_tokens_to_first_solution": _median_or_none(tokens_to_solution),
                "median_verifier_calls_to_first_solution": _median_or_none(
                    verifier_calls_to_solution
                ),
                "total_attempts": sum(len(result.attempts) for result in group),
                "total_tokens": sum(_final_cumulative_tokens(result) for result in group),
                "total_verifier_calls": sum(
                    _final_cumulative_verifier_calls(result) for result in group
                ),
            }
        )
    return tuple(rows)


def write_summary_json(path: Path, summary: Sequence[dict[str, Any]]) -> Path:
    """Write summary metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(summary), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_success_plot(path: Path, results: Sequence[SearchResult]) -> Path | None:
    """Write fraction-solved versus token budget plot."""
    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_name, []).append(result)
    if not grouped:
        return None
    curves = {
        policy_name: success_curve_by_token_budget(tuple(policy_results))
        for policy_name, policy_results in grouped.items()
    }
    for curve in curves.values():
        assert_monotone_nondecreasing(curve)
    return plot_success_curve_by_token_budget(
        curves,
        path,
        title="HF toy eval success by token budget",
    )


def model_id_for_result(result: SearchResult) -> str | None:
    """Return the model ID recorded on a result's attempts."""
    for attempt in result.attempts:
        if attempt.model_id:
            return attempt.model_id
        metadata_model_id = attempt.metadata.get("model_id")
        if isinstance(metadata_model_id, str):
            return metadata_model_id
    return None


def _final_cumulative_tokens(result: SearchResult) -> int:
    if not result.attempts:
        return 0
    return result.attempts[-1].cumulative_tokens


def _final_cumulative_verifier_calls(result: SearchResult) -> int:
    if not result.attempts:
        return 0
    return result.attempts[-1].cumulative_verifier_calls


def _median_or_none(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "unknown"


__all__ = [
    "DEFAULT_HF_TOY_POLICIES",
    "DEFAULT_HF_TOY_OUTPUT_ROOT",
    "DEFAULT_HF_TOY_TASK_IDS",
    "HFToyEvalArtifacts",
    "HFToyEvalConfig",
    "ProviderFactory",
    "default_budget_for",
    "default_output_dir_for_run",
    "make_policy",
    "model_id_for_result",
    "run_hf_toy_eval",
    "select_tasks",
    "summarize_results",
    "write_attempts_jsonl",
    "write_success_plot",
    "write_summary_json",
]
