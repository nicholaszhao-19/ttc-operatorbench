"""Task-level hidden-correctness and cost analysis for stopped trajectories."""

from __future__ import annotations

import random
from collections.abc import Sequence
from statistics import median
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ttc_operatorbench.core.candidate_pool import CandidateGrade
from ttc_operatorbench.core.schema import SchemaModel
from ttc_operatorbench.core.trajectory import (
    TrajectoryOperator,
    WidthDepthTrajectoryPool,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Rate = Annotated[float, Field(ge=0.0, le=1.0)]
ConfirmationOutcome = Literal[
    "strong_confirmation",
    "suggestive_only",
    "failed_confirmation",
]


class TrajectoryTaskObservation(SchemaModel):
    """One task outcome after hidden labels are joined post-search."""

    pool_id: NonEmptyStr
    task_id: NonEmptyStr
    width: PositiveInt
    depth: PositiveInt
    calls: PositiveInt
    generation_tokens: NonNegativeInt
    generation_latency_seconds: NonNegativeFloat
    repair_calls: NonNegativeInt
    public_passed: bool
    selected_candidate_index: NonNegativeInt
    selected_operator: TrajectoryOperator
    hidden_passed: bool
    false_accept: bool

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.hidden_passed and not self.public_passed:
            raise ValueError("hidden correctness requires a public base pass")
        if self.false_accept != (self.public_passed and not self.hidden_passed):
            raise ValueError("false_accept must match public and hidden outcomes")
        if self.repair_calls > self.calls:
            raise ValueError("repair calls cannot exceed total calls")
        return self


class TrajectoryPolicySummary(SchemaModel):
    """Aggregate correctness, uncertainty, and realized generation cost."""

    pool_id: NonEmptyStr
    width: PositiveInt
    depth: PositiveInt
    task_count: PositiveInt
    maximum_calls_per_task: PositiveInt
    public_pass_rate: Rate
    hidden_pass_rate: Rate
    hidden_pass_ci_low: Rate
    hidden_pass_ci_high: Rate
    false_accept_rate: Rate | None
    selected_by_repair_count: NonNegativeInt
    hidden_solved_by_repair_count: NonNegativeInt
    total_calls: PositiveInt
    mean_calls: NonNegativeFloat
    median_calls: NonNegativeFloat
    full_budget_task_count: NonNegativeInt
    fixed_n16_call_fraction: Rate
    total_generation_tokens: NonNegativeInt
    mean_generation_tokens: NonNegativeFloat
    generation_latency_seconds: NonNegativeFloat
    empty_candidate_rate: Rate
    possible_truncation_rate: Rate


class TrajectoryPolicyAnalysis(SchemaModel):
    """Task rows and summary for one fixed width-depth policy."""

    observations: tuple[TrajectoryTaskObservation, ...]
    summary: TrajectoryPolicySummary
    bootstrap_resamples: PositiveInt
    bootstrap_seed: NonNegativeInt


class TrajectoryPolicyComparison(SchemaModel):
    """Paired task-level challenger difference from a baseline policy."""

    baseline_pool_id: NonEmptyStr
    challenger_pool_id: NonEmptyStr
    task_count: PositiveInt
    hidden_pass_rate_difference: float = Field(ge=-1.0, le=1.0)
    hidden_pass_ci_low: float = Field(ge=-1.0, le=1.0)
    hidden_pass_ci_high: float = Field(ge=-1.0, le=1.0)
    mean_call_difference: float
    mean_call_ci_low: float
    mean_call_ci_high: float
    mean_token_difference: float
    mean_token_ci_low: float
    mean_token_ci_high: float
    hidden_win_count: NonNegativeInt
    hidden_loss_count: NonNegativeInt
    hidden_tie_count: NonNegativeInt
    meets_engineering_gate: bool


def analyze_width_depth_trajectory(
    pool: WidthDepthTrajectoryPool,
    plus_grades: Sequence[CandidateGrade],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> TrajectoryPolicyAnalysis:
    """Join hidden labels after routing and summarize task-level outcomes."""
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap settings must be positive/nonnegative")
    plus_index = _plus_grade_index(pool, plus_grades)
    manifest = pool.header.candidate_manifest
    observations: list[TrajectoryTaskObservation] = []
    empty_candidates = 0
    possible_truncations = 0
    for task_id in manifest.task_ids:
        steps = pool.steps_for_task(task_id)
        selected = next((step for step in steps if step.selected), steps[0])
        public_passed = selected.selected
        plus_grade = plus_index[
            (
                selected.candidate.pool_id,
                selected.candidate.task_id,
                selected.candidate.candidate_index,
            )
        ]
        hidden_passed = plus_grade.verification_passed
        empty_candidates += sum(not step.candidate.sanitized_code.strip() for step in steps)
        possible_truncations += sum(
            step.candidate.generation.output_tokens >= manifest.max_output_tokens
            for step in steps
        )
        observations.append(
            TrajectoryTaskObservation(
                pool_id=manifest.pool_id,
                task_id=task_id,
                width=pool.header.width,
                depth=pool.header.depth,
                calls=len(steps),
                generation_tokens=sum(
                    step.candidate.generation.total_tokens for step in steps
                ),
                generation_latency_seconds=sum(
                    step.candidate.generation.latency_seconds for step in steps
                ),
                repair_calls=sum(step.operator == "repair" for step in steps),
                public_passed=public_passed,
                selected_candidate_index=selected.candidate.candidate_index,
                selected_operator=selected.operator,
                hidden_passed=hidden_passed,
                false_accept=public_passed and not hidden_passed,
            )
        )
    rows = tuple(observations)
    hidden_values = [float(row.hidden_passed) for row in rows]
    interval = _bootstrap_mean_interval(
        hidden_values,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    public_rows = [row for row in rows if row.public_passed]
    calls = [row.calls for row in rows]
    tokens = [row.generation_tokens for row in rows]
    candidate_count = sum(calls)
    summary = TrajectoryPolicySummary(
        pool_id=manifest.pool_id,
        width=pool.header.width,
        depth=pool.header.depth,
        task_count=len(rows),
        maximum_calls_per_task=manifest.pool_size,
        public_pass_rate=_mean([float(row.public_passed) for row in rows]),
        hidden_pass_rate=_mean(hidden_values),
        hidden_pass_ci_low=interval[0],
        hidden_pass_ci_high=interval[1],
        false_accept_rate=(
            _mean([float(row.false_accept) for row in public_rows])
            if public_rows
            else None
        ),
        selected_by_repair_count=sum(
            row.public_passed and row.selected_operator == "repair" for row in rows
        ),
        hidden_solved_by_repair_count=sum(
            row.hidden_passed and row.selected_operator == "repair" for row in rows
        ),
        total_calls=sum(calls),
        mean_calls=_mean([float(value) for value in calls]),
        median_calls=float(median(calls)),
        full_budget_task_count=calls.count(manifest.pool_size),
        fixed_n16_call_fraction=min(1.0, sum(calls) / (16 * len(rows))),
        total_generation_tokens=sum(tokens),
        mean_generation_tokens=_mean([float(value) for value in tokens]),
        generation_latency_seconds=sum(
            row.generation_latency_seconds for row in rows
        ),
        empty_candidate_rate=empty_candidates / candidate_count,
        possible_truncation_rate=possible_truncations / candidate_count,
    )
    return TrajectoryPolicyAnalysis(
        observations=rows,
        summary=summary,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def compare_trajectory_policies(
    baseline: TrajectoryPolicyAnalysis,
    challenger: TrajectoryPolicyAnalysis,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> TrajectoryPolicyComparison:
    """Return paired challenger-minus-baseline correctness and cost differences."""
    baseline_rows = {row.task_id: row for row in baseline.observations}
    challenger_rows = {row.task_id: row for row in challenger.observations}
    if set(baseline_rows) != set(challenger_rows):
        raise ValueError("policy analyses must contain the same task IDs")
    task_ids = sorted(baseline_rows)
    hidden_differences = [
        float(challenger_rows[task_id].hidden_passed)
        - float(baseline_rows[task_id].hidden_passed)
        for task_id in task_ids
    ]
    call_differences = [
        float(challenger_rows[task_id].calls - baseline_rows[task_id].calls)
        for task_id in task_ids
    ]
    token_differences = [
        float(
            challenger_rows[task_id].generation_tokens
            - baseline_rows[task_id].generation_tokens
        )
        for task_id in task_ids
    ]
    hidden_interval = _bootstrap_mean_interval(
        hidden_differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    call_interval = _bootstrap_mean_interval(
        call_differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1,
    )
    token_interval = _bootstrap_mean_interval(
        token_differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 2,
    )
    hidden_difference = _mean(hidden_differences)
    return TrajectoryPolicyComparison(
        baseline_pool_id=baseline.summary.pool_id,
        challenger_pool_id=challenger.summary.pool_id,
        task_count=len(task_ids),
        hidden_pass_rate_difference=hidden_difference,
        hidden_pass_ci_low=hidden_interval[0],
        hidden_pass_ci_high=hidden_interval[1],
        mean_call_difference=_mean(call_differences),
        mean_call_ci_low=call_interval[0],
        mean_call_ci_high=call_interval[1],
        mean_token_difference=_mean(token_differences),
        mean_token_ci_low=token_interval[0],
        mean_token_ci_high=token_interval[1],
        hidden_win_count=sum(value > 0 for value in hidden_differences),
        hidden_loss_count=sum(value < 0 for value in hidden_differences),
        hidden_tie_count=sum(value == 0 for value in hidden_differences),
        meets_engineering_gate=(
            hidden_difference >= 0.03 and challenger.summary.mean_calls <= 8.0
        ),
    )


def classify_confirmation(
    comparison: TrajectoryPolicyComparison,
) -> ConfirmationOutcome:
    """Apply the frozen confirmation rule to one paired comparison."""
    if (
        comparison.hidden_pass_rate_difference > 0.0
        and comparison.hidden_pass_ci_low > 0.0
    ):
        return "strong_confirmation"
    if comparison.hidden_pass_rate_difference > 0.0:
        return "suggestive_only"
    return "failed_confirmation"


def validate_comparable_trajectory_pools(
    pools: Sequence[WidthDepthTrajectoryPool],
) -> None:
    """Require matched data, model, sampling, hardware, and repository provenance."""
    if len(pools) < 2:
        raise ValueError("at least two trajectory pools are required")
    baseline = pools[0].header.candidate_manifest
    matched_fields = (
        "dataset_name",
        "dataset_version",
        "dataset_sha256",
        "repository_commit",
        "task_ids",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "provider_name",
        "prompt_style",
        "temperature",
        "top_p",
        "max_output_tokens",
        "pool_seed",
        "hardware",
        "dependencies",
    )
    for pool in pools[1:]:
        manifest = pool.header.candidate_manifest
        mismatched = [
            field
            for field in matched_fields
            if getattr(manifest, field) != getattr(baseline, field)
        ]
        if mismatched:
            raise ValueError(f"trajectory pools are not comparable: {mismatched}")


def development_winner(
    analyses: Sequence[TrajectoryPolicyAnalysis],
) -> TrajectoryPolicyAnalysis:
    """Apply the preregistered deterministic development tie-break rule."""
    if not analyses:
        raise ValueError("at least one policy analysis is required")
    return min(
        analyses,
        key=lambda analysis: (
            -analysis.summary.hidden_pass_rate,
            analysis.summary.total_generation_tokens,
            analysis.summary.total_calls,
            -analysis.summary.width,
            analysis.summary.depth,
            analysis.summary.pool_id,
        ),
    )


def _plus_grade_index(
    pool: WidthDepthTrajectoryPool,
    plus_grades: Sequence[CandidateGrade],
) -> dict[tuple[str, str, int], CandidateGrade]:
    expected = {
        (
            step.candidate.pool_id,
            step.candidate.task_id,
            step.candidate.candidate_index,
        ): step.candidate
        for step in pool.steps
    }
    observed: dict[tuple[str, str, int], CandidateGrade] = {}
    for grade in plus_grades:
        candidate = expected.get(grade.key)
        if grade.scope != "plus" or candidate is None:
            raise ValueError(f"unexpected hidden grade: {grade.key}")
        if grade.sanitized_code_sha256 != candidate.sanitized_code_sha256:
            raise ValueError(f"hidden grade digest mismatch: {grade.key}")
        if grade.key in observed:
            raise ValueError(f"duplicate hidden grade: {grade.key}")
        observed[grade.key] = grade
    if set(observed) != set(expected):
        raise ValueError("hidden grades must cover every trajectory candidate exactly")
    return observed


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    return means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


__all__ = [
    "ConfirmationOutcome",
    "TrajectoryPolicyAnalysis",
    "TrajectoryPolicyComparison",
    "TrajectoryPolicySummary",
    "TrajectoryTaskObservation",
    "analyze_width_depth_trajectory",
    "classify_confirmation",
    "compare_trajectory_policies",
    "development_winner",
    "validate_comparable_trajectory_pools",
]
