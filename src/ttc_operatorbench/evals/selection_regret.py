"""Selection-regret analysis over immutable, externally graded candidate pools."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from math import comb
from statistics import median
from typing import Annotated, Self

from pydantic import Field, model_validator

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    CandidateRecord,
)
from ttc_operatorbench.core.schema import SchemaModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Rate = Annotated[float, Field(ge=0.0, le=1.0)]

CandidateSelector = Callable[[tuple[CandidateRecord, ...], tuple[bool, ...]], int]


class SelectionObservation(SchemaModel):
    """One task/policy/budget outcome with hidden labels joined post-selection."""

    pool_id: NonEmptyStr
    task_id: NonEmptyStr
    selector_name: NonEmptyStr
    k: PositiveInt
    selected_index: NonNegativeInt
    selected_base_passed: bool
    selected_plus_passed: bool
    correct_candidates_in_pool: NonNegativeInt
    prefix_has_correct_candidate: bool
    unbiased_pass_at_k: Rate
    selection_regret: Rate

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selected_index >= self.k:
            raise ValueError("selected_index must be below k")
        if self.selected_plus_passed and not self.selected_base_passed:
            raise ValueError("plus correctness requires base correctness")
        if self.selected_plus_passed and not self.prefix_has_correct_candidate:
            raise ValueError("a selected correct candidate must be present in the prefix")
        return self


class SelectionSummary(SchemaModel):
    """Aggregate task-level estimate and paired-bootstrap intervals."""

    selector_name: NonEmptyStr
    k: PositiveInt
    task_count: PositiveInt
    selected_base_pass_rate: Rate
    selected_plus_pass_rate: Rate
    selected_plus_ci_low: Rate
    selected_plus_ci_high: Rate
    false_accept_rate: Rate | None
    prefix_oracle_pass_rate: Rate
    unbiased_pass_at_k: Rate
    selection_regret: Rate
    selection_regret_ci_low: Rate
    selection_regret_ci_high: Rate


class SelectorComparison(SchemaModel):
    """Paired task-level difference in hidden correctness between selectors."""

    baseline_selector: NonEmptyStr
    challenger_selector: NonEmptyStr
    k: PositiveInt
    task_count: PositiveInt
    selected_plus_rate_difference: float = Field(ge=-1.0, le=1.0)
    ci_low: float = Field(ge=-1.0, le=1.0)
    ci_high: float = Field(ge=-1.0, le=1.0)


class CoverageGain(SchemaModel):
    """Paired task-level gain in unbiased Pass@k over a reference k."""

    reference_k: PositiveInt
    k: PositiveInt
    task_count: PositiveInt
    unbiased_pass_at_k_gain: Rate
    ci_low: Rate
    ci_high: Rate


class StoppingEfficiency(SchemaModel):
    """Cost of stopping generation at the first public base-test pass."""

    max_k: PositiveInt
    task_count: PositiveInt
    total_candidate_calls: PositiveInt
    fixed_candidate_calls: PositiveInt
    mean_candidate_calls: NonNegativeFloat
    median_candidate_calls: NonNegativeFloat
    used_full_budget_count: NonNegativeInt
    no_base_pass_count: NonNegativeInt
    candidate_call_savings_rate: Rate
    total_tokens: NonNegativeInt
    fixed_total_tokens: NonNegativeInt
    token_savings_rate: Rate
    generation_latency_seconds: NonNegativeFloat
    fixed_generation_latency_seconds: NonNegativeFloat
    stop_counts: dict[str, NonNegativeInt]


class SelectionAnalysis(SchemaModel):
    """Complete observation and summary rows for one pool analysis."""

    observations: tuple[SelectionObservation, ...]
    summaries: tuple[SelectionSummary, ...]
    comparisons: tuple[SelectorComparison, ...]
    coverage_gains: tuple[CoverageGain, ...]
    stopping_efficiency: StoppingEfficiency
    bootstrap_resamples: PositiveInt
    bootstrap_seed: NonNegativeInt


def first_sample_selector(
    candidates: tuple[CandidateRecord, ...],
    base_passes: tuple[bool, ...],
) -> int:
    """Select the first candidate without using verifier outcomes."""
    del base_passes
    if not candidates:
        raise ValueError("selector requires at least one candidate")
    return 0


def first_base_pass_selector(
    candidates: tuple[CandidateRecord, ...],
    base_passes: tuple[bool, ...],
) -> int:
    """Select the first base-passing candidate, falling back to candidate zero."""
    if not candidates or len(candidates) != len(base_passes):
        raise ValueError("candidate and base-pass sequences must be nonempty and aligned")
    return next((index for index, passed in enumerate(base_passes) if passed), 0)


def analyze_selection_regret(
    pool: CandidatePool,
    base_grades: Sequence[CandidateGrade],
    plus_grades: Sequence[CandidateGrade],
    *,
    selectors: Mapping[str, CandidateSelector] | None = None,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
    k_values: Sequence[int] = (1, 2, 4, 8, 16),
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> SelectionAnalysis:
    """Run public-only selectors and join hidden labels after each decision."""
    if bootstrap_resamples <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap settings must be positive/nonnegative")
    active_selectors = selectors or {
        "first_sample": first_sample_selector,
        "first_base_pass": first_base_pass_selector,
    }
    if not active_selectors:
        raise ValueError("at least one selector is required")
    active_comparisons = _comparison_pairs(active_selectors, comparison_pairs)
    base_index = _grade_index(pool, base_grades, scope="base")
    plus_index = _grade_index(pool, plus_grades, scope="plus")
    eligible_k = tuple(sorted({k for k in k_values if 0 < k <= pool.manifest.pool_size}))
    if not eligible_k:
        raise ValueError("no k value is eligible for the candidate pool")

    observations: list[SelectionObservation] = []
    for task_id in pool.manifest.task_ids:
        candidates = pool.candidates_for_task(task_id)
        all_plus = tuple(
            plus_index[(candidate.pool_id, task_id, candidate.candidate_index)].verification_passed
            for candidate in candidates
        )
        correct_count = sum(all_plus)
        for k in eligible_k:
            prefix = candidates[:k]
            prefix_has_correct = any(all_plus[:k])
            base_passes = tuple(
                base_index[
                    (candidate.pool_id, task_id, candidate.candidate_index)
                ].verification_passed
                for candidate in prefix
            )
            oracle = _pass_at_k(pool.manifest.pool_size, correct_count, k)
            for selector_name, selector in active_selectors.items():
                selected_index = selector(prefix, base_passes)
                if not 0 <= selected_index < k:
                    raise ValueError(
                        f"selector {selector_name} returned invalid index {selected_index}"
                    )
                selected = prefix[selected_index]
                selected_plus = plus_index[
                    (selected.pool_id, task_id, selected.candidate_index)
                ].verification_passed
                observations.append(
                    SelectionObservation(
                        pool_id=pool.manifest.pool_id,
                        task_id=task_id,
                        selector_name=selector_name,
                        k=k,
                        selected_index=selected.candidate_index,
                        selected_base_passed=base_passes[selected_index],
                        selected_plus_passed=selected_plus,
                        correct_candidates_in_pool=correct_count,
                        prefix_has_correct_candidate=prefix_has_correct,
                        unbiased_pass_at_k=oracle,
                        selection_regret=float(prefix_has_correct) - float(selected_plus),
                    )
                )
    summaries = _summarize_observations(
        tuple(observations),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    comparisons = _compare_selectors(
        tuple(observations),
        active_comparisons,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    coverage_gains = _coverage_gains(
        tuple(observations),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    stopping_efficiency = _first_base_pass_stopping_efficiency(
        pool,
        base_index,
        max_k=max(eligible_k),
    )
    return SelectionAnalysis(
        observations=tuple(observations),
        summaries=summaries,
        comparisons=comparisons,
        coverage_gains=coverage_gains,
        stopping_efficiency=stopping_efficiency,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _comparison_pairs(
    selectors: Mapping[str, CandidateSelector],
    comparison_pairs: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    pairs = (
        tuple(combinations(selectors, 2))
        if comparison_pairs is None
        else tuple(comparison_pairs)
    )
    for baseline, challenger in pairs:
        if baseline == challenger:
            raise ValueError("selector comparisons require two distinct selectors")
        if baseline not in selectors or challenger not in selectors:
            raise ValueError(f"unknown selector comparison: {(baseline, challenger)}")
    if len(set(pairs)) != len(pairs):
        raise ValueError("selector comparison pairs must be unique")
    return pairs


def _grade_index(
    pool: CandidatePool,
    grades: Sequence[CandidateGrade],
    *,
    scope: str,
) -> dict[tuple[str, str, int], CandidateGrade]:
    expected = {
        (candidate.pool_id, candidate.task_id, candidate.candidate_index): candidate
        for candidate in pool.candidates
    }
    observed: dict[tuple[str, str, int], CandidateGrade] = {}
    for grade in grades:
        if grade.scope != scope:
            raise ValueError(f"expected only {scope} grades")
        candidate = expected.get(grade.key)
        if candidate is None or candidate.sanitized_code_sha256 != grade.sanitized_code_sha256:
            raise ValueError(f"grade does not match candidate pool: {grade.key}")
        if grade.key in observed:
            raise ValueError(f"duplicate grade: {grade.key}")
        observed[grade.key] = grade
    if set(observed) != set(expected):
        raise ValueError(f"{scope} grades must cover the complete candidate pool")
    return observed


def _pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    if correct_count == 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - comb(sample_count - correct_count, k) / comb(sample_count, k)


def _summarize_observations(
    observations: tuple[SelectionObservation, ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[SelectionSummary, ...]:
    grouped: dict[tuple[str, int], list[SelectionObservation]] = {}
    for observation in observations:
        grouped.setdefault((observation.selector_name, observation.k), []).append(observation)

    summaries: list[SelectionSummary] = []
    for group_index, ((selector_name, k), group) in enumerate(sorted(grouped.items())):
        base_values = [float(item.selected_base_passed) for item in group]
        plus_values = [float(item.selected_plus_passed) for item in group]
        prefix_oracle_values = [float(item.prefix_has_correct_candidate) for item in group]
        pass_at_k_values = [item.unbiased_pass_at_k for item in group]
        regret_values = [item.selection_regret for item in group]
        false_accepts = [
            float(not item.selected_plus_passed)
            for item in group
            if item.selected_base_passed
        ]
        plus_interval = _bootstrap_mean_interval(
            plus_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 2 * group_index,
        )
        regret_interval = _bootstrap_mean_interval(
            regret_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 2 * group_index + 1,
        )
        summaries.append(
            SelectionSummary(
                selector_name=selector_name,
                k=k,
                task_count=len(group),
                selected_base_pass_rate=_mean(base_values),
                selected_plus_pass_rate=_mean(plus_values),
                selected_plus_ci_low=plus_interval[0],
                selected_plus_ci_high=plus_interval[1],
                false_accept_rate=_mean(false_accepts) if false_accepts else None,
                prefix_oracle_pass_rate=_mean(prefix_oracle_values),
                unbiased_pass_at_k=_mean(pass_at_k_values),
                selection_regret=_mean(regret_values),
                selection_regret_ci_low=regret_interval[0],
                selection_regret_ci_high=regret_interval[1],
            )
        )
    return tuple(summaries)


def _compare_selectors(
    observations: tuple[SelectionObservation, ...],
    comparison_pairs: tuple[tuple[str, str], ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[SelectorComparison, ...]:
    by_selector_k_task = {
        (item.selector_name, item.k, item.task_id): item for item in observations
    }
    k_values = sorted({item.k for item in observations})
    task_ids = sorted({item.task_id for item in observations})
    comparisons_out: list[SelectorComparison] = []
    for pair_index, (baseline, challenger) in enumerate(comparison_pairs):
        for k_index, k in enumerate(k_values):
            differences: list[float] = []
            for task_id in task_ids:
                try:
                    baseline_item = by_selector_k_task[(baseline, k, task_id)]
                    challenger_item = by_selector_k_task[(challenger, k, task_id)]
                except KeyError as error:
                    raise ValueError("selector observations are not task-paired") from error
                differences.append(
                    float(challenger_item.selected_plus_passed)
                    - float(baseline_item.selected_plus_passed)
                )
            interval = _bootstrap_mean_interval(
                differences,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 1_000_000 + pair_index * len(k_values) + k_index,
            )
            comparisons_out.append(
                SelectorComparison(
                    baseline_selector=baseline,
                    challenger_selector=challenger,
                    k=k,
                    task_count=len(differences),
                    selected_plus_rate_difference=_mean(differences),
                    ci_low=interval[0],
                    ci_high=interval[1],
                )
            )
    return tuple(comparisons_out)


def _coverage_gains(
    observations: tuple[SelectionObservation, ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[CoverageGain, ...]:
    coverage_by_task_k: dict[tuple[str, int], float] = {}
    for observation in observations:
        key = (observation.task_id, observation.k)
        previous = coverage_by_task_k.setdefault(key, observation.unbiased_pass_at_k)
        if previous != observation.unbiased_pass_at_k:
            raise ValueError("selectors disagree on task-level Pass@k")
    task_ids = sorted({task_id for task_id, _ in coverage_by_task_k})
    k_values = sorted({k for _, k in coverage_by_task_k})
    reference_k = k_values[0]
    gains: list[CoverageGain] = []
    for k_index, k in enumerate(k_values[1:]):
        differences = [
            coverage_by_task_k[(task_id, k)]
            - coverage_by_task_k[(task_id, reference_k)]
            for task_id in task_ids
        ]
        interval = _bootstrap_mean_interval(
            differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 2_000_000 + k_index,
        )
        gains.append(
            CoverageGain(
                reference_k=reference_k,
                k=k,
                task_count=len(task_ids),
                unbiased_pass_at_k_gain=_mean(differences),
                ci_low=interval[0],
                ci_high=interval[1],
            )
        )
    return tuple(gains)


def _first_base_pass_stopping_efficiency(
    pool: CandidatePool,
    base_index: Mapping[tuple[str, str, int], CandidateGrade],
    *,
    max_k: int,
) -> StoppingEfficiency:
    calls: list[int] = []
    token_costs: list[int] = []
    latency_costs: list[float] = []
    fixed_total_tokens = 0
    fixed_total_latency = 0.0
    no_base_pass_count = 0
    for task_id in pool.manifest.task_ids:
        candidates = pool.candidates_for_task(task_id)[:max_k]
        first_pass = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if base_index[
                    (candidate.pool_id, candidate.task_id, candidate.candidate_index)
                ].verification_passed
            ),
            None,
        )
        used = max_k if first_pass is None else first_pass + 1
        no_base_pass_count += first_pass is None
        calls.append(used)
        token_costs.append(
            sum(candidate.generation.total_tokens for candidate in candidates[:used])
        )
        latency_costs.append(
            sum(candidate.generation.latency_seconds for candidate in candidates[:used])
        )
        fixed_total_tokens += sum(candidate.generation.total_tokens for candidate in candidates)
        fixed_total_latency += sum(
            candidate.generation.latency_seconds for candidate in candidates
        )
    total_calls = sum(calls)
    fixed_calls = max_k * len(calls)
    total_tokens = sum(token_costs)
    stop_counts = {str(value): calls.count(value) for value in sorted(set(calls))}
    return StoppingEfficiency(
        max_k=max_k,
        task_count=len(calls),
        total_candidate_calls=total_calls,
        fixed_candidate_calls=fixed_calls,
        mean_candidate_calls=total_calls / len(calls),
        median_candidate_calls=float(median(calls)),
        used_full_budget_count=calls.count(max_k),
        no_base_pass_count=no_base_pass_count,
        candidate_call_savings_rate=1.0 - total_calls / fixed_calls,
        total_tokens=total_tokens,
        fixed_total_tokens=fixed_total_tokens,
        token_savings_rate=1.0 - total_tokens / fixed_total_tokens,
        generation_latency_seconds=sum(latency_costs),
        fixed_generation_latency_seconds=fixed_total_latency,
        stop_counts=stop_counts,
    )


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
    "CandidateSelector",
    "SelectionAnalysis",
    "SelectionObservation",
    "SelectionSummary",
    "analyze_selection_regret",
    "first_base_pass_selector",
    "first_sample_selector",
]
