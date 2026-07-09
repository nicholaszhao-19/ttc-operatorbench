"""Metrics for verifier-guided search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import median

from ttc_operatorbench.core.schema import AttemptLog, SearchResult

BudgetCurve = dict[int, float]
TimeBudgetCurve = dict[float, float]
SuccessCurve = Mapping[int, float] | Mapping[float, float]


def _selected_attempt(result: SearchResult) -> AttemptLog | None:
    if result.selected_attempt_id is None:
        return None
    for attempt in result.attempts:
        if attempt.attempt_id == result.selected_attempt_id:
            return attempt
    return None


def _hidden_passed(attempt: AttemptLog | None) -> bool:
    return bool(
        attempt is not None
        and attempt.hidden_verification is not None
        and attempt.hidden_verification.verification_passed
    )


def _count_results(results: Sequence[SearchResult]) -> int:
    return len(results)


def solve_rate(results: Sequence[SearchResult]) -> float:
    """Return the fraction of tasks solved."""
    total = _count_results(results)
    if total == 0:
        return 0.0
    solved = sum(1 for result in results if result.success)
    return solved / total


def public_solve_rate(results: Sequence[SearchResult]) -> float:
    """Return the public-test solve rate used by policy-visible feedback."""
    return solve_rate(results)


def _first_oracle_hidden_attempt(result: SearchResult) -> AttemptLog | None:
    for attempt in result.attempts:
        if _hidden_passed(attempt):
            return attempt
    return None


def hidden_success(result: SearchResult) -> bool:
    """Return whether the selected answer passed hidden evaluation tests."""
    return _hidden_passed(_selected_attempt(result))


def oracle_hidden_success(result: SearchResult) -> bool:
    """Return whether any generated attempt passed hidden evaluation tests."""
    return _first_oracle_hidden_attempt(result) is not None


def hidden_solve_rate(results: Sequence[SearchResult]) -> float:
    """Return the fraction of tasks whose selected answer passed hidden tests."""
    total = _count_results(results)
    if total == 0:
        return 0.0
    solved = sum(1 for result in results if hidden_success(result))
    return solved / total


def oracle_hidden_solve_rate(results: Sequence[SearchResult]) -> float:
    """Return the fraction of tasks with any hidden-test-passing attempt."""
    total = _count_results(results)
    if total == 0:
        return 0.0
    solved = sum(1 for result in results if oracle_hidden_success(result))
    return solved / total


def public_hidden_gap(results: Sequence[SearchResult]) -> float:
    """Return public solve rate minus hidden solve rate."""
    return public_solve_rate(results) - hidden_solve_rate(results)


def overfit_rate(results: Sequence[SearchResult]) -> float:
    """Return fraction of tasks solved publicly without a hidden-test pass."""
    total = _count_results(results)
    if total == 0:
        return 0.0
    overfit = sum(1 for result in results if result.success and not hidden_success(result))
    return overfit / total


def tokens_to_first_solution(result: SearchResult) -> int | None:
    """Return cumulative tokens at the selected solution, if solved."""
    if not result.success:
        return None
    attempt = _selected_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_tokens


def tokens_to_first_hidden_solution(result: SearchResult) -> int | None:
    """Return cumulative tokens at selected hidden-test-passing answer, if any."""
    attempt = _selected_attempt(result)
    if attempt is None or not _hidden_passed(attempt):
        return None
    return attempt.cumulative_tokens


def tokens_to_first_oracle_hidden_solution(result: SearchResult) -> int | None:
    """Return cumulative tokens at first hidden-test-passing attempt, if any."""
    attempt = _first_oracle_hidden_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_tokens


def verifier_calls_to_first_solution(result: SearchResult) -> int | None:
    """Return cumulative verifier calls at the selected solution, if solved."""
    if not result.success:
        return None
    attempt = _selected_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_verifier_calls


def verifier_calls_to_first_hidden_solution(result: SearchResult) -> int | None:
    """Return verifier calls at selected hidden-test-passing answer, if any."""
    attempt = _selected_attempt(result)
    if attempt is None or not _hidden_passed(attempt):
        return None
    return attempt.cumulative_verifier_calls


def verifier_calls_to_first_oracle_hidden_solution(result: SearchResult) -> int | None:
    """Return verifier calls at first hidden-test-passing attempt, if any."""
    attempt = _first_oracle_hidden_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_verifier_calls


def wall_clock_to_first_solution(result: SearchResult) -> float | None:
    """Return cumulative seconds at the selected solution, if solved."""
    if not result.success:
        return None
    attempt = _selected_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_seconds


def wall_clock_to_first_hidden_solution(result: SearchResult) -> float | None:
    """Return cumulative seconds at selected hidden-test-passing answer, if any."""
    attempt = _selected_attempt(result)
    if attempt is None or not _hidden_passed(attempt):
        return None
    return attempt.cumulative_seconds


def wall_clock_to_first_oracle_hidden_solution(result: SearchResult) -> float | None:
    """Return cumulative seconds at first hidden-test-passing attempt, if any."""
    attempt = _first_oracle_hidden_attempt(result)
    if attempt is None:
        return None
    return attempt.cumulative_seconds


def _unique_int_budgets(values: Iterable[int]) -> tuple[int, ...]:
    budgets = sorted({0, *values})
    return tuple(budget for budget in budgets if budget >= 0)


def _unique_time_budgets(values: Iterable[float]) -> tuple[float, ...]:
    budgets = sorted({0.0, *values})
    return tuple(budget for budget in budgets if budget >= 0.0)


def _attempt_token_budgets(results: Sequence[SearchResult]) -> tuple[int, ...]:
    return _unique_int_budgets(
        attempt.cumulative_tokens for result in results for attempt in result.attempts
    )


def _attempt_verifier_budgets(results: Sequence[SearchResult]) -> tuple[int, ...]:
    return _unique_int_budgets(
        attempt.cumulative_verifier_calls for result in results for attempt in result.attempts
    )


def _attempt_time_budgets(results: Sequence[SearchResult]) -> tuple[float, ...]:
    return _unique_time_budgets(
        attempt.cumulative_seconds for result in results for attempt in result.attempts
    )


def success_curve_by_token_budget(
    results: Sequence[SearchResult],
    budgets: Sequence[int] | None = None,
) -> BudgetCurve:
    """Return token-budget success curve as ``budget -> fraction solved``."""
    if not results:
        return {}
    curve_budgets = _unique_int_budgets(budgets or _attempt_token_budgets(results))
    first_solution_tokens = [tokens_to_first_solution(result) for result in results]
    return {
        budget: sum(
            1
            for solution_tokens in first_solution_tokens
            if solution_tokens is not None and solution_tokens <= budget
        )
        / len(results)
        for budget in curve_budgets
    }


def hidden_success_curve_by_token_budget(
    results: Sequence[SearchResult],
    budgets: Sequence[int] | None = None,
) -> BudgetCurve:
    """Return hidden-test success curve as ``token_budget -> fraction solved``."""
    if not results:
        return {}
    curve_budgets = _unique_int_budgets(budgets or _attempt_token_budgets(results))
    first_solution_tokens = [tokens_to_first_hidden_solution(result) for result in results]
    return {
        budget: sum(
            1
            for solution_tokens in first_solution_tokens
            if solution_tokens is not None and solution_tokens <= budget
        )
        / len(results)
        for budget in curve_budgets
    }


def success_curve_by_verifier_budget(
    results: Sequence[SearchResult],
    budgets: Sequence[int] | None = None,
) -> BudgetCurve:
    """Return verifier-call-budget success curve as ``budget -> fraction solved``."""
    if not results:
        return {}
    curve_budgets = _unique_int_budgets(budgets or _attempt_verifier_budgets(results))
    first_solution_calls = [verifier_calls_to_first_solution(result) for result in results]
    return {
        budget: sum(
            1
            for solution_calls in first_solution_calls
            if solution_calls is not None and solution_calls <= budget
        )
        / len(results)
        for budget in curve_budgets
    }


def hidden_success_curve_by_verifier_budget(
    results: Sequence[SearchResult],
    budgets: Sequence[int] | None = None,
) -> BudgetCurve:
    """Return hidden-test success curve as ``verifier_budget -> fraction solved``."""
    if not results:
        return {}
    curve_budgets = _unique_int_budgets(budgets or _attempt_verifier_budgets(results))
    first_solution_calls = [verifier_calls_to_first_hidden_solution(result) for result in results]
    return {
        budget: sum(
            1
            for solution_calls in first_solution_calls
            if solution_calls is not None and solution_calls <= budget
        )
        / len(results)
        for budget in curve_budgets
    }


def success_curve_by_time_budget(
    results: Sequence[SearchResult],
    budgets: Sequence[float] | None = None,
) -> TimeBudgetCurve:
    """Return wall-clock-budget success curve as ``budget -> fraction solved``."""
    if not results:
        return {}
    curve_budgets = _unique_time_budgets(budgets or _attempt_time_budgets(results))
    first_solution_seconds = [wall_clock_to_first_solution(result) for result in results]
    return {
        budget: sum(
            1
            for solution_seconds in first_solution_seconds
            if solution_seconds is not None and solution_seconds <= budget
        )
        / len(results)
        for budget in curve_budgets
    }


def area_under_success_curve(curve: SuccessCurve) -> float:
    """Return trapezoidal area under a success curve."""
    points = sorted((float(budget), success_fraction) for budget, success_fraction in curve.items())
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (left_budget, left_value), (right_budget, right_value) in zip(
        points,
        points[1:],
        strict=False,
    ):
        area += (right_budget - left_budget) * (left_value + right_value) / 2.0
    return area


def median_tokens_to_solution(results: Sequence[SearchResult]) -> float | None:
    """Return median cumulative tokens for solved tasks."""
    solved_tokens = [
        solution_tokens
        for result in results
        if (solution_tokens := tokens_to_first_solution(result)) is not None
    ]
    if not solved_tokens:
        return None
    return float(median(solved_tokens))


def median_tokens_to_hidden_solution(results: Sequence[SearchResult]) -> float | None:
    """Return median cumulative tokens for hidden-solved tasks."""
    solved_tokens = [
        solution_tokens
        for result in results
        if (solution_tokens := tokens_to_first_hidden_solution(result)) is not None
    ]
    if not solved_tokens:
        return None
    return float(median(solved_tokens))


def group_results_by_policy(results: Sequence[SearchResult]) -> dict[str, tuple[SearchResult, ...]]:
    """Group search results by policy name."""
    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_name, []).append(result)
    return {policy_name: tuple(policy_results) for policy_name, policy_results in grouped.items()}


def assert_monotone_nondecreasing(curve: SuccessCurve) -> None:
    """Raise ValueError if a success curve decreases."""
    previous_value = 0.0
    for _, value in sorted(curve.items()):
        if value < previous_value:
            raise ValueError("success curve must be monotone nondecreasing")
        previous_value = value


__all__ = [
    "BudgetCurve",
    "SuccessCurve",
    "TimeBudgetCurve",
    "area_under_success_curve",
    "assert_monotone_nondecreasing",
    "group_results_by_policy",
    "hidden_solve_rate",
    "hidden_success",
    "hidden_success_curve_by_token_budget",
    "hidden_success_curve_by_verifier_budget",
    "median_tokens_to_solution",
    "median_tokens_to_hidden_solution",
    "oracle_hidden_solve_rate",
    "oracle_hidden_success",
    "overfit_rate",
    "public_hidden_gap",
    "public_solve_rate",
    "solve_rate",
    "success_curve_by_time_budget",
    "success_curve_by_token_budget",
    "success_curve_by_verifier_budget",
    "tokens_to_first_solution",
    "tokens_to_first_hidden_solution",
    "tokens_to_first_oracle_hidden_solution",
    "verifier_calls_to_first_solution",
    "verifier_calls_to_first_hidden_solution",
    "verifier_calls_to_first_oracle_hidden_solution",
    "wall_clock_to_first_solution",
    "wall_clock_to_first_hidden_solution",
    "wall_clock_to_first_oracle_hidden_solution",
]
