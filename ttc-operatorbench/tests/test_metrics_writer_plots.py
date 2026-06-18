"""Tests for metrics, JSONL logging, and plots."""

from pathlib import Path

import pytest

from ttc_operatorbench.core.schema import Budget, SearchResult, VerificationResult
from ttc_operatorbench.evals.metrics import (
    area_under_success_curve,
    assert_monotone_nondecreasing,
    cost_to_first_solution,
    group_results_by_policy,
    hidden_solve_rate,
    hidden_success_curve_by_token_budget,
    hidden_success_curve_by_verifier_budget,
    median_tokens_to_hidden_solution,
    median_tokens_to_solution,
    overfit_rate,
    public_hidden_gap,
    solve_rate,
    success_curve_by_cost_budget,
    success_curve_by_time_budget,
    success_curve_by_token_budget,
    success_curve_by_verifier_budget,
    tokens_to_first_hidden_solution,
    tokens_to_first_solution,
    verifier_calls_to_first_solution,
    wall_clock_to_first_solution,
)
from ttc_operatorbench.evals.plots import plot_success_curve_by_token_budget
from ttc_operatorbench.logging.writer import read_search_results_jsonl, write_search_results_jsonl
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.baselines import BestOfNPolicy, GreedyPolicy
from ttc_operatorbench.tasks.toy_code import get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

CORRECT_IS_EVEN = "def is_even(n):\n    return n % 2 == 0"
WRONG_IS_EVEN = "def is_even(n):\n    return True"


def run_result(policy_name: str, generations: tuple[str, ...]) -> SearchResult:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: generations})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    policy = GreedyPolicy() if policy_name == "greedy" else BestOfNPolicy(n=3)
    return policy.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=3, max_verifier_calls=3, max_tokens=1_000),
        run_id=policy_name,
    )


def with_hidden_verification(result: SearchResult, *, passed: bool) -> SearchResult:
    attempts = tuple(
        attempt.model_copy(
            update={
                "hidden_verification": VerificationResult(
                    verification_passed=passed and attempt.verification_passed,
                    verification_score=1.0 if passed and attempt.verification_passed else 0.0,
                    scope="hidden",
                    error_type=None if passed and attempt.verification_passed else "test_failure",
                )
            }
        )
        for attempt in result.attempts
    )
    return result.model_copy(update={"attempts": attempts})


def with_attempt_cost(result: SearchResult, cost: float) -> SearchResult:
    attempts = tuple(
        attempt.model_copy(update={"cumulative_cost": cost})
        for attempt in result.attempts
    )
    return result.model_copy(update={"attempts": attempts, "total_cost": cost})


def test_metrics_capture_first_solution_costs() -> None:
    failed = run_result("greedy", (WRONG_IS_EVEN,))
    solved = run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN))
    results = (failed, solved)

    assert solve_rate(results) == 0.5
    assert tokens_to_first_solution(failed) is None
    assert tokens_to_first_solution(solved) == solved.attempts[1].cumulative_tokens
    assert verifier_calls_to_first_solution(solved) == 2
    assert wall_clock_to_first_solution(solved) is not None
    assert median_tokens_to_solution(results) == float(solved.attempts[1].cumulative_tokens)


def test_success_curves_are_monotone_nondecreasing() -> None:
    failed = run_result("greedy", (WRONG_IS_EVEN,))
    solved = run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN))
    budgets = [0, failed.attempts[0].cumulative_tokens, solved.attempts[1].cumulative_tokens]

    token_curve = success_curve_by_token_budget((failed, solved), budgets)
    verifier_curve = success_curve_by_verifier_budget((failed, solved), [0, 1, 2])
    time_curve = success_curve_by_time_budget(
        (failed, solved),
        [0.0, solved.attempts[1].cumulative_seconds],
    )

    assert_monotone_nondecreasing(token_curve)
    assert_monotone_nondecreasing(verifier_curve)
    assert_monotone_nondecreasing(time_curve)
    assert list(token_curve.values()) == sorted(token_curve.values())
    assert area_under_success_curve(token_curve) >= 0.0


def test_cost_success_curve_uses_cumulative_attempt_cost() -> None:
    failed = with_attempt_cost(run_result("greedy", (WRONG_IS_EVEN,)), 1.0)
    solved = with_attempt_cost(run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN)), 3.0)

    curve = success_curve_by_cost_budget((failed, solved), [0.0, 1.0, 3.0])

    assert cost_to_first_solution(failed) is None
    assert cost_to_first_solution(solved) == 3.0
    assert curve == {0.0: 0.0, 1.0: 0.0, 3.0: 0.5}


def test_hidden_metrics_capture_public_hidden_gap_and_overfit() -> None:
    public_only = with_hidden_verification(
        run_result("greedy", (CORRECT_IS_EVEN,)),
        passed=False,
    )
    hidden_solved = with_hidden_verification(
        run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN)),
        passed=True,
    )
    results = (public_only, hidden_solved)

    assert solve_rate(results) == 1.0
    assert hidden_solve_rate(results) == 0.5
    assert public_hidden_gap(results) == 0.5
    assert overfit_rate(results) == 0.5
    assert tokens_to_first_hidden_solution(public_only) is None
    assert (
        tokens_to_first_hidden_solution(hidden_solved)
        == hidden_solved.attempts[1].cumulative_tokens
    )
    assert median_tokens_to_hidden_solution(results) == float(
        hidden_solved.attempts[1].cumulative_tokens
    )


def test_hidden_success_curves_are_monotone_nondecreasing() -> None:
    public_only = with_hidden_verification(
        run_result("greedy", (CORRECT_IS_EVEN,)),
        passed=False,
    )
    hidden_solved = with_hidden_verification(
        run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN)),
        passed=True,
    )
    budgets = [
        0,
        public_only.attempts[0].cumulative_tokens,
        hidden_solved.attempts[1].cumulative_tokens,
    ]

    token_curve = hidden_success_curve_by_token_budget((public_only, hidden_solved), budgets)
    verifier_curve = hidden_success_curve_by_verifier_budget((public_only, hidden_solved), [0, 1])

    assert_monotone_nondecreasing(token_curve)
    assert_monotone_nondecreasing(verifier_curve)


def test_monotone_assertion_rejects_decrease() -> None:
    with pytest.raises(ValueError):
        assert_monotone_nondecreasing({0: 1.0, 1: 0.5})


def test_jsonl_logs_reproduce_metrics(tmp_path: Path) -> None:
    results = (
        run_result("greedy", (WRONG_IS_EVEN,)),
        run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN)),
    )
    path = tmp_path / "toy_eval.jsonl"

    write_search_results_jsonl(path, results)
    reloaded = read_search_results_jsonl(path)

    assert reloaded == results
    assert solve_rate(reloaded) == solve_rate(results)
    assert success_curve_by_token_budget(reloaded) == success_curve_by_token_budget(results)


def test_greedy_vs_best_of_n_plot_exists(tmp_path: Path) -> None:
    results = (
        run_result("greedy", (WRONG_IS_EVEN,)),
        run_result("best_of_n", (WRONG_IS_EVEN, CORRECT_IS_EVEN)),
    )
    grouped = group_results_by_policy(results)
    budgets = sorted(
        {
            attempt.cumulative_tokens
            for policy_results in grouped.values()
            for result in policy_results
            for attempt in result.attempts
        }
    )
    curves = {
        policy_name: success_curve_by_token_budget(policy_results, budgets)
        for policy_name, policy_results in grouped.items()
    }
    output_path = tmp_path / "greedy_vs_best_of_n.png"

    plot_success_curve_by_token_budget(curves, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
