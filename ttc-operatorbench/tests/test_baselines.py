"""Tests for verifier-guided baseline policies."""

import pytest

from ttc_operatorbench.core.schema import Budget, Generation, SamplingConfig, SearchResult, Task
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.baselines import (
    BaselinePolicy,
    BestOfNPolicy,
    GreedyPolicy,
    LocalRevisionBasicPolicy,
    PlanThenCodePolicy,
    RepairOnlyPolicy,
    best_of_n_success_probability,
)
from ttc_operatorbench.tasks.toy_code import ToyTaskId, get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

CORRECT_IS_EVEN = "def is_even(n):\n    return n % 2 == 0"
WRONG_IS_EVEN = "def is_even(n):\n    return True"
PLAN = "Use modulo by two and return a boolean."


class SeedRecordingProvider:
    """Provider that records policy-supplied sampling seeds."""

    seed = 123

    def __init__(self) -> None:
        self.seeds_by_task: dict[str, list[int | None]] = {}

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        sampling_config = sampling or SamplingConfig()
        self.seeds_by_task.setdefault(task.task_id, []).append(sampling_config.seed)
        return Generation(
            prompt=task.prompt,
            generation_text=f"def {task.metadata['entrypoint']}(*args):\n    return None",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_seconds=0.0,
            model_name="seed-recorder",
            provider_name="seed-recorder",
            metadata={"seed": sampling_config.seed},
        )


def run_policy(
    policy: BaselinePolicy,
    generations: tuple[str, ...],
    budget: Budget | None = None,
) -> SearchResult:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: generations})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    return policy.run(
        task,
        provider,
        verifier,
        budget or Budget(max_attempts=5, max_verifier_calls=5, max_tokens=1_000),
    )


def assert_budget_respected(result: SearchResult) -> None:
    if result.budget.max_attempts is not None:
        assert len(result.attempts) <= result.budget.max_attempts
    for attempt in result.attempts:
        if result.budget.max_tokens is not None:
            assert attempt.cumulative_tokens <= result.budget.max_tokens
        if result.budget.max_verifier_calls is not None:
            assert attempt.cumulative_verifier_calls <= result.budget.max_verifier_calls
        if result.budget.max_seconds is not None:
            assert attempt.cumulative_seconds <= result.budget.max_seconds


def assert_all_attempts_logged(result: SearchResult, expected_attempts: int) -> None:
    assert len(result.attempts) == expected_attempts
    assert all(attempt.task_id == result.task_id for attempt in result.attempts)
    assert all(attempt.policy_name == result.policy_name for attempt in result.attempts)
    assert all(attempt.provider_name == "dummy" for attempt in result.attempts)


def test_best_of_n_success_probability() -> None:
    assert best_of_n_success_probability(0.25, 3) == pytest.approx(1.0 - 0.75**3)

    with pytest.raises(ValueError):
        best_of_n_success_probability(-0.1, 3)

    with pytest.raises(ValueError):
        best_of_n_success_probability(0.5, 0)


def test_greedy_one_generation_one_verification() -> None:
    result = run_policy(GreedyPolicy(), (CORRECT_IS_EVEN,))

    assert result.success is True
    assert result.policy_name == "greedy"
    assert_all_attempts_logged(result, expected_attempts=1)
    assert result.attempts[0].verification_passed is True
    assert result.attempts[0].cumulative_verifier_calls == 1
    assert_budget_respected(result)


def test_best_of_n_logs_until_first_success() -> None:
    result = run_policy(BestOfNPolicy(n=3), (WRONG_IS_EVEN, CORRECT_IS_EVEN))

    assert result.success is True
    assert result.policy_name == "best_of_n"
    assert_all_attempts_logged(result, expected_attempts=2)
    assert [attempt.verification_passed for attempt in result.attempts] == [False, True]
    assert_budget_respected(result)


def test_repair_only_uses_feedback_and_repairs() -> None:
    result = run_policy(RepairOnlyPolicy(max_repairs=1), (WRONG_IS_EVEN, CORRECT_IS_EVEN))

    assert result.success is True
    assert_all_attempts_logged(result, expected_attempts=2)
    assert result.attempts[0].operator_name == "repair_only/draft"
    assert result.attempts[1].operator_name == "repair_only/repair"
    assert "Verifier error type: test_failure" in result.attempts[1].prompt
    assert_budget_respected(result)


def test_plan_then_code_logs_plan_and_code() -> None:
    result = run_policy(PlanThenCodePolicy(), (PLAN, CORRECT_IS_EVEN))

    assert result.success is True
    assert_all_attempts_logged(result, expected_attempts=2)
    assert result.attempts[0].operator_name == "plan_then_code/plan"
    assert result.attempts[0].error_type == "not_verified_plan"
    assert result.attempts[0].cumulative_verifier_calls == 0
    assert result.attempts[1].operator_name == "plan_then_code/code"
    assert "Plan:" in result.attempts[1].prompt
    assert_budget_respected(result)


def test_local_revision_basic_revises_failed_candidate() -> None:
    result = run_policy(
        LocalRevisionBasicPolicy(max_revisions=1),
        (WRONG_IS_EVEN, CORRECT_IS_EVEN),
    )

    assert result.success is True
    assert_all_attempts_logged(result, expected_attempts=2)
    assert result.attempts[0].operator_name == "local_revision_basic/draft"
    assert result.attempts[1].operator_name == "local_revision_basic/revise"
    assert "Current local candidate:" in result.attempts[1].prompt
    assert_budget_respected(result)


def test_policy_stops_at_attempt_budget() -> None:
    result = run_policy(
        BestOfNPolicy(n=3),
        (WRONG_IS_EVEN, CORRECT_IS_EVEN),
        Budget(max_attempts=1, max_verifier_calls=5, max_tokens=1_000),
    )

    assert result.success is False
    assert_all_attempts_logged(result, expected_attempts=1)
    assert_budget_respected(result)


def test_policy_attempt_seeds_are_stable_across_task_order() -> None:
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    budget = Budget(max_attempts=2, max_verifier_calls=2, max_tokens=1_000)

    def run_order(task_ids: tuple[ToyTaskId, ...]) -> dict[str, list[int | None]]:
        provider = SeedRecordingProvider()
        policy = BestOfNPolicy(n=2)
        for task_id in task_ids:
            policy.run(
                get_toy_task(task_id),
                provider,
                verifier,
                budget,
                run_id="seed_protocol:seed_recorder:seed_123:two_call",
            )
        return provider.seeds_by_task

    first_order = run_order(("is_even", "factorial"))
    reversed_order = run_order(("factorial", "is_even"))

    assert first_order["is_even"] == reversed_order["is_even"]
    assert first_order["factorial"] == reversed_order["factorial"]
    assert first_order["is_even"][0] != first_order["factorial"][0]
    assert len(set(first_order["is_even"])) == 2
