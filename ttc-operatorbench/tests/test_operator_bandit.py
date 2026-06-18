"""Tests for the adaptive operator bandit scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ttc_operatorbench.core.schema import (
    Budget,
    Generation,
    SamplingConfig,
    Task,
    VerificationResult,
)
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.operator_bandit import (
    FixedOperatorOrderScheduler,
    OperatorBanditScheduler,
    OperatorContext,
    OperatorStepResult,
)
from ttc_operatorbench.tasks.toy_code import get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

CORRECT_IS_EVEN = "def is_even(n):\n    return n % 2 == 0"
WRONG_IS_EVEN = "def is_even(n):\n    return True"


class NoopProvider:
    """Provider placeholder for synthetic operator tests."""

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        del sampling
        return Generation(
            prompt=task.prompt,
            generation_text="",
            input_tokens=1,
            output_tokens=0,
            total_tokens=1,
            latency_seconds=0.0,
            model_name="noop",
            provider_name="noop",
        )


class CostedNoopProvider(NoopProvider):
    """Provider placeholder with nonzero accounting cost rates."""

    input_token_cost = 0.0
    output_token_cost = 0.0
    verifier_call_cost = 2.0
    fixed_attempt_cost = 3.0


class NoopVerifier:
    """Verifier placeholder for synthetic operator tests."""

    def verify_generation(self, task: Task, generation: Generation) -> VerificationResult:
        del task, generation
        return VerificationResult(verification_passed=False, verification_score=0.0)


@dataclass
class SyntheticOperator:
    """Deterministic synthetic operator arm."""

    name: str
    cost: int
    success_on: tuple[int, ...] = ()
    error_type: str | None = "wrong_answer"
    enabled: bool = True
    calls: int = 0

    def can_run(self, context: OperatorContext) -> bool:
        return self.enabled and context.ledger.can_record_synthetic(
            total_tokens=self.cost,
            verifier_called=True,
        )

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        start_index = len(context.attempts)
        self.calls += 1
        success = self.calls in self.success_on
        context.record_synthetic_attempt(
            operator_name=self.name,
            total_tokens=self.cost,
            verification_passed=success,
            error_type=None if success else self.error_type,
        )
        return context.step_since(start_index)


def run_synthetic(
    scheduler: OperatorBanditScheduler,
    budget: Budget,
) -> tuple[str, ...]:
    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        budget,
    )
    return tuple(attempt.operator_name for attempt in result.attempts)


def test_scheduler_initializes_with_multiple_operators() -> None:
    scheduler = OperatorBanditScheduler()

    assert tuple(operator.name for operator in scheduler.operators) == (
        "direct_sample",
        "repair_from_error",
        "plan_then_code",
        "local_revision",
    )
    assert set(scheduler.operator_statistics) == {
        "direct_sample",
        "repair_from_error",
        "plan_then_code",
        "local_revision",
    }


def test_scheduler_selects_valid_operators() -> None:
    disabled = SyntheticOperator("disabled", cost=10, enabled=False)
    enabled = SyntheticOperator("enabled", cost=10)
    scheduler = OperatorBanditScheduler(operators=(disabled, enabled), exploration_weight=1.0)

    names = run_synthetic(scheduler, Budget(max_attempts=1, max_tokens=100))

    assert names == ("enabled",)


def test_scheduler_updates_operator_statistics() -> None:
    cheap = SyntheticOperator("cheap_good", cost=10, success_on=(1,))
    scheduler = OperatorBanditScheduler(operators=(cheap,), exploration_weight=0.0)

    run_synthetic(scheduler, Budget(max_attempts=1, max_tokens=100))

    stats = scheduler.operator_statistics["cheap_good"]
    assert stats.n == 1
    assert stats.successes == 1
    assert stats.total_cost == 10.0
    assert stats.mean_success == 1.0
    assert stats.mean_cost == 10.0


def test_scheduler_shifts_toward_cheap_successful_operator() -> None:
    cheap_good = SyntheticOperator("cheap_good", cost=10, success_on=(2,))
    expensive_bad = SyntheticOperator("expensive_bad", cost=100)
    medium_random = SyntheticOperator("medium_random", cost=50)
    scheduler = OperatorBanditScheduler(
        operators=(cheap_good, expensive_bad, medium_random),
        exploration_weight=1.0,
    )

    run_synthetic(scheduler, Budget(max_attempts=5, max_tokens=1_000))

    assert cheap_good.calls > expensive_bad.calls
    assert scheduler.operator_statistics["cheap_good"].successes == 1


def test_scheduler_explores_initially_with_exploration_weight() -> None:
    operators = (
        SyntheticOperator("cheap_good", cost=10),
        SyntheticOperator("expensive_bad", cost=100),
        SyntheticOperator("medium_random", cost=50),
    )
    scheduler = OperatorBanditScheduler(operators=operators, exploration_weight=1.0)

    names = run_synthetic(scheduler, Budget(max_attempts=3, max_tokens=1_000))

    assert set(names) == {"cheap_good", "expensive_bad", "medium_random"}


def test_scheduler_respects_max_attempts() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=10),),
        exploration_weight=0.0,
    )

    names = run_synthetic(scheduler, Budget(max_attempts=2, max_tokens=1_000))

    assert len(names) == 2


def test_scheduler_respects_max_tokens() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("medium_random", cost=60),),
        exploration_weight=0.0,
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=5, max_tokens=100),
    )

    assert len(result.attempts) == 1
    assert result.total_tokens == 60
    assert all(attempt.cumulative_tokens <= 100 for attempt in result.attempts)


def test_scheduler_respects_max_verifier_calls() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=10),),
        exploration_weight=0.0,
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=5, max_verifier_calls=1, max_tokens=1_000),
    )

    assert len(result.attempts) == 1
    assert result.total_verifier_calls == 1


def test_scheduler_stops_immediately_after_verified_success() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(
            SyntheticOperator("cheap_good", cost=10, success_on=(1,)),
            SyntheticOperator("expensive_bad", cost=100),
        ),
        exploration_weight=1.0,
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=5, max_tokens=1_000),
    )

    assert result.success is True
    assert len(result.attempts) == 1
    assert result.selected_attempt_id == result.attempts[0].attempt_id


def test_scheduler_logs_operator_name_for_every_attempt() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=10),),
        exploration_weight=0.0,
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=2, max_tokens=1_000),
    )

    assert all(attempt.operator_name == "cheap_good" for attempt in result.attempts)


def test_scheduler_returns_valid_search_result_when_no_operator_succeeds() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=10),),
        exploration_weight=0.0,
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=2, max_tokens=1_000),
    )

    assert result.policy_name == "operator_bandit"
    assert result.success is False
    assert result.selected_attempt_id is None
    assert len(result.attempts) == 2
    assert result.metadata["operator_statistics"]["cheap_good"]["n"] == 2


@pytest.mark.parametrize("error_type", ["syntax_error", "wrong_answer", "timeout"])
def test_scheduler_handles_error_types(error_type: str) -> None:
    first = SyntheticOperator("direct_sample", cost=10, error_type=error_type)
    repair = SyntheticOperator("repair_from_error", cost=10)
    scheduler = OperatorBanditScheduler(
        operators=(first, repair),
        exploration_weight=0.0,
    )

    names = run_synthetic(scheduler, Budget(max_attempts=2, max_tokens=100))

    assert names[0] == "direct_sample"
    if error_type in {"syntax_error", "wrong_answer"}:
        assert names[1] == "repair_from_error"
    assert scheduler.operator_statistics["direct_sample"].last_error_type == error_type


def test_scheduler_accepts_empty_error_bonus_mapping() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(
            SyntheticOperator("direct_sample", cost=10, error_type="syntax_error"),
            SyntheticOperator("repair_from_error", cost=10),
        ),
        exploration_weight=0.0,
        error_type_bonuses={},
    )

    names = run_synthetic(scheduler, Budget(max_attempts=2, max_tokens=100))

    assert names == ("direct_sample", "direct_sample")


def test_unit_cost_ablation_records_unit_operator_cost() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=25),),
        exploration_weight=0.0,
        cost_metric="unit",
        policy_name="operator_bandit_unit_cost",
    )

    run_synthetic(scheduler, Budget(max_attempts=1, max_tokens=100))

    assert scheduler.operator_statistics["cheap_good"].total_cost == 1.0


def test_cost_metric_records_realized_accounting_cost() -> None:
    scheduler = OperatorBanditScheduler(
        operators=(SyntheticOperator("cheap_good", cost=25),),
        exploration_weight=0.0,
        cost_metric="cost",
        policy_name="operator_bandit_cost",
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        CostedNoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=1, max_tokens=100),
    )

    assert result.total_cost == 5.0
    assert scheduler.operator_statistics["cheap_good"].total_cost == 5.0
    assert len(result.decision_log) == 1
    assert result.decision_log[0].chosen_operator_name == "cheap_good"
    assert result.decision_log[0].delta_cost == 5.0


def test_fixed_operator_order_cycles_valid_operators() -> None:
    scheduler = FixedOperatorOrderScheduler(
        operators=(
            SyntheticOperator("direct_sample", cost=10),
            SyntheticOperator("repair_from_error", cost=10),
            SyntheticOperator("plan_then_code", cost=10),
        )
    )

    result = scheduler.run(
        get_toy_task("is_even"),
        NoopProvider(),
        NoopVerifier(),
        Budget(max_attempts=3, max_tokens=100),
    )

    assert result.policy_name == "fixed_operator_order"
    assert tuple(attempt.operator_name for attempt in result.attempts) == (
        "direct_sample",
        "repair_from_error",
        "plan_then_code",
    )
    assert tuple(decision.chosen_operator_name for decision in result.decision_log) == (
        "direct_sample",
        "repair_from_error",
        "plan_then_code",
    )


def test_operator_bandit_integration_with_toy_task_and_dummy_provider() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: (WRONG_IS_EVEN, CORRECT_IS_EVEN)})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    scheduler = OperatorBanditScheduler(exploration_weight=1.0)

    result = scheduler.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=4, max_verifier_calls=4, max_tokens=2_000),
    )

    assert result.success is True
    assert result.policy_name == "operator_bandit"
    assert len(result.attempts) == 2
    assert [attempt.operator_name for attempt in result.attempts] == [
        "direct_sample",
        "repair_from_error",
    ]
    assert [decision.chosen_operator_name for decision in result.decision_log] == [
        "direct_sample",
        "repair_from_error",
    ]
    assert result.decision_log[0].previous_failure_category == "no_attempt"
    assert result.decision_log[1].previous_failure_category == "public_test_failure"
    assert result.decision_log[1].outcome_success is True
    assert result.total_tokens <= 2_000
    assert result.total_verifier_calls <= 4
