"""Tests for core benchmark schemas."""

import json

import pytest
from pydantic import ValidationError

from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    Generation,
    OperatorResult,
    SamplingConfig,
    SearchResult,
    Task,
    VerificationResult,
)


def make_attempt(
    attempt_id: str,
    *,
    selected: bool = False,
    verification_passed: bool = False,
    cumulative_tokens: int = 8,
) -> AttemptLog:
    return AttemptLog(
        attempt_id=attempt_id,
        task_id="task-1",
        operator_name="draft",
        prompt="Solve the task.",
        generation_text="candidate",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        latency_seconds=0.25,
        verification_passed=verification_passed,
        verification_score=1.0 if verification_passed else 0.0,
        cumulative_tokens=cumulative_tokens,
        cumulative_verifier_calls=1,
        cumulative_seconds=0.25,
        selected=selected,
    )


def test_schemas_serialize_to_json() -> None:
    task = Task(
        task_id="task-1",
        prompt="Solve the task.",
        public_tests=("assert candidate() == 1",),
        hidden_tests=("assert candidate() == 2",),
        task_family="unit",
        difficulty_label="tiny",
    )
    sampling = SamplingConfig(temperature=0.2, top_p=0.95, max_output_tokens=128, seed=7)
    generation = Generation(
        prompt=task.prompt,
        generation_text="candidate",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        latency_seconds=0.25,
        sampling=sampling,
        model_name="test-model",
    )
    verification = VerificationResult(
        verification_passed=True,
        verification_score=1.0,
        scope="public",
    )
    hidden_verification = VerificationResult(
        verification_passed=True,
        verification_score=1.0,
        scope="hidden",
    )
    attempt = make_attempt("attempt-1", selected=True, verification_passed=True).model_copy(
        update={
            "public_verification": verification,
            "hidden_verification": hidden_verification,
        }
    )
    result = SearchResult(
        task_id=task.task_id,
        policy_name="adaptive",
        budget=Budget(max_attempts=3, max_tokens=100),
        attempts=(attempt,),
        selected_attempt_id=attempt.attempt_id,
        success=True,
    )
    operator_result = OperatorResult(
        operator_name=attempt.operator_name,
        generation=generation,
        verification=verification,
        attempt_log=attempt,
    )

    payload = json.loads(result.model_dump_json())
    operator_payload = json.loads(operator_result.model_dump_json())

    assert payload["attempts"][0]["task_id"] == task.task_id
    assert payload["attempts"][0]["public_verification"]["scope"] == "public"
    assert payload["attempts"][0]["hidden_verification"]["scope"] == "hidden"
    assert operator_payload["verification"]["verification_passed"] is True


def test_attempt_log_validates_public_verification_matches_scalar_fields() -> None:
    with pytest.raises(ValidationError):
        AttemptLog(
            attempt_id="attempt-1",
            task_id="task-1",
            operator_name="draft",
            prompt="Solve the task.",
            generation_text="candidate",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            latency_seconds=0.25,
            verification_passed=True,
            verification_score=1.0,
            public_verification=VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                scope="public",
            ),
            cumulative_tokens=8,
            cumulative_verifier_calls=1,
            cumulative_seconds=0.25,
        )


def test_attempt_log_validates_verification_scopes() -> None:
    with pytest.raises(ValidationError):
        AttemptLog(
            attempt_id="attempt-1",
            task_id="task-1",
            operator_name="draft",
            prompt="Solve the task.",
            generation_text="candidate",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            latency_seconds=0.25,
            verification_passed=True,
            verification_score=1.0,
            public_verification=VerificationResult(
                verification_passed=True,
                verification_score=1.0,
                scope="hidden",
            ),
            cumulative_tokens=8,
            cumulative_verifier_calls=1,
            cumulative_seconds=0.25,
        )

    with pytest.raises(ValidationError):
        AttemptLog(
            attempt_id="attempt-2",
            task_id="task-1",
            operator_name="draft",
            prompt="Solve the task.",
            generation_text="candidate",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            latency_seconds=0.25,
            verification_passed=True,
            verification_score=1.0,
            hidden_verification=VerificationResult(
                verification_passed=True,
                verification_score=1.0,
                scope="public",
            ),
            cumulative_tokens=8,
            cumulative_verifier_calls=1,
            cumulative_seconds=0.25,
        )


def test_invalid_budgets_fail() -> None:
    with pytest.raises(ValidationError):
        Budget()

    with pytest.raises(ValidationError):
        Budget(max_tokens=0)

    with pytest.raises(ValidationError):
        Budget(max_seconds=-1.0)


def test_attempt_log_validates_cost_trace() -> None:
    with pytest.raises(ValidationError):
        AttemptLog(
            attempt_id="attempt-1",
            task_id="task-1",
            operator_name="draft",
            prompt="Solve the task.",
            generation_text="candidate",
            input_tokens=5,
            output_tokens=3,
            total_tokens=7,
            latency_seconds=0.25,
            verification_passed=False,
            verification_score=0.0,
            cumulative_tokens=8,
            cumulative_verifier_calls=1,
            cumulative_seconds=0.25,
        )

    with pytest.raises(ValidationError):
        make_attempt("attempt-1", cumulative_tokens=7)


def test_search_result_stores_all_attempts() -> None:
    failed_attempt = make_attempt("attempt-1")
    selected_attempt = make_attempt("attempt-2", selected=True, verification_passed=True)

    result = SearchResult(
        task_id="task-1",
        policy_name="adaptive",
        budget=Budget(max_attempts=2),
        attempts=(failed_attempt, selected_attempt),
        selected_attempt_id=selected_attempt.attempt_id,
        success=True,
    )

    assert result.attempts == (failed_attempt, selected_attempt)
    assert [attempt.attempt_id for attempt in result.attempts] == ["attempt-1", "attempt-2"]


def test_search_result_rejects_selected_attempt_not_in_attempts() -> None:
    with pytest.raises(ValidationError):
        SearchResult(
            task_id="task-1",
            policy_name="adaptive",
            budget=Budget(max_attempts=2),
            attempts=(make_attempt("attempt-1"),),
            selected_attempt_id="missing",
            success=True,
        )


def test_search_result_validates_terminal_decision_costs() -> None:
    selected_attempt = make_attempt("attempt-1", selected=True, verification_passed=True)

    result = SearchResult(
        task_id="task-1",
        policy_name="batch_selector",
        budget=Budget(max_attempts=2),
        attempts=(selected_attempt,),
        selected_attempt_id=selected_attempt.attempt_id,
        success=True,
        total_tokens=16,
        total_verifier_calls=2,
        total_seconds=0.5,
        decision_tokens=16,
        decision_verifier_calls=2,
        decision_seconds=0.5,
    )

    assert result.decision_tokens == 16

    invalid_payload = result.model_dump()
    invalid_payload["decision_tokens"] = 7
    with pytest.raises(ValidationError):
        SearchResult.model_validate(invalid_payload)

    with pytest.raises(ValidationError):
        SearchResult(
            task_id="task-1",
            policy_name="batch_selector",
            budget=Budget(max_attempts=2),
            attempts=(selected_attempt,),
            selected_attempt_id=selected_attempt.attempt_id,
            success=True,
            total_tokens=16,
            total_verifier_calls=2,
            total_seconds=0.5,
            decision_tokens=16,
        )
