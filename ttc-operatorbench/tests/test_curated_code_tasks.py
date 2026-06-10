"""Tests for the curated local code task suite."""

from __future__ import annotations

import pytest

from ttc_operatorbench.core.schema import Budget, Task
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.baselines import GreedyPolicy, RepairOnlyPolicy
from ttc_operatorbench.search.operator_bandit import OperatorBanditScheduler
from ttc_operatorbench.tasks.curated_code import (
    CURATED_REFERENCE_CANDIDATES,
    curated_task_ids,
    list_curated_tasks,
)
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY, PUBLIC_TESTS_KEY
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier


def test_curated_task_ids_are_stable_and_in_target_range() -> None:
    task_ids = curated_task_ids()

    assert len(task_ids) == 20
    assert 20 <= len(task_ids) <= 50
    assert task_ids[:3] == ("count_vowels", "sum_squares", "flatten_once")
    assert task_ids[-1] == "count_words"


def test_curated_tasks_are_loadable_and_have_public_verifier_inputs() -> None:
    tasks = list_curated_tasks()

    assert len(tasks) == len(curated_task_ids())
    for task in tasks:
        assert task.metadata["suite"] == "curated_code"
        assert task.allowed_verifier_inputs[ENTRYPOINT_KEY] == task.metadata["entrypoint"]
        assert task.allowed_verifier_inputs[PUBLIC_TESTS_KEY]
        assert task.task_id in CURATED_REFERENCE_CANDIDATES


@pytest.mark.parametrize("task", list_curated_tasks(), ids=lambda task: task.task_id)
def test_curated_reference_candidates_pass_public_verifier(task: Task) -> None:
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(task, CURATED_REFERENCE_CANDIDATES[task.task_id])

    assert result.verification_passed is True


def test_curated_tasks_run_with_required_dummy_policies() -> None:
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    budget = Budget(max_attempts=4, max_verifier_calls=4, max_tokens=5_000)
    assert budget.max_tokens is not None
    assert budget.max_verifier_calls is not None

    for task in list_curated_tasks():
        correct = CURATED_REFERENCE_CANDIDATES[task.task_id]
        wrong = f"def {task.metadata['entrypoint']}(*args):\n    return None"
        policy_runs = (
            (GreedyPolicy(), (correct,)),
            (RepairOnlyPolicy(max_repairs=1), (wrong, correct)),
            (OperatorBanditScheduler(exploration_weight=1.0), (wrong, correct)),
        )
        for policy, generations in policy_runs:
            provider = DummyModelProvider({task.task_id: generations})
            result = policy.run(task, provider, verifier, budget)

            assert result.attempts
            assert result.total_tokens <= budget.max_tokens
            assert result.total_verifier_calls <= budget.max_verifier_calls
