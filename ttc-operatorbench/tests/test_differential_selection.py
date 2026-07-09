"""Tests for differential-selection and bottleneck-aware policies."""

from __future__ import annotations

from ttc_operatorbench.core.schema import Budget
from ttc_operatorbench.evals.metrics import tokens_to_first_solution
from ttc_operatorbench.models.dummy import DummyModelProvider
from ttc_operatorbench.search.differential_selection import (
    BottleneckAwareControllerPolicy,
    DifferentialSelectionPolicy,
    ProbeCall,
    build_probe_calls,
    collect_behavior_trace,
    select_consensus_medoid,
)
from ttc_operatorbench.tasks.toy_code import get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier

CORRECT_IS_EVEN = "def is_even(n):\n    return n % 2 == 0"
CORRECT_IS_EVEN_ALT = "def is_even(n):\n    return (n & 1) == 0"
WRONG_ALWAYS_TRUE = "def is_even(n):\n    return True"
WRONG_ALWAYS_FALSE = "def is_even(n):\n    return False"


def test_probe_calls_are_extracted_and_mutated_from_public_call_shapes() -> None:
    probes = build_probe_calls(get_toy_task("is_even"), limit=8)

    assert ProbeCall(args=(0,)) in probes
    assert ProbeCall(args=(1,)) in probes
    assert len(probes) <= 8


def test_behavior_trace_distinguishes_candidate_outputs() -> None:
    task = get_toy_task("is_even")
    probes = (ProbeCall(args=(2,)), ProbeCall(args=(3,)))

    correct_trace = collect_behavior_trace(task, CORRECT_IS_EVEN, probes)
    wrong_trace = collect_behavior_trace(task, WRONG_ALWAYS_TRUE, probes)

    assert correct_trace != wrong_trace
    assert correct_trace == ("ok:True", "ok:False")


def test_consensus_medoid_selects_largest_behavior_cluster() -> None:
    traces = (
        ("ok:True", "ok:False"),
        ("ok:True", "ok:False"),
        ("ok:True", "ok:True"),
    )

    selection = select_consensus_medoid(traces)

    assert selection.selected_index in {0, 1}
    assert sorted(selection.cluster_sizes) == [1, 2]
    assert selection.largest_cluster_ratio == 2 / 3


def test_diffcodegen_selector_selects_behavior_consensus_candidate() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider(
        {task.task_id: (CORRECT_IS_EVEN, CORRECT_IS_EVEN_ALT, WRONG_ALWAYS_TRUE)}
    )
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    policy = DifferentialSelectionPolicy(n=3)

    result = policy.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=3, max_verifier_calls=3, max_tokens=1_000),
    )

    assert result.policy_name == "diffcodegen_select"
    assert result.success is True
    assert result.metadata["candidate_count"] == 3
    assert sorted(result.metadata["cluster_sizes"]) == [1, 2]
    assert result.metadata["selected_candidate_index"] in {0, 1}


def test_diffcodegen_selector_exposes_common_bug_failure_mode() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider(
        {task.task_id: (WRONG_ALWAYS_TRUE, WRONG_ALWAYS_TRUE, CORRECT_IS_EVEN)}
    )
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    policy = DifferentialSelectionPolicy(n=3)

    result = policy.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=3, max_verifier_calls=3, max_tokens=1_000),
    )

    assert result.selected_attempt_id is not None
    assert result.success is False
    assert result.attempts[0].selected is True
    assert sorted(result.metadata["cluster_sizes"]) == [1, 2]
    assert tokens_to_first_solution(result) is None


def test_bottleneck_controller_stops_early_when_consensus_is_confident() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider(
        {task.task_id: (CORRECT_IS_EVEN, CORRECT_IS_EVEN_ALT, WRONG_ALWAYS_TRUE)}
    )
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    policy = BottleneckAwareControllerPolicy(min_samples=2, max_samples=4)

    result = policy.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=4, max_verifier_calls=4, max_tokens=1_000),
    )

    assert len(result.attempts) == 2
    assert result.success is True
    assert result.metadata["initial_regime"] == "stop_confident"
    assert "stop_early" in result.metadata["controller_actions"]


def test_bottleneck_controller_repairs_when_no_candidate_is_plausible() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: (WRONG_ALWAYS_FALSE, CORRECT_IS_EVEN)})
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    policy = BottleneckAwareControllerPolicy(min_samples=1, max_samples=3)

    result = policy.run(
        task,
        provider,
        verifier,
        Budget(max_attempts=3, max_verifier_calls=3, max_tokens=1_000),
    )

    assert result.success is True
    assert [attempt.operator_name for attempt in result.attempts] == [
        "bottleneck_controller/sample",
        "bottleneck_controller/repair",
    ]
    assert result.metadata["initial_regime"] == "coverage_failure"
    assert result.metadata["selection_override"] == "single_public_pass"
