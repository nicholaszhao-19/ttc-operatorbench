"""Tests for the policy-safe HumanEval+ task projection."""

import json

import pytest

from ttc_operatorbench.tasks.evalplus import (
    EVALPLUS_DATASET_NAME,
    EVALPLUS_HUMANEVAL_VERSION,
    evalplus_dataset_sha256,
    evalplus_task_split,
    tasks_from_evalplus_problems,
)


def fixture_problems() -> dict[str, dict[str, object]]:
    return {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n",
            "entry_point": "add",
            "canonical_solution": "SECRET_CANONICAL_SOLUTION",
            "base_input": ["SECRET_BASE_INPUT"],
            "plus_input": ["SECRET_PLUS_INPUT"],
            "expected_output": ["SECRET_EXPECTED_OUTPUT"],
        }
    }


def test_evalplus_projection_drops_all_evaluation_only_fields() -> None:
    task = tasks_from_evalplus_problems(fixture_problems())[0]
    serialized = task.model_dump_json()

    assert task.task_id == "HumanEval/0"
    assert task.task_family == EVALPLUS_DATASET_NAME
    assert task.public_tests == ()
    assert task.hidden_tests == ()
    assert task.allowed_verifier_inputs == {"entrypoint": "add"}
    assert task.metadata["split"] == evalplus_task_split(task.task_id)
    for secret in (
        "SECRET_CANONICAL_SOLUTION",
        "SECRET_BASE_INPUT",
        "SECRET_PLUS_INPUT",
        "SECRET_EXPECTED_OUTPUT",
    ):
        assert secret not in serialized


def test_evalplus_projection_is_stable_and_json_serializable() -> None:
    first = tasks_from_evalplus_problems(fixture_problems())
    second = tasks_from_evalplus_problems(fixture_problems())

    assert first == second
    assert (
        json.loads(first[0].model_dump_json())["metadata"]["dataset_version"]
        == EVALPLUS_HUMANEVAL_VERSION
    )


def test_evalplus_dataset_hash_covers_private_evaluation_payload() -> None:
    original = fixture_problems()
    changed = fixture_problems()
    changed["HumanEval/0"]["plus_input"] = ["DIFFERENT_PLUS_INPUT"]

    assert len(evalplus_dataset_sha256(original)) == 64
    assert evalplus_dataset_sha256(original) == evalplus_dataset_sha256(fixture_problems())
    assert evalplus_dataset_sha256(original) != evalplus_dataset_sha256(changed)


def test_evalplus_split_is_deterministic_and_has_both_partitions() -> None:
    task_ids = tuple(f"HumanEval/{index}" for index in range(164))
    first = tuple(evalplus_task_split(task_id) for task_id in task_ids)
    second = tuple(evalplus_task_split(task_id) for task_id in task_ids)

    assert first == second
    assert "development" in first
    assert "evaluation" in first
    assert 20 <= first.count("development") <= 45


def test_evalplus_projection_rejects_malformed_records() -> None:
    problems = fixture_problems()
    problems["HumanEval/0"]["entry_point"] = ""

    with pytest.raises(ValueError, match="entry_point"):
        tasks_from_evalplus_problems(problems)

    with pytest.raises(ValueError, match="must not be empty"):
        tasks_from_evalplus_problems({})
