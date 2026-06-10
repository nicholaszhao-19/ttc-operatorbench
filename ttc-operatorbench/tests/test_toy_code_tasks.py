"""Tests for public and hidden toy code task definitions."""

from ttc_operatorbench.tasks.toy_code import (
    HIDDEN_TESTS_KEY,
    PUBLIC_TESTS_KEY,
    list_toy_tasks,
    toy_task_ids,
)


def test_toy_task_ids_are_stable() -> None:
    assert toy_task_ids() == (
        "is_even",
        "factorial",
        "reverse_string",
        "is_prime",
        "fibonacci",
        "gcd",
        "palindrome",
    )


def test_toy_tasks_include_public_and_hidden_verifier_tests() -> None:
    tasks = list_toy_tasks()

    assert len(tasks) == 7
    for task in tasks:
        public_tests = task.allowed_verifier_inputs[PUBLIC_TESTS_KEY]
        hidden_tests = task.allowed_verifier_inputs[HIDDEN_TESTS_KEY]
        assert task.prompt
        assert task.metadata["suite"] == "toy_code"
        assert task.task_family == "toy_code"
        assert task.difficulty_label == "toy"
        assert isinstance(public_tests, tuple)
        assert isinstance(hidden_tests, tuple)
        assert public_tests
        assert hidden_tests
        assert task.public_tests == public_tests
        assert task.hidden_tests == hidden_tests
        assert all(test.startswith("assert ") for test in public_tests)
        assert all(test.startswith("assert ") for test in hidden_tests)
