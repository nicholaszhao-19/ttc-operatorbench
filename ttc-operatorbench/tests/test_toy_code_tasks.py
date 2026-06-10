"""Tests for public toy code task definitions."""

from ttc_operatorbench.tasks.toy_code import PUBLIC_TESTS_KEY, list_toy_tasks, toy_task_ids


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


def test_toy_tasks_include_public_verifier_tests() -> None:
    tasks = list_toy_tasks()

    assert len(tasks) == 7
    for task in tasks:
        public_tests = task.allowed_verifier_inputs[PUBLIC_TESTS_KEY]
        assert task.prompt
        assert task.metadata["suite"] == "toy_code"
        assert isinstance(public_tests, tuple)
        assert public_tests
        assert all(test.startswith("assert ") for test in public_tests)
