"""Tests for the Python unit-test verifier."""

import pytest

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.toy_code import ToyTaskId, get_toy_task
from ttc_operatorbench.verifiers.python_unit_tests import (
    PythonUnitTestVerifier,
    extract_python_code,
)

CORRECT_CANDIDATES: tuple[tuple[ToyTaskId, str], ...] = (
    ("is_even", "def is_even(n):\n    return n % 2 == 0"),
    (
        "factorial",
        "def factorial(n):\n"
        "    result = 1\n"
        "    for value in range(2, n + 1):\n"
        "        result *= value\n"
        "    return result",
    ),
    ("reverse_string", "def reverse_string(s):\n    return s[::-1]"),
    (
        "is_prime",
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for divisor in range(2, int(n ** 0.5) + 1):\n"
        "        if n % divisor == 0:\n"
        "            return False\n"
        "    return True",
    ),
    (
        "fibonacci",
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a",
    ),
    (
        "gcd",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)",
    ),
    ("palindrome", "def palindrome(s):\n    return s == s[::-1]"),
)


def test_extract_python_code_from_markdown_fence() -> None:
    candidate = """
Here is the solution:

```python
def is_even(n):
    return n % 2 == 0
```
"""

    assert extract_python_code(candidate) == "def is_even(n):\n    return n % 2 == 0"


def test_extract_python_code_from_prose_prefix() -> None:
    candidate = """
Here is a direct implementation:

def is_even(n):
    return n % 2 == 0

This function checks divisibility by two.
"""

    assert (
        extract_python_code(candidate, entrypoint="is_even")
        == "def is_even(n):\n    return n % 2 == 0"
    )


def test_verifier_accepts_parseable_code_with_prose_prefix() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(
        task,
        "Here is the function:\n\ndef is_even(n):\n    return n % 2 == 0\n\nNo tests are included.",
    )

    assert result.verification_passed is True
    assert result.error_type is None


def test_correct_candidate_passes_public_tests() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(task, "def is_even(n):\n    return n % 2 == 0")

    assert result.verification_passed is True
    assert result.verification_score == 1.0
    assert result.scope == "public"
    assert result.error_type is None


@pytest.mark.parametrize(("task_id", "candidate"), CORRECT_CANDIDATES)
def test_correct_candidates_pass_all_toy_tasks(task_id: ToyTaskId, candidate: str) -> None:
    task = get_toy_task(task_id)
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    public_result = verifier.verify_public_text(task, candidate)
    hidden_result = verifier.verify_hidden_text(task, candidate)

    assert public_result.verification_passed is True
    assert public_result.scope == "public"
    assert public_result.error_type is None
    assert hidden_result.verification_passed is True
    assert hidden_result.scope == "hidden"
    assert hidden_result.error_type is None


def test_public_passing_candidate_can_fail_hidden_tests() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    public_only_candidate = "def is_even(n):\n    return n in {0, -4}"

    public_result = verifier.verify_public_text(task, public_only_candidate)
    hidden_result = verifier.verify_hidden_text(task, public_only_candidate)

    assert public_result.verification_passed is True
    assert hidden_result.verification_passed is False
    assert hidden_result.scope == "hidden"
    assert hidden_result.error_type == "test_failure"


def test_missing_hidden_tests_are_classified() -> None:
    task = Task(
        task_id="no-hidden",
        prompt="Write a function f().",
        public_tests=("assert True",),
    )
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_hidden_text(task, "def f():\n    return None")

    assert result.verification_passed is False
    assert result.scope == "hidden"
    assert result.error_type == "missing_hidden_tests"


def test_wrong_candidate_fails_public_tests() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(task, "def is_even(n):\n    return True")

    assert result.verification_passed is False
    assert result.verification_score == 0.0
    assert result.error_type == "test_failure"
    assert "AssertionError" in result.stderr


def test_syntax_error_is_classified() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(task, "def is_even(n)\n    return n % 2 == 0")

    assert result.verification_passed is False
    assert result.error_type == "syntax_error"


def test_infinite_loop_times_out() -> None:
    task = get_toy_task("is_even")
    verifier = PythonUnitTestVerifier(timeout_seconds=0.1)

    result = verifier.verify_text(task, "def is_even(n):\n    while True:\n        pass")

    assert result.verification_passed is False
    assert result.error_type == "timeout"


def test_verifier_runs_candidate_in_temp_cwd_without_repo_pythonpath() -> None:
    task = Task(
        task_id="cwd-isolation",
        prompt="Write probe helpers.",
        public_tests=(
            "assert repo_readme_visible() is False",
            "assert repo_package_importable() is False",
        ),
        metadata={"entrypoint": "repo_readme_visible"},
    )
    candidate = """
def repo_readme_visible():
    from pathlib import Path
    return Path('README.md').exists()

def repo_package_importable():
    try:
        import ttc_operatorbench  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
"""
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)

    result = verifier.verify_text(task, candidate)

    assert result.verification_passed is True
    assert result.latency_seconds > 0.0
