"""Public toy coding tasks for verifier-first development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ttc_operatorbench.core.schema import Task

ToyTaskId = Literal[
    "is_even",
    "factorial",
    "reverse_string",
    "is_prime",
    "fibonacci",
    "gcd",
    "palindrome",
]

PUBLIC_TESTS_KEY = "public_tests"
ENTRYPOINT_KEY = "entrypoint"


@dataclass(frozen=True)
class ToyCodeTaskSpec:
    """Definition for a toy code task with public verifier tests."""

    task_id: ToyTaskId
    entrypoint: str
    prompt: str
    public_tests: tuple[str, ...]

    def to_task(self) -> Task:
        """Convert the toy spec into the shared task schema."""
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            metadata={
                "suite": "toy_code",
                "entrypoint": self.entrypoint,
            },
            allowed_verifier_inputs={
                ENTRYPOINT_KEY: self.entrypoint,
                PUBLIC_TESTS_KEY: self.public_tests,
            },
        )


TOY_CODE_TASK_SPECS: tuple[ToyCodeTaskSpec, ...] = (
    ToyCodeTaskSpec(
        task_id="is_even",
        entrypoint="is_even",
        prompt="Write a Python function is_even(n) that returns True exactly when n is even.",
        public_tests=(
            "assert is_even(0) is True",
            "assert is_even(1) is False",
            "assert is_even(-4) is True",
            "assert is_even(9) is False",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="factorial",
        entrypoint="factorial",
        prompt="Write a Python function factorial(n) that returns n! for nonnegative integers.",
        public_tests=(
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="reverse_string",
        entrypoint="reverse_string",
        prompt="Write a Python function reverse_string(s) that returns the reverse of s.",
        public_tests=(
            "assert reverse_string('') == ''",
            "assert reverse_string('abc') == 'cba'",
            "assert reverse_string('racecar') == 'racecar'",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="is_prime",
        entrypoint="is_prime",
        prompt="Write a Python function is_prime(n) that returns True exactly when n is prime.",
        public_tests=(
            "assert is_prime(1) is False",
            "assert is_prime(2) is True",
            "assert is_prime(17) is True",
            "assert is_prime(21) is False",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="fibonacci",
        entrypoint="fibonacci",
        prompt="Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
        public_tests=(
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(7) == 13",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="gcd",
        entrypoint="gcd",
        prompt="Write a Python function gcd(a, b) that returns the greatest common divisor.",
        public_tests=(
            "assert gcd(12, 8) == 4",
            "assert gcd(7, 3) == 1",
            "assert gcd(0, 5) == 5",
        ),
    ),
    ToyCodeTaskSpec(
        task_id="palindrome",
        entrypoint="palindrome",
        prompt=(
            "Write a Python function palindrome(s) that returns True exactly when s is a "
            "palindrome."
        ),
        public_tests=(
            "assert palindrome('') is True",
            "assert palindrome('racecar') is True",
            "assert palindrome('python') is False",
        ),
    ),
)

_TOY_CODE_TASKS_BY_ID = {spec.task_id: spec for spec in TOY_CODE_TASK_SPECS}


def list_toy_tasks() -> tuple[Task, ...]:
    """Return all toy code tasks as shared task schemas."""
    return tuple(spec.to_task() for spec in TOY_CODE_TASK_SPECS)


def get_toy_task(task_id: ToyTaskId) -> Task:
    """Return one toy code task by identifier."""
    return _TOY_CODE_TASKS_BY_ID[task_id].to_task()


def toy_task_ids() -> tuple[ToyTaskId, ...]:
    """Return the stable toy task identifiers."""
    return tuple(spec.task_id for spec in TOY_CODE_TASK_SPECS)


__all__ = [
    "ENTRYPOINT_KEY",
    "PUBLIC_TESTS_KEY",
    "TOY_CODE_TASK_SPECS",
    "ToyCodeTaskSpec",
    "ToyTaskId",
    "get_toy_task",
    "list_toy_tasks",
    "toy_task_ids",
]
