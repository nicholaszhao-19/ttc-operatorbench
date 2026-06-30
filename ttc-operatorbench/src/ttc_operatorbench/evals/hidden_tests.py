"""Hidden-test loading and sealing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.toy_code import HIDDEN_TESTS_KEY


def load_external_hidden_tests(path: Path) -> dict[str, tuple[str, ...]]:
    """Load hidden tests from JSON mapping or JSONL rows."""
    if path.suffix == ".jsonl":
        return _load_hidden_tests_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("sealed hidden-test JSON must be an object keyed by task_id")
    return {
        str(task_id): _tests_from_payload(raw_tests, task_id=str(task_id))
        for task_id, raw_tests in payload.items()
    }


def policy_visible_task(task: Task) -> Task:
    """Return a task with hidden tests removed before policy execution."""
    allowed_inputs = {
        key: value
        for key, value in task.allowed_verifier_inputs.items()
        if key != HIDDEN_TESTS_KEY
    }
    return task.model_copy(update={"hidden_tests": (), "allowed_verifier_inputs": allowed_inputs})


def task_with_hidden_tests(task: Task, hidden_tests: tuple[str, ...]) -> Task:
    """Return a task carrying hidden tests for evaluation-only grading."""
    allowed_inputs = {**task.allowed_verifier_inputs, HIDDEN_TESTS_KEY: hidden_tests}
    return task.model_copy(
        update={"hidden_tests": hidden_tests, "allowed_verifier_inputs": allowed_inputs}
    )


def hidden_test_fingerprint(hidden_tests: tuple[str, ...]) -> str | None:
    """Return a stable hash for hidden tests without revealing their source."""
    if not hidden_tests:
        return None
    payload = json.dumps(list(hidden_tests), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hidden_test_count(task: Task) -> int:
    """Return the number of hidden tests carried by a task."""
    return len(task.hidden_tests)


def _load_hidden_tests_jsonl(path: Path) -> dict[str, tuple[str, ...]]:
    hidden_tests: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number}: expected object")
            task_id = payload.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"line {line_number}: missing task_id")
            hidden_tests[task_id] = _tests_from_payload(
                payload.get("hidden_tests"),
                task_id=task_id,
            )
    return hidden_tests


def _tests_from_payload(raw_tests: Any, *, task_id: str) -> tuple[str, ...]:
    if not isinstance(raw_tests, list):
        raise ValueError(f"hidden tests for {task_id} must be a list of strings")
    tests = tuple(test for test in raw_tests if isinstance(test, str) and test.strip())
    if len(tests) != len(raw_tests):
        raise ValueError(f"hidden tests for {task_id} must all be non-empty strings")
    return tests


__all__ = [
    "hidden_test_count",
    "hidden_test_fingerprint",
    "load_external_hidden_tests",
    "policy_visible_task",
    "task_with_hidden_tests",
]
