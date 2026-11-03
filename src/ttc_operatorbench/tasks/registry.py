"""Task-suite registry for local experiment protocols."""

from __future__ import annotations

from typing import Literal

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.curated_code import (
    curated_task_ids,
    get_curated_task,
    list_curated_tasks,
)
from ttc_operatorbench.tasks.toy_code import get_toy_task, list_toy_tasks, toy_task_ids

TaskSuite = Literal["toy_code", "curated_code"]


def list_task_ids(task_suite: TaskSuite) -> tuple[str, ...]:
    """Return stable task identifiers for a suite."""
    if task_suite == "toy_code":
        return toy_task_ids()
    if task_suite == "curated_code":
        return curated_task_ids()
    raise ValueError(f"unsupported task suite: {task_suite}")


def get_task(task_suite: TaskSuite, task_id: str) -> Task:
    """Return one task from a named suite."""
    if task_suite == "toy_code":
        return get_toy_task(task_id)  # type: ignore[arg-type]
    if task_suite == "curated_code":
        return get_curated_task(task_id)
    raise ValueError(f"unsupported task suite: {task_suite}")


def list_tasks(task_suite: TaskSuite) -> tuple[Task, ...]:
    """Return all tasks from a named suite."""
    if task_suite == "toy_code":
        return list_toy_tasks()
    if task_suite == "curated_code":
        return list_curated_tasks()
    raise ValueError(f"unsupported task suite: {task_suite}")


def validate_task_ids(task_suite: TaskSuite, task_ids: tuple[str, ...]) -> None:
    """Raise ValueError if any task ids are invalid for the selected suite."""
    known = set(list_task_ids(task_suite))
    unknown = sorted(set(task_ids) - known)
    if unknown:
        raise ValueError(f"unknown task ids for suite {task_suite}: {unknown}")


__all__ = [
    "TaskSuite",
    "get_task",
    "list_task_ids",
    "list_tasks",
    "validate_task_ids",
]
