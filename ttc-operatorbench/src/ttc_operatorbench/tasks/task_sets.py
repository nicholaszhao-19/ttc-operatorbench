"""Validated frozen task-set files for reproducible benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path


def read_task_ids_file(path: Path) -> tuple[str, ...]:
    """Read a unique, ordered, nonempty JSON list of task IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"task IDs file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("task IDs file must contain a nonempty JSON list")
    if any(not isinstance(task_id, str) or not task_id for task_id in payload):
        raise ValueError("task IDs file entries must be nonempty strings")
    task_ids = tuple(payload)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs file must not contain duplicates")
    return task_ids


__all__ = ["read_task_ids_file"]
