"""Policy-safe HumanEval+ task loading."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal, cast

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY

EVALPLUS_PACKAGE_VERSION = "0.3.1"
EVALPLUS_HUMANEVAL_VERSION = "v0.1.10"
EVALPLUS_DATASET_NAME = "humaneval_plus"
EVALPLUS_MBPP_VERSION = "v0.2.0"
EVALPLUS_MBPP_DATASET_NAME = "mbpp_plus"
EVALPLUS_SPLIT_SALT = "ttc-operatorbench-evalplus-v1:"

EvalPlusSplit = Literal["development", "evaluation"]


def evalplus_task_split(task_id: str) -> EvalPlusSplit:
    """Return the frozen hash-based development/evaluation split."""
    digest = hashlib.sha256(f"{EVALPLUS_SPLIT_SALT}{task_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 5
    return "development" if bucket == 0 else "evaluation"


def tasks_from_evalplus_problems(
    problems: Mapping[str, Mapping[str, Any]],
    *,
    dataset_version: str = EVALPLUS_HUMANEVAL_VERSION,
) -> tuple[Task, ...]:
    """Project EvalPlus records onto the strictly policy-visible task schema."""
    tasks: list[Task] = []
    for mapping_key, problem in sorted(problems.items()):
        task_id = _required_string(problem, "task_id")
        if task_id != mapping_key:
            raise ValueError(f"EvalPlus task key does not match task_id: {mapping_key}")
        prompt = _required_string(problem, "prompt")
        entrypoint = _required_string(problem, "entry_point")
        tasks.append(
            Task(
                task_id=task_id,
                prompt=prompt,
                task_family=EVALPLUS_DATASET_NAME,
                difficulty_label="unlabeled",
                metadata={
                    "suite": EVALPLUS_DATASET_NAME,
                    "dataset_version": dataset_version,
                    "entrypoint": entrypoint,
                    "split": evalplus_task_split(task_id),
                },
                allowed_verifier_inputs={ENTRYPOINT_KEY: entrypoint},
            )
        )
    if not tasks:
        raise ValueError("EvalPlus problem mapping must not be empty")
    return tuple(tasks)


def tasks_from_mbpp_plus_problems(
    problems: Mapping[str, Mapping[str, Any]],
    *,
    dataset_version: str = EVALPLUS_MBPP_VERSION,
) -> tuple[Task, ...]:
    """Project MBPP+ records onto policy-visible confirmation tasks."""
    tasks: list[Task] = []
    for mapping_key, problem in sorted(problems.items()):
        task_id = _required_string(problem, "task_id")
        if task_id != mapping_key:
            raise ValueError(f"EvalPlus task key does not match task_id: {mapping_key}")
        prompt = _required_string(problem, "prompt")
        entrypoint = _required_string(problem, "entry_point")
        tasks.append(
            Task(
                task_id=task_id,
                prompt=prompt,
                task_family=EVALPLUS_MBPP_DATASET_NAME,
                difficulty_label="unlabeled",
                metadata={
                    "suite": EVALPLUS_MBPP_DATASET_NAME,
                    "dataset_version": dataset_version,
                    "entrypoint": entrypoint,
                    "split": "confirmation",
                },
                allowed_verifier_inputs={ENTRYPOINT_KEY: entrypoint},
            )
        )
    if not tasks:
        raise ValueError("EvalPlus problem mapping must not be empty")
    return tuple(tasks)


def evalplus_dataset_sha256(problems: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash the complete canonical dataset payload without exposing it to policies."""
    serialized = json.dumps(
        problems,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, set):
        return {"__set__": sorted(value, key=repr)}
    raise TypeError(f"unsupported EvalPlus value for hashing: {type(value).__name__}")


@lru_cache(maxsize=1)
def load_humaneval_plus_tasks() -> tuple[Task, ...]:
    """Load HumanEval+ through the optional pinned EvalPlus package."""
    try:
        evalplus_data = importlib.import_module("evalplus.data")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EvalPlus is optional; install the pinned evalplus dependency group first"
        ) from exc
    loader = getattr(evalplus_data, "get_human_eval_plus", None)
    if not callable(loader):
        raise RuntimeError("evalplus.data.get_human_eval_plus is unavailable")
    problems = cast(Mapping[str, Mapping[str, Any]], loader())
    return tasks_from_evalplus_problems(problems)


@lru_cache(maxsize=1)
def load_mbpp_plus_tasks() -> tuple[Task, ...]:
    """Load MBPP+ through the optional pinned EvalPlus package."""
    try:
        evalplus_data = importlib.import_module("evalplus.data")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EvalPlus is optional; install the pinned evalplus dependency group first"
        ) from exc
    loader = getattr(evalplus_data, "get_mbpp_plus", None)
    if not callable(loader):
        raise RuntimeError("evalplus.data.get_mbpp_plus is unavailable")
    problems = cast(
        Mapping[str, Mapping[str, Any]],
        loader(version=EVALPLUS_MBPP_VERSION),
    )
    return tasks_from_mbpp_plus_problems(problems)


def _required_string(problem: Mapping[str, Any], key: str) -> str:
    value = problem.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"EvalPlus problem requires nonempty string field: {key}")
    return value


__all__ = [
    "EVALPLUS_DATASET_NAME",
    "EVALPLUS_HUMANEVAL_VERSION",
    "EVALPLUS_MBPP_DATASET_NAME",
    "EVALPLUS_MBPP_VERSION",
    "EVALPLUS_PACKAGE_VERSION",
    "EVALPLUS_SPLIT_SALT",
    "EvalPlusSplit",
    "evalplus_dataset_sha256",
    "evalplus_task_split",
    "load_humaneval_plus_tasks",
    "load_mbpp_plus_tasks",
    "tasks_from_mbpp_plus_problems",
    "tasks_from_evalplus_problems",
]
