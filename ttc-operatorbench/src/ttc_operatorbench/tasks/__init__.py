"""Task definitions and loaders."""

from ttc_operatorbench.tasks.curated_code import (
    CURATED_CODE_TASK_SPECS,
    CURATED_REFERENCE_CANDIDATES,
    CuratedCodeTaskSpec,
    curated_task_ids,
    get_curated_task,
    list_curated_tasks,
)
from ttc_operatorbench.tasks.registry import (
    TaskSuite,
    get_task,
    list_task_ids,
    list_tasks,
    validate_task_ids,
)
from ttc_operatorbench.tasks.toy_code import (
    TOY_CODE_TASK_SPECS,
    ToyCodeTaskSpec,
    get_toy_task,
    list_toy_tasks,
    toy_task_ids,
)

__all__ = [
    "CURATED_CODE_TASK_SPECS",
    "CURATED_REFERENCE_CANDIDATES",
    "TOY_CODE_TASK_SPECS",
    "CuratedCodeTaskSpec",
    "TaskSuite",
    "ToyCodeTaskSpec",
    "curated_task_ids",
    "get_curated_task",
    "get_task",
    "get_toy_task",
    "list_curated_tasks",
    "list_task_ids",
    "list_tasks",
    "list_toy_tasks",
    "toy_task_ids",
    "validate_task_ids",
]
