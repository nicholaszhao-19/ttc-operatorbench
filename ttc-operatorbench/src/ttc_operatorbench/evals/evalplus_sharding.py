"""Task-preserving candidate shards for bounded EvalPlus containers."""

from __future__ import annotations

from ttc_operatorbench.core.candidate_pool import CandidateRecord

MBPP_TASKS_PER_CONTAINER = 10


def shard_candidates_by_task(
    candidates: tuple[CandidateRecord, ...],
    *,
    max_tasks_per_shard: int | None,
) -> tuple[tuple[CandidateRecord, ...], ...]:
    """Split candidates without separating candidates from the same task."""
    if not candidates:
        raise ValueError("candidate batch must not be empty")
    if max_tasks_per_shard is None:
        return (candidates,)
    if max_tasks_per_shard <= 0:
        raise ValueError("max_tasks_per_shard must be positive")
    task_ids = tuple(sorted({candidate.task_id for candidate in candidates}))
    shards: list[tuple[CandidateRecord, ...]] = []
    for offset in range(0, len(task_ids), max_tasks_per_shard):
        shard_task_ids = set(task_ids[offset : offset + max_tasks_per_shard])
        shards.append(
            tuple(
                candidate
                for candidate in candidates
                if candidate.task_id in shard_task_ids
            )
        )
    return tuple(shards)


__all__ = ["MBPP_TASKS_PER_CONTAINER", "shard_candidates_by_task"]
