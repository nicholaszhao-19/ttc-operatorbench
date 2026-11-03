"""Deterministic model-provider replay from an immutable candidate pool."""

from __future__ import annotations

from dataclasses import dataclass, field

from ttc_operatorbench.core.candidate_pool import CandidatePool
from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task


class CandidatePoolExhaustedError(RuntimeError):
    """Raised when a policy requests more candidates than the frozen pool."""


@dataclass
class CandidatePoolReplayProvider:
    """Replay each task's candidates in canonical index order."""

    pool: CandidatePool
    provider_name: str = "candidate_pool_replay"
    _next_index_by_task: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        """Return the next frozen candidate for a task."""
        candidate_index = self._next_index_by_task.get(task.task_id, 0)
        task_candidates = self.pool.candidates_for_task(task.task_id)
        if candidate_index >= len(task_candidates):
            raise CandidatePoolExhaustedError(
                f"candidate pool exhausted for {task.task_id} at index {candidate_index}"
            )
        candidate = task_candidates[candidate_index]
        self._next_index_by_task[task.task_id] = candidate_index + 1
        requested_sampling = sampling.model_dump(mode="json") if sampling is not None else None
        return candidate.generation.model_copy(
            update={
                "provider_name": self.provider_name,
                "metadata": {
                    **candidate.generation.metadata,
                    "source_provider_name": candidate.generation.provider_name,
                    "candidate_pool_id": candidate.pool_id,
                    "candidate_task_id": candidate.task_id,
                    "candidate_index": candidate.candidate_index,
                    "candidate_raw_completion_sha256": candidate.raw_completion_sha256,
                    "candidate_sanitized_code_sha256": candidate.sanitized_code_sha256,
                    "requested_replay_sampling": requested_sampling,
                },
            }
        )


__all__ = ["CandidatePoolExhaustedError", "CandidatePoolReplayProvider"]
