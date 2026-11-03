"""Fixed-sample candidate-pool generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ttc_operatorbench.core.candidate_pool import (
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
)
from ttc_operatorbench.core.schema import SamplingConfig, Task
from ttc_operatorbench.search.baselines import ModelProvider

CandidateSanitizer = Callable[[Task, str], str]


class CandidateGenerationError(RuntimeError):
    """Raised with task/index context when fixed-pool generation fails."""


def generate_candidate_pool(
    manifest: CandidatePoolManifest,
    tasks: Sequence[Task],
    provider: ModelProvider,
    sanitizer: CandidateSanitizer,
) -> CandidatePool:
    """Generate every manifest candidate before returning a valid immutable pool."""
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("tasks must have unique task_id values")
    if set(tasks_by_id) != set(manifest.task_ids):
        raise ValueError("tasks must exactly match manifest.task_ids")

    candidates: list[CandidateRecord] = []
    for task_id in manifest.task_ids:
        task = tasks_by_id[task_id]
        for candidate_index in range(manifest.pool_size):
            sampling = SamplingConfig(
                temperature=manifest.temperature,
                top_p=manifest.top_p,
                do_sample=True,
                max_output_tokens=manifest.max_output_tokens,
                seed=manifest.pool_seed,
                seed_offset=candidate_index,
            )
            try:
                generation = provider.generate(task, sampling)
                sanitized_code = sanitizer(task, generation.generation_text)
            except Exception as exc:
                raise CandidateGenerationError(
                    f"candidate generation failed for {task_id}/{candidate_index}"
                ) from exc
            candidates.append(
                CandidateRecord(
                    pool_id=manifest.pool_id,
                    task_id=task_id,
                    candidate_index=candidate_index,
                    generation=generation,
                    sanitized_code=sanitized_code,
                    prompt_sha256=sha256_text(generation.prompt),
                    raw_completion_sha256=sha256_text(generation.generation_text),
                    sanitized_code_sha256=sha256_text(sanitized_code),
                    metadata={
                        "task_split": task.metadata.get("split"),
                        "requested_sampling": sampling.model_dump(mode="json"),
                    },
                )
            )
    return CandidatePool(manifest=manifest, candidates=tuple(candidates))


__all__ = [
    "CandidateGenerationError",
    "CandidateSanitizer",
    "generate_candidate_pool",
]
