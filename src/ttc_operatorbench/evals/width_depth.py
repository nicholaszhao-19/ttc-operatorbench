"""Fixed width-depth generation with public-verifier stopping and repair."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
)
from ttc_operatorbench.core.schema import SamplingConfig, Task
from ttc_operatorbench.core.trajectory import (
    TrajectoryOperator,
    TrajectoryStep,
    WidthDepthTrajectoryHeader,
    WidthDepthTrajectoryPool,
)
from ttc_operatorbench.search.baselines import ModelProvider

CandidateSanitizer = Callable[[Task, str], str]


class PublicBatchEvaluator(Protocol):
    """Evaluate one candidate batch using public signals only."""

    def evaluate(
        self,
        batch_id: str,
        candidates: tuple[CandidateRecord, ...],
    ) -> tuple[CandidateGrade, ...]:
        """Return one base grade for every candidate in canonical order."""


class WidthDepthGenerationError(RuntimeError):
    """Raised with task and call context when trajectory generation fails."""


@dataclass
class _TaskState:
    task: Task
    steps: list[TrajectoryStep] = field(default_factory=list)
    root_heads: dict[int, TrajectoryStep] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return bool(self.steps and self.steps[-1].selected)


@dataclass(frozen=True)
class _PendingStep:
    candidate: CandidateRecord
    operator: TrajectoryOperator
    root_index: int
    depth: int
    round_index: int
    parent_candidate_index: int | None


def run_width_depth_search(
    manifest: CandidatePoolManifest,
    tasks: Sequence[Task],
    provider: ModelProvider,
    sanitizer: CandidateSanitizer,
    evaluator: PublicBatchEvaluator,
    *,
    width: int,
    depth: int,
) -> WidthDepthTrajectoryPool:
    """Run one deterministic fixed width-depth policy with stop-on-public-pass."""
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    if width * depth != manifest.pool_size:
        raise ValueError("width * depth must equal manifest.pool_size")
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks) or set(tasks_by_id) != set(manifest.task_ids):
        raise ValueError("tasks must uniquely and exactly match manifest.task_ids")
    states = {
        task_id: _TaskState(task=tasks_by_id[task_id])
        for task_id in sorted(manifest.task_ids)
    }
    round_index = 0

    for root_index in range(width):
        pending = tuple(
            _generate_pending_step(
                manifest,
                state,
                provider,
                sanitizer,
                operator="sample",
                root_index=root_index,
                depth=0,
                round_index=round_index,
                parent=None,
            )
            for state in states.values()
            if not state.resolved
        )
        if not pending:
            break
        _grade_and_attach(pending, states, evaluator, batch_id=f"root-{root_index}")
        round_index += 1

    for repair_depth in range(1, depth):
        for root_index in range(width):
            pending = tuple(
                _generate_pending_step(
                    manifest,
                    state,
                    provider,
                    sanitizer,
                    operator="repair",
                    root_index=root_index,
                    depth=repair_depth,
                    round_index=round_index,
                    parent=state.root_heads[root_index],
                )
                for state in states.values()
                if not state.resolved
            )
            if not pending:
                break
            _grade_and_attach(
                pending,
                states,
                evaluator,
                batch_id=f"repair-{repair_depth}-{root_index}",
            )
            round_index += 1
        if all(state.resolved for state in states.values()):
            break

    steps = tuple(
        step
        for task_id in sorted(states)
        for step in states[task_id].steps
    )
    return WidthDepthTrajectoryPool(
        header=WidthDepthTrajectoryHeader(
            width=width,
            depth=depth,
            candidate_manifest=manifest,
        ),
        steps=steps,
    )


def _generate_pending_step(
    manifest: CandidatePoolManifest,
    state: _TaskState,
    provider: ModelProvider,
    sanitizer: CandidateSanitizer,
    *,
    operator: TrajectoryOperator,
    root_index: int,
    depth: int,
    round_index: int,
    parent: TrajectoryStep | None,
) -> _PendingStep:
    candidate_index = len(state.steps)
    task_for_generation = (
        state.task
        if parent is None
        else state.task.model_copy(update={"prompt": _repair_prompt(state.task, parent)})
    )
    sampling = SamplingConfig(
        temperature=manifest.temperature,
        top_p=manifest.top_p,
        do_sample=True,
        max_output_tokens=manifest.max_output_tokens,
        seed=manifest.pool_seed,
        seed_offset=candidate_index,
    )
    try:
        generation = provider.generate(task_for_generation, sampling)
        sanitized_code = sanitizer(state.task, generation.generation_text)
    except Exception as exc:
        raise WidthDepthGenerationError(
            f"trajectory generation failed for {state.task.task_id}/{candidate_index}"
        ) from exc
    parent_index = None if parent is None else parent.candidate.candidate_index
    candidate = CandidateRecord(
        pool_id=manifest.pool_id,
        task_id=state.task.task_id,
        candidate_index=candidate_index,
        generation=generation,
        sanitized_code=sanitized_code,
        prompt_sha256=sha256_text(generation.prompt),
        raw_completion_sha256=sha256_text(generation.generation_text),
        sanitized_code_sha256=sha256_text(sanitized_code),
        metadata={
            "trajectory_operator": operator,
            "trajectory_root_index": root_index,
            "trajectory_depth": depth,
            "trajectory_round_index": round_index,
            "trajectory_parent_candidate_index": parent_index,
            "requested_sampling": sampling.model_dump(mode="json"),
        },
    )
    return _PendingStep(
        candidate=candidate,
        operator=operator,
        root_index=root_index,
        depth=depth,
        round_index=round_index,
        parent_candidate_index=parent_index,
    )


def _grade_and_attach(
    pending: tuple[_PendingStep, ...],
    states: Mapping[str, _TaskState],
    evaluator: PublicBatchEvaluator,
    *,
    batch_id: str,
) -> None:
    candidates = tuple(item.candidate for item in pending)
    grades = evaluator.evaluate(batch_id, candidates)
    grade_index = _validated_grade_index(candidates, grades)
    for item in pending:
        grade = grade_index[
            (item.candidate.pool_id, item.candidate.task_id, item.candidate.candidate_index)
        ]
        step = TrajectoryStep(
            candidate=item.candidate,
            public_grade=grade,
            operator=item.operator,
            root_index=item.root_index,
            depth=item.depth,
            round_index=item.round_index,
            parent_candidate_index=item.parent_candidate_index,
            selected=grade.verification_passed,
        )
        state = states[item.candidate.task_id]
        state.steps.append(step)
        state.root_heads[item.root_index] = step


def _validated_grade_index(
    candidates: tuple[CandidateRecord, ...],
    grades: tuple[CandidateGrade, ...],
) -> dict[tuple[str, str, int], CandidateGrade]:
    expected = {
        (candidate.pool_id, candidate.task_id, candidate.candidate_index): candidate
        for candidate in candidates
    }
    observed: dict[tuple[str, str, int], CandidateGrade] = {}
    for grade in grades:
        candidate = expected.get(grade.key)
        if grade.scope != "base" or candidate is None:
            raise ValueError(f"public evaluator returned an unexpected grade: {grade.key}")
        if grade.sanitized_code_sha256 != candidate.sanitized_code_sha256:
            raise ValueError(f"public grade digest mismatch: {grade.key}")
        if grade.key in observed:
            raise ValueError(f"public evaluator returned a duplicate grade: {grade.key}")
        observed[grade.key] = grade
    if set(observed) != set(expected):
        raise ValueError("public evaluator must grade every batch candidate exactly once")
    return observed


def _repair_prompt(task: Task, parent: TrajectoryStep) -> str:
    feedback = parent.public_grade.public_feedback
    failed_inputs = [] if feedback is None else list(feedback.failed_inputs)
    feedback_json = json.dumps(failed_inputs, ensure_ascii=True, sort_keys=True)
    return (
        f"{task.prompt}\n\n"
        "Previous candidate:\n```python\n"
        f"{parent.candidate.sanitized_code}\n```\n\n"
        f"Public verifier status: {parent.public_grade.status}\n"
        f"Public verifier error type: {parent.public_grade.error_type}\n"
        f"Failing public inputs (JSON): {feedback_json}\n\n"
        "Repair the candidate for the original problem. Return only valid Python code. "
        "Do not include Markdown fences, explanations, print calls, or tests."
    )


__all__ = [
    "PublicBatchEvaluator",
    "WidthDepthGenerationError",
    "run_width_depth_search",
]
