"""Typed provenance for variable-length width-depth search trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePoolManifest,
    CandidateRecord,
)
from ttc_operatorbench.core.schema import SchemaModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
TrajectoryOperator = Literal["sample", "repair"]


class TrajectoryStep(SchemaModel):
    """One generated and publicly graded node in a search trajectory."""

    candidate: CandidateRecord
    public_grade: CandidateGrade
    operator: TrajectoryOperator
    root_index: NonNegativeInt
    depth: NonNegativeInt
    round_index: NonNegativeInt
    parent_candidate_index: NonNegativeInt | None = None
    selected: bool = False

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        candidate_key = (
            self.candidate.pool_id,
            self.candidate.task_id,
            self.candidate.candidate_index,
        )
        if self.public_grade.key != candidate_key:
            raise ValueError("public grade identity must match trajectory candidate")
        if self.public_grade.scope != "base":
            raise ValueError("trajectory steps may contain only public base grades")
        if self.public_grade.sanitized_code_sha256 != self.candidate.sanitized_code_sha256:
            raise ValueError("public grade digest must match trajectory candidate")
        if self.operator == "sample":
            if self.depth != 0 or self.parent_candidate_index is not None:
                raise ValueError("sample roots require depth zero and no parent")
        elif self.depth == 0 or self.parent_candidate_index is None:
            raise ValueError("repair steps require positive depth and a parent")
        if (
            self.parent_candidate_index is not None
            and self.parent_candidate_index >= self.candidate.candidate_index
        ):
            raise ValueError("repair parent must precede its child")
        if self.selected and not self.public_grade.verification_passed:
            raise ValueError("only a public-passing step may be selected")
        return self


class WidthDepthTrajectoryHeader(SchemaModel):
    """Frozen policy and generation provenance for one trajectory pool."""

    schema_version: Literal["1"] = "1"
    policy_name: NonEmptyStr = "stop_then_escalate"
    width: PositiveInt
    depth: PositiveInt
    candidate_manifest: CandidatePoolManifest

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.width * self.depth != self.candidate_manifest.pool_size:
            raise ValueError("width * depth must equal manifest pool_size")
        return self


class WidthDepthTrajectoryPool(SchemaModel):
    """Complete stopped trajectories for every task in one fixed policy run."""

    header: WidthDepthTrajectoryHeader
    steps: tuple[TrajectoryStep, ...]

    @model_validator(mode="after")
    def validate_trajectories(self) -> Self:
        task_ids = set(self.header.candidate_manifest.task_ids)
        if not self.steps:
            raise ValueError("trajectory pool must contain steps")
        if {step.candidate.task_id for step in self.steps} != task_ids:
            raise ValueError("trajectory steps must cover every manifest task")
        ordered_keys = tuple(
            (step.candidate.task_id, step.candidate.candidate_index) for step in self.steps
        )
        if ordered_keys != tuple(sorted(ordered_keys)):
            raise ValueError("trajectory steps must use canonical task/index order")
        for task_id in sorted(task_ids):
            task_steps = tuple(
                step for step in self.steps if step.candidate.task_id == task_id
            )
            self._validate_task(task_id, task_steps)
        return self

    def _validate_task(
        self,
        task_id: str,
        task_steps: tuple[TrajectoryStep, ...],
    ) -> None:
        indexes = tuple(step.candidate.candidate_index for step in task_steps)
        if indexes != tuple(range(len(task_steps))):
            raise ValueError(f"trajectory indexes must be contiguous for {task_id}")
        max_calls = self.header.candidate_manifest.pool_size
        if len(task_steps) > max_calls:
            raise ValueError(f"trajectory exceeds maximum calls for {task_id}")
        by_index = {step.candidate.candidate_index: step for step in task_steps}
        passing = [step for step in task_steps if step.public_grade.verification_passed]
        selected = [step for step in task_steps if step.selected]
        if passing:
            if len(passing) != 1 or selected != passing or passing[0] != task_steps[-1]:
                raise ValueError(f"trajectory must stop on its first public pass for {task_id}")
        elif selected or len(task_steps) != max_calls:
            raise ValueError(f"unresolved trajectory must exhaust its budget for {task_id}")
        for step in task_steps:
            if step.operator != "repair":
                continue
            parent_index = step.parent_candidate_index
            if parent_index is None:
                raise ValueError(f"repair parent is missing for {task_id}")
            parent = by_index.get(parent_index)
            if parent is None:
                raise ValueError(f"repair parent is missing for {task_id}")
            if parent.root_index != step.root_index or parent.depth + 1 != step.depth:
                raise ValueError(f"repair lineage is inconsistent for {task_id}")

    def steps_for_task(self, task_id: str) -> tuple[TrajectoryStep, ...]:
        """Return one task trajectory in call order."""
        if task_id not in self.header.candidate_manifest.task_ids:
            raise KeyError(f"task is not in trajectory pool: {task_id}")
        return tuple(
            step for step in self.steps if step.candidate.task_id == task_id
        )


def write_trajectory_pool(
    directory: Path,
    pool: WidthDepthTrajectoryPool,
) -> tuple[Path, Path]:
    """Write a trajectory header and deterministic step JSONL."""
    directory.mkdir(parents=True, exist_ok=True)
    header_path = directory / "trajectory_manifest.json"
    steps_path = directory / "trajectory_steps.jsonl"
    header_path.write_text(
        json.dumps(pool.header.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with steps_path.open("w", encoding="utf-8") as file:
        for step in pool.steps:
            file.write(step.model_dump_json())
            file.write("\n")
    return header_path, steps_path


def read_trajectory_pool(directory: Path) -> WidthDepthTrajectoryPool:
    """Read and validate a trajectory pool directory."""
    header = WidthDepthTrajectoryHeader.model_validate_json(
        (directory / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    steps = tuple(
        TrajectoryStep.model_validate_json(line)
        for line in (directory / "trajectory_steps.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    return WidthDepthTrajectoryPool(header=header, steps=steps)


__all__ = [
    "TrajectoryOperator",
    "TrajectoryStep",
    "WidthDepthTrajectoryHeader",
    "WidthDepthTrajectoryPool",
    "read_trajectory_pool",
    "write_trajectory_pool",
]
