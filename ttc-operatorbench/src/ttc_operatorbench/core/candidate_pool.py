"""Immutable candidate-pool and external-grade contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from ttc_operatorbench.core.schema import Generation, SchemaModel

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GradeScope = Literal["base", "plus"]
GradeStatus = Literal["pass", "fail", "timeout", "error"]


def sha256_text(value: str) -> str:
    """Return a lowercase SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CandidatePoolManifest(SchemaModel):
    """Frozen provenance shared by every candidate in one pool."""

    schema_version: Literal["1"] = "1"
    pool_id: NonEmptyStr
    dataset_name: NonEmptyStr
    dataset_version: NonEmptyStr
    dataset_sha256: Sha256Hex
    repository_commit: NonEmptyStr
    task_ids: tuple[NonEmptyStr, ...]
    model_id: NonEmptyStr
    model_revision: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    provider_name: NonEmptyStr
    prompt_style: NonEmptyStr
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_output_tokens: PositiveInt
    pool_size: PositiveInt
    pool_seed: NonNegativeInt
    created_at_utc: NonEmptyStr
    hardware: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        if not self.task_ids:
            raise ValueError("task_ids must not be empty")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        return self


class CandidateRecord(SchemaModel):
    """One generated candidate with content-addressed identity."""

    pool_id: NonEmptyStr
    task_id: NonEmptyStr
    candidate_index: NonNegativeInt
    generation: Generation
    sanitized_code: str
    prompt_sha256: Sha256Hex
    raw_completion_sha256: Sha256Hex
    sanitized_code_sha256: Sha256Hex
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        if self.prompt_sha256 != sha256_text(self.generation.prompt):
            raise ValueError("prompt_sha256 must match generation.prompt")
        if self.raw_completion_sha256 != sha256_text(self.generation.generation_text):
            raise ValueError("raw_completion_sha256 must match generation text")
        if self.sanitized_code_sha256 != sha256_text(self.sanitized_code):
            raise ValueError("sanitized_code_sha256 must match sanitized_code")
        return self


class CandidatePool(SchemaModel):
    """Complete ordered candidates for every task in a manifest."""

    manifest: CandidatePoolManifest
    candidates: tuple[CandidateRecord, ...]

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        expected = {
            (task_id, candidate_index)
            for task_id in self.manifest.task_ids
            for candidate_index in range(self.manifest.pool_size)
        }
        observed: set[tuple[str, int]] = set()
        for candidate in self.candidates:
            if candidate.pool_id != self.manifest.pool_id:
                raise ValueError("candidate pool_id must match manifest")
            key = (candidate.task_id, candidate.candidate_index)
            if key in observed:
                raise ValueError("candidate task/index pairs must be unique")
            observed.add(key)
        ordered_keys = tuple(
            (candidate.task_id, candidate.candidate_index) for candidate in self.candidates
        )
        if ordered_keys != tuple(sorted(ordered_keys)):
            raise ValueError("candidates must use canonical task/index order")
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"candidate pool must be complete; missing={missing}, unexpected={unexpected}"
            )
        return self

    def candidates_for_task(self, task_id: str) -> tuple[CandidateRecord, ...]:
        """Return one task's candidates in frozen index order."""
        if task_id not in self.manifest.task_ids:
            raise KeyError(f"task is not in candidate pool: {task_id}")
        return tuple(
            sorted(
                (candidate for candidate in self.candidates if candidate.task_id == task_id),
                key=lambda candidate: candidate.candidate_index,
            )
        )


class CandidateGrade(SchemaModel):
    """One external evaluator outcome keyed to a candidate digest."""

    pool_id: NonEmptyStr
    task_id: NonEmptyStr
    candidate_index: NonNegativeInt
    sanitized_code_sha256: Sha256Hex
    scope: GradeScope
    status: GradeStatus
    verification_passed: bool
    runtime_seconds: NonNegativeFloat = 0.0
    error_type: str | None = None
    stdout: str = ""
    stderr: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.verification_passed != (self.status == "pass"):
            raise ValueError("verification_passed must agree with grade status")
        return self

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the stable pool/task/index lookup key."""
        return (self.pool_id, self.task_id, self.candidate_index)


def write_candidate_pool(directory: Path, pool: CandidatePool) -> tuple[Path, Path]:
    """Write a manifest and deterministic candidate JSONL file."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    candidates_path = directory / "candidates.jsonl"
    manifest_path.write_text(
        json.dumps(pool.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with candidates_path.open("w", encoding="utf-8") as file:
        for candidate in pool.candidates:
            file.write(candidate.model_dump_json())
            file.write("\n")
    return manifest_path, candidates_path


def read_candidate_pool(directory: Path) -> CandidatePool:
    """Read and validate a candidate pool directory."""
    manifest = CandidatePoolManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    candidates = tuple(
        CandidateRecord.model_validate_json(line)
        for line in (directory / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return CandidatePool(manifest=manifest, candidates=candidates)


def write_candidate_grades(path: Path, grades: tuple[CandidateGrade, ...]) -> Path:
    """Write grades in stable task/index order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(grades, key=lambda grade: (grade.task_id, grade.candidate_index))
    with path.open("w", encoding="utf-8") as file:
        for grade in ordered:
            file.write(grade.model_dump_json())
            file.write("\n")
    return path


def read_candidate_grades(path: Path) -> tuple[CandidateGrade, ...]:
    """Read candidate grades from JSONL."""
    return tuple(
        CandidateGrade.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


__all__ = [
    "CandidateGrade",
    "CandidatePool",
    "CandidatePoolManifest",
    "CandidateRecord",
    "GradeScope",
    "GradeStatus",
    "read_candidate_grades",
    "read_candidate_pool",
    "sha256_text",
    "write_candidate_grades",
    "write_candidate_pool",
]
