"""Pinned, resource-limited EvalPlus Docker integration."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    CandidateRecord,
    GradeScope,
    GradeStatus,
    PublicFailureFeedback,
    sha256_text,
)
from ttc_operatorbench.core.schema import Task

EVALPLUS_PACKAGE_VERSION = "0.3.1"
EVALPLUS_DOCKER_IMAGE = (
    "ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
)
PUBLIC_FAILURE_INPUT_LIMIT = 3
EvalPlusDataset = Literal["humaneval", "mbpp"]


class DockerUnavailableError(RuntimeError):
    """Raised when an opt-in external evaluation has no Docker runtime."""


@dataclass(frozen=True)
class EvalPlusDockerConfig:
    """Security and resource limits for one evaluator container."""

    image: str = EVALPLUS_DOCKER_IMAGE
    cpus: float = 2.0
    memory: str = "4g"
    pids_limit: int = 256
    timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError("image must not be empty")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.pids_limit <= 0:
            raise ValueError("pids_limit must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True)
class EvalPlusGradeBundle:
    """Separated public and hidden grades parsed from one official evaluation."""

    base_grades: tuple[CandidateGrade, ...]
    plus_grades: tuple[CandidateGrade, ...]
    official_dataset_hash: str


@dataclass(frozen=True)
class EvalPlusBaseGradeBundle:
    """Public-only grades parsed from one official base-only evaluation."""

    base_grades: tuple[CandidateGrade, ...]
    official_dataset_hash: str


def load_humaneval_plus_problems() -> dict[str, dict[str, Any]]:
    """Load full private evaluator records from exactly EvalPlus 0.3.1."""
    return load_evalplus_problems("humaneval")


def load_mbpp_plus_problems() -> dict[str, dict[str, Any]]:
    """Load full MBPP+ v0.2.0 evaluator records from exactly EvalPlus 0.3.1."""
    return load_evalplus_problems("mbpp")


def load_evalplus_problems(dataset: EvalPlusDataset) -> dict[str, dict[str, Any]]:
    """Load one supported full evaluator dataset through the pinned package."""
    _require_evalplus_version()
    evalplus_data = importlib.import_module("evalplus.data")
    loader_name = "get_human_eval_plus" if dataset == "humaneval" else "get_mbpp_plus"
    loader = getattr(evalplus_data, loader_name, None)
    if not callable(loader):
        raise RuntimeError(f"evalplus.data.{loader_name} is unavailable")
    if dataset == "mbpp":
        return cast(dict[str, dict[str, Any]], loader(version="v0.2.0"))
    return cast(dict[str, dict[str, Any]], loader())


def evalplus_dataset_from_manifest_name(dataset_name: str) -> EvalPlusDataset:
    """Map an internal dataset name to the official evaluator CLI key."""
    if dataset_name == "humaneval_plus":
        return "humaneval"
    if dataset_name == "mbpp_plus":
        return "mbpp"
    raise ValueError(f"unsupported EvalPlus manifest dataset: {dataset_name}")


def sanitize_evalplus_candidate(task: Task, candidate_text: str) -> str:
    """Sanitize one completion using the pinned official implementation."""
    _require_evalplus_version()
    sanitize_module = importlib.import_module("evalplus.sanitize")
    sanitizer = getattr(sanitize_module, "sanitize", None)
    if not callable(sanitizer):
        raise RuntimeError("evalplus.sanitize.sanitize is unavailable")
    entrypoint = task.metadata.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValueError("EvalPlus task requires an entrypoint for sanitization")
    sanitized = sanitizer(candidate_text, entrypoint=entrypoint)
    if not isinstance(sanitized, str):
        raise RuntimeError("EvalPlus sanitizer returned a non-string value")
    return sanitized


def _require_evalplus_version() -> None:
    try:
        installed_version = importlib.metadata.version("evalplus")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "EvalPlus is optional; install the pinned evalplus dependency group first"
        ) from exc
    if installed_version != EVALPLUS_PACKAGE_VERSION:
        raise RuntimeError(
            f"EvalPlus {EVALPLUS_PACKAGE_VERSION} is required; found {installed_version}"
        )


def write_evalplus_samples(path: Path, pool: CandidatePool) -> Path:
    """Write the official EvalPlus task/solution JSONL format."""
    return write_evalplus_candidate_samples(path, pool.candidates)


def write_evalplus_candidate_samples(
    path: Path,
    candidates: Sequence[CandidateRecord],
) -> Path:
    """Write an arbitrary ordered candidate batch in official sample format."""
    if not candidates:
        raise ValueError("EvalPlus candidate batch must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(
                json.dumps(
                    {
                        "task_id": candidate.task_id,
                        "solution": candidate.sanitized_code,
                    },
                    sort_keys=True,
                )
            )
            file.write("\n")
    return path


def write_evalplus_sample_index(path: Path, pool: CandidatePool) -> Path:
    """Write the private sidecar mapping evaluator rows to candidate identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample_index, candidate in enumerate(pool.candidates):
            file.write(
                json.dumps(
                    {
                        "sample_index": sample_index,
                        "pool_id": candidate.pool_id,
                        "task_id": candidate.task_id,
                        "candidate_index": candidate.candidate_index,
                        "sanitized_code_sha256": candidate.sanitized_code_sha256,
                    },
                    sort_keys=True,
                )
            )
            file.write("\n")
    return path


def write_evalplus_dataset_override(
    path: Path,
    problems: dict[str, dict[str, Any]],
    task_ids: tuple[str, ...],
    *,
    dataset: EvalPlusDataset = "humaneval",
) -> Path:
    """Write a private evaluator-only subset containing full EvalPlus records."""
    missing = sorted(set(task_ids) - set(problems))
    if missing:
        raise ValueError(f"EvalPlus override is missing tasks: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for task_id in task_ids:
            file.write(
                json.dumps(
                    _json_safe_problem_record(dataset, task_id, problems[task_id]),
                    sort_keys=True,
                )
            )
            file.write("\n")
    return path


def _json_safe_problem_record(
    dataset: EvalPlusDataset,
    task_id: str,
    problem: dict[str, Any],
) -> dict[str, Any]:
    if dataset == "humaneval":
        return problem
    _require_evalplus_version()
    mbpp_data = importlib.import_module("evalplus.data.mbpp")
    serializer = getattr(mbpp_data, "mbpp_serialize_inputs", None)
    if not callable(serializer):
        raise RuntimeError("evalplus.data.mbpp.mbpp_serialize_inputs is unavailable")
    record = dict(problem)
    for field in ("base_input", "plus_input"):
        if field not in record:
            raise ValueError(f"MBPP+ problem requires field: {field}")
        record[field] = serializer(task_id, record[field])
    return record


def build_evalplus_docker_command(
    work_directory: Path,
    samples_filename: str,
    *,
    base_only: bool,
    dataset: EvalPlusDataset = "humaneval",
    dataset_filename: str,
    output_directory: Path,
    config: EvalPlusDockerConfig | None = None,
) -> tuple[str, ...]:
    """Build a shell-free command with read-only inputs and isolated output."""
    resolved_work_directory = work_directory.resolve()
    if not resolved_work_directory.is_dir():
        raise ValueError("work_directory must exist")
    if Path(samples_filename).name != samples_filename:
        raise ValueError("samples_filename must be a basename")
    samples_path = resolved_work_directory / samples_filename
    if not samples_path.is_file():
        raise ValueError(f"samples file does not exist: {samples_path}")
    if Path(dataset_filename).name != dataset_filename:
        raise ValueError("dataset_filename must be a basename")
    dataset_path = resolved_work_directory / dataset_filename
    if not dataset_path.is_file():
        raise ValueError(f"dataset override does not exist: {dataset_path}")
    resolved_output_directory = output_directory.resolve()
    if not resolved_output_directory.is_dir():
        raise ValueError("output_directory must exist")
    runner_path = Path(__file__).with_name("evalplus_container_runner.py").resolve()
    if not runner_path.is_file():
        raise RuntimeError("trusted EvalPlus container runner is missing")
    limits = config or EvalPlusDockerConfig()
    # EvalPlus 0.3.1 crashes if EVALPLUS_TIMEOUT_PER_TASK is set because it
    # compares the string environment value with floats. Keep its native 60s limit.
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        str(limits.pids_limit),
        "--cpus",
        str(limits.cpus),
        "--memory",
        limits.memory,
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--env",
        "HOME=/tmp/evalplus-home",
        "--env",
        "XDG_CACHE_HOME=/tmp/evalplus-cache",
        "--mount",
        f"type=bind,src={samples_path},dst=/input/samples.jsonl,readonly",
        "--mount",
        f"type=bind,src={dataset_path},dst=/input/private_dataset.jsonl,readonly",
        "--mount",
        f"type=bind,src={runner_path},dst=/runner/evalplus_container_runner.py,readonly",
        "--mount",
        f"type=bind,src={resolved_output_directory},dst=/output",
        "--workdir",
        "/tmp",
        limits.image,
        "python",
        "/runner/evalplus_container_runner.py",
        "--dataset",
        dataset,
        "--samples",
        "/input/samples.jsonl",
        "--dataset-file",
        "/input/private_dataset.jsonl",
        "--output",
        "/output/samples_eval_results.json",
    ]
    if base_only:
        command.append("--base-only")
    return tuple(command)


def run_evalplus_docker(
    work_directory: Path,
    samples_filename: str,
    *,
    base_only: bool,
    dataset: EvalPlusDataset = "humaneval",
    dataset_filename: str,
    output_directory: Path,
    config: EvalPlusDockerConfig | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the pinned evaluator only when Docker is explicitly available."""
    if shutil.which("docker") is None:
        raise DockerUnavailableError(
            "Docker is required; generated benchmark code must not run on the host"
        )
    limits = config or EvalPlusDockerConfig()
    command = build_evalplus_docker_command(
        work_directory,
        samples_filename,
        base_only=base_only,
        dataset=dataset,
        dataset_filename=dataset_filename,
        output_directory=output_directory,
        config=limits,
    )
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=limits.timeout_seconds,
        check=False,
    )


def parse_evalplus_results(path: Path, pool: CandidatePool) -> EvalPlusGradeBundle:
    """Parse official v0.3.1 results and separate base from base-plus-extra grades."""
    return parse_evalplus_candidate_results(path, pool.candidates)


def parse_evalplus_base_results(path: Path, pool: CandidatePool) -> EvalPlusBaseGradeBundle:
    """Parse policy-visible base outcomes for one complete fixed candidate pool."""
    return parse_evalplus_base_candidate_results(path, pool.candidates)


def parse_evalplus_candidate_results(
    path: Path,
    candidates: Sequence[CandidateRecord],
) -> EvalPlusGradeBundle:
    """Parse base and plus outcomes for an arbitrary ordered candidate batch."""
    official_hash, rows = _validated_evalplus_rows(path, candidates)
    base_grades = tuple(
        _candidate_grade(
            candidate,
            scope="base",
            status_value=result.get("base_status"),
            public_feedback=_public_feedback(result),
        )
        for candidate, result in rows
    )
    plus_grades: list[CandidateGrade] = []
    for candidate, result in rows:
        base_status = result.get("base_status")
        plus_status = result.get("plus_status")
        if base_status == "pass" and plus_status is None:
            raise ValueError("base-passing full EvalPlus results require plus_status")
        combined_status = plus_status if base_status == "pass" else base_status
        plus_grades.append(
            _candidate_grade(
                candidate,
                scope="plus",
                status_value=combined_status,
                metadata={"base_status": base_status, "plus_status": plus_status},
            )
        )
    return EvalPlusGradeBundle(
        base_grades=base_grades,
        plus_grades=tuple(plus_grades),
        official_dataset_hash=official_hash,
    )


def parse_evalplus_base_candidate_results(
    path: Path,
    candidates: Sequence[CandidateRecord],
) -> EvalPlusBaseGradeBundle:
    """Parse only public base outcomes for an arbitrary ordered candidate batch."""
    official_hash, rows = _validated_evalplus_rows(path, candidates)
    grades = tuple(
        _candidate_grade(
            candidate,
            scope="base",
            status_value=result.get("base_status"),
            public_feedback=_public_feedback(result),
        )
        for candidate, result in rows
    )
    return EvalPlusBaseGradeBundle(
        base_grades=grades,
        official_dataset_hash=official_hash,
    )


def _validated_evalplus_rows(
    path: Path,
    candidates: Sequence[CandidateRecord],
) -> tuple[str, tuple[tuple[CandidateRecord, dict[str, Any]], ...]]:
    ordered_candidates = tuple(candidates)
    if not ordered_candidates:
        raise ValueError("EvalPlus candidate batch must not be empty")
    candidate_keys = [
        (candidate.pool_id, candidate.task_id, candidate.candidate_index)
        for candidate in ordered_candidates
    ]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("EvalPlus candidate batch contains duplicate identities")
    if candidate_keys != sorted(candidate_keys, key=lambda key: (key[1], key[2])):
        raise ValueError("EvalPlus candidate batch must use canonical task/index order")

    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    official_hash = raw.get("hash")
    evaluations = raw.get("eval")
    if not isinstance(official_hash, str) or not official_hash:
        raise ValueError("EvalPlus results require a dataset hash")
    if not isinstance(evaluations, dict):
        raise ValueError("EvalPlus results require an eval mapping")
    task_ids = tuple(dict.fromkeys(candidate.task_id for candidate in ordered_candidates))
    if set(evaluations) != set(task_ids):
        raise ValueError("EvalPlus result tasks must exactly match the candidate batch")

    rows: list[tuple[CandidateRecord, dict[str, Any]]] = []
    for task_id in task_ids:
        task_results = evaluations.get(task_id)
        task_candidates = tuple(
            candidate for candidate in ordered_candidates if candidate.task_id == task_id
        )
        if not isinstance(task_results, list) or len(task_results) != len(task_candidates):
            raise ValueError(f"EvalPlus result count mismatch for {task_id}")
        for result_index, result in enumerate(task_results):
            if not isinstance(result, dict):
                raise ValueError(f"invalid EvalPlus result for {task_id}/{result_index}")
            candidate = task_candidates[result_index]
            solution = result.get("solution")
            if (
                not isinstance(solution, str)
                or sha256_text(solution) != candidate.sanitized_code_sha256
            ):
                raise ValueError(
                    f"EvalPlus solution digest mismatch for "
                    f"{task_id}/{candidate.candidate_index}"
                )
            rows.append((candidate, result))
    return official_hash, tuple(rows)


def _public_feedback(result: dict[str, Any]) -> PublicFailureFeedback | None:
    status = _grade_status(result.get("base_status"))
    if status == "pass":
        return None
    failed_inputs = result.get("base_fail_tests", [])
    if not isinstance(failed_inputs, list):
        raise ValueError("EvalPlus base_fail_tests must be a list")
    retained = tuple(failed_inputs[:PUBLIC_FAILURE_INPUT_LIMIT])
    return PublicFailureFeedback(
        status=status,
        failed_inputs=retained,
        total_failed_inputs=len(failed_inputs),
        feedback_truncated=len(failed_inputs) > len(retained),
    )


def _grade_status(status_value: object) -> GradeStatus:
    return (
        cast(GradeStatus, status_value)
        if status_value in {"pass", "fail", "timeout"}
        else "error"
    )


def _candidate_grade(
    candidate: CandidateRecord,
    *,
    scope: GradeScope,
    status_value: object,
    public_feedback: PublicFailureFeedback | None = None,
    metadata: dict[str, Any] | None = None,
) -> CandidateGrade:
    status = _grade_status(status_value)
    passed = status == "pass"
    return CandidateGrade(
        pool_id=candidate.pool_id,
        task_id=candidate.task_id,
        candidate_index=candidate.candidate_index,
        sanitized_code_sha256=candidate.sanitized_code_sha256,
        scope=scope,
        status=status,
        verification_passed=passed,
        error_type=None if passed else f"evalplus_{status}",
        public_feedback=public_feedback,
        metadata=metadata or {"status": status_value},
    )


__all__ = [
    "DockerUnavailableError",
    "EVALPLUS_DOCKER_IMAGE",
    "EVALPLUS_PACKAGE_VERSION",
    "EvalPlusDataset",
    "PUBLIC_FAILURE_INPUT_LIMIT",
    "EvalPlusBaseGradeBundle",
    "EvalPlusGradeBundle",
    "EvalPlusDockerConfig",
    "build_evalplus_docker_command",
    "evalplus_dataset_from_manifest_name",
    "load_evalplus_problems",
    "load_humaneval_plus_problems",
    "load_mbpp_plus_problems",
    "parse_evalplus_base_candidate_results",
    "parse_evalplus_base_results",
    "parse_evalplus_candidate_results",
    "parse_evalplus_results",
    "run_evalplus_docker",
    "sanitize_evalplus_candidate",
    "write_evalplus_candidate_samples",
    "write_evalplus_sample_index",
    "write_evalplus_samples",
    "write_evalplus_dataset_override",
]
