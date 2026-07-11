"""Post-search hidden EvalPlus grading for immutable width-depth trajectories."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ttc_operatorbench.core.candidate_pool import CandidateGrade, write_candidate_grades
from ttc_operatorbench.core.trajectory import WidthDepthTrajectoryPool
from ttc_operatorbench.systems.evalplus import (
    EvalPlusDockerConfig,
    build_evalplus_docker_command,
    evalplus_dataset_from_manifest_name,
    parse_evalplus_candidate_results,
    run_evalplus_docker,
    write_evalplus_candidate_samples,
    write_evalplus_dataset_override,
)
from ttc_operatorbench.tasks.evalplus import evalplus_dataset_sha256


@dataclass(frozen=True)
class EvalPlusTrajectoryHiddenResult:
    """Separated grades and artifact directory from one post-search evaluation."""

    output_directory: Path
    base_grades: tuple[CandidateGrade, ...]
    plus_grades: tuple[CandidateGrade, ...]


def evaluate_evalplus_trajectory_hidden(
    trajectory_directory: Path,
    pool: WidthDepthTrajectoryPool,
    problems: dict[str, dict[str, Any]],
    *,
    config: EvalPlusDockerConfig | None = None,
) -> EvalPlusTrajectoryHiddenResult:
    """Grade a completed trajectory without exposing hidden labels to search."""
    trajectory_directory = trajectory_directory.resolve()
    output_directory = trajectory_directory / "hidden_evaluation"
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite hidden evaluation: {output_directory}")
    output_directory.mkdir(parents=True)
    limits = config or EvalPlusDockerConfig()

    manifest = pool.header.candidate_manifest
    dataset = evalplus_dataset_from_manifest_name(manifest.dataset_name)
    observed_dataset_sha256 = evalplus_dataset_sha256(problems)
    if observed_dataset_sha256 != manifest.dataset_sha256:
        raise ValueError("loaded EvalPlus dataset does not match trajectory manifest")
    candidates = tuple(step.candidate for step in pool.steps)
    samples_path = write_evalplus_candidate_samples(
        output_directory / "samples.jsonl",
        candidates,
    )
    dataset_path = write_evalplus_dataset_override(
        output_directory / "private_dataset.jsonl",
        problems,
        manifest.task_ids,
        dataset=dataset,
    )

    with tempfile.TemporaryDirectory(
        prefix=".evalplus-hidden-output-",
        dir=output_directory,
    ) as temporary_output:
        temporary_output_directory = Path(temporary_output)
        command = build_evalplus_docker_command(
            output_directory,
            samples_path.name,
            base_only=False,
            dataset=dataset,
            dataset_filename=dataset_path.name,
            output_directory=temporary_output_directory,
            config=limits,
        )
        started_at = time.perf_counter()
        completed = run_evalplus_docker(
            output_directory,
            samples_path.name,
            base_only=False,
            dataset=dataset,
            dataset_filename=dataset_path.name,
            output_directory=temporary_output_directory,
            config=limits,
        )
        elapsed_seconds = time.perf_counter() - started_at
        stdout_path = output_directory / "evalplus_stdout.log"
        stderr_path = output_directory / "evalplus_stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"EvalPlus hidden evaluation failed with exit code {completed.returncode}; "
                f"see {stderr_path}"
            )
        temporary_results = temporary_output_directory / "samples_eval_results.json"
        if not temporary_results.is_file():
            raise RuntimeError(
                "EvalPlus hidden evaluation exited successfully without a results file; "
                f"see {stderr_path}"
            )
        results_path = output_directory / "samples_eval_results.json"
        shutil.copyfile(temporary_results, results_path)

    bundle = parse_evalplus_candidate_results(results_path, candidates)
    _validate_base_recheck(pool, bundle.base_grades)
    base_grades_path = write_candidate_grades(
        output_directory / "base_recheck_grades.jsonl",
        bundle.base_grades,
    )
    plus_grades_path = write_candidate_grades(
        output_directory / "hidden_plus_grades.jsonl",
        bundle.plus_grades,
    )
    evaluation_manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "trajectory_pool_id": manifest.pool_id,
        "candidate_count": len(candidates),
        "task_ids": list(manifest.task_ids),
        "docker_image": limits.image,
        "dataset": dataset,
        "base_only": False,
        "search_was_complete_before_hidden_evaluation": True,
        "base_recheck_matches_search": True,
        "command": list(command),
        "elapsed_seconds": elapsed_seconds,
        "official_dataset_hash": bundle.official_dataset_hash,
        "input_sha256": {
            "trajectory_manifest": _sha256_file(
                trajectory_directory / "trajectory_manifest.json"
            ),
            "trajectory_steps": _sha256_file(
                trajectory_directory / "trajectory_steps.jsonl"
            ),
            "samples": _sha256_file(samples_path),
            "dataset_override": _sha256_file(dataset_path),
        },
        "output_sha256": {
            "results": _sha256_file(results_path),
            "base_recheck_grades": _sha256_file(base_grades_path),
            "hidden_plus_grades": _sha256_file(plus_grades_path),
        },
    }
    (output_directory / "evaluator_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvalPlusTrajectoryHiddenResult(
        output_directory=output_directory,
        base_grades=bundle.base_grades,
        plus_grades=bundle.plus_grades,
    )


def _validate_base_recheck(
    pool: WidthDepthTrajectoryPool,
    observed_grades: tuple[CandidateGrade, ...],
) -> None:
    expected = {step.public_grade.key: step.public_grade for step in pool.steps}
    observed = {grade.key: grade for grade in observed_grades}
    if len(observed) != len(observed_grades) or set(observed) != set(expected):
        raise ValueError("hidden evaluation base grades do not cover the trajectory exactly")
    for key, expected_grade in expected.items():
        observed_grade = observed[key]
        if (
            observed_grade.status != expected_grade.status
            or observed_grade.verification_passed != expected_grade.verification_passed
            or observed_grade.error_type != expected_grade.error_type
            or observed_grade.public_feedback != expected_grade.public_feedback
            or observed_grade.sanitized_code_sha256
            != expected_grade.sanitized_code_sha256
        ):
            raise ValueError(f"hidden evaluation base recheck disagrees with search: {key}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EvalPlusTrajectoryHiddenResult",
    "evaluate_evalplus_trajectory_hidden",
]
