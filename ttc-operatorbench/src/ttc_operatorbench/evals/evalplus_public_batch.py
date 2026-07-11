"""Immutable public-only EvalPlus batch evaluation for search trajectories."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidateRecord,
    write_candidate_grades,
)
from ttc_operatorbench.evals.evalplus_sharding import shard_candidates_by_task
from ttc_operatorbench.systems.evalplus import (
    EvalPlusDataset,
    EvalPlusDockerConfig,
    build_evalplus_docker_command,
    parse_evalplus_base_candidate_results,
    run_evalplus_docker,
    write_evalplus_candidate_samples,
    write_evalplus_dataset_override,
)

_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class EvalPlusPublicBatchEvaluator:
    """Grade trajectory batches with base tests in the pinned container."""

    def __init__(
        self,
        run_directory: Path,
        problems: dict[str, dict[str, Any]],
        *,
        dataset: EvalPlusDataset = "humaneval",
        max_tasks_per_container: int | None = None,
        config: EvalPlusDockerConfig | None = None,
    ):
        self.run_directory = run_directory.resolve()
        self.problems = problems
        self.dataset = dataset
        self.max_tasks_per_container = max_tasks_per_container
        self.config = config or EvalPlusDockerConfig()
        self.batches_directory = self.run_directory / "public_batches"
        self.batches_directory.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        batch_id: str,
        candidates: tuple[CandidateRecord, ...],
    ) -> tuple[CandidateGrade, ...]:
        """Run one base-only batch and persist its complete audit trail."""
        if not _BATCH_ID_PATTERN.fullmatch(batch_id):
            raise ValueError("batch_id must contain only letters, digits, dot, dash, underscore")
        if not candidates:
            raise ValueError("public candidate batch must not be empty")
        batch_directory = self.batches_directory / batch_id
        if batch_directory.exists():
            raise FileExistsError(f"refusing to overwrite public batch: {batch_directory}")
        batch_directory.mkdir()
        shards = shard_candidates_by_task(
            candidates,
            max_tasks_per_shard=self.max_tasks_per_container,
        )
        if len(shards) == 1:
            single_grades, _ = self._evaluate_container_batch(
                batch_directory,
                batch_id,
                shards[0],
                manifest_filename="batch_manifest.json",
            )
            return single_grades

        shards_directory = batch_directory / "shards"
        shards_directory.mkdir()
        grades: list[CandidateGrade] = []
        shard_summaries: list[dict[str, object]] = []
        for shard_index, shard in enumerate(shards):
            shard_directory = shards_directory / f"{shard_index:03d}"
            shard_directory.mkdir()
            shard_grades, manifest_path = self._evaluate_container_batch(
                shard_directory,
                batch_id,
                shard,
                manifest_filename="shard_manifest.json",
                shard_index=shard_index,
            )
            grades.extend(shard_grades)
            shard_summaries.append(
                {
                    "shard_index": shard_index,
                    "relative_directory": str(shard_directory.relative_to(batch_directory)),
                    "candidate_count": len(shard),
                    "task_ids": sorted({candidate.task_id for candidate in shard}),
                    "manifest_sha256": _sha256_file(manifest_path),
                }
            )
        ordered_grades = tuple(
            sorted(grades, key=lambda grade: (grade.task_id, grade.candidate_index))
        )
        grades_path = write_candidate_grades(
            batch_directory / "base_grades.jsonl",
            ordered_grades,
        )
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "batch_id": batch_id,
            "candidate_count": len(candidates),
            "task_ids": sorted({candidate.task_id for candidate in candidates}),
            "docker_image": self.config.image,
            "dataset": self.dataset,
            "base_only": True,
            "shard_count": len(shards),
            "max_tasks_per_container": self.max_tasks_per_container,
            "shards": shard_summaries,
            "base_grades_sha256": _sha256_file(grades_path),
        }
        (batch_directory / "batch_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ordered_grades

    def _evaluate_container_batch(
        self,
        directory: Path,
        batch_id: str,
        candidates: tuple[CandidateRecord, ...],
        *,
        manifest_filename: str,
        shard_index: int | None = None,
    ) -> tuple[tuple[CandidateGrade, ...], Path]:
        """Run and audit one bounded base-only evaluator container."""
        samples_path = write_evalplus_candidate_samples(
            directory / "samples.jsonl",
            candidates,
        )
        task_ids = tuple(sorted({candidate.task_id for candidate in candidates}))
        dataset_path = write_evalplus_dataset_override(
            directory / "private_dataset.jsonl",
            self.problems,
            task_ids,
            dataset=self.dataset,
        )

        with tempfile.TemporaryDirectory(
            prefix=".evalplus-public-output-",
            dir=directory,
        ) as temporary_output:
            temporary_output_directory = Path(temporary_output)
            command = build_evalplus_docker_command(
                directory,
                samples_path.name,
                base_only=True,
                dataset=self.dataset,
                dataset_filename=dataset_path.name,
                output_directory=temporary_output_directory,
                config=self.config,
            )
            started_at = time.perf_counter()
            completed = run_evalplus_docker(
                directory,
                samples_path.name,
                base_only=True,
                dataset=self.dataset,
                dataset_filename=dataset_path.name,
                output_directory=temporary_output_directory,
                config=self.config,
            )
            elapsed_seconds = time.perf_counter() - started_at
            stdout_path = directory / "evalplus_stdout.log"
            stderr_path = directory / "evalplus_stderr.log"
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"EvalPlus public batch failed with exit code {completed.returncode}; "
                    f"see {stderr_path}"
                )
            temporary_results = temporary_output_directory / "samples_eval_results.json"
            if not temporary_results.is_file():
                raise RuntimeError(
                    "EvalPlus public batch exited successfully without a results file; "
                    f"see {stderr_path}"
                )
            shutil.copyfile(temporary_results, directory / "samples_eval_results.json")

        results_path = directory / "samples_eval_results.json"
        bundle = parse_evalplus_base_candidate_results(results_path, candidates)
        grades_path = write_candidate_grades(
            directory / "base_grades.jsonl",
            bundle.base_grades,
        )
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "batch_id": batch_id,
            "shard_index": shard_index,
            "candidate_count": len(candidates),
            "task_ids": list(task_ids),
            "docker_image": self.config.image,
            "dataset": self.dataset,
            "base_only": True,
            "command": list(command),
            "elapsed_seconds": elapsed_seconds,
            "official_dataset_hash": bundle.official_dataset_hash,
            "samples_sha256": _sha256_file(samples_path),
            "dataset_override_sha256": _sha256_file(dataset_path),
            "results_sha256": _sha256_file(results_path),
            "base_grades_sha256": _sha256_file(grades_path),
        }
        manifest_path = directory / manifest_filename
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return bundle.base_grades, manifest_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["EvalPlusPublicBatchEvaluator"]
