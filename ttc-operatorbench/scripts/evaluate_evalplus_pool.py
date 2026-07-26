"""Evaluate one candidate pool in pinned EvalPlus Docker and split its grades."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import (
    read_candidate_pool,
    write_candidate_grades,
)
from ttc_operatorbench.systems.evalplus import (
    EVALPLUS_DOCKER_IMAGE,
    EvalPlusDockerConfig,
    build_evalplus_docker_command,
    parse_evalplus_results,
    run_evalplus_docker,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--samples-filename", default="samples.jsonl")
    parser.add_argument("--dataset-filename", default="private_dataset.jsonl")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    args = parser.parse_args(argv)
    if args.cpus <= 0 or args.timeout_seconds <= 0:
        parser.error("CPU and timeout limits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pool_directory = args.pool_dir.resolve()
    pool = read_candidate_pool(pool_directory)
    base_grades_path = pool_directory / "base_grades.jsonl"
    plus_grades_path = pool_directory / "hidden_plus_grades.jsonl"
    evaluator_manifest_path = pool_directory / "evaluator_manifest.json"
    results_path = pool_directory / "samples_eval_results.json"
    for output_path in (
        base_grades_path,
        plus_grades_path,
        evaluator_manifest_path,
        results_path,
    ):
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite evaluator artifact: {output_path}")

    config = EvalPlusDockerConfig(
        cpus=args.cpus,
        memory=args.memory,
        timeout_seconds=args.timeout_seconds,
    )
    with tempfile.TemporaryDirectory(
        prefix=".evalplus-output-",
        dir=pool_directory,
    ) as temporary_output:
        temporary_output_directory = Path(temporary_output)
        command = build_evalplus_docker_command(
            pool_directory,
            args.samples_filename,
            base_only=False,
            dataset_filename=args.dataset_filename,
            output_directory=temporary_output_directory,
            config=config,
        )
        started_at = time.perf_counter()
        completed = run_evalplus_docker(
            pool_directory,
            args.samples_filename,
            base_only=False,
            dataset_filename=args.dataset_filename,
            output_directory=temporary_output_directory,
            config=config,
        )
        elapsed_seconds = time.perf_counter() - started_at
        temporary_results_path = temporary_output_directory / "samples_eval_results.json"
        if completed.returncode == 0:
            bundle = parse_evalplus_results(temporary_results_path, pool)
            shutil.copyfile(temporary_results_path, results_path)
    stdout_path = pool_directory / "evalplus_stdout.log"
    stderr_path = pool_directory / "evalplus_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"EvalPlus container failed with exit code {completed.returncode}; see {stderr_path}"
        )

    samples_path = pool_directory / args.samples_filename
    write_candidate_grades(base_grades_path, bundle.base_grades)
    write_candidate_grades(plus_grades_path, bundle.plus_grades)
    evaluator_manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_pool_id": pool.manifest.pool_id,
        "docker_image": EVALPLUS_DOCKER_IMAGE,
        "command": list(command),
        "elapsed_seconds": elapsed_seconds,
        "official_dataset_hash": bundle.official_dataset_hash,
        "samples_sha256": _sha256_file(samples_path),
        "results_sha256": _sha256_file(results_path),
        "base_grades_sha256": _sha256_file(base_grades_path),
        "hidden_plus_grades_sha256": _sha256_file(plus_grades_path),
    }
    evaluator_manifest_path.write_text(
        json.dumps(evaluator_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote base grades to {base_grades_path}")
    print(f"wrote hidden plus grades to {plus_grades_path}")
    print(f"wrote evaluator manifest to {evaluator_manifest_path}")
    return 0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
