"""Run one immutable public-only EvalPlus width-depth search trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import CandidatePoolManifest
from ttc_operatorbench.core.provenance import (
    git_provenance,
    git_toplevel,
    package_version,
)
from ttc_operatorbench.core.trajectory import (
    WidthDepthTrajectoryHeader,
    WidthDepthTrajectoryPool,
    write_trajectory_pool,
)
from ttc_operatorbench.evals.evalplus_public_batch import EvalPlusPublicBatchEvaluator
from ttc_operatorbench.evals.width_depth import run_width_depth_search
from ttc_operatorbench.models.hf_provider import CODE_ONLY_PROMPT_STYLE, HuggingFaceModelProvider
from ttc_operatorbench.systems.evalplus import (
    EvalPlusDockerConfig,
    load_humaneval_plus_problems,
    sanitize_evalplus_candidate,
)
from ttc_operatorbench.tasks.evalplus import (
    EVALPLUS_DATASET_NAME,
    EVALPLUS_HUMANEVAL_VERSION,
    evalplus_dataset_sha256,
    tasks_from_evalplus_problems,
)
from ttc_operatorbench.tasks.task_sets import read_task_ids_file

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PILOT_TASK_LIMIT = 5
_PILOT_CALL_LIMIT = 4
_MAX_CALL_LIMIT = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--task-offset", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/width_depth"))
    args = parser.parse_args(argv)

    if not _RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, dash, and underscore")
    if args.task_ids_file is not None and (
        args.task_offset is not None or args.max_tasks is not None
    ):
        parser.error("--task-ids-file cannot be combined with task offset or limit")
    if args.task_ids_file is None:
        args.task_offset = 5 if args.task_offset is None else args.task_offset
        args.max_tasks = 5 if args.max_tasks is None else args.max_tasks
    if (args.task_offset is not None and args.task_offset < 0) or args.seed < 0:
        parser.error("task offset and seed must be nonnegative")
    if (args.max_tasks is not None and args.max_tasks <= 0) or args.width <= 0 or args.depth <= 0:
        parser.error("task, width, and depth values must be positive")
    if args.max_output_tokens <= 0 or args.cpus <= 0 or args.timeout_seconds <= 0:
        parser.error("output-token, CPU, and timeout limits must be positive")
    max_calls = args.width * args.depth
    if max_calls > _MAX_CALL_LIMIT:
        parser.error(f"width * depth must not exceed {_MAX_CALL_LIMIT}")
    if (
        not args.allow_large_run
        and (
            (args.max_tasks is not None and args.max_tasks > _PILOT_TASK_LIMIT)
            or max_calls > _PILOT_CALL_LIMIT
        )
    ):
        parser.error(
            "runs above the engineering pilot limits require --allow-large-run"
        )
    if not _COMMIT_RE.fullmatch(args.model_revision):
        parser.error("--model-revision must be an exact 40-character commit SHA")
    args.tokenizer_revision = args.tokenizer_revision or args.model_revision
    if not _COMMIT_RE.fullmatch(args.tokenizer_revision):
        parser.error("--tokenizer-revision must be an exact 40-character commit SHA")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.getenv(REAL_MODEL_TESTS_ENV) != "1":
        raise RuntimeError(f"set {REAL_MODEL_TESTS_ENV}=1 to enable model generation")

    output_directory = (args.output_root / args.run_id).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"trajectory directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    problems = load_humaneval_plus_problems()
    development_tasks = tuple(
        task
        for task in tasks_from_evalplus_problems(problems)
        if task.metadata.get("split") == "development"
    )
    task_selection_metadata: dict[str, object]
    if args.task_ids_file is not None:
        task_ids_path = args.task_ids_file.resolve()
        requested_task_ids = read_task_ids_file(task_ids_path)
        development_by_id = {task.task_id: task for task in development_tasks}
        missing = sorted(set(requested_task_ids) - set(development_by_id))
        if missing:
            raise RuntimeError(f"task list contains non-development or unknown tasks: {missing}")
        tasks = tuple(development_by_id[task_id] for task_id in requested_task_ids)
        if len(tasks) > _PILOT_TASK_LIMIT and not args.allow_large_run:
            raise RuntimeError("task files above the pilot limit require --allow-large-run")
        task_selection_metadata = {
            "task_ids_file": str(task_ids_path),
            "task_ids_file_sha256": hashlib.sha256(task_ids_path.read_bytes()).hexdigest(),
        }
    else:
        if args.task_offset is None or args.max_tasks is None:
            raise RuntimeError("internal task selection error")
        end = args.task_offset + args.max_tasks
        tasks = development_tasks[args.task_offset:end]
        if len(tasks) != args.max_tasks:
            raise RuntimeError(
                f"requested {args.max_tasks} development tasks at offset {args.task_offset}, "
                f"but found {len(tasks)}"
            )
        task_selection_metadata = {"task_offset": args.task_offset}

    repository = git_provenance(git_toplevel(Path(__file__).resolve().parents[1]))
    if repository.dirty and not args.allow_dirty:
        raise RuntimeError(
            "repository has uncommitted changes; commit them or use --allow-dirty "
            "for an engineering-only run"
        )
    max_calls = args.width * args.depth
    manifest = CandidatePoolManifest(
        pool_id=args.run_id,
        dataset_name=EVALPLUS_DATASET_NAME,
        dataset_version=EVALPLUS_HUMANEVAL_VERSION,
        dataset_sha256=evalplus_dataset_sha256(problems),
        repository_commit=repository.commit,
        task_ids=tuple(task.task_id for task in tasks),
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        provider_name="huggingface",
        prompt_style=CODE_ONLY_PROMPT_STYLE,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        pool_size=max_calls,
        pool_seed=args.seed,
        created_at_utc=datetime.now(UTC).isoformat(),
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": args.device,
            "dtype": args.dtype,
        },
        dependencies={
            "evalplus": package_version("evalplus"),
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        metadata={
            "protocol": "stop_then_escalate_v1",
            "split": "development",
            "repository_dirty": repository.dirty,
            "repository_state_sha256": repository.state_sha256,
            **task_selection_metadata,
        },
    )
    header = WidthDepthTrajectoryHeader(
        width=args.width,
        depth=args.depth,
        candidate_manifest=manifest,
    )
    (output_directory / "run_plan.json").write_text(
        json.dumps(header.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provider = HuggingFaceModelProvider(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    evaluator = EvalPlusPublicBatchEvaluator(
        output_directory,
        problems,
        config=EvalPlusDockerConfig(
            cpus=args.cpus,
            memory=args.memory,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    pool = run_width_depth_search(
        manifest,
        tasks,
        provider,
        sanitize_evalplus_candidate,
        evaluator,
        width=args.width,
        depth=args.depth,
    )
    manifest_path, steps_path = write_trajectory_pool(output_directory, pool)
    summary_path = _write_public_summary(output_directory, pool)
    print(f"wrote trajectory manifest to {manifest_path}")
    print(f"wrote trajectory steps to {steps_path}")
    print(f"wrote public-only summary to {summary_path}")
    return 0


def _write_public_summary(
    output_directory: Path,
    pool: WidthDepthTrajectoryPool,
) -> Path:
    steps = pool.steps
    task_ids = pool.header.candidate_manifest.task_ids
    resolved = sum(
        bool(pool.steps_for_task(task_id)[-1].selected)
        for task_id in task_ids
    )
    empty_candidates = sum(not step.candidate.sanitized_code.strip() for step in steps)
    token_limit = pool.header.candidate_manifest.max_output_tokens
    possible_truncations = sum(
        step.candidate.generation.output_tokens >= token_limit for step in steps
    )
    summary = {
        "scope": "public_base_only",
        "task_count": len(task_ids),
        "maximum_calls_per_task": pool.header.candidate_manifest.pool_size,
        "actual_model_calls": len(steps),
        "mean_model_calls_per_task": len(steps) / len(task_ids),
        "total_generation_tokens": sum(
            step.candidate.generation.total_tokens for step in steps
        ),
        "publicly_resolved_tasks": resolved,
        "public_resolution_rate": resolved / len(task_ids),
        "empty_candidate_count": empty_candidates,
        "empty_candidate_rate": empty_candidates / len(steps),
        "possible_truncation_count": possible_truncations,
        "possible_truncation_rate": possible_truncations / len(steps),
    }
    path = output_directory / "public_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
if __name__ == "__main__":
    sys.exit(main())
