"""Generate one immutable, revision-pinned HumanEval+ candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import (
    CandidatePoolManifest,
    write_candidate_pool,
)
from ttc_operatorbench.evals.candidate_generation import generate_candidate_pool
from ttc_operatorbench.models.hf_provider import CODE_ONLY_PROMPT_STYLE, HuggingFaceModelProvider
from ttc_operatorbench.systems.evalplus import (
    load_humaneval_plus_problems,
    sanitize_evalplus_candidate,
    write_evalplus_dataset_override,
    write_evalplus_sample_index,
    write_evalplus_samples,
)
from ttc_operatorbench.tasks.evalplus import (
    EVALPLUS_DATASET_NAME,
    EVALPLUS_HUMANEVAL_VERSION,
    evalplus_dataset_sha256,
    tasks_from_evalplus_problems,
)

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    task_limit = parser.add_mutually_exclusive_group()
    task_limit.add_argument("--max-tasks", type=int)
    task_limit.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-evaluation-split", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/candidate_pools"))
    args = parser.parse_args(argv)
    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("task limit must be positive")
    if args.pool_size <= 0 or args.max_output_tokens <= 0:
        parser.error("task, pool, and output-token limits must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not _COMMIT_RE.fullmatch(args.model_revision):
        parser.error("--model-revision must be an exact 40-character commit SHA")
    args.tokenizer_revision = args.tokenizer_revision or args.model_revision
    if not _COMMIT_RE.fullmatch(args.tokenizer_revision):
        parser.error("--tokenizer-revision must be an exact 40-character commit SHA")
    if args.split == "evaluation" and not args.allow_evaluation_split:
        parser.error("evaluation generation requires --allow-evaluation-split")
    if args.max_tasks is None and not args.all_tasks:
        args.max_tasks = 5
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.getenv(REAL_MODEL_TESTS_ENV) != "1":
        raise RuntimeError(f"set {REAL_MODEL_TESTS_ENV}=1 to enable model generation")

    output_directory = args.output_root / args.pool_id
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"candidate pool directory is not empty: {output_directory}")

    problems = load_humaneval_plus_problems()
    all_tasks = tasks_from_evalplus_problems(problems)
    split_tasks = tuple(
        task for task in all_tasks if task.metadata.get("split") == args.split
    )
    tasks = split_tasks if args.all_tasks else split_tasks[: args.max_tasks]
    if not args.all_tasks and len(tasks) != args.max_tasks:
        raise RuntimeError(f"requested {args.max_tasks} tasks but found {len(tasks)}")

    repository_root = _git_toplevel(Path(__file__).resolve().parents[1])
    repository_commit, repository_dirty, repository_state_sha256 = _git_provenance(
        repository_root
    )
    if repository_dirty and not args.allow_dirty:
        raise RuntimeError(
            "repository has uncommitted changes; commit them or use --allow-dirty "
            "for an engineering-only pool"
        )
    manifest = CandidatePoolManifest(
        pool_id=args.pool_id,
        dataset_name=EVALPLUS_DATASET_NAME,
        dataset_version=EVALPLUS_HUMANEVAL_VERSION,
        dataset_sha256=evalplus_dataset_sha256(problems),
        repository_commit=repository_commit,
        task_ids=tuple(task.task_id for task in tasks),
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        provider_name="huggingface",
        prompt_style=CODE_ONLY_PROMPT_STYLE,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        pool_size=args.pool_size,
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
            "python": platform.python_version(),
            "evalplus": _package_version("evalplus"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        metadata={
            "split": args.split,
            "protocol": "evalplus_selection_regret_v1",
            "repository_dirty": repository_dirty,
            "repository_state_sha256": repository_state_sha256,
        },
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
    pool = generate_candidate_pool(manifest, tasks, provider, sanitize_evalplus_candidate)

    manifest_path, candidates_path = write_candidate_pool(output_directory, pool)
    samples_path = write_evalplus_samples(output_directory / "samples.jsonl", pool)
    sample_index_path = write_evalplus_sample_index(
        output_directory / "sample_index.jsonl",
        pool,
    )
    dataset_path = write_evalplus_dataset_override(
        output_directory / "private_dataset.jsonl",
        problems,
        manifest.task_ids,
    )
    print(f"wrote manifest to {manifest_path}")
    print(f"wrote candidates to {candidates_path}")
    print(f"wrote EvalPlus samples to {samples_path}")
    print(f"wrote sample index to {sample_index_path}")
    print(f"wrote private evaluator dataset to {dataset_path}")
    return 0


def _git_toplevel(start_directory: Path) -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _git_provenance(repository_root: Path) -> tuple[str, bool, str]:
    commit = _git_output(repository_root, "rev-parse", "HEAD").decode().strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise RuntimeError("git rev-parse did not return a full commit SHA")
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _git_output(repository_root, "diff", "--binary", "HEAD", "--")
    untracked = _git_output(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    state_hash = hashlib.sha256()
    state_hash.update(b"commit\0" + commit.encode() + b"\0diff\0" + diff)
    for relative_bytes in sorted(path for path in untracked if path):
        relative_path = Path(relative_bytes.decode("utf-8"))
        full_path = (repository_root / relative_path).resolve()
        full_path.relative_to(repository_root)
        if not full_path.is_file():
            continue
        state_hash.update(b"\0untracked\0" + relative_bytes + b"\0")
        state_hash.update(full_path.read_bytes())
    return commit, bool(status.strip()), state_hash.hexdigest()


def _git_output(repository_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required optional package is not installed: {package}") from exc


if __name__ == "__main__":
    sys.exit(main())
