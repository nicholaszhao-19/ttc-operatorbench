"""Trusted in-container wrapper that isolates EvalPlus inputs from its output."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--parallel-workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="evalplus-run-", dir="/tmp") as directory:
        run_directory = Path(directory)
        samples_path = run_directory / "samples.jsonl"
        dataset_path = run_directory / "private_dataset.jsonl"
        shutil.copyfile(args.samples, samples_path)
        shutil.copyfile(args.dataset_file, dataset_path)
        environment = dict(os.environ)
        override_variable = (
            "HUMANEVAL_OVERRIDE_PATH" if args.dataset == "humaneval" else "MBPP_OVERRIDE_PATH"
        )
        environment[override_variable] = str(dataset_path)
        command = [
            "evalplus.evaluate",
            "--dataset",
            args.dataset,
            "--samples",
            str(samples_path),
            "--parallel",
            str(args.parallel_workers),
        ]
        if args.base_only:
            command.append("--base-only")
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            return completed.returncode
        results_path = run_directory / "samples_eval_results.json"
        if not results_path.is_file():
            raise RuntimeError("EvalPlus completed without writing its result file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(results_path, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
