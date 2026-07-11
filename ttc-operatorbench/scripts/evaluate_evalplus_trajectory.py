"""Join hidden EvalPlus labels only after a search trajectory is complete."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ttc_operatorbench.core.trajectory import read_trajectory_pool
from ttc_operatorbench.evals.evalplus_trajectory_hidden import (
    evaluate_evalplus_trajectory_hidden,
)
from ttc_operatorbench.systems.evalplus import (
    EvalPlusDockerConfig,
    load_humaneval_plus_problems,
)

HIDDEN_EVAL_ENV = "RUN_HIDDEN_EVAL"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    args = parser.parse_args(argv)
    if args.cpus <= 0 or args.timeout_seconds <= 0:
        parser.error("CPU and timeout limits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.getenv(HIDDEN_EVAL_ENV) != "1":
        raise RuntimeError(
            f"set {HIDDEN_EVAL_ENV}=1 only after every compared public trajectory is complete"
        )
    trajectory_directory = args.trajectory_dir.resolve()
    pool = read_trajectory_pool(trajectory_directory)
    result = evaluate_evalplus_trajectory_hidden(
        trajectory_directory,
        pool,
        load_humaneval_plus_problems(),
        config=EvalPlusDockerConfig(
            cpus=args.cpus,
            memory=args.memory,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    print(f"wrote base recheck and hidden grades to {result.output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
