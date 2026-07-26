"""Run a config-driven TTC OperatorBench experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.evals.experiment import (  # noqa: E402
    ExperimentArtifacts,
    run_experiment_from_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/toy_protocol.yaml"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the configured experiment."""
    args = parse_args(argv)
    artifacts = run_experiment_from_config(
        args.config,
        run_id=args.run_id,
        output_root=args.output_root,
        report_root=args.report_root,
    )
    print_artifacts(artifacts)
    return 0


def print_artifacts(artifacts: ExperimentArtifacts) -> None:
    """Print the stable artifact summary shared with the package command."""
    print(f"wrote attempts to {artifacts.attempts_path}")
    print(f"wrote search results to {artifacts.search_results_path}")
    print(f"wrote summary to {artifacts.summary_path}")
    print(f"wrote decision to {artifacts.decision_path}")
    print(f"wrote report to {artifacts.report_path}")
    if artifacts.skipped_models:
        print(f"skipped {len(artifacts.skipped_models)} model(s)")


if __name__ == "__main__":
    sys.exit(main())
