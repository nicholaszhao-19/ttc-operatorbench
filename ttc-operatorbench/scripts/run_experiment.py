"""Run a config-driven TTC OperatorBench experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.cli import run_experiment_from_options  # noqa: E402


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    """Run the configured experiment."""
    args = parse_args()
    artifacts = run_experiment_from_options(
        config_path=args.config,
        run_id=args.run_id,
        output_root=args.output_root,
        report_root=args.report_root,
    )
    print(f"wrote attempts to {artifacts.attempts_path}")
    print(f"wrote search results to {artifacts.search_results_path}")
    print(f"wrote summary to {artifacts.summary_path}")
    print(f"wrote run manifest to {artifacts.run_manifest_path}")
    print(f"wrote failure taxonomy to {artifacts.failure_taxonomy_path}")
    print(f"wrote decision log to {artifacts.decision_log_path}")
    print(f"wrote state-action analysis to {artifacts.state_action_analysis_path}")
    print(f"wrote state-action analysis CSV to {artifacts.state_action_analysis_csv_path}")
    print(f"wrote decision to {artifacts.decision_path}")
    print(f"wrote report to {artifacts.report_path}")
    if artifacts.skipped_models:
        print(f"skipped {len(artifacts.skipped_models)} model(s)")


if __name__ == "__main__":
    main()
