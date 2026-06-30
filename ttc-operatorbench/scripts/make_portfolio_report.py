"""Create a portfolio report from completed TTC OperatorBench runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.cli import (  # noqa: E402
    DEFAULT_PORTFOLIO_RUN_IDS,
    write_portfolio_from_options,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_PORTFOLIO_RUN_IDS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/runs"))
    parser.add_argument("--output", type=Path, default=Path("reports/portfolio_report.md"))
    return parser.parse_args()


def main() -> None:
    """Write the requested portfolio report."""
    args = parse_args()
    report_path, _ = write_portfolio_from_options(
        run_ids=tuple(args.runs),
        output_root=args.output_root,
        report_root=args.report_root,
        output=args.output,
    )
    print(f"wrote portfolio report to {report_path}")


if __name__ == "__main__":
    main()
