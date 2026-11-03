"""Create toy evaluation plots from JSONL logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.evals.metrics import (  # noqa: E402
    group_results_by_policy,
    success_curve_by_token_budget,
)
from ttc_operatorbench.evals.plots import plot_success_curve_by_token_budget  # noqa: E402
from ttc_operatorbench.logging.writer import read_search_results_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/toy_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/greedy_vs_best_of_n.png"))
    return parser.parse_args()


def main() -> None:
    """Read toy eval logs and make the baseline comparison plot."""
    args = parse_args()
    results = read_search_results_jsonl(args.input)
    grouped = group_results_by_policy(results)
    comparison_labels = ("greedy", "best_of_n")
    comparison_results = {
        label: grouped[label] for label in comparison_labels if label in grouped
    }
    budgets = sorted(
        {
            attempt.cumulative_tokens
            for policy_results in comparison_results.values()
            for result in policy_results
            for attempt in result.attempts
        }
    )
    curves = {
        label: success_curve_by_token_budget(policy_results, budgets)
        for label, policy_results in comparison_results.items()
    }
    output_path = plot_success_curve_by_token_budget(curves, args.output)
    print(f"wrote plot to {output_path}")


if __name__ == "__main__":
    main()
