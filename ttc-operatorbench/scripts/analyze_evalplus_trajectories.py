"""Compare hidden correctness and cost for matched width-depth trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import read_candidate_grades
from ttc_operatorbench.core.trajectory import read_trajectory_pool
from ttc_operatorbench.evals.trajectory_analysis import (
    TrajectoryPolicyAnalysis,
    TrajectoryPolicyComparison,
    analyze_width_depth_trajectory,
    compare_trajectory_policies,
    development_winner,
    validate_comparable_trajectory_pools,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if len(args.trajectory_dir) < 2:
        parser.error("at least two --trajectory-dir values are required")
    if args.bootstrap_resamples <= 0 or args.bootstrap_seed < 0:
        parser.error("bootstrap resamples must be positive and seed nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory = args.output_dir.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {output_directory}")

    trajectory_directories = tuple(path.resolve() for path in args.trajectory_dir)
    pools = tuple(read_trajectory_pool(path) for path in trajectory_directories)
    validate_comparable_trajectory_pools(pools)
    if pools[0].header.width != 16 or pools[0].header.depth != 1:
        raise ValueError("the first trajectory must be the preregistered 16x1 baseline")
    analyses = tuple(
        analyze_width_depth_trajectory(
            pool,
            read_candidate_grades(
                directory / "hidden_evaluation" / "hidden_plus_grades.jsonl"
            ),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        for index, (directory, pool) in enumerate(
            zip(trajectory_directories, pools, strict=True)
        )
    )
    baseline = analyses[0]
    comparisons = tuple(
        compare_trajectory_policies(
            baseline,
            challenger,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed + 10_000 + index * 10,
        )
        for index, challenger in enumerate(analyses[1:])
    )
    winner = development_winner(analyses)
    output_directory.mkdir(parents=True)

    observations_path = output_directory / "task_observations.jsonl"
    with observations_path.open("w", encoding="utf-8") as file:
        for analysis in analyses:
            for observation in analysis.observations:
                file.write(observation.model_dump_json())
                file.write("\n")
    summary_path = output_directory / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": args.bootstrap_seed,
                "baseline_pool_id": baseline.summary.pool_id,
                "development_winner_pool_id": winner.summary.pool_id,
                "winner_rule": [
                    "highest_hidden_pass_rate",
                    "lowest_total_generation_tokens",
                    "lowest_total_calls",
                    "greatest_width",
                ],
                "input_sha256": {
                    pool.header.candidate_manifest.pool_id: {
                        "trajectory_manifest": _sha256_file(
                            directory / "trajectory_manifest.json"
                        ),
                        "trajectory_steps": _sha256_file(
                            directory / "trajectory_steps.jsonl"
                        ),
                        "hidden_plus_grades": _sha256_file(
                            directory
                            / "hidden_evaluation"
                            / "hidden_plus_grades.jsonl"
                        ),
                    }
                    for directory, pool in zip(
                        trajectory_directories, pools, strict=True
                    )
                },
                "policies": [
                    analysis.summary.model_dump(mode="json") for analysis in analyses
                ],
                "comparisons": [
                    comparison.model_dump(mode="json")
                    for comparison in comparisons
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_directory / "comparison_report.md"
    report_path.write_text(
        _render_report(analyses, comparisons, winner),
        encoding="utf-8",
    )
    print(f"wrote task observations to {observations_path}")
    print(f"wrote comparison summary to {summary_path}")
    print(f"wrote comparison report to {report_path}")
    return 0


def _render_report(
    analyses: tuple[TrajectoryPolicyAnalysis, ...],
    comparisons: tuple[TrajectoryPolicyComparison, ...],
    winner: TrajectoryPolicyAnalysis,
) -> str:
    lines = [
        "# Width-Depth Development Comparison",
        "",
        "Hidden labels were joined only after every public trajectory was complete.",
        "The first policy is the paired stop-only sampling baseline.",
        "",
        "## Policy Results",
        "",
        "| Policy | w x d | Hidden pass | 95% CI | Public pass | False accept | "
        "Mean calls | Tokens | Repair hidden solves |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for analysis in analyses:
        row = analysis.summary
        false_accept = "NA" if row.false_accept_rate is None else _percent(row.false_accept_rate)
        lines.append(
            f"| `{row.pool_id}` | {row.width} x {row.depth} | "
            f"{_percent(row.hidden_pass_rate)} | "
            f"[{_percent(row.hidden_pass_ci_low)}, {_percent(row.hidden_pass_ci_high)}] | "
            f"{_percent(row.public_pass_rate)} | {false_accept} | "
            f"{row.mean_calls:.2f} | {row.total_generation_tokens:,} | "
            f"{row.hidden_solved_by_repair_count} |"
        )
    lines.extend(
        [
            "",
            "## Paired Differences",
            "",
            "Positive accuracy values favor the challenger. Negative cost values are cheaper.",
            "",
            "| Challenger | Hidden difference | 95% CI | Mean-call difference | "
            "Mean-token difference | Win/loss/tie | Engineering gate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        lines.append(
            f"| `{comparison.challenger_pool_id}` | "
            f"{_signed_percent(comparison.hidden_pass_rate_difference)} | "
            f"[{_signed_percent(comparison.hidden_pass_ci_low)}, "
            f"{_signed_percent(comparison.hidden_pass_ci_high)}] | "
            f"{comparison.mean_call_difference:+.2f} | "
            f"{comparison.mean_token_difference:+.1f} | "
            f"{comparison.hidden_win_count}/{comparison.hidden_loss_count}/"
            f"{comparison.hidden_tie_count} | {comparison.meets_engineering_gate} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Development Choice",
            "",
            f"The deterministic tie-break rule selects `{winner.summary.pool_id}`.",
            "This is a development-set choice, not confirmation evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _signed_percent(value: float) -> str:
    return f"{100.0 * value:+.1f}%"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
