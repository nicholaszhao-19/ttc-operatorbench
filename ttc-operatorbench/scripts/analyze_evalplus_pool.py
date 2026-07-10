"""Analyze coverage and selection regret for one externally graded pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    read_candidate_grades,
    read_candidate_pool,
)
from ttc_operatorbench.evals.selection_regret import (
    SelectionAnalysis,
    analyze_selection_regret,
)
from ttc_operatorbench.systems.evalplus import EVALPLUS_DOCKER_IMAGE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--base-grades-filename", default="base_grades.jsonl")
    parser.add_argument("--plus-grades-filename", default="hidden_plus_grades.jsonl")
    parser.add_argument("--output-stem", default="selection")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(k <= 0 for k in args.k_values):
        parser.error("k values must be positive")
    if args.bootstrap_resamples <= 0 or args.bootstrap_seed < 0:
        parser.error("bootstrap resamples must be positive and seed nonnegative")
    if Path(args.output_stem).name != args.output_stem or args.output_stem in {"", ".", ".."}:
        parser.error("output stem must be a nonempty basename")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pool_directory = args.pool_dir.resolve()
    observations_path = pool_directory / f"{args.output_stem}_observations.jsonl"
    summary_path = pool_directory / f"{args.output_stem}_summary.json"
    report_path = pool_directory / f"{args.output_stem}_report.md"
    for output_path in (observations_path, summary_path, report_path):
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite analysis artifact: {output_path}")

    base_grades_path = pool_directory / args.base_grades_filename
    plus_grades_path = pool_directory / args.plus_grades_filename
    pool = read_candidate_pool(pool_directory)
    base_grades = read_candidate_grades(base_grades_path)
    plus_grades = read_candidate_grades(plus_grades_path)
    analysis = analyze_selection_regret(
        pool,
        base_grades,
        plus_grades,
        k_values=args.k_values,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    with observations_path.open("w", encoding="utf-8") as file:
        for observation in analysis.observations:
            file.write(observation.model_dump_json())
            file.write("\n")
    diagnostics = _diagnostics(pool_directory, pool, base_grades, plus_grades)
    summary_payload = {
        "candidate_pool_id": pool.manifest.pool_id,
        "candidate_pool_size": pool.manifest.pool_size,
        "task_count": len(pool.manifest.task_ids),
        "bootstrap_resamples": analysis.bootstrap_resamples,
        "bootstrap_seed": analysis.bootstrap_seed,
        "input_sha256": {
            "manifest": _sha256_file(pool_directory / "manifest.json"),
            "candidates": _sha256_file(pool_directory / "candidates.jsonl"),
            "base_grades": _sha256_file(base_grades_path),
            "hidden_plus_grades": _sha256_file(plus_grades_path),
        },
        "diagnostics": diagnostics,
        "summaries": [row.model_dump(mode="json") for row in analysis.summaries],
        "comparisons": [
            row.model_dump(mode="json") for row in analysis.comparisons
        ],
        "coverage_gains": [
            row.model_dump(mode="json") for row in analysis.coverage_gains
        ],
        "stopping_efficiency": analysis.stopping_efficiency.model_dump(mode="json"),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _render_report(pool.manifest.pool_id, analysis, diagnostics),
        encoding="utf-8",
    )
    print(f"wrote observations to {observations_path}")
    print(f"wrote summary to {summary_path}")
    print(f"wrote report to {report_path}")
    return 0


def _render_report(
    pool_id: str,
    analysis: SelectionAnalysis,
    diagnostics: dict[str, object],
) -> str:
    lines = [
        "# EvalPlus Coverage and Selection Regret",
        "",
        f"Candidate pool: `{pool_id}`",
        "",
        "Hidden correctness is joined only after each selector has made its decision.",
        "",
        "## Selector Results",
        "",
        "| Selector | k | Base pass | Hidden pass | Prefix oracle | Pass@k | "
        "Regret | False accept |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary_row in analysis.summaries:
        false_accept = (
            "NA"
            if summary_row.false_accept_rate is None
            else _percent(summary_row.false_accept_rate)
        )
        lines.append(
            "| "
            f"`{summary_row.selector_name}` | {summary_row.k} | "
            f"{_percent(summary_row.selected_base_pass_rate)} | "
            f"{_percent(summary_row.selected_plus_pass_rate)} | "
            f"{_percent(summary_row.prefix_oracle_pass_rate)} | "
            f"{_percent(summary_row.unbiased_pass_at_k)} | "
            f"{_percent(summary_row.selection_regret)} | {false_accept} |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "Positive values favor the challenger. Intervals use paired task bootstrap resampling.",
            "",
            "| Baseline | Challenger | k | Hidden-pass difference | 95% interval |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison_row in analysis.comparisons:
        lines.append(
            "| "
            f"`{comparison_row.baseline_selector}` | "
            f"`{comparison_row.challenger_selector}` | {comparison_row.k} | "
            f"{_percent(comparison_row.selected_plus_rate_difference)} | "
            f"[{_percent(comparison_row.ci_low)}, {_percent(comparison_row.ci_high)}] |"
        )
    lines.extend(
        [
            "",
            "## Coverage Scaling",
            "",
            "| Reference k | k | Pass@k gain | 95% interval |",
            "|---:|---:|---:|---:|",
        ]
    )
    for gain_row in analysis.coverage_gains:
        lines.append(
            f"| {gain_row.reference_k} | {gain_row.k} | "
            f"{_percent(gain_row.unbiased_pass_at_k_gain)} | "
            f"[{_percent(gain_row.ci_low)}, {_percent(gain_row.ci_high)}] |"
        )
    pilot_checks = diagnostics["automatic_pilot_checks_passed"]
    pilot_checks_text = "Not applicable" if pilot_checks is None else str(pilot_checks)
    stopping = analysis.stopping_efficiency
    lines.extend(
        [
            "",
            "## Stopping Efficiency",
            "",
            "Stopping at the first base-test pass selects the same answer as "
            f"`first_base_pass` at k={stopping.max_k}.",
            "",
            f"- Mean candidate calls: {stopping.mean_candidate_calls:.2f} / "
            f"{stopping.max_k}",
            f"- Median candidate calls: {stopping.median_candidate_calls:.1f}",
            f"- Candidate calls saved: {_percent(stopping.candidate_call_savings_rate)}",
            f"- Generation tokens saved: {_percent(stopping.token_savings_rate)}",
            f"- Tasks using the full budget: {stopping.used_full_budget_count} / "
            f"{stopping.task_count}",
            f"- Tasks with no base pass: {stopping.no_base_pass_count} / "
            f"{stopping.task_count}",
            "",
            "## Engineering Diagnostics",
            "",
            f"- Candidates: {diagnostics['candidate_count']}",
            f"- Empty sanitized outputs: {_percent_value(diagnostics['empty_sanitized_rate'])}",
            f"- Outputs at token cap: {_percent_value(diagnostics['output_cap_rate'])}",
            "- Candidate hidden pass rate: "
            f"{_percent_value(diagnostics['plus_candidate_pass_rate'])}",
            "- Sandboxed evaluator command verified: "
            f"{diagnostics['sandbox_command_verified']}",
            f"- Clean repository provenance: {diagnostics['repository_clean']}",
            f"- Automatic pilot checks passed: {pilot_checks_text}",
            "- Runtime affordability: manual review required",
            "",
            f"Bootstrap: {analysis.bootstrap_resamples:,} task-level resamples, "
            f"seed {analysis.bootstrap_seed}.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostics(
    pool_directory: Path,
    pool: CandidatePool,
    base_grades: tuple[CandidateGrade, ...],
    plus_grades: tuple[CandidateGrade, ...],
) -> dict[str, object]:
    candidate_count = len(pool.candidates)
    empty_raw_count = sum(
        not candidate.generation.generation_text.strip() for candidate in pool.candidates
    )
    empty_sanitized_count = sum(
        not candidate.sanitized_code.strip() for candidate in pool.candidates
    )
    output_cap_count = sum(
        candidate.generation.output_tokens >= pool.manifest.max_output_tokens
        for candidate in pool.candidates
    )
    base_pass_rate = sum(grade.verification_passed for grade in base_grades) / candidate_count
    plus_pass_rate = sum(grade.verification_passed for grade in plus_grades) / candidate_count
    evaluator_manifest_path = pool_directory / "evaluator_manifest.json"
    evaluator_manifest = (
        json.loads(evaluator_manifest_path.read_text(encoding="utf-8"))
        if evaluator_manifest_path.is_file()
        else {}
    )
    command = evaluator_manifest.get("command", [])
    sandbox_verified = _sandbox_command_verified(command, evaluator_manifest.get("docker_image"))
    repository_clean = pool.manifest.metadata.get("repository_dirty") is False
    empty_raw_rate = empty_raw_count / candidate_count
    empty_sanitized_rate = empty_sanitized_count / candidate_count
    output_cap_rate = output_cap_count / candidate_count
    is_pilot = len(pool.manifest.task_ids) == 5 and pool.manifest.pool_size == 4
    automatic_checks = (
        (
            empty_raw_rate < 0.05
            and empty_sanitized_rate < 0.05
            and output_cap_rate < 0.05
            and 0.0 < plus_pass_rate < 1.0
            and sandbox_verified
            and repository_clean
        )
        if is_pilot
        else None
    )
    return {
        "candidate_count": candidate_count,
        "empty_raw_count": empty_raw_count,
        "empty_raw_rate": empty_raw_rate,
        "empty_sanitized_count": empty_sanitized_count,
        "empty_sanitized_rate": empty_sanitized_rate,
        "output_cap_count": output_cap_count,
        "output_cap_rate": output_cap_rate,
        "unique_sanitized_codes": len(
            {candidate.sanitized_code_sha256 for candidate in pool.candidates}
        ),
        "base_candidate_pass_rate": base_pass_rate,
        "plus_candidate_pass_rate": plus_pass_rate,
        "total_generation_seconds": sum(
            candidate.generation.latency_seconds for candidate in pool.candidates
        ),
        "evaluator_elapsed_seconds": evaluator_manifest.get("elapsed_seconds"),
        "sandbox_command_verified": sandbox_verified,
        "repository_clean": repository_clean,
        "automatic_pilot_checks_passed": automatic_checks,
        "runtime_affordability_requires_manual_review": True,
    }


def _sandbox_command_verified(command: object, image: object) -> bool:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return False
    mounts = [command[index + 1] for index, item in enumerate(command[:-1]) if item == "--mount"]
    return (
        image == EVALPLUS_DOCKER_IMAGE
        and _contains_pair(command, "--platform", "linux/amd64")
        and _contains_pair(command, "--network", "none")
        and "--read-only" in command
        and _contains_pair(command, "--cap-drop", "ALL")
        and any("dst=/input/samples.jsonl,readonly" in mount for mount in mounts)
        and any("dst=/input/private_dataset.jsonl,readonly" in mount for mount in mounts)
        and any("dst=/output" in mount and "readonly" not in mount for mount in mounts)
        and not any(":/work:rw" in mount for mount in mounts)
    )


def _contains_pair(values: list[str], first: str, second: str) -> bool:
    return any(
        values[index] == first and values[index + 1] == second
        for index in range(len(values) - 1)
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _percent_value(value: object) -> str:
    if not isinstance(value, int | float):
        raise TypeError("diagnostic percentage must be numeric")
    return _percent(float(value))


if __name__ == "__main__":
    sys.exit(main())
