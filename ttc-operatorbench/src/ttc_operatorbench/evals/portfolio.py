"""Aggregate completed experiment runs into a portfolio-style report."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_ATTEMPT_FIELDS = (
    "model_id",
    "policy_name",
    "operator_name",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "verification_passed",
    "cumulative_tokens",
    "cumulative_verifier_calls",
)


@dataclass(frozen=True)
class PortfolioRun:
    """Loaded artifact bundle for one completed experiment run."""

    run_id: str
    output_dir: Path
    report_dir: Path
    config: dict[str, Any]
    decision: dict[str, Any]
    summary: tuple[dict[str, Any], ...]
    attempts_preview: tuple[dict[str, Any], ...]
    search_results_preview: tuple[dict[str, Any], ...]
    artifact_status: dict[str, bool]


def load_portfolio_runs(
    *,
    run_ids: Sequence[str],
    output_root: Path = Path("outputs/runs"),
    report_root: Path = Path("reports/runs"),
) -> tuple[PortfolioRun, ...]:
    """Load completed run artifacts by run identifier."""
    return tuple(
        load_portfolio_run(run_id, output_root=output_root, report_root=report_root)
        for run_id in run_ids
    )


def load_portfolio_run(
    run_id: str,
    *,
    output_root: Path = Path("outputs/runs"),
    report_root: Path = Path("reports/runs"),
) -> PortfolioRun:
    """Load one completed run artifact bundle."""
    output_dir = output_root / run_id
    report_dir = report_root / run_id
    if not output_dir.exists():
        raise FileNotFoundError(f"missing run output directory: {output_dir}")
    return PortfolioRun(
        run_id=run_id,
        output_dir=output_dir,
        report_dir=report_dir,
        config=_read_json_mapping(output_dir / "config_snapshot.yaml"),
        decision=_read_json_mapping(output_dir / "decision.json"),
        summary=_read_json_rows(output_dir / "summary.json"),
        attempts_preview=_read_jsonl_preview(output_dir / "attempts.jsonl", limit=5),
        search_results_preview=_read_jsonl_preview(output_dir / "search_results.jsonl", limit=20),
        artifact_status={
            "run_manifest_present": (output_dir / "run_manifest.json").exists(),
            "failure_taxonomy_present": (output_dir / "failure_taxonomy.json").exists(),
            "failure_taxonomy_csv_present": (output_dir / "failure_taxonomy.csv").exists(),
            "decision_log_present": (output_dir / "decision_log.jsonl").exists(),
            "state_action_analysis_present": (
                output_dir / "state_action_analysis.json"
            ).exists(),
            "state_action_analysis_csv_present": (
                output_dir / "state_action_analysis.csv"
            ).exists(),
        },
    )


def write_portfolio_report(path: Path, runs: Sequence[PortfolioRun]) -> Path:
    """Write a Markdown report that compares completed experiment runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TTC OperatorBench Portfolio Report",
        "",
        "This report aggregates completed experiment artifacts. It is descriptive, not a "
        "model leaderboard.",
        "",
        "## Run Status",
        "",
    ]
    for run in runs:
        lines.extend(_run_status_lines(run))

    lines.extend(["", "## Budget Decisions", ""])
    for run in runs:
        lines.extend(_budget_decision_lines(run))

    lines.extend(["", "## Summary Rows", ""])
    for run in runs:
        lines.extend(_summary_lines(run))

    lines.extend(["", "## Artifact Checks", ""])
    for run in runs:
        lines.extend(_artifact_check_lines(run))

    failure_lines = _failure_example_lines(runs)
    if failure_lines:
        lines.extend(["", "## Unsuccessful Examples", "", *failure_lines])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_status_lines(run: PortfolioRun) -> list[str]:
    models = _models(run.config)
    model_labels = ", ".join(
        f"{model.get('model_id', 'unknown')}[{model.get('model_tier', 'unknown')}]"
        for model in models
    )
    task_ids = _string_sequence(run.config.get("task_ids"))
    policies = _string_sequence(run.config.get("policies"))
    budgets = [
        str(budget.get("name", "unknown"))
        for budget in _mapping_sequence(run.config.get("budgets"))
    ]
    return [
        f"- {run.run_id}: verdict={run.decision.get('verdict', 'unknown')}, "
        f"experiment={run.config.get('experiment_id', 'unknown')}, "
        f"task_suite={run.config.get('task_suite', 'unknown')}, "
        f"task_count={len(task_ids)}, models={model_labels or 'unknown'}",
        f"  policies={', '.join(policies) or 'unknown'}; budgets={', '.join(budgets) or 'unknown'}",
    ]


def _budget_decision_lines(run: PortfolioRun) -> list[str]:
    comparisons = run.decision.get("budget_comparisons")
    if not isinstance(comparisons, list):
        return [f"- {run.run_id}: no budget comparisons"]
    lines: list[str] = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        relationship = comparison.get("relationship")
        if not isinstance(relationship, str):
            relationship = _comparison_relationship(comparison)
        lines.append(
            f"- {run.run_id} / {comparison.get('budget_name', 'unknown')}: "
            f"{comparison.get('status', 'unknown')} "
            f"scope={comparison.get('metric_scope', run.decision.get('metric_scope', 'public'))} "
            f"relationship={relationship} "
            f"(best_baseline={comparison.get('best_baseline_policy', 'unknown')})"
        )
    return lines or [f"- {run.run_id}: no parseable budget comparisons"]


def _summary_lines(run: PortfolioRun) -> list[str]:
    if not run.summary:
        return [f"- {run.run_id}: no summary rows"]
    lines: list[str] = []
    for row in run.summary:
        public_solve_rate = _format_float(row.get("public_solve_rate", row.get("solve_rate")))
        lines.append(
            f"- {run.run_id} / {row.get('model_id', 'unknown')} / "
            f"{row.get('model_tier', 'unknown')} / {row.get('policy_name', 'unknown')} / "
            f"{row.get('budget_name', 'unknown')}: "
            f"public_solve_rate={public_solve_rate}, "
            f"hidden_solve_rate={_format_float(row.get('hidden_solve_rate'))}, "
            f"overfit_rate={_format_float(row.get('overfit_rate'))}, "
            f"token_auc={_format_float(row.get('token_auc'))}, "
            f"cost_auc={_format_float(row.get('cost_auc'))}, "
            f"hidden_token_auc={_format_float(row.get('hidden_token_auc'))}, "
            f"hidden_cost_auc={_format_float(row.get('hidden_cost_auc'))}, "
            f"tasks={row.get('number_of_tasks', 'unknown')}, "
            f"attempts={row.get('total_attempts', 'unknown')}"
        )
    return lines


def _artifact_check_lines(run: PortfolioRun) -> list[str]:
    if not run.attempts_preview:
        return [f"- {run.run_id}: no attempts logged"]
    first_attempt = run.attempts_preview[0]
    missing = [field for field in REQUIRED_ATTEMPT_FIELDS if field not in first_attempt]
    token_accounting_ok = (
        isinstance(first_attempt.get("input_tokens"), int)
        and isinstance(first_attempt.get("output_tokens"), int)
        and isinstance(first_attempt.get("total_tokens"), int)
        and first_attempt["total_tokens"]
        == first_attempt["input_tokens"] + first_attempt["output_tokens"]
    )
    budget_name_present = _budget_name_present(first_attempt)
    return [
        f"- {run.run_id}: attempts_logged={len(run.attempts_preview)}, "
        f"missing_fields={missing or 'none'}, "
        f"token_accounting_ok={token_accounting_ok}, "
        f"budget_name_present={budget_name_present}, "
        f"run_manifest_present={run.artifact_status['run_manifest_present']}, "
        f"failure_taxonomy_present={run.artifact_status['failure_taxonomy_present']}, "
        f"decision_log_present={run.artifact_status['decision_log_present']}, "
        "state_action_analysis_present="
        f"{run.artifact_status['state_action_analysis_present']}"
    ]


def _failure_example_lines(runs: Sequence[PortfolioRun], limit: int = 3) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for run in runs:
        for result in run.search_results_preview:
            if bool(result.get("success")):
                continue
            attempts = result.get("attempts")
            last_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
            last_error = (
                last_attempt.get("error_type") if isinstance(last_attempt, Mapping) else None
            )
            key = (
                run.run_id,
                str(result.get("task_id", "unknown")),
                str(result.get("policy_name", "unknown")),
                str(last_error) if last_error is not None else None,
            )
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {run.run_id} / {result.get('task_id', 'unknown')} / "
                f"{result.get('policy_name', 'unknown')}: last_error={last_error}"
            )
            if len(lines) >= limit:
                return lines
    return lines


def _comparison_relationship(comparison: Mapping[str, Any]) -> str:
    decision_score = comparison.get("decision_policy_metrics")
    baseline_score = comparison.get("best_baseline_metrics")
    if not isinstance(decision_score, Mapping) or not isinstance(baseline_score, Mapping):
        return "unknown"
    decision_solve_rate = _score_value(decision_score.get("solve_rate"))
    baseline_solve_rate = _score_value(baseline_score.get("solve_rate"))
    decision_token_auc = _score_value(decision_score.get("token_auc"))
    baseline_token_auc = _score_value(baseline_score.get("token_auc"))
    if decision_solve_rate == 0.0 and baseline_solve_rate == 0.0:
        return "inconclusive"
    if (
        decision_solve_rate > baseline_solve_rate
        or decision_solve_rate == baseline_solve_rate
        and decision_token_auc > baseline_token_auc
    ):
        return "win"
    if decision_solve_rate == baseline_solve_rate and decision_token_auc == baseline_token_auc:
        return "tie"
    return "loss"


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object in {path}")
    return dict(payload)


def _read_json_rows(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array in {path}")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"expected JSON object rows in {path}")
        rows.append(dict(item))
    return tuple(rows)


def _read_jsonl_preview(path: Path, *, limit: int) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object rows in {path}")
            rows.append(dict(payload))
            if len(rows) >= limit:
                break
    return tuple(rows)


def _models(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return _mapping_sequence(config.get("models"))


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _budget_name_present(first_attempt: Mapping[str, Any]) -> bool:
    metadata = first_attempt.get("metadata")
    result_metadata = first_attempt.get("result_metadata")
    return (
        isinstance(metadata, Mapping)
        and "budget_name" in metadata
        or isinstance(result_metadata, Mapping)
        and "budget_name" in result_metadata
    )


def _format_float(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "n/a"


def _score_value(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


__all__ = [
    "PortfolioRun",
    "REQUIRED_ATTEMPT_FIELDS",
    "load_portfolio_run",
    "load_portfolio_runs",
    "write_portfolio_report",
]
