"""Decision-log extraction and state-action aggregation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ttc_operatorbench.core.schema import DecisionLog, SearchResult

DECISION_RECORD_FIELDS = (
    "decision_id",
    "task_id",
    "policy_name",
    "run_id",
    "step_index",
    "chosen_operator_name",
    "valid_operator_names",
    "previous_operator_name",
    "previous_error_type",
    "previous_failure_category",
    "repeated_error_count",
    "state_attempts",
    "state_tokens",
    "state_verifier_calls",
    "state_seconds",
    "state_cost",
    "remaining_attempts",
    "remaining_tokens",
    "remaining_verifier_calls",
    "remaining_seconds",
    "remaining_cost",
    "remaining_attempt_bucket",
    "remaining_cost_bucket",
    "operator_scores",
    "produced_attempt_ids",
    "produced_attempt_count",
    "delta_tokens",
    "delta_verifier_calls",
    "delta_seconds",
    "delta_cost",
    "outcome_success",
    "outcome_error_type",
    "outcome_failure_category",
    "budget_exhausted_after",
    "model_name",
    "model_id",
    "model_tier",
    "budget_name",
    "task_family",
    "seed",
)

STATE_ACTION_GROUP_FIELDS = (
    "model_name",
    "model_id",
    "model_tier",
    "policy_name",
    "budget_name",
    "task_family",
    "previous_failure_category",
    "previous_error_type",
    "remaining_attempt_bucket",
    "remaining_cost_bucket",
    "chosen_operator_name",
)

STATE_ACTION_FIELDS = (
    *STATE_ACTION_GROUP_FIELDS,
    "decision_count",
    "outcome_success_count",
    "outcome_success_rate",
    "budget_exhausted_after_count",
    "mean_delta_tokens",
    "mean_delta_verifier_calls",
    "mean_delta_seconds",
    "mean_delta_cost",
    "success_per_1k_tokens",
    "success_per_verifier_call",
    "success_per_cost_unit",
)


def decision_log_records(results: Sequence[SearchResult]) -> tuple[dict[str, Any], ...]:
    """Return denormalized decision-log rows for JSONL artifacts."""
    records: list[dict[str, Any]] = []
    for result in results:
        for decision in result.decision_log:
            records.append(_decision_record(result, decision))
    return tuple(records)


def aggregate_state_action_analysis(
    results: Sequence[SearchResult],
) -> tuple[dict[str, Any], ...]:
    """Aggregate decision logs by visible state and chosen operator."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in decision_log_records(results):
        grouped[tuple(record[field] for field in STATE_ACTION_GROUP_FIELDS)].append(record)

    rows: list[dict[str, Any]] = []
    for key, records in sorted(grouped.items()):
        row = dict(zip(STATE_ACTION_GROUP_FIELDS, key, strict=True))
        successes = sum(1 for record in records if record["outcome_success"])
        total_tokens = sum(int(record["delta_tokens"]) for record in records)
        total_calls = sum(int(record["delta_verifier_calls"]) for record in records)
        total_cost = sum(float(record["delta_cost"]) for record in records)
        count = len(records)
        row.update(
            {
                "decision_count": count,
                "outcome_success_count": successes,
                "outcome_success_rate": successes / count if count else 0.0,
                "budget_exhausted_after_count": sum(
                    1 for record in records if record["budget_exhausted_after"]
                ),
                "mean_delta_tokens": total_tokens / count if count else 0.0,
                "mean_delta_verifier_calls": total_calls / count if count else 0.0,
                "mean_delta_seconds": _mean_float(records, "delta_seconds"),
                "mean_delta_cost": total_cost / count if count else 0.0,
                "success_per_1k_tokens": successes / (total_tokens / 1000)
                if total_tokens > 0
                else None,
                "success_per_verifier_call": successes / total_calls if total_calls > 0 else None,
                "success_per_cost_unit": successes / total_cost if total_cost > 0 else None,
            }
        )
        rows.append(row)
    return tuple(rows)


def write_decision_log_jsonl(path: Path, results: Sequence[SearchResult]) -> Path:
    """Write one JSONL row per operator decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in decision_log_records(results):
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")
    return path


def write_state_action_analysis_json(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write state-action aggregate rows as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_state_action_analysis_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write state-action aggregate rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STATE_ACTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in STATE_ACTION_FIELDS})
    return path


def _decision_record(result: SearchResult, decision: DecisionLog) -> dict[str, Any]:
    row = decision.model_dump()
    row["valid_operator_names"] = list(decision.valid_operator_names)
    row["operator_scores"] = dict(decision.operator_scores)
    row["produced_attempt_ids"] = list(decision.produced_attempt_ids)
    row.update(
        {
            "remaining_attempt_bucket": _attempt_bucket(decision.remaining_attempts),
            "remaining_cost_bucket": _cost_bucket(decision.remaining_cost),
            "model_name": result.metadata.get("model_name", "unknown"),
            "model_id": result.metadata.get("model_id", "unknown"),
            "model_tier": result.metadata.get("model_tier", "unknown"),
            "budget_name": result.metadata.get("budget_name", "unknown"),
            "task_family": result.metadata.get(
                "task_suite",
                result.metadata.get("task_family", ""),
            ),
            "seed": result.metadata.get("seed"),
        }
    )
    return row


def _attempt_bucket(remaining_attempts: int | None) -> str:
    if remaining_attempts is None:
        return "unbounded"
    if remaining_attempts <= 0:
        return "exhausted"
    if remaining_attempts == 1:
        return "last_attempt"
    if remaining_attempts <= 3:
        return "low"
    return "ample"


def _cost_bucket(remaining_cost: float | None) -> str:
    if remaining_cost is None:
        return "unbounded"
    if remaining_cost <= 0.0:
        return "exhausted"
    if remaining_cost <= 1.0:
        return "low"
    if remaining_cost <= 5.0:
        return "medium"
    return "ample"


def _mean_float(records: Sequence[Mapping[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record[key]) for record in records) / len(records)


__all__ = [
    "DECISION_RECORD_FIELDS",
    "STATE_ACTION_FIELDS",
    "aggregate_state_action_analysis",
    "decision_log_records",
    "write_decision_log_jsonl",
    "write_state_action_analysis_csv",
    "write_state_action_analysis_json",
]
