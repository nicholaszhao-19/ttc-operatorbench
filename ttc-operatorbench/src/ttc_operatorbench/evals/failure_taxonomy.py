"""Post-hoc failure taxonomy for experiment artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from ttc_operatorbench.core.schema import AttemptLog, SearchResult

FailureCategory = Literal[
    "success",
    "public_test_failure",
    "hidden_overfit",
    "syntax_or_parse_error",
    "runtime_error",
    "timeout",
    "empty_or_non_code",
    "missing_tests",
    "budget_exhausted",
    "unverified_plan",
    "no_attempt",
]

GROUP_FIELDS = (
    "model_name",
    "model_id",
    "model_tier",
    "policy_name",
    "budget_name",
    "operator_name",
    "task_family",
    "public_success",
    "hidden_success",
    "failure_category",
)


def result_with_failure_categories(result: SearchResult) -> SearchResult:
    """Attach post-hoc failure categories to attempts."""
    attempts = tuple(
        attempt.model_copy(update={"failure_category": classify_attempt(result, attempt)})
        for attempt in result.attempts
    )
    return result.model_copy(update={"attempts": attempts})


def classify_attempt(result: SearchResult, attempt: AttemptLog) -> FailureCategory:
    """Return a normalized post-hoc failure category for one attempt."""
    public_success = _public_success(attempt)
    hidden_success = _hidden_success(attempt)
    error_type = _error_type(attempt)
    if public_success and hidden_success is not False:
        return "success"
    if public_success and hidden_success is False:
        return "hidden_overfit"
    if error_type == "not_verified_plan":
        return "unverified_plan"
    if error_type in {"syntax_error", "parse_error"}:
        return "syntax_or_parse_error"
    if error_type == "runtime_error":
        return "runtime_error"
    if error_type == "timeout":
        return "timeout"
    if error_type in {"empty_code", "empty_generation", "no_code"}:
        return "empty_or_non_code"
    if error_type is not None and error_type.startswith("missing_"):
        return "missing_tests"
    if _is_last_attempt(result, attempt) and _budget_exhausted(result):
        return "budget_exhausted"
    return "public_test_failure"


def aggregate_failure_taxonomy(results: Sequence[SearchResult]) -> tuple[dict[str, Any], ...]:
    """Return grouped failure taxonomy rows for experiment outputs."""
    counter: Counter[tuple[Any, ...]] = Counter()
    for record in _taxonomy_records(results):
        counter[tuple(record[field] for field in GROUP_FIELDS)] += 1
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items()):
        row = dict(zip(GROUP_FIELDS, key, strict=True))
        row["count"] = count
        rows.append(row)
    return tuple(rows)


def failure_taxonomy_examples(
    results: Sequence[SearchResult],
    *,
    limit: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Return representative non-success examples."""
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for result in results:
        for record in _taxonomy_records((result,)):
            category = str(record["failure_category"])
            if category == "success":
                continue
            key = (
                str(record["policy_name"]),
                str(record["budget_name"]),
                str(record["task_id"]),
                category,
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(record)
            if len(examples) >= limit:
                return tuple(examples)
    return tuple(examples)


def write_failure_taxonomy_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write taxonomy rows as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_failure_taxonomy_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write taxonomy rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (*GROUP_FIELDS, "count")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def _taxonomy_records(results: Sequence[SearchResult]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for result in results:
        if not result.attempts:
            records.append(_result_record(result, failure_category="no_attempt"))
            continue
        for attempt in result.attempts:
            records.append(_attempt_record(result, attempt))
    return tuple(records)


def _attempt_record(result: SearchResult, attempt: AttemptLog) -> dict[str, Any]:
    category = attempt.failure_category or classify_attempt(result, attempt)
    return {
        **_result_dimensions(result),
        "task_id": result.task_id,
        "operator_name": attempt.operator_name,
        "public_success": _public_success(attempt),
        "hidden_success": _hidden_success(attempt),
        "failure_category": category,
        "error_type": attempt.error_type,
        "attempt_id": attempt.attempt_id,
    }


def _result_record(result: SearchResult, *, failure_category: FailureCategory) -> dict[str, Any]:
    return {
        **_result_dimensions(result),
        "task_id": result.task_id,
        "operator_name": "search_result",
        "public_success": False,
        "hidden_success": None,
        "failure_category": failure_category,
        "error_type": None,
        "attempt_id": None,
    }


def _result_dimensions(result: SearchResult) -> dict[str, Any]:
    return {
        "model_name": result.metadata.get("model_name", "unknown"),
        "model_id": result.metadata.get("model_id", "unknown"),
        "model_tier": result.metadata.get("model_tier", "unknown"),
        "policy_name": result.policy_name,
        "budget_name": result.metadata.get("budget_name", "unknown"),
        "task_family": result.metadata.get("task_suite", result.metadata.get("task_family", "")),
    }


def _public_success(attempt: AttemptLog) -> bool:
    if attempt.public_verification is not None:
        return attempt.public_verification.verification_passed
    return attempt.verification_passed


def _hidden_success(attempt: AttemptLog) -> bool | None:
    if attempt.hidden_verification is None:
        return None
    return attempt.hidden_verification.verification_passed


def _error_type(attempt: AttemptLog) -> str | None:
    if attempt.error_type is not None:
        return attempt.error_type
    if attempt.public_verification is not None:
        return attempt.public_verification.error_type
    return None


def _is_last_attempt(result: SearchResult, attempt: AttemptLog) -> bool:
    return bool(result.attempts) and result.attempts[-1].attempt_id == attempt.attempt_id


def _budget_exhausted(result: SearchResult) -> bool:
    budget = result.budget
    return any(
        (
            budget.max_attempts is not None and len(result.attempts) >= budget.max_attempts,
            budget.max_tokens is not None and result.total_tokens >= budget.max_tokens,
            budget.max_verifier_calls is not None
            and result.total_verifier_calls >= budget.max_verifier_calls,
            budget.max_seconds is not None and result.total_seconds >= budget.max_seconds,
            budget.max_cost is not None and result.total_cost >= budget.max_cost,
        )
    )


__all__ = [
    "FailureCategory",
    "aggregate_failure_taxonomy",
    "classify_attempt",
    "failure_taxonomy_examples",
    "result_with_failure_categories",
    "write_failure_taxonomy_csv",
    "write_failure_taxonomy_json",
]
