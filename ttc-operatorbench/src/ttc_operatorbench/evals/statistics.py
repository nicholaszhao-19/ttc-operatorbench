"""Deterministic paired policy-comparison statistics."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ttc_operatorbench.core.schema import SearchResult
from ttc_operatorbench.evals.metrics import (
    area_under_success_curve,
    hidden_solve_rate,
    hidden_success,
    hidden_success_curve_by_cost_budget,
    hidden_success_curve_by_token_budget,
    hidden_success_curve_by_verifier_budget,
    public_hidden_gap,
    solve_rate,
    success_curve_by_cost_budget,
    success_curve_by_token_budget,
    success_curve_by_verifier_budget,
)

BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 0


def paired_policy_comparisons(
    results: Sequence[SearchResult],
    *,
    decision_policy: str,
    baseline_policies: Sequence[str],
    metric_scope: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], ...]:
    """Compare a decision policy with each baseline on matched result keys."""
    by_policy_key = {
        (result.policy_name, _pair_key(result)): result
        for result in results
        if result.policy_name == decision_policy or result.policy_name in baseline_policies
    }
    comparisons: list[dict[str, Any]] = []
    for baseline_policy in baseline_policies:
        pairs: list[tuple[SearchResult, SearchResult]] = []
        for result in results:
            if result.policy_name != decision_policy:
                continue
            baseline = by_policy_key.get((baseline_policy, _pair_key(result)))
            if baseline is not None:
                pairs.append((result, baseline))
        if not pairs:
            continue
        comparisons.append(
            _comparison_for_pairs(
                pairs,
                decision_policy=decision_policy,
                baseline_policy=baseline_policy,
                metric_scope=metric_scope,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    return tuple(comparisons)


def _comparison_for_pairs(
    pairs: Sequence[tuple[SearchResult, SearchResult]],
    *,
    decision_policy: str,
    baseline_policy: str,
    metric_scope: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    decision_results = tuple(pair[0] for pair in pairs)
    baseline_results = tuple(pair[1] for pair in pairs)
    deltas = _metric_deltas(decision_results, baseline_results, metric_scope=metric_scope)
    outcomes = [_pair_outcome(decision, baseline, metric_scope) for decision, baseline in pairs]
    intervals = _bootstrap_intervals(
        tuple(pairs),
        metric_scope=metric_scope,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "decision_policy": decision_policy,
        "baseline_policy": baseline_policy,
        "metric_scope": metric_scope,
        "paired_count": len(pairs),
        "win_count": sum(1 for outcome in outcomes if outcome == "win"),
        "tie_count": sum(1 for outcome in outcomes if outcome == "tie"),
        "loss_count": sum(1 for outcome in outcomes if outcome == "loss"),
        **deltas,
        "confidence_intervals": intervals,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _metric_deltas(
    decision_results: Sequence[SearchResult],
    baseline_results: Sequence[SearchResult],
    *,
    metric_scope: str,
) -> dict[str, float]:
    combined = (*decision_results, *baseline_results)
    token_budgets = _token_budget_grid(combined)
    verifier_budgets = _verifier_budget_grid(combined)
    cost_budgets = _cost_budget_grid(combined)
    if metric_scope == "hidden":
        decision_primary = hidden_solve_rate(decision_results)
        baseline_primary = hidden_solve_rate(baseline_results)
    else:
        decision_primary = solve_rate(decision_results)
        baseline_primary = solve_rate(baseline_results)
    decision_token_auc = _token_auc(decision_results, token_budgets, metric_scope)
    baseline_token_auc = _token_auc(baseline_results, token_budgets, metric_scope)
    decision_verifier_auc = _verifier_auc(decision_results, verifier_budgets, metric_scope)
    baseline_verifier_auc = _verifier_auc(baseline_results, verifier_budgets, metric_scope)
    decision_cost_auc = _cost_auc(decision_results, cost_budgets, metric_scope)
    baseline_cost_auc = _cost_auc(baseline_results, cost_budgets, metric_scope)
    return {
        "primary_solve_rate_delta": decision_primary - baseline_primary,
        "public_solve_rate_delta": solve_rate(decision_results) - solve_rate(baseline_results),
        "hidden_solve_rate_delta": hidden_solve_rate(decision_results)
        - hidden_solve_rate(baseline_results),
        "public_hidden_gap_delta": public_hidden_gap(decision_results)
        - public_hidden_gap(baseline_results),
        "token_auc_delta": decision_token_auc - baseline_token_auc,
        "verifier_call_auc_delta": decision_verifier_auc - baseline_verifier_auc,
        "cost_auc_delta": decision_cost_auc - baseline_cost_auc,
    }


def _bootstrap_intervals(
    pairs: tuple[tuple[SearchResult, SearchResult], ...],
    *,
    metric_scope: str,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    sampled_deltas: dict[str, list[float]] = {
        "primary_solve_rate_delta": [],
        "token_auc_delta": [],
        "verifier_call_auc_delta": [],
        "cost_auc_delta": [],
    }
    if not pairs:
        return {key: (0.0, 0.0) for key in sampled_deltas}
    for _ in range(samples):
        sample = tuple(rng.choice(pairs) for _ in pairs)
        metrics = _metric_deltas(
            tuple(pair[0] for pair in sample),
            tuple(pair[1] for pair in sample),
            metric_scope=metric_scope,
        )
        for key in sampled_deltas:
            sampled_deltas[key].append(metrics[key])
    return {key: _percentile_interval(values) for key, values in sampled_deltas.items()}


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    low_index = int(0.025 * (len(ordered) - 1))
    high_index = int(0.975 * (len(ordered) - 1))
    return (ordered[low_index], ordered[high_index])


def _pair_outcome(decision: SearchResult, baseline: SearchResult, metric_scope: str) -> str:
    decision_success = hidden_success(decision) if metric_scope == "hidden" else decision.success
    baseline_success = hidden_success(baseline) if metric_scope == "hidden" else baseline.success
    if decision_success and not baseline_success:
        return "win"
    if baseline_success and not decision_success:
        return "loss"
    return "tie"


def _token_auc(results: Sequence[SearchResult], budgets: Sequence[int], metric_scope: str) -> float:
    if metric_scope == "hidden":
        return area_under_success_curve(hidden_success_curve_by_token_budget(results, budgets))
    return area_under_success_curve(success_curve_by_token_budget(results, budgets))


def _verifier_auc(
    results: Sequence[SearchResult],
    budgets: Sequence[int],
    metric_scope: str,
) -> float:
    if metric_scope == "hidden":
        return area_under_success_curve(hidden_success_curve_by_verifier_budget(results, budgets))
    return area_under_success_curve(success_curve_by_verifier_budget(results, budgets))


def _cost_auc(
    results: Sequence[SearchResult],
    budgets: Sequence[float],
    metric_scope: str,
) -> float:
    if metric_scope == "hidden":
        return area_under_success_curve(hidden_success_curve_by_cost_budget(results, budgets))
    return area_under_success_curve(success_curve_by_cost_budget(results, budgets))


def _pair_key(result: SearchResult) -> tuple[str, str, str, str, str]:
    return (
        str(result.metadata.get("model_name", "")),
        str(result.metadata.get("budget_name", "")),
        str(result.metadata.get("seed", "")),
        str(result.metadata.get("task_suite", "")),
        result.task_id,
    )


def _token_budget_grid(results: Sequence[SearchResult]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                0,
                *(attempt.cumulative_tokens for result in results for attempt in result.attempts),
            }
        )
    )


def _verifier_budget_grid(results: Sequence[SearchResult]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                0,
                *(
                    attempt.cumulative_verifier_calls
                    for result in results
                    for attempt in result.attempts
                ),
            }
        )
    )


def _cost_budget_grid(results: Sequence[SearchResult]) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                0.0,
                *(attempt.cumulative_cost for result in results for attempt in result.attempts),
            }
        )
    )


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "paired_policy_comparisons",
]
