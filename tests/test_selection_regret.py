"""Tests for public-only selector analysis and task-level uncertainty."""

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
)
from ttc_operatorbench.core.schema import Generation
from ttc_operatorbench.evals.selection_regret import analyze_selection_regret


def analysis_pool() -> CandidatePool:
    task_ids = ("HumanEval/0", "HumanEval/1")
    candidates: list[CandidateRecord] = []
    for task_id in task_ids:
        for index in range(4):
            code = f"def f():\n    return {index}"
            candidates.append(
                CandidateRecord(
                    pool_id="analysis-pool",
                    task_id=task_id,
                    candidate_index=index,
                    generation=Generation(
                        prompt="def f():",
                        generation_text=code,
                        input_tokens=2,
                        output_tokens=4,
                        total_tokens=6,
                        latency_seconds=0.1,
                    ),
                    sanitized_code=code,
                    prompt_sha256=sha256_text("def f():"),
                    raw_completion_sha256=sha256_text(code),
                    sanitized_code_sha256=sha256_text(code),
                )
            )
    return CandidatePool(
        manifest=CandidatePoolManifest(
            pool_id="analysis-pool",
            dataset_name="humaneval_plus",
            dataset_version="v0.1.10",
            dataset_sha256="a" * 64,
            repository_commit="deadbeef",
            task_ids=task_ids,
            model_id="test-model",
            model_revision="revision",
            tokenizer_revision="revision",
            provider_name="dummy",
            prompt_style="raw",
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=256,
            pool_size=4,
            pool_seed=0,
            created_at_utc="2026-07-10T00:00:00Z",
        ),
        candidates=tuple(candidates),
    )


def analysis_grades(
    pool: CandidatePool,
    *,
    scope: str,
) -> tuple[CandidateGrade, ...]:
    plus_correct = {("HumanEval/0", 1), ("HumanEval/1", 2)}
    base_correct = plus_correct | {("HumanEval/1", 1)}
    correct = base_correct if scope == "base" else plus_correct
    return tuple(
        CandidateGrade(
            pool_id=candidate.pool_id,
            task_id=candidate.task_id,
            candidate_index=candidate.candidate_index,
            sanitized_code_sha256=candidate.sanitized_code_sha256,
            scope=scope,  # type: ignore[arg-type]
            status="pass" if (candidate.task_id, candidate.candidate_index) in correct else "fail",
            verification_passed=(candidate.task_id, candidate.candidate_index) in correct,
            error_type=(
                None
                if (candidate.task_id, candidate.candidate_index) in correct
                else "test_failure"
            ),
        )
        for candidate in pool.candidates
    )


def test_selection_analysis_uses_only_base_outcomes_during_selection() -> None:
    pool = analysis_pool()
    captured_base_passes: list[tuple[bool, ...]] = []

    def capturing_selector(
        candidates: tuple[CandidateRecord, ...],
        base_passes: tuple[bool, ...],
    ) -> int:
        assert len(candidates) == len(base_passes)
        captured_base_passes.append(base_passes)
        return next((index for index, passed in enumerate(base_passes) if passed), 0)

    analysis = analyze_selection_regret(
        pool,
        analysis_grades(pool, scope="base"),
        analysis_grades(pool, scope="plus"),
        selectors={"capturing": capturing_selector},
        k_values=(1, 2, 4),
        bootstrap_resamples=200,
    )

    assert captured_base_passes
    assert len(analysis.observations) == 6
    task_zero_k2 = next(
        item
        for item in analysis.observations
        if item.task_id == "HumanEval/0" and item.k == 2
    )
    assert task_zero_k2.selected_index == 1
    assert task_zero_k2.selected_plus_passed is True
    assert task_zero_k2.prefix_has_correct_candidate is True
    assert task_zero_k2.unbiased_pass_at_k == 0.5


def test_selection_summary_reports_false_acceptance_and_regret() -> None:
    pool = analysis_pool()

    analysis = analyze_selection_regret(
        pool,
        analysis_grades(pool, scope="base"),
        analysis_grades(pool, scope="plus"),
        k_values=(2, 4, 8),
        bootstrap_resamples=200,
        bootstrap_seed=7,
    )

    first_base_k2 = next(
        row
        for row in analysis.summaries
        if row.selector_name == "first_base_pass" and row.k == 2
    )
    assert first_base_k2.task_count == 2
    assert first_base_k2.selected_base_pass_rate == 1.0
    assert first_base_k2.selected_plus_pass_rate == 0.5
    assert first_base_k2.false_accept_rate == 0.5
    assert first_base_k2.prefix_oracle_pass_rate == 0.5
    assert first_base_k2.unbiased_pass_at_k == 0.5
    assert first_base_k2.selection_regret == 0.0
    assert {row.k for row in analysis.summaries} == {2, 4}
    first_base_difference_k2 = next(
        comparison
        for comparison in analysis.comparisons
        if comparison.challenger_selector == "first_base_pass" and comparison.k == 2
    )
    assert first_base_difference_k2.baseline_selector == "first_sample"
    assert first_base_difference_k2.selected_plus_rate_difference == 0.5
    coverage_gain_k4 = next(row for row in analysis.coverage_gains if row.k == 4)
    assert coverage_gain_k4.reference_k == 2
    assert coverage_gain_k4.unbiased_pass_at_k_gain == 0.5
    assert analysis.stopping_efficiency.max_k == 4
    assert analysis.stopping_efficiency.total_candidate_calls == 4
    assert analysis.stopping_efficiency.fixed_candidate_calls == 8
    assert analysis.stopping_efficiency.candidate_call_savings_rate == 0.5
