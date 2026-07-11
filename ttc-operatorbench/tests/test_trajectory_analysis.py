"""Tests for task-paired width-depth hidden-correctness analysis."""

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePoolManifest,
    CandidateRecord,
    PublicFailureFeedback,
)
from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.core.trajectory import WidthDepthTrajectoryPool
from ttc_operatorbench.evals.trajectory_analysis import (
    TrajectoryPolicyComparison,
    analyze_width_depth_trajectory,
    classify_confirmation,
    compare_trajectory_policies,
    development_winner,
    validate_comparable_trajectory_pools,
)
from ttc_operatorbench.evals.width_depth import run_width_depth_search
from ttc_operatorbench.models.dummy import DummyModelProvider


class PrefixEvaluator:
    def evaluate(
        self,
        batch_id: str,
        candidates: tuple[CandidateRecord, ...],
    ) -> tuple[CandidateGrade, ...]:
        del batch_id
        return tuple(base_grade(candidate) for candidate in candidates)


def test_paired_trajectory_analysis_detects_repair_gain() -> None:
    tasks = (
        Task(task_id="task-a", prompt="solve a"),
        Task(task_id="task-b", prompt="solve b"),
    )
    baseline = run_width_depth_search(
        manifest("baseline", ("task-a", "task-b"), width=2, depth=1),
        tasks,
        DummyModelProvider(
            {
                "task-a": ("FAIL root 0", "FAIL root 1"),
                "task-b": ("PASS root",),
            }
        ),
        lambda _task, text: text,
        PrefixEvaluator(),
        width=2,
        depth=1,
    )
    challenger = run_width_depth_search(
        manifest("challenger", ("task-a", "task-b"), width=1, depth=2),
        tasks,
        DummyModelProvider(
            {
                "task-a": ("FAIL root", "PASS repair"),
                "task-b": ("PASS root",),
            }
        ),
        lambda _task, text: text,
        PrefixEvaluator(),
        width=1,
        depth=2,
    )
    validate_comparable_trajectory_pools((baseline, challenger))
    baseline_analysis = analyze_width_depth_trajectory(
        baseline,
        plus_grades(baseline),
        bootstrap_resamples=200,
    )
    challenger_analysis = analyze_width_depth_trajectory(
        challenger,
        plus_grades(challenger),
        bootstrap_resamples=200,
    )
    comparison = compare_trajectory_policies(
        baseline_analysis,
        challenger_analysis,
        bootstrap_resamples=200,
    )

    assert baseline_analysis.summary.hidden_pass_rate == 0.5
    assert challenger_analysis.summary.hidden_pass_rate == 1.0
    assert challenger_analysis.summary.hidden_solved_by_repair_count == 1
    assert comparison.hidden_pass_rate_difference == 0.5
    assert comparison.hidden_win_count == 1
    assert comparison.hidden_loss_count == 0
    assert comparison.meets_engineering_gate is True
    assert development_winner(
        (baseline_analysis, challenger_analysis)
    ) == challenger_analysis


def test_confirmation_rule_distinguishes_strong_suggestive_and_failed() -> None:
    comparison = _example_comparison()

    strong = comparison.model_copy(
        update={"hidden_pass_rate_difference": 0.04, "hidden_pass_ci_low": 0.01}
    )
    suggestive = comparison.model_copy(
        update={"hidden_pass_rate_difference": 0.04, "hidden_pass_ci_low": 0.0}
    )
    failed = comparison.model_copy(
        update={"hidden_pass_rate_difference": -0.03, "hidden_pass_ci_low": -0.08}
    )

    assert classify_confirmation(strong) == "strong_confirmation"
    assert classify_confirmation(suggestive) == "suggestive_only"
    assert classify_confirmation(failed) == "failed_confirmation"


def _example_comparison() -> TrajectoryPolicyComparison:
    tasks = (Task(task_id="task-a", prompt="solve a"),)
    baseline = run_width_depth_search(
        manifest("baseline-rule", ("task-a",), width=1, depth=1),
        tasks,
        DummyModelProvider({"task-a": ("PASS root",)}),
        lambda _task, text: text,
        PrefixEvaluator(),
        width=1,
        depth=1,
    )
    challenger = run_width_depth_search(
        manifest("challenger-rule", ("task-a",), width=1, depth=1),
        tasks,
        DummyModelProvider({"task-a": ("PASS root",)}),
        lambda _task, text: text,
        PrefixEvaluator(),
        width=1,
        depth=1,
    )
    return compare_trajectory_policies(
        analyze_width_depth_trajectory(
            baseline,
            plus_grades(baseline),
            bootstrap_resamples=10,
        ),
        analyze_width_depth_trajectory(
            challenger,
            plus_grades(challenger),
            bootstrap_resamples=10,
        ),
        bootstrap_resamples=10,
    )


def manifest(
    pool_id: str,
    task_ids: tuple[str, ...],
    *,
    width: int,
    depth: int,
) -> CandidatePoolManifest:
    return CandidatePoolManifest(
        pool_id=pool_id,
        dataset_name="test",
        dataset_version="1",
        dataset_sha256="a" * 64,
        repository_commit="deadbeef",
        task_ids=task_ids,
        model_id="dummy",
        model_revision="revision",
        tokenizer_revision="revision",
        provider_name="dummy",
        prompt_style="raw",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=16,
        pool_size=width * depth,
        pool_seed=0,
        created_at_utc="2026-07-11T00:00:00Z",
    )


def base_grade(candidate: CandidateRecord) -> CandidateGrade:
    passed = candidate.sanitized_code.startswith("PASS")
    return CandidateGrade(
        pool_id=candidate.pool_id,
        task_id=candidate.task_id,
        candidate_index=candidate.candidate_index,
        sanitized_code_sha256=candidate.sanitized_code_sha256,
        scope="base",
        status="pass" if passed else "fail",
        verification_passed=passed,
        error_type=None if passed else "evalplus_fail",
        public_feedback=(
            None
            if passed
            else PublicFailureFeedback(status="fail", total_failed_inputs=0)
        ),
    )


def plus_grades(pool: WidthDepthTrajectoryPool) -> tuple[CandidateGrade, ...]:
    steps = pool.steps
    return tuple(
        CandidateGrade(
            pool_id=step.candidate.pool_id,
            task_id=step.candidate.task_id,
            candidate_index=step.candidate.candidate_index,
            sanitized_code_sha256=step.candidate.sanitized_code_sha256,
            scope="plus",
            status="pass" if step.public_grade.verification_passed else "fail",
            verification_passed=step.public_grade.verification_passed,
            error_type=None if step.public_grade.verification_passed else "evalplus_fail",
        )
        for step in steps
    )
