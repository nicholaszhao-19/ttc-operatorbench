"""Public-only replay verification and post-selection hidden grading."""

from __future__ import annotations

from collections.abc import Iterable

from ttc_operatorbench.core.candidate_pool import CandidateGrade, CandidatePool
from ttc_operatorbench.core.schema import (
    AttemptLog,
    Generation,
    SearchResult,
    Task,
    VerificationResult,
)


def _validated_grade_index(
    pool: CandidatePool,
    grades: Iterable[CandidateGrade],
    *,
    required_scope: str,
) -> dict[tuple[str, str, int], CandidateGrade]:
    candidates = {
        (candidate.pool_id, candidate.task_id, candidate.candidate_index): candidate
        for candidate in pool.candidates
    }
    index: dict[tuple[str, str, int], CandidateGrade] = {}
    for grade in grades:
        if grade.scope != required_scope:
            raise ValueError(f"expected only {required_scope} grades")
        if grade.key in index:
            raise ValueError(f"duplicate candidate grade: {grade.key}")
        candidate = candidates.get(grade.key)
        if candidate is None:
            raise ValueError(f"grade does not reference candidate pool: {grade.key}")
        if grade.sanitized_code_sha256 != candidate.sanitized_code_sha256:
            raise ValueError(f"grade candidate digest mismatch: {grade.key}")
        index[grade.key] = grade
    if set(index) != set(candidates):
        missing = sorted(set(candidates) - set(index))
        raise ValueError(f"grades must cover the complete candidate pool; missing={missing}")
    return index


class PublicCandidateGradeVerifier:
    """Expose cached base outcomes without retaining any plus grades."""

    def __init__(self, pool: CandidatePool, base_grades: Iterable[CandidateGrade]):
        self._pool_id = pool.manifest.pool_id
        self._grades = _validated_grade_index(pool, base_grades, required_scope="base")

    def verify_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Return the cached base grade for one replayed generation."""
        key = _candidate_key(task, generation, expected_pool_id=self._pool_id)
        grade = self._grades.get(key)
        if grade is None:
            raise ValueError(f"no base grade for replayed candidate: {key}")
        return _verification_from_grade(grade, scope="public", verifier_name="evalplus_base")


def attach_hidden_candidate_grades(
    result: SearchResult,
    pool: CandidatePool,
    plus_grades: Iterable[CandidateGrade],
) -> SearchResult:
    """Attach plus outcomes after a selector has returned its result."""
    grade_index = _validated_grade_index(pool, plus_grades, required_scope="plus")
    attempts = tuple(
        _attempt_with_hidden_grade(attempt, pool.manifest.pool_id, grade_index)
        for attempt in result.attempts
    )
    metadata = {
        **result.metadata,
        "hidden_tests_available": True,
        "hidden_grading_policy_visible": False,
        "hidden_grade_source": "evalplus_base_plus_extra",
    }
    return result.model_copy(update={"attempts": attempts, "metadata": metadata})


def _attempt_with_hidden_grade(
    attempt: AttemptLog,
    pool_id: str,
    grades: dict[tuple[str, str, int], CandidateGrade],
) -> AttemptLog:
    metadata = attempt.metadata
    key = (
        metadata.get("candidate_pool_id"),
        attempt.task_id,
        metadata.get("candidate_index"),
    )
    if key[0] != pool_id or not isinstance(key[2], int):
        raise ValueError(f"attempt is not traceable to candidate pool: {attempt.attempt_id}")
    typed_key = (pool_id, attempt.task_id, key[2])
    grade = grades.get(typed_key)
    if grade is None:
        raise ValueError(f"no plus grade for attempt: {attempt.attempt_id}")
    hidden = _verification_from_grade(
        grade,
        scope="hidden",
        verifier_name="evalplus_base_plus_extra",
    )
    return attempt.model_copy(update={"hidden_verification": hidden})


def _candidate_key(
    task: Task,
    generation: Generation,
    *,
    expected_pool_id: str,
) -> tuple[str, str, int]:
    metadata = generation.metadata
    pool_id = metadata.get("candidate_pool_id")
    task_id = metadata.get("candidate_task_id")
    candidate_index = metadata.get("candidate_index")
    if pool_id != expected_pool_id:
        raise ValueError("generation candidate_pool_id does not match verifier pool")
    if task_id != task.task_id:
        raise ValueError("generation candidate_task_id does not match task")
    if not isinstance(candidate_index, int):
        raise ValueError("generation candidate_index is missing")
    return (expected_pool_id, task.task_id, candidate_index)


def _verification_from_grade(
    grade: CandidateGrade,
    *,
    scope: str,
    verifier_name: str,
) -> VerificationResult:
    return VerificationResult(
        verification_passed=grade.verification_passed,
        verification_score=1.0 if grade.verification_passed else 0.0,
        scope=scope,  # type: ignore[arg-type]
        verifier_name=verifier_name,
        stdout=grade.stdout,
        stderr=grade.stderr,
        error_type=grade.error_type,
    )


__all__ = ["PublicCandidateGradeVerifier", "attach_hidden_candidate_grades"]
