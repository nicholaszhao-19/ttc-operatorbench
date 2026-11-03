"""Tests for immutable candidate-pool contracts and persistence."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    PublicFailureFeedback,
    read_candidate_grades,
    read_candidate_pool,
    sha256_text,
    write_candidate_grades,
    write_candidate_pool,
)
from ttc_operatorbench.core.schema import Generation, SamplingConfig


def manifest() -> CandidatePoolManifest:
    return CandidatePoolManifest(
        pool_id="pool-1",
        dataset_name="humaneval_plus",
        dataset_version="0.3.1",
        dataset_sha256="a" * 64,
        repository_commit="deadbeef",
        task_ids=("HumanEval/0",),
        model_id="test-model",
        model_revision="revision-1",
        tokenizer_revision="revision-1",
        provider_name="dummy",
        prompt_style="raw",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=256,
        pool_size=2,
        pool_seed=0,
        created_at_utc="2026-07-10T00:00:00Z",
    )


def candidate(index: int, text: str) -> CandidateRecord:
    prompt = "def add(a, b):"
    sanitized = f"{prompt}\n    {text}"
    generation = Generation(
        prompt=prompt,
        generation_text=sanitized,
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        latency_seconds=0.1,
        sampling=SamplingConfig(
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            max_output_tokens=256,
            seed=index,
        ),
        model_name="test-model",
        provider_name="dummy",
    )
    return CandidateRecord(
        pool_id="pool-1",
        task_id="HumanEval/0",
        candidate_index=index,
        generation=generation,
        sanitized_code=sanitized,
        prompt_sha256=sha256_text(prompt),
        raw_completion_sha256=sha256_text(sanitized),
        sanitized_code_sha256=sha256_text(sanitized),
    )


def complete_pool() -> CandidatePool:
    return CandidatePool(
        manifest=manifest(),
        candidates=(candidate(0, "return a + b"), candidate(1, "return a - b")),
    )


def test_candidate_pool_requires_every_task_index_exactly_once() -> None:
    with pytest.raises(ValidationError, match="candidate pool must be complete"):
        CandidatePool(manifest=manifest(), candidates=(candidate(0, "return a + b"),))

    with pytest.raises(ValidationError, match="task/index pairs must be unique"):
        CandidatePool(
            manifest=manifest(),
            candidates=(candidate(0, "return a + b"), candidate(0, "return a - b")),
        )

    with pytest.raises(ValidationError, match="canonical task/index order"):
        CandidatePool(
            manifest=manifest(),
            candidates=(candidate(1, "return a - b"), candidate(0, "return a + b")),
        )


def test_candidate_record_rejects_content_hash_mismatch() -> None:
    with pytest.raises(ValidationError, match="raw_completion_sha256"):
        candidate(0, "return a + b").model_copy(
            update={"raw_completion_sha256": "b" * 64}
        ).__class__.model_validate(
            {
                **candidate(0, "return a + b").model_dump(),
                "raw_completion_sha256": "b" * 64,
            }
        )


def test_candidate_pool_round_trip_is_ordered_and_validated(tmp_path: Path) -> None:
    pool = complete_pool()

    write_candidate_pool(tmp_path, pool)
    restored = read_candidate_pool(tmp_path)

    assert restored == pool
    assert [item.candidate_index for item in restored.candidates_for_task("HumanEval/0")] == [
        0,
        1,
    ]
    lines = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"candidate_index":0' in lines[0]
    assert '"candidate_index":1' in lines[1]


def test_candidate_grades_round_trip_and_validate_status(tmp_path: Path) -> None:
    grades = (
        CandidateGrade(
            pool_id="pool-1",
            task_id="HumanEval/0",
            candidate_index=0,
            sanitized_code_sha256=candidate(0, "return a + b").sanitized_code_sha256,
            scope="base",
            status="pass",
            verification_passed=True,
        ),
        CandidateGrade(
            pool_id="pool-1",
            task_id="HumanEval/0",
            candidate_index=1,
            sanitized_code_sha256=candidate(1, "return a - b").sanitized_code_sha256,
            scope="base",
            status="fail",
            verification_passed=False,
            error_type="test_failure",
            public_feedback=PublicFailureFeedback(
                status="fail",
                failed_inputs=([1, 2],),
                total_failed_inputs=1,
            ),
        ),
    )

    path = write_candidate_grades(tmp_path / "base_grades.jsonl", grades)

    assert read_candidate_grades(path) == grades
    with pytest.raises(ValidationError, match="must agree"):
        CandidateGrade(
            pool_id="pool-1",
            task_id="HumanEval/0",
            candidate_index=0,
            sanitized_code_sha256="a" * 64,
            scope="base",
            status="pass",
            verification_passed=False,
        )

    with pytest.raises(ValidationError, match="hidden plus grades"):
        CandidateGrade(
            pool_id="pool-1",
            task_id="HumanEval/0",
            candidate_index=1,
            sanitized_code_sha256="a" * 64,
            scope="plus",
            status="fail",
            verification_passed=False,
            public_feedback=PublicFailureFeedback(status="fail"),
        )

    with pytest.raises(ValidationError, match="feedback_truncated"):
        PublicFailureFeedback(
            status="fail",
            failed_inputs=([1],),
            total_failed_inputs=2,
            feedback_truncated=False,
        )
