"""Tests for candidate-pool replay and public/hidden grade separation."""

import pytest

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
)
from ttc_operatorbench.core.schema import Budget, Generation, SamplingConfig, Task
from ttc_operatorbench.models.replay import (
    CandidatePoolExhaustedError,
    CandidatePoolReplayProvider,
)
from ttc_operatorbench.search.baselines import BestOfNPolicy
from ttc_operatorbench.verifiers.candidate_grades import (
    PublicCandidateGradeVerifier,
    attach_hidden_candidate_grades,
)


def replay_pool() -> CandidatePool:
    prompt = "def add(a, b):"
    texts = ("def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b")
    candidates = tuple(
        CandidateRecord(
            pool_id="replay-pool",
            task_id="HumanEval/0",
            candidate_index=index,
            generation=Generation(
                prompt=prompt,
                generation_text=text,
                input_tokens=3,
                output_tokens=5,
                total_tokens=8,
                latency_seconds=0.1,
                sampling=SamplingConfig(seed=index),
                model_name="test-model",
                provider_name="source-provider",
            ),
            sanitized_code=text,
            prompt_sha256=sha256_text(prompt),
            raw_completion_sha256=sha256_text(text),
            sanitized_code_sha256=sha256_text(text),
        )
        for index, text in enumerate(texts)
    )
    return CandidatePool(
        manifest=CandidatePoolManifest(
            pool_id="replay-pool",
            dataset_name="humaneval_plus",
            dataset_version="0.3.1",
            dataset_sha256="a" * 64,
            repository_commit="deadbeef",
            task_ids=("HumanEval/0",),
            model_id="test-model",
            model_revision="revision-1",
            tokenizer_revision="revision-1",
            provider_name="source-provider",
            prompt_style="raw",
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=256,
            pool_size=2,
            pool_seed=0,
            created_at_utc="2026-07-10T00:00:00Z",
        ),
        candidates=candidates,
    )


def grades(pool: CandidatePool, scope: str) -> tuple[CandidateGrade, ...]:
    return tuple(
        CandidateGrade(
            pool_id=candidate.pool_id,
            task_id=candidate.task_id,
            candidate_index=candidate.candidate_index,
            sanitized_code_sha256=candidate.sanitized_code_sha256,
            scope=scope,  # type: ignore[arg-type]
            status="pass" if candidate.candidate_index == 1 else "fail",
            verification_passed=candidate.candidate_index == 1,
            error_type=None if candidate.candidate_index == 1 else "test_failure",
        )
        for candidate in pool.candidates
    )


def replay_task() -> Task:
    return Task(
        task_id="HumanEval/0",
        prompt="def add(a, b):",
        task_family="humaneval_plus",
        metadata={"entrypoint": "add"},
        allowed_verifier_inputs={"entrypoint": "add"},
    )


def test_replay_provider_returns_identical_order_for_each_selector() -> None:
    pool = replay_pool()
    first_provider = CandidatePoolReplayProvider(pool)
    second_provider = CandidatePoolReplayProvider(pool)

    first = tuple(first_provider.generate(replay_task()).generation_text for _ in range(2))
    second = tuple(second_provider.generate(replay_task()).generation_text for _ in range(2))

    assert first == second
    assert first[0].endswith("a - b")
    assert first[1].endswith("a + b")
    with pytest.raises(CandidatePoolExhaustedError):
        first_provider.generate(replay_task())


def test_public_verifier_cannot_access_plus_grades() -> None:
    pool = replay_pool()
    verifier = PublicCandidateGradeVerifier(pool, grades(pool, "base"))

    assert not hasattr(verifier, "verify_hidden_generation")
    assert not hasattr(verifier, "plus_grades")


def test_best_of_n_replay_selects_from_base_then_hidden_is_attached() -> None:
    pool = replay_pool()
    provider = CandidatePoolReplayProvider(pool)
    public_verifier = PublicCandidateGradeVerifier(pool, grades(pool, "base"))

    result = BestOfNPolicy(n=2).run(
        replay_task(),
        provider,
        public_verifier,
        Budget(max_attempts=2, max_verifier_calls=2, max_tokens=1_000),
        run_id="replay",
    )

    assert len(result.attempts) == 2
    assert result.selected_attempt_id == result.attempts[1].attempt_id
    assert all(attempt.hidden_verification is None for attempt in result.attempts)

    graded = attach_hidden_candidate_grades(result, pool, grades(pool, "plus"))

    assert graded.metadata["hidden_grading_policy_visible"] is False
    assert graded.attempts[0].hidden_verification is not None
    assert graded.attempts[0].hidden_verification.verification_passed is False
    assert graded.attempts[1].hidden_verification is not None
    assert graded.attempts[1].hidden_verification.verification_passed is True


def test_grade_verifier_rejects_digest_mismatch() -> None:
    pool = replay_pool()
    mismatched = grades(pool, "base")[0].model_copy(
        update={"sanitized_code_sha256": "b" * 64}
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        PublicCandidateGradeVerifier(pool, (mismatched, grades(pool, "base")[1]))
