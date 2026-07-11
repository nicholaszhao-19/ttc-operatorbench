"""Tests for post-search hidden evaluation and base-outcome rechecking."""

import json
import subprocess
from pathlib import Path

import pytest

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePoolManifest,
    CandidateRecord,
    PublicFailureFeedback,
    sha256_text,
)
from ttc_operatorbench.core.schema import Generation
from ttc_operatorbench.core.trajectory import (
    TrajectoryStep,
    WidthDepthTrajectoryHeader,
    WidthDepthTrajectoryPool,
    write_trajectory_pool,
)
from ttc_operatorbench.evals.evalplus_trajectory_hidden import (
    evaluate_evalplus_trajectory_hidden,
)
from ttc_operatorbench.tasks.evalplus import evalplus_dataset_sha256


def test_hidden_evaluation_rechecks_base_before_writing_plus_grades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems: dict[str, dict[str, object]] = {
        "HumanEval/0": {"task_id": "HumanEval/0"}
    }
    pool = trajectory_pool(problems)
    write_trajectory_pool(tmp_path, pool)

    def fake_run(
        work_directory: Path,
        samples_filename: str,
        *,
        base_only: bool,
        dataset_filename: str,
        output_directory: Path,
        config: object,
    ) -> subprocess.CompletedProcess[str]:
        del dataset_filename, config
        assert base_only is False
        samples = [
            json.loads(line)
            for line in (work_directory / samples_filename).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        results = []
        for index, sample in enumerate(samples):
            passed = index == 1
            results.append(
                {
                    "solution": sample["solution"],
                    "base_status": "pass" if passed else "fail",
                    "base_fail_tests": [] if passed else [["public-input"]],
                    "plus_status": "pass" if passed else None,
                    "plus_fail_tests": [],
                }
            )
        (output_directory / "samples_eval_results.json").write_text(
            json.dumps(
                {
                    "hash": "official-hash",
                    "eval": {"HumanEval/0": results},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "ttc_operatorbench.evals.evalplus_trajectory_hidden.run_evalplus_docker",
        fake_run,
    )
    result = evaluate_evalplus_trajectory_hidden(tmp_path, pool, problems)

    assert [grade.verification_passed for grade in result.plus_grades] == [False, True]
    manifest = json.loads(
        (result.output_directory / "evaluator_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base_only"] is False
    assert manifest["search_was_complete_before_hidden_evaluation"] is True
    assert "--base-only" not in manifest["command"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_evalplus_trajectory_hidden(tmp_path, pool, problems)


def trajectory_pool(problems: dict[str, dict[str, object]]) -> WidthDepthTrajectoryPool:
    task_id = "HumanEval/0"
    candidates = tuple(candidate(index) for index in range(2))
    fail_feedback = PublicFailureFeedback(
        status="fail",
        failed_inputs=(["public-input"],),
        total_failed_inputs=1,
    )
    grades = (
        grade(candidates[0], passed=False, feedback=fail_feedback),
        grade(candidates[1], passed=True),
    )
    manifest = CandidatePoolManifest(
        pool_id="trajectory",
        dataset_name="humaneval_plus",
        dataset_version="test",
        dataset_sha256=evalplus_dataset_sha256(problems),
        repository_commit="deadbeef",
        task_ids=(task_id,),
        model_id="dummy",
        model_revision="revision",
        tokenizer_revision="revision",
        provider_name="dummy",
        prompt_style="raw",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=16,
        pool_size=2,
        pool_seed=0,
        created_at_utc="2026-07-11T00:00:00Z",
    )
    return WidthDepthTrajectoryPool(
        header=WidthDepthTrajectoryHeader(
            width=2,
            depth=1,
            candidate_manifest=manifest,
        ),
        steps=(
            TrajectoryStep(
                candidate=candidates[0],
                public_grade=grades[0],
                operator="sample",
                root_index=0,
                depth=0,
                round_index=0,
            ),
            TrajectoryStep(
                candidate=candidates[1],
                public_grade=grades[1],
                operator="sample",
                root_index=1,
                depth=0,
                round_index=1,
                selected=True,
            ),
        ),
    )


def candidate(index: int) -> CandidateRecord:
    code = f"def f():\n    return {index}"
    prompt = "def f():"
    return CandidateRecord(
        pool_id="trajectory",
        task_id="HumanEval/0",
        candidate_index=index,
        generation=Generation(
            prompt=prompt,
            generation_text=code,
            input_tokens=2,
            output_tokens=4,
            total_tokens=6,
            latency_seconds=0.1,
        ),
        sanitized_code=code,
        prompt_sha256=sha256_text(prompt),
        raw_completion_sha256=sha256_text(code),
        sanitized_code_sha256=sha256_text(code),
    )


def grade(
    candidate_record: CandidateRecord,
    *,
    passed: bool,
    feedback: PublicFailureFeedback | None = None,
) -> CandidateGrade:
    return CandidateGrade(
        pool_id=candidate_record.pool_id,
        task_id=candidate_record.task_id,
        candidate_index=candidate_record.candidate_index,
        sanitized_code_sha256=candidate_record.sanitized_code_sha256,
        scope="base",
        status="pass" if passed else "fail",
        verification_passed=passed,
        error_type=None if passed else "evalplus_fail",
        public_feedback=feedback,
    )
