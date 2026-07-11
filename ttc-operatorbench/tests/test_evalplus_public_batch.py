"""Tests for immutable base-only EvalPlus trajectory batches."""

import json
import subprocess
from pathlib import Path

import pytest

from ttc_operatorbench.core.candidate_pool import CandidateRecord, sha256_text
from ttc_operatorbench.core.schema import Generation
from ttc_operatorbench.evals.evalplus_public_batch import EvalPlusPublicBatchEvaluator
from ttc_operatorbench.systems.evalplus import EvalPlusDockerConfig


def test_public_batch_persists_base_only_audit_trail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        batch_candidate("HumanEval/0", 2, "def f():\n    return 0"),
        batch_candidate("HumanEval/1", 4, "def g():\n    return 1"),
    )

    def fake_run(
        work_directory: Path,
        samples_filename: str,
        *,
        base_only: bool,
        dataset: str,
        dataset_filename: str,
        output_directory: Path,
        config: object,
    ) -> subprocess.CompletedProcess[str]:
        del dataset_filename, config
        assert base_only is True
        assert dataset == "humaneval"
        samples = [
            json.loads(line)
            for line in (work_directory / samples_filename).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        evaluations = {
            sample["task_id"]: [
                {
                    "solution": sample["solution"],
                    "base_status": "fail",
                    "base_fail_tests": [[sample["task_id"]]],
                    "plus_status": "pass",
                    "plus_fail_tests": ["SECRET_PLUS_INPUT"],
                }
            ]
            for sample in samples
        }
        (output_directory / "samples_eval_results.json").write_text(
            json.dumps({"hash": "official-hash", "eval": evaluations}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "ttc_operatorbench.evals.evalplus_public_batch.run_evalplus_docker",
        fake_run,
    )
    evaluator = EvalPlusPublicBatchEvaluator(
        tmp_path,
        {
            "HumanEval/0": {"task_id": "HumanEval/0"},
            "HumanEval/1": {"task_id": "HumanEval/1"},
        },
        config=EvalPlusDockerConfig(image="example/evalplus@sha256:test"),
    )

    grades = evaluator.evaluate("root-0", candidates)

    assert [grade.candidate_index for grade in grades] == [2, 4]
    assert all(grade.scope == "base" for grade in grades)
    assert grades[0].public_feedback is not None
    assert grades[0].public_feedback.failed_inputs == (["HumanEval/0"],)
    batch_directory = tmp_path / "public_batches" / "root-0"
    manifest = json.loads(
        (batch_directory / "batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base_only"] is True
    assert manifest["docker_image"] == "example/evalplus@sha256:test"
    assert "--base-only" in manifest["command"]
    assert "SECRET_PLUS_INPUT" not in (batch_directory / "base_grades.jsonl").read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.evaluate("root-0", candidates)


def test_public_batch_preserves_logs_when_results_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = batch_candidate("HumanEval/0", 0, "def f():\n    return 0")

    def fake_run_without_results(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="container stdout",
            stderr="container stderr",
        )

    monkeypatch.setattr(
        "ttc_operatorbench.evals.evalplus_public_batch.run_evalplus_docker",
        fake_run_without_results,
    )
    evaluator = EvalPlusPublicBatchEvaluator(
        tmp_path,
        {"HumanEval/0": {"task_id": "HumanEval/0"}},
    )

    with pytest.raises(RuntimeError, match="without a results file"):
        evaluator.evaluate("root-0", (candidate,))

    batch_directory = tmp_path / "public_batches" / "root-0"
    assert (batch_directory / "evalplus_stdout.log").read_text(
        encoding="utf-8"
    ) == "container stdout"
    assert (batch_directory / "evalplus_stderr.log").read_text(
        encoding="utf-8"
    ) == "container stderr"


def batch_candidate(task_id: str, index: int, code: str) -> CandidateRecord:
    prompt = f"solve {task_id}"
    return CandidateRecord(
        pool_id="trajectory-pool",
        task_id=task_id,
        candidate_index=index,
        generation=Generation(
            prompt=prompt,
            generation_text=code,
            input_tokens=3,
            output_tokens=5,
            total_tokens=8,
            latency_seconds=0.1,
        ),
        sanitized_code=code,
        prompt_sha256=sha256_text(prompt),
        raw_completion_sha256=sha256_text(code),
        sanitized_code_sha256=sha256_text(code),
    )
