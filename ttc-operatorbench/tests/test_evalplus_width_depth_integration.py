"""Dry integration for width-depth routing through the EvalPlus batch boundary."""

import json
import subprocess
from pathlib import Path

import pytest

from ttc_operatorbench.core.candidate_pool import CandidatePoolManifest
from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.core.trajectory import write_trajectory_pool
from ttc_operatorbench.evals.evalplus_public_batch import EvalPlusPublicBatchEvaluator
from ttc_operatorbench.evals.width_depth import run_width_depth_search
from ttc_operatorbench.models.dummy import DummyModelProvider


def test_width_depth_routes_through_base_only_evalplus_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = (
        Task(task_id="HumanEval/0", prompt="solve zero"),
        Task(task_id="HumanEval/1", prompt="solve one"),
    )
    provider = DummyModelProvider(
        {
            "HumanEval/0": ("PASS root",),
            "HumanEval/1": ("FAIL root 0", "FAIL root 1", "PASS repair"),
        }
    )

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
        assert base_only is True
        samples = [
            json.loads(line)
            for line in (work_directory / samples_filename).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        evaluations: dict[str, list[dict[str, object]]] = {}
        for sample in samples:
            solution = str(sample["solution"])
            passed = solution.startswith("PASS")
            evaluations.setdefault(str(sample["task_id"]), []).append(
                {
                    "solution": solution,
                    "base_status": "pass" if passed else "fail",
                    "base_fail_tests": [] if passed else [["public-input"]],
                    "plus_status": "fail",
                    "plus_fail_tests": ["SECRET_PLUS_INPUT"],
                }
            )
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
        {task.task_id: {"task_id": task.task_id} for task in tasks},
    )
    pool = run_width_depth_search(
        manifest(tuple(task.task_id for task in tasks)),
        tasks,
        provider,
        lambda _task, text: text,
        evaluator,
        width=2,
        depth=2,
    )

    assert len(pool.steps_for_task("HumanEval/0")) == 1
    assert len(pool.steps_for_task("HumanEval/1")) == 3
    assert pool.steps_for_task("HumanEval/1")[-1].selected is True
    assert sorted(path.name for path in (tmp_path / "public_batches").iterdir()) == [
        "repair-1-0",
        "root-0",
        "root-1",
    ]
    write_trajectory_pool(tmp_path, pool)
    policy_state = (tmp_path / "trajectory_steps.jsonl").read_text(encoding="utf-8")
    assert "SECRET_PLUS_INPUT" not in policy_state
    assert "public-input" in policy_state


def manifest(task_ids: tuple[str, ...]) -> CandidatePoolManifest:
    return CandidatePoolManifest(
        pool_id="dry-integration",
        dataset_name="humaneval_plus",
        dataset_version="test",
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
        max_output_tokens=256,
        pool_size=4,
        pool_seed=0,
        created_at_utc="2026-07-11T00:00:00Z",
    )
