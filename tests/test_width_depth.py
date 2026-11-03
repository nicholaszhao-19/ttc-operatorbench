"""Tests for fixed width-depth routing, stopping, repair, and provenance."""

from pathlib import Path

import pytest

from ttc_operatorbench.core.candidate_pool import (
    CandidateGrade,
    CandidatePoolManifest,
    CandidateRecord,
    PublicFailureFeedback,
)
from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task
from ttc_operatorbench.core.trajectory import (
    read_trajectory_pool,
    write_trajectory_pool,
)
from ttc_operatorbench.evals.width_depth import run_width_depth_search


class ScriptedProvider:
    def __init__(self, outputs: dict[str, list[str]]):
        self.outputs = outputs
        self.counts = {task_id: 0 for task_id in outputs}
        self.prompts: dict[str, list[str]] = {task_id: [] for task_id in outputs}

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        index = self.counts[task.task_id]
        self.counts[task.task_id] += 1
        self.prompts[task.task_id].append(task.prompt)
        text = self.outputs[task.task_id][index]
        return Generation(
            prompt=task.prompt,
            generation_text=text,
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            latency_seconds=0.1,
            sampling=sampling or SamplingConfig(),
            model_name="scripted",
            provider_name="scripted",
        )


class ScriptedPublicEvaluator:
    def __init__(self) -> None:
        self.batches: list[tuple[str, tuple[str, ...]]] = []

    def evaluate(
        self,
        batch_id: str,
        candidates: tuple[CandidateRecord, ...],
    ) -> tuple[CandidateGrade, ...]:
        self.batches.append((batch_id, tuple(candidate.task_id for candidate in candidates)))
        grades: list[CandidateGrade] = []
        for candidate in candidates:
            passed = candidate.sanitized_code.startswith("PASS")
            grades.append(
                CandidateGrade(
                    pool_id=candidate.pool_id,
                    task_id=candidate.task_id,
                    candidate_index=candidate.candidate_index,
                    sanitized_code_sha256=candidate.sanitized_code_sha256,
                    scope="base",
                    status="pass" if passed else "fail",
                    verification_passed=passed,
                    error_type=None if passed else "test_failure",
                    public_feedback=(
                        None
                        if passed
                        else PublicFailureFeedback(
                            status="fail",
                            failed_inputs=([candidate.candidate_index],),
                            total_failed_inputs=1,
                        )
                    ),
                )
            )
        return tuple(grades)


def test_width_depth_stops_and_repairs_with_complete_lineage(tmp_path: Path) -> None:
    tasks = tuple(
        Task(task_id=task_id, prompt=f"solve {task_id}")
        for task_id in ("task-a", "task-b", "task-c")
    )
    provider = ScriptedProvider(
        {
            "task-a": ["PASS root"],
            "task-b": ["FAIL root 0", "FAIL root 1", "PASS repair"],
            "task-c": ["FAIL root 0", "FAIL root 1", "FAIL repair 0", "FAIL repair 1"],
        }
    )
    evaluator = ScriptedPublicEvaluator()

    pool = run_width_depth_search(
        trajectory_manifest(tuple(task.task_id for task in tasks)),
        tasks,
        provider,
        lambda _task, text: text,
        evaluator,
        width=2,
        depth=2,
    )

    assert [len(pool.steps_for_task(task_id)) for task_id in ("task-a", "task-b", "task-c")] == [
        1,
        3,
        4,
    ]
    task_b = pool.steps_for_task("task-b")
    assert task_b[-1].operator == "repair"
    assert task_b[-1].parent_candidate_index == 0
    assert task_b[-1].root_index == 0
    assert task_b[-1].depth == 1
    assert task_b[-1].selected is True
    assert "Failing public inputs (JSON): [[0]]" in provider.prompts["task-b"][-1]
    assert evaluator.batches == [
        ("root-0", ("task-a", "task-b", "task-c")),
        ("root-1", ("task-b", "task-c")),
        ("repair-1-0", ("task-b", "task-c")),
        ("repair-1-1", ("task-c",)),
    ]

    write_trajectory_pool(tmp_path, pool)
    assert read_trajectory_pool(tmp_path) == pool


@pytest.mark.parametrize("width,depth", [(16, 1), (8, 2), (4, 4), (2, 8)])
def test_preregistered_policies_exhaust_exactly_sixteen_calls(
    width: int,
    depth: int,
) -> None:
    task = Task(task_id="task-a", prompt="solve task-a")
    provider = ScriptedProvider({task.task_id: ["FAIL"] * 16})

    pool = run_width_depth_search(
        trajectory_manifest((task.task_id,), pool_size=16),
        (task,),
        provider,
        lambda _task, text: text,
        ScriptedPublicEvaluator(),
        width=width,
        depth=depth,
    )

    steps = pool.steps_for_task(task.task_id)
    assert len(steps) == 16
    assert provider.counts[task.task_id] == 16
    for index, step in enumerate(steps):
        if index < width:
            assert (step.operator, step.root_index, step.depth) == (
                "sample",
                index,
                0,
            )
            assert step.parent_candidate_index is None
            continue
        repair_depth, root_index = divmod(index - width, width)
        assert (step.operator, step.root_index, step.depth) == (
            "repair",
            root_index,
            repair_depth + 1,
        )
        expected_parent = (
            root_index
            if repair_depth == 0
            else width + (repair_depth - 1) * width + root_index
        )
        assert step.parent_candidate_index == expected_parent


def trajectory_manifest(
    task_ids: tuple[str, ...],
    *,
    pool_size: int = 4,
) -> CandidatePoolManifest:
    return CandidatePoolManifest(
        pool_id="trajectory-pool",
        dataset_name="test",
        dataset_version="1",
        dataset_sha256="a" * 64,
        repository_commit="deadbeef",
        task_ids=task_ids,
        model_id="scripted",
        model_revision="revision",
        tokenizer_revision="revision",
        provider_name="scripted",
        prompt_style="raw",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=256,
        pool_size=pool_size,
        pool_seed=0,
        created_at_utc="2026-07-11T00:00:00Z",
    )
