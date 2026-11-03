"""Tests for fixed-sample candidate-pool generation."""

from ttc_operatorbench.core.candidate_pool import CandidatePoolManifest, sha256_text
from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task
from ttc_operatorbench.evals.candidate_generation import generate_candidate_pool


class SamplingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SamplingConfig]] = []

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        assert sampling is not None
        self.calls.append((task.task_id, sampling))
        text = f"def f():\n    return {sampling.seed_offset}"
        return Generation(
            prompt=task.prompt,
            generation_text=text,
            input_tokens=2,
            output_tokens=4,
            total_tokens=6,
            latency_seconds=0.1,
            sampling=sampling,
            model_name="sampling-model",
            provider_name="sampling-provider",
        )


def pool_manifest() -> CandidatePoolManifest:
    return CandidatePoolManifest(
        pool_id="generated-pool",
        dataset_name="humaneval_plus",
        dataset_version="v0.1.10",
        dataset_sha256="a" * 64,
        repository_commit="deadbeef",
        task_ids=("HumanEval/0", "HumanEval/1"),
        model_id="sampling-model",
        model_revision="revision",
        tokenizer_revision="revision",
        provider_name="sampling-provider",
        prompt_style="raw",
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=256,
        pool_size=3,
        pool_seed=100,
        created_at_utc="2026-07-10T00:00:00Z",
    )


def pool_tasks() -> tuple[Task, ...]:
    return (
        Task(task_id="HumanEval/1", prompt="prompt one", metadata={"split": "evaluation"}),
        Task(task_id="HumanEval/0", prompt="prompt zero", metadata={"split": "development"}),
    )


def test_generate_candidate_pool_uses_manifest_order_and_fixed_seed_offsets() -> None:
    provider = SamplingProvider()

    pool = generate_candidate_pool(
        pool_manifest(),
        pool_tasks(),
        provider,
        sanitizer=lambda _task, text: text.strip(),
    )

    assert [(item.task_id, item.candidate_index) for item in pool.candidates] == [
        ("HumanEval/0", 0),
        ("HumanEval/0", 1),
        ("HumanEval/0", 2),
        ("HumanEval/1", 0),
        ("HumanEval/1", 1),
        ("HumanEval/1", 2),
    ]
    assert [sampling.seed_offset for _, sampling in provider.calls] == [0, 1, 2, 0, 1, 2]
    assert all(sampling.seed == 100 for _, sampling in provider.calls)
    assert all(sampling.do_sample is True for _, sampling in provider.calls)
    assert pool.candidates[0].raw_completion_sha256 == sha256_text(
        pool.candidates[0].generation.generation_text
    )
    assert pool.candidates[0].metadata["task_split"] == "development"


def test_generate_candidate_pool_requires_exact_task_set() -> None:
    provider = SamplingProvider()

    try:
        generate_candidate_pool(
            pool_manifest(),
            pool_tasks()[:1],
            provider,
            sanitizer=lambda _task, text: text,
        )
    except ValueError as exc:
        assert "exactly match" in str(exc)
    else:
        raise AssertionError("expected task-set validation failure")
