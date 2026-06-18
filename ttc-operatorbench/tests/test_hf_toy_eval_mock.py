"""Mocked tests for the HF validation runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.evals.hf_toy_eval import (
    HFToyEvalConfig,
    default_output_dir_for_run,
    run_hf_toy_eval,
)
from ttc_operatorbench.evals.metrics import (
    assert_monotone_nondecreasing,
    success_curve_by_token_budget,
)
from ttc_operatorbench.models.dummy import DummyModelProvider

CORRECT_CANDIDATES = {
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "factorial": (
        "def factorial(n):\n"
        "    result = 1\n"
        "    for value in range(2, n + 1):\n"
        "        result *= value\n"
        "    return result"
    ),
}


def wrong_candidate(task: Task) -> str:
    entrypoint = task.metadata["entrypoint"]
    return f"def {entrypoint}(*args):\n    return None"


def mock_provider_factory(policy_name: str, task: Task) -> DummyModelProvider:
    correct = CORRECT_CANDIDATES.get(task.task_id)
    wrong = wrong_candidate(task)
    script: tuple[str, ...]
    if policy_name == "greedy":
        generation = correct if task.task_id == "is_even" and correct is not None else wrong
        script = (generation,)
    else:
        script = (wrong, correct or wrong)
    return DummyModelProvider(
        {task.task_id: script},
        provider_name="huggingface-mock",
        model_name="mock-hf-model",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_default_output_dir_for_run_scopes_model_and_policies(tmp_path: Path) -> None:
    output_dir = default_output_dir_for_run(
        "Qwen/Qwen3-0.6B",
        ("greedy", "operator_bandit"),
        root=tmp_path,
    )

    assert output_dir == tmp_path / "Qwen_Qwen3-0.6B" / "greedy__operator_bandit"


def test_mocked_hf_toy_eval_writes_attempts_summary_and_plot(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUN_REAL_MODEL_TESTS", raising=False)
    config = HFToyEvalConfig(
        model_id="mock-hf-model",
        output_dir=tmp_path,
        max_tasks=2,
        policies=("greedy",),
    )

    artifacts = run_hf_toy_eval(config, mock_provider_factory)

    assert artifacts.attempts_path.exists()
    assert artifacts.search_results_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.plot_path is not None
    assert artifacts.plot_path.exists()

    attempts = read_jsonl(artifacts.attempts_path)
    assert len(attempts) == 2
    for attempt in attempts:
        assert attempt["model_id"] == "mock-hf-model"
        assert attempt["input_tokens"] >= 0
        assert attempt["output_tokens"] >= 0
        assert attempt["total_tokens"] == attempt["input_tokens"] + attempt["output_tokens"]
        assert "verification_passed" in attempt
        assert "verification_score" in attempt
        assert "cumulative_tokens" in attempt
        assert "cumulative_verifier_calls" in attempt
        assert "cumulative_seconds" in attempt

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary == [
        {
            "median_tokens_to_first_solution": attempts[0]["cumulative_tokens"],
            "median_verifier_calls_to_first_solution": 1.0,
            "model_id": "mock-hf-model",
            "number_of_tasks": 2,
            "policy_name": "greedy",
            "solve_rate": 0.5,
            "solved_count": 1,
            "total_attempts": 2,
            "total_tokens": attempts[0]["cumulative_tokens"] + attempts[1]["cumulative_tokens"],
            "total_verifier_calls": 2,
        }
    ]


def test_selected_policies_respect_budget_and_success_curve_is_monotone(tmp_path: Path) -> None:
    config = HFToyEvalConfig(
        model_id="mock-hf-model",
        output_dir=tmp_path,
        max_tasks=1,
        policies=("best_of_n_2", "repair_only"),
    )

    artifacts = run_hf_toy_eval(config, mock_provider_factory)

    for result in artifacts.results:
        assert result.budget.max_attempts is not None
        assert result.budget.max_tokens is not None
        assert result.budget.max_verifier_calls is not None
        assert len(result.attempts) <= result.budget.max_attempts
        for attempt in result.attempts:
            assert attempt.cumulative_tokens <= result.budget.max_tokens
            assert attempt.cumulative_verifier_calls <= result.budget.max_verifier_calls

    curve = success_curve_by_token_budget(artifacts.results)
    assert_monotone_nondecreasing(curve)


def test_hf_toy_eval_script_skips_without_real_model_gate(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("RUN_REAL_MODEL_TESTS", None)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_hf_toy_eval.py",
            "--output-dir",
            str(tmp_path / "hf_toy_eval"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0
    assert "set RUN_REAL_MODEL_TESTS=1" in completed.stdout
    assert not (tmp_path / "hf_toy_eval" / "attempts.jsonl").exists()


def test_hf_toy_eval_script_help_documents_scoped_default_output_dir() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_hf_toy_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "outputs/hf_toy_eval/<model>/<policies>" in completed.stdout
