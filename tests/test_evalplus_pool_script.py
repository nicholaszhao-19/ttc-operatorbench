"""CLI safety tests for HumanEval+ candidate-pool generation."""

import os
import subprocess
import sys
from pathlib import Path


def test_generate_evalplus_pool_help_is_network_and_model_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_evalplus_pool.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--model-revision" in completed.stdout
    assert "--allow-evaluation-split" in completed.stdout
    assert "--all-tasks" in completed.stdout
    assert "--allow-dirty" in completed.stdout


def test_generate_evalplus_pool_requires_explicit_real_model_gate(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("RUN_REAL_MODEL_TESTS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_evalplus_pool.py",
            "--pool-id",
            "blocked",
            "--model-revision",
            "a" * 40,
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode != 0
    assert "set RUN_REAL_MODEL_TESTS=1" in completed.stderr


def test_evaluate_evalplus_pool_help_is_docker_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_evalplus_pool.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--pool-dir" in completed.stdout
    assert "--timeout-seconds" in completed.stdout


def test_analyze_evalplus_pool_help_is_grade_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_evalplus_pool.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--pool-dir" in completed.stdout
    assert "--bootstrap-resamples" in completed.stdout
    assert "--output-stem" in completed.stdout
