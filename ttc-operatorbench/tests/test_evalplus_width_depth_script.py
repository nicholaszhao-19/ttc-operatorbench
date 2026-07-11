"""CLI safety tests for the public-only width-depth runner."""

import os
import subprocess
import sys
from pathlib import Path


def test_width_depth_help_is_network_model_and_docker_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_evalplus_width_depth.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--width" in completed.stdout
    assert "--depth" in completed.stdout
    assert "--task-offset" in completed.stdout
    assert "--allow-large-run" in completed.stdout


def test_width_depth_requires_explicit_real_model_gate(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("RUN_REAL_MODEL_TESTS", None)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_evalplus_width_depth.py",
            "--run-id",
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
    assert not (tmp_path / "blocked").exists()


def test_width_depth_requires_explicit_large_run_gate(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_evalplus_width_depth.py",
            "--run-id",
            "too-large",
            "--model-revision",
            "a" * 40,
            "--width",
            "8",
            "--depth",
            "2",
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "require --allow-large-run" in completed.stderr
    assert not (tmp_path / "too-large").exists()
