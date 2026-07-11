"""CLI safety tests for post-search hidden trajectory grading."""

import os
import subprocess
import sys
from pathlib import Path


def test_hidden_trajectory_help_is_docker_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_evalplus_trajectory.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--trajectory-dir" in completed.stdout
    assert "--timeout-seconds" in completed.stdout


def test_hidden_trajectory_requires_explicit_gate(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("RUN_HIDDEN_EVAL", None)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_evalplus_trajectory.py",
            "--trajectory-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode != 0
    assert "set RUN_HIDDEN_EVAL=1" in completed.stderr
