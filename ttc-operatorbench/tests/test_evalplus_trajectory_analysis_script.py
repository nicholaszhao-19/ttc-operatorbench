"""CLI tests for matched trajectory analysis."""

import subprocess
import sys


def test_trajectory_analysis_help_is_grade_free() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_evalplus_trajectories.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--trajectory-dir" in completed.stdout
    assert "--bootstrap-resamples" in completed.stdout
