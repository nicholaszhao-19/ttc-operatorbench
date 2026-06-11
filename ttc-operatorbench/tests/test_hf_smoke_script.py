"""Tests for the explicit HF smoke script gate."""

import subprocess
import sys


def test_hf_smoke_script_skips_without_env_gate() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_hf_smoke.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "set RUN_REAL_MODEL_TESTS=1" in completed.stdout
