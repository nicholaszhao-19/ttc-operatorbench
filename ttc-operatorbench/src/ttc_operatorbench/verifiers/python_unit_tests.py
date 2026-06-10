"""Verifier that runs candidate Python code against public unit tests."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from ttc_operatorbench.core.schema import Generation, Task, VerificationResult
from ttc_operatorbench.tasks.toy_code import PUBLIC_TESTS_KEY

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(candidate_text: str) -> str:
    """Extract Python code from Markdown fences, or return stripped raw text."""
    fenced_blocks = _CODE_FENCE_RE.findall(candidate_text)
    if fenced_blocks:
        return "\n\n".join(block.strip() for block in fenced_blocks).strip()
    return candidate_text.strip()


def _normalize_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def _public_tests(task: Task) -> tuple[str, ...]:
    raw_tests = task.allowed_verifier_inputs.get(PUBLIC_TESTS_KEY)
    if isinstance(raw_tests, str):
        return (raw_tests,)
    if isinstance(raw_tests, Iterable):
        return tuple(test for test in raw_tests if isinstance(test, str) and test.strip())
    return ()


def _script_for(candidate_code: str, public_tests: tuple[str, ...]) -> str:
    test_block = "\n".join(public_tests)
    return f"{candidate_code}\n\n# Public verifier tests\n{test_block}\n"


def _classify_failure(stderr: str) -> str:
    if "SyntaxError:" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "test_failure"
    return "runtime_error"


class PythonUnitTestVerifier:
    """Run extracted candidate code in a subprocess with public tests appended."""

    def __init__(self, timeout_seconds: float = 2.0, verifier_name: str = "python_unit_tests"):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.verifier_name = verifier_name

    def verify_text(self, task: Task, candidate_text: str) -> VerificationResult:
        """Verify raw candidate text for a task."""
        public_tests = _public_tests(task)
        if not public_tests:
            return VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                verifier_name=self.verifier_name,
                error_type="missing_public_tests",
                stderr="Task has no public verifier tests.",
            )

        candidate_code = extract_python_code(candidate_text)
        if not candidate_code:
            return VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                verifier_name=self.verifier_name,
                error_type="empty_code",
                stderr="Candidate did not contain Python code.",
            )

        with tempfile.TemporaryDirectory(prefix="ttc_operatorbench_") as tmp_dir:
            candidate_path = Path(tmp_dir) / "candidate.py"
            candidate_path.write_text(_script_for(candidate_code, public_tests), encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(candidate_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return VerificationResult(
                    verification_passed=False,
                    verification_score=0.0,
                    verifier_name=self.verifier_name,
                    stdout=_normalize_stream(exc.stdout),
                    stderr=_normalize_stream(exc.stderr),
                    error_type="timeout",
                )

        if completed.returncode == 0:
            return VerificationResult(
                verification_passed=True,
                verification_score=1.0,
                verifier_name=self.verifier_name,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error_type=None,
            )

        return VerificationResult(
            verification_passed=False,
            verification_score=0.0,
            verifier_name=self.verifier_name,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type=_classify_failure(completed.stderr),
        )

    def verify_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Verify a structured generation for a task."""
        return self.verify_text(task, generation.generation_text)


__all__ = ["PythonUnitTestVerifier", "extract_python_code"]
