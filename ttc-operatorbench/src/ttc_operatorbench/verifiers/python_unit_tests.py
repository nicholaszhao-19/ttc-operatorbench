"""Verifier that runs candidate Python code against public unit tests."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ttc_operatorbench.core.schema import Generation, Task, VerificationResult
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY, HIDDEN_TESTS_KEY, PUBLIC_TESTS_KEY

VerificationScope = Literal["public", "hidden"]

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_CODE_START_RE = re.compile(
    r"(?m)^(?:async\s+def|def|class|from\s+\S+\s+import|import)\b"
)


def extract_python_code(candidate_text: str, *, entrypoint: str | None = None) -> str:
    """Extract Python code from fences, parseable text, or a code-like suffix."""
    fenced_blocks = _CODE_FENCE_RE.findall(candidate_text)
    if fenced_blocks:
        return "\n\n".join(block.strip() for block in fenced_blocks).strip()

    raw_text = candidate_text.strip()
    if _is_parseable_python(raw_text):
        return raw_text

    for start in _candidate_code_starts(raw_text, entrypoint=entrypoint):
        suffix = raw_text[start:].strip()
        parseable_prefix = _parseable_prefix(suffix)
        if parseable_prefix is not None:
            return parseable_prefix
    return raw_text


def _candidate_code_starts(candidate_text: str, *, entrypoint: str | None) -> tuple[int, ...]:
    starts: list[int] = []
    if entrypoint:
        pattern = re.compile(rf"(?m)^def\s+{re.escape(entrypoint)}\s*\(")
        starts.extend(match.start() for match in pattern.finditer(candidate_text))
    starts.extend(match.start() for match in _CODE_START_RE.finditer(candidate_text))
    return tuple(dict.fromkeys(starts))


def _parseable_prefix(candidate_text: str) -> str | None:
    lines = candidate_text.splitlines()
    for end_index in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end_index]).strip()
        if _is_parseable_python(candidate):
            return candidate
    return None


def _is_parseable_python(candidate_text: str) -> bool:
    if not candidate_text:
        return False
    try:
        ast.parse(candidate_text)
    except SyntaxError:
        return False
    return True


def _normalize_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def _tests_from_raw(raw_tests: object) -> tuple[str, ...]:
    if isinstance(raw_tests, str):
        return (raw_tests,)
    if isinstance(raw_tests, Iterable):
        return tuple(test for test in raw_tests if isinstance(test, str) and test.strip())
    return ()


def _tests_for_scope(task: Task, scope: VerificationScope) -> tuple[str, ...]:
    if scope == "public" and task.public_tests:
        return task.public_tests
    if scope == "hidden" and task.hidden_tests:
        return task.hidden_tests
    key = PUBLIC_TESTS_KEY if scope == "public" else HIDDEN_TESTS_KEY
    return _tests_from_raw(task.allowed_verifier_inputs.get(key))


def _script_for(candidate_code: str, tests: tuple[str, ...], *, scope: VerificationScope) -> str:
    test_block = "\n".join(tests)
    return f"{candidate_code}\n\n# {scope.title()} verifier tests\n{test_block}\n"


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
        """Verify raw candidate text against public tests."""
        return self.verify_public_text(task, candidate_text)

    def verify_public_text(self, task: Task, candidate_text: str) -> VerificationResult:
        """Verify raw candidate text against policy-visible public tests."""
        return self.verify_candidate_text(task, candidate_text, scope="public")

    def verify_hidden_text(self, task: Task, candidate_text: str) -> VerificationResult:
        """Verify raw candidate text against hidden evaluation tests."""
        return self.verify_candidate_text(task, candidate_text, scope="hidden")

    def verify_candidate_text(
        self,
        task: Task,
        candidate_text: str,
        *,
        scope: VerificationScope = "public",
    ) -> VerificationResult:
        """Verify raw candidate text for a task and test scope."""
        tests = _tests_for_scope(task, scope)
        if not tests:
            return VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                scope=scope,
                verifier_name=self.verifier_name,
                error_type=f"missing_{scope}_tests",
                stderr=f"Task has no {scope} verifier tests.",
            )

        entrypoint = task.allowed_verifier_inputs.get(ENTRYPOINT_KEY)
        if not isinstance(entrypoint, str):
            metadata_entrypoint = task.metadata.get(ENTRYPOINT_KEY)
            entrypoint = metadata_entrypoint if isinstance(metadata_entrypoint, str) else None
        candidate_code = extract_python_code(candidate_text, entrypoint=entrypoint)
        if not candidate_code:
            return VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                scope=scope,
                verifier_name=self.verifier_name,
                error_type="empty_code",
                stderr="Candidate did not contain Python code.",
            )

        with tempfile.TemporaryDirectory(prefix="ttc_operatorbench_") as tmp_dir:
            candidate_path = Path(tmp_dir) / "candidate.py"
            candidate_path.write_text(
                _script_for(candidate_code, tests, scope=scope),
                encoding="utf-8",
            )
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
                    scope=scope,
                    verifier_name=self.verifier_name,
                    stdout=_normalize_stream(exc.stdout),
                    stderr=_normalize_stream(exc.stderr),
                    error_type="timeout",
                )

        if completed.returncode == 0:
            return VerificationResult(
                verification_passed=True,
                verification_score=1.0,
                scope=scope,
                verifier_name=self.verifier_name,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error_type=None,
            )

        return VerificationResult(
            verification_passed=False,
            verification_score=0.0,
            scope=scope,
            verifier_name=self.verifier_name,
            stdout=completed.stdout,
            stderr=completed.stderr,
            error_type=_classify_failure(completed.stderr),
        )

    def verify_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Verify a structured generation against public tests."""
        return self.verify_public_generation(task, generation)

    def verify_public_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Verify a structured generation against policy-visible public tests."""
        return self.verify_public_text(task, generation.generation_text)

    def verify_hidden_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Verify a structured generation against hidden evaluation tests."""
        return self.verify_hidden_text(task, generation.generation_text)


__all__ = ["PythonUnitTestVerifier", "VerificationScope", "extract_python_code"]
