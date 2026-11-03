"""Reusable repository and dependency provenance for immutable experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class GitProvenance:
    """Commit and content hash for one repository state."""

    commit: str
    dirty: bool
    state_sha256: str


def git_toplevel(start_directory: Path) -> Path:
    """Return the containing Git worktree root."""
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(completed.stdout.strip()).resolve()


def git_provenance(repository_root: Path) -> GitProvenance:
    """Hash the commit, tracked diff, and untracked file contents."""
    commit = _git_output(repository_root, "rev-parse", "HEAD").decode().strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise RuntimeError("git rev-parse did not return a full commit SHA")
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _git_output(repository_root, "diff", "--binary", "HEAD", "--")
    untracked = _git_output(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    state_hash = hashlib.sha256()
    state_hash.update(b"commit\0" + commit.encode() + b"\0diff\0" + diff)
    for relative_bytes in sorted(path for path in untracked if path):
        relative_path = Path(relative_bytes.decode("utf-8"))
        full_path = (repository_root / relative_path).resolve()
        full_path.relative_to(repository_root)
        if not full_path.is_file():
            continue
        state_hash.update(b"\0untracked\0" + relative_bytes + b"\0")
        state_hash.update(full_path.read_bytes())
    return GitProvenance(
        commit=commit,
        dirty=bool(status.strip()),
        state_sha256=state_hash.hexdigest(),
    )


def package_version(package: str) -> str:
    """Return an installed package version or fail with experiment context."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required optional package is not installed: {package}") from exc


def _git_output(repository_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return completed.stdout


__all__ = ["GitProvenance", "git_provenance", "git_toplevel", "package_version"]
