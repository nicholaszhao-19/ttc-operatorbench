"""Run manifest generation for reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ttc_operatorbench import __version__


def build_run_manifest(
    *,
    config_payload: dict[str, Any],
    run_id: str,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """Build a run manifest tied to code, command, and config state."""
    config_json = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    status = _git(["status", "--short"])
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_version": __version__,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "git": {
            "branch": _git(["branch", "--show-current"]),
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(status.strip()),
            "status_short": status.splitlines(),
        },
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write a stable JSON run manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(_checkout_root()), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _checkout_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = ["build_run_manifest", "write_run_manifest"]
