"""Tests for committed result-bundle integrity checks."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ttc_operatorbench.evals.result_bundle import (
    ResultBundleManifest,
    verify_result_bundle,
)

SOURCE_COMMIT = "b19fdb3c61383f9f95d028ff380ccf914ea1c66a"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_verify_result_bundle_accepts_valid_files(tmp_path: Path) -> None:
    summary = b'{"accuracy": 0.75}\n'
    observations = b'{"task_id":"one","passed":true}\n{"task_id":"two","passed":false}\n'
    (tmp_path / "summary.json").write_bytes(summary)
    (tmp_path / "observations.jsonl").write_bytes(observations)
    manifest_path = _write_manifest(tmp_path, summary, observations)

    result = verify_result_bundle(manifest_path)

    assert result.bundle_id == "test_bundle"
    assert result.artifact_count == 2
    assert result.record_count == 3
    assert result.total_bytes == len(summary) + len(observations)


def test_verify_result_bundle_rejects_changed_file(tmp_path: Path) -> None:
    summary = b'{"accuracy": 0.75}\n'
    observations = b'{"task_id":"one","passed":true}\n'
    (tmp_path / "summary.json").write_bytes(summary)
    (tmp_path / "observations.jsonl").write_bytes(observations)
    manifest_path = _write_manifest(tmp_path, summary, observations)
    (tmp_path / "summary.json").write_text('{"accuracy": 1.0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="byte-size mismatch|SHA-256 mismatch"):
        verify_result_bundle(manifest_path)


def test_result_bundle_manifest_rejects_parent_path() -> None:
    with pytest.raises(ValidationError, match="normalized and relative"):
        ResultBundleManifest.model_validate(
            {
                "bundle_id": "unsafe",
                "description": "Unsafe test bundle.",
                "source_commits": [SOURCE_COMMIT],
                "artifacts": [
                    {
                        "path": "../private.json",
                        "format": "json",
                        "sha256": "0" * 64,
                        "bytes": 0,
                        "record_count": 1,
                        "description": "Unsafe path.",
                    }
                ],
            }
        )


def test_committed_stop_then_escalate_bundle_is_intact() -> None:
    result = verify_result_bundle(
        PROJECT_ROOT / "artifacts/results/stop_then_escalate_v1/manifest.json"
    )

    assert result.bundle_id == "stop_then_escalate_v1"
    assert result.artifact_count == 6
    assert result.record_count == 1637
    assert result.total_bytes == 544666


def _write_manifest(
    directory: Path,
    summary: bytes,
    observations: bytes,
) -> Path:
    manifest = {
        "schema_version": "1",
        "bundle_id": "test_bundle",
        "description": "Test bundle.",
        "source_commits": [SOURCE_COMMIT],
        "artifacts": [
            {
                "path": "summary.json",
                "format": "json",
                "sha256": hashlib.sha256(summary).hexdigest(),
                "bytes": len(summary),
                "record_count": 1,
                "description": "Aggregate result.",
            },
            {
                "path": "observations.jsonl",
                "format": "jsonl",
                "sha256": hashlib.sha256(observations).hexdigest(),
                "bytes": len(observations),
                "record_count": 2 if observations.count(b"\n") == 2 else 1,
                "description": "Task observations.",
            },
        ],
        "limitations": [],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
