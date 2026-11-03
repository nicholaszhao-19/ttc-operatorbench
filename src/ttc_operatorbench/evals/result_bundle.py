"""Validation for compact, committed research-result bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResultArtifact(BaseModel):
    """One derived data file tracked by a result-bundle manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    format: Literal["json", "jsonl"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    record_count: int = Field(ge=0)
    description: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("artifact paths must be normalized and relative")
        return value


class ResultBundleManifest(BaseModel):
    """Manifest for a reviewable subset of derived research outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    bundle_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_commits: tuple[str, ...] = Field(min_length=1)
    artifacts: tuple[ResultArtifact, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()

    @field_validator("source_commits")
    @classmethod
    def validate_source_commits(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        valid_characters = set("0123456789abcdef")
        if any(
            len(value) != 40 or not set(value).issubset(valid_characters)
            for value in values
        ):
            raise ValueError("source commits must be exact 40-character lowercase SHAs")
        if len(set(values)) != len(values):
            raise ValueError("source commits must be unique")
        return values

    @field_validator("artifacts")
    @classmethod
    def validate_unique_artifacts(
        cls,
        artifacts: tuple[ResultArtifact, ...],
    ) -> tuple[ResultArtifact, ...]:
        paths = tuple(artifact.path for artifact in artifacts)
        if len(set(paths)) != len(paths):
            raise ValueError("artifact paths must be unique")
        return artifacts


@dataclass(frozen=True)
class ResultBundleVerification:
    """Successful verification totals for one bundle."""

    bundle_id: str
    artifact_count: int
    record_count: int
    total_bytes: int


def load_result_bundle_manifest(path: Path) -> ResultBundleManifest:
    """Load and validate one result-bundle manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResultBundleManifest.model_validate(data)


def verify_result_bundle(manifest_path: Path) -> ResultBundleVerification:
    """Verify hashes, sizes, formats, and record counts for a result bundle."""
    manifest_path = manifest_path.resolve()
    manifest = load_result_bundle_manifest(manifest_path)
    bundle_directory = manifest_path.parent

    for artifact in manifest.artifacts:
        artifact_path = (bundle_directory / artifact.path).resolve()
        if not artifact_path.is_relative_to(bundle_directory):
            raise ValueError(f"artifact escapes bundle directory: {artifact.path}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing result artifact: {artifact.path}")
        contents = artifact_path.read_bytes()
        if len(contents) != artifact.bytes:
            raise ValueError(
                f"byte-size mismatch for {artifact.path}: "
                f"expected {artifact.bytes}, found {len(contents)}"
            )
        digest = hashlib.sha256(contents).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact.path}: "
                f"expected {artifact.sha256}, found {digest}"
            )
        record_count = _validate_records(contents, artifact.format, artifact.path)
        if record_count != artifact.record_count:
            raise ValueError(
                f"record-count mismatch for {artifact.path}: "
                f"expected {artifact.record_count}, found {record_count}"
            )

    return ResultBundleVerification(
        bundle_id=manifest.bundle_id,
        artifact_count=len(manifest.artifacts),
        record_count=sum(artifact.record_count for artifact in manifest.artifacts),
        total_bytes=sum(artifact.bytes for artifact in manifest.artifacts),
    )


def _validate_records(contents: bytes, format_name: str, path: str) -> int:
    text = contents.decode("utf-8")
    if format_name == "json":
        json.loads(text)
        return 1
    records = tuple(line for line in text.splitlines() if line.strip())
    for line_number, line in enumerate(records, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL in {path} at record {line_number}") from error
    return len(records)


__all__ = [
    "ResultArtifact",
    "ResultBundleManifest",
    "ResultBundleVerification",
    "load_result_bundle_manifest",
    "verify_result_bundle",
]
