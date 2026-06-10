"""Tests for static model roster configuration."""

import json
from pathlib import Path

ROSTER_PATH = Path("configs/models/model_roster.yaml")
QWEN_TINY_PATH = Path("configs/models/qwen_tiny.yaml")
REQUIRED_KEYS = {"model_id", "tier", "role", "enabled_by_default", "stage"}


def test_model_roster_parses_without_loading_models() -> None:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))

    assert isinstance(roster, list)
    assert roster


def test_model_roster_entries_have_required_fields() -> None:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))

    for entry in roster:
        assert REQUIRED_KEYS <= set(entry)


def test_future_large_and_medium_models_disabled_by_default() -> None:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))

    for entry in roster:
        if entry["stage"] in {"future_large_eval", "future_medium_eval"}:
            assert entry["enabled_by_default"] is False


def test_qwen_tiny_config_is_explicit_smoke_only() -> None:
    config = json.loads(QWEN_TINY_PATH.read_text(encoding="utf-8"))

    assert config["model_id"] == "Qwen/Qwen3-0.6B"
    assert config["enabled_by_default"] is False
    assert config["stage"] == "explicit_smoke"
