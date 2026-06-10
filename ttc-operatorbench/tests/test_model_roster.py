"""Tests for static model roster configuration."""

import json
from pathlib import Path

ROSTER_PATH = Path("configs/models/model_roster.yaml")
QWEN_TINY_PATH = Path("configs/models/qwen_tiny.yaml")
REQUIRED_KEYS = {"model_id", "tier", "role", "enabled_by_default", "stage"}
LOCAL_MODEST_MODELS = {
    "qwen25_coder_05b": (
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "small_coder_sanity",
    ),
    "qwen25_coder_15b": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "small_coder",
    ),
    "qwen25_coder_7b": (
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "strong_local_candidate",
    ),
    "starcoder2_3b": (
        "bigcode/starcoder2-3b",
        "optional_control",
    ),
}


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


def test_local_modest_models_are_config_only_and_gated() -> None:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    by_model_id = {entry["model_id"]: entry for entry in roster}

    for model_id, tier in LOCAL_MODEST_MODELS.values():
        entry = by_model_id[model_id]
        assert entry["tier"] == tier
        assert entry["enabled_by_default"] is False
        assert entry["requires_real_model_gate"] is True
        assert entry["stage"] in {"local_modest_eval", "local_modest_optional"}


def test_qwen_tiny_config_is_explicit_smoke_only() -> None:
    config = json.loads(QWEN_TINY_PATH.read_text(encoding="utf-8"))

    assert config["model_id"] == "Qwen/Qwen3-0.6B"
    assert config["requires_real_model_gate"] is True
    assert config["enabled_by_default"] is False
    assert config["stage"] == "explicit_smoke"
