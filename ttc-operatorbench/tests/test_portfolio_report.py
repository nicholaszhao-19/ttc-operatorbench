"""Tests for portfolio report aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ttc_operatorbench.evals.portfolio import (
    REQUIRED_ATTEMPT_FIELDS,
    load_portfolio_runs,
    write_portfolio_report,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_run(output_root: Path, run_id: str) -> None:
    output_dir = output_root / run_id
    write_json(
        output_dir / "config_snapshot.yaml",
        {
            "experiment_id": "hf_unit_protocol",
            "task_suite": "curated_code",
            "task_ids": ["count_vowels"],
            "policies": ["greedy", "operator_bandit"],
            "budgets": [{"name": "one_call"}, {"name": "two_call"}],
            "models": [
                {
                    "name": "qwen25_coder_05b",
                    "provider": "huggingface",
                    "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    "model_tier": "small_coder_sanity",
                }
            ],
        },
    )
    write_json(
        output_dir / "decision.json",
        {
            "verdict": "promising",
            "metric_scope": "hidden",
            "rationale": "Synthetic unit run.",
            "budget_comparisons": [
                {
                    "budget_name": "one_call",
                    "status": "promising",
                    "metric_scope": "hidden",
                    "best_baseline_policy": "greedy",
                }
            ],
        },
    )
    write_json(
        output_dir / "summary.json",
        [
            {
                "model_name": "qwen25_coder_05b",
                "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "model_tier": "small_coder_sanity",
                "policy_name": "operator_bandit",
                "budget_name": "one_call",
                "number_of_tasks": 1,
                "solve_rate": 1.0,
                "public_solve_rate": 1.0,
                "hidden_solve_rate": 1.0,
                "overfit_rate": 0.0,
                "token_auc": 1.0,
                "hidden_token_auc": 1.0,
                "total_attempts": 1,
            }
        ],
    )
    attempt: dict[str, Any] = {
        field: "present"
        for field in REQUIRED_ATTEMPT_FIELDS
        if field not in {"input_tokens", "output_tokens", "total_tokens"}
    }
    attempt.update(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "metadata": {"budget_name": "one_call"},
            "result_metadata": {"budget_name": "one_call"},
        }
    )
    write_jsonl(output_dir / "attempts.jsonl", [attempt])
    write_jsonl(
        output_dir / "search_results.jsonl",
        [
            {
                "task_id": "count_vowels",
                "policy_name": "greedy",
                "success": False,
                "attempts": [{"error_type": "runtime_error"}],
            }
        ],
    )


def test_portfolio_report_loads_runs_and_writes_markdown(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    make_run(output_root, "hf_unit")

    runs = load_portfolio_runs(
        run_ids=("hf_unit",),
        output_root=output_root,
        report_root=report_root,
    )
    report_path = write_portfolio_report(tmp_path / "portfolio.md", runs)
    report = report_path.read_text(encoding="utf-8")

    assert "hf_unit: verdict=promising" in report
    assert "Qwen/Qwen2.5-Coder-0.5B-Instruct[small_coder_sanity]" in report
    assert "scope=hidden" in report
    assert "relationship=unknown" in report
    assert "public_solve_rate=1.000" in report
    assert "hidden_solve_rate=1.000" in report
    assert "overfit_rate=0.000" in report
    assert "operator_bandit / one_call" in report
    assert "missing_fields=none" in report
    assert "token_accounting_ok=True" in report
    assert "budget_name_present=True" in report
    assert "run_manifest_present=False" in report
    assert "failure_taxonomy_present=False" in report
    assert "decision_log_present=False" in report
    assert "state_action_analysis_present=False" in report
    assert "last_error=runtime_error" in report


def test_make_portfolio_report_script(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report_root = tmp_path / "reports"
    make_run(output_root, "hf_unit")
    output_path = tmp_path / "report.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/make_portfolio_report.py",
            "--runs",
            "hf_unit",
            "--output-root",
            str(output_root),
            "--report-root",
            str(report_root),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "wrote portfolio report" in completed.stdout
    assert output_path.exists()
