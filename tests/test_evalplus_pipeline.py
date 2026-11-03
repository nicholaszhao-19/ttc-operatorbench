"""End-to-end dry run from immutable candidates to selection analysis."""

import json
import subprocess
import sys
from pathlib import Path

from ttc_operatorbench.core.candidate_pool import (
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
    write_candidate_grades,
    write_candidate_pool,
)
from ttc_operatorbench.core.schema import Generation
from ttc_operatorbench.systems.evalplus import (
    parse_evalplus_results,
    write_evalplus_samples,
)


def test_dry_pipeline_preserves_pool_identity_and_analyzes_grades(tmp_path: Path) -> None:
    pool = _two_candidate_pool()
    write_candidate_pool(tmp_path, pool)
    write_evalplus_samples(tmp_path / "samples.jsonl", pool)
    results = {
        "hash": "official-test-hash",
        "eval": {
            "HumanEval/0": [
                {
                    "solution": candidate.sanitized_code,
                    "base_status": "pass" if candidate.candidate_index == 1 else "fail",
                    "plus_status": "pass" if candidate.candidate_index == 1 else "fail",
                }
                for candidate in pool.candidates
            ]
        },
    }
    results_path = tmp_path / "samples_eval_results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    grades = parse_evalplus_results(results_path, pool)
    write_candidate_grades(tmp_path / "base_grades.jsonl", grades.base_grades)
    write_candidate_grades(tmp_path / "hidden_plus_grades.jsonl", grades.plus_grades)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_evalplus_pool.py",
            "--pool-dir",
            str(tmp_path),
            "--k-values",
            "1",
            "2",
            "--bootstrap-resamples",
            "100",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "selection_summary.json").read_text(encoding="utf-8"))
    first_base_k2 = next(
        row
        for row in summary["summaries"]
        if row["selector_name"] == "first_base_pass" and row["k"] == 2
    )
    assert first_base_k2["selected_plus_pass_rate"] == 1.0
    assert first_base_k2["prefix_oracle_pass_rate"] == 1.0
    assert first_base_k2["unbiased_pass_at_k"] == 1.0
    assert (tmp_path / "selection_observations.jsonl").is_file()
    assert (tmp_path / "selection_report.md").is_file()


def _two_candidate_pool() -> CandidatePool:
    prompt = "def add(a, b):"
    candidates: list[CandidateRecord] = []
    for candidate_index, body in enumerate(("return 0", "return a + b")):
        code = f"def add(a, b):\n    {body}"
        candidates.append(
            CandidateRecord(
                pool_id="dry-pool",
                task_id="HumanEval/0",
                candidate_index=candidate_index,
                generation=Generation(
                    prompt=prompt,
                    generation_text=code,
                    input_tokens=3,
                    output_tokens=5,
                    total_tokens=8,
                    latency_seconds=0.1,
                ),
                sanitized_code=code,
                prompt_sha256=sha256_text(prompt),
                raw_completion_sha256=sha256_text(code),
                sanitized_code_sha256=sha256_text(code),
            )
        )
    return CandidatePool(
        manifest=CandidatePoolManifest(
            pool_id="dry-pool",
            dataset_name="humaneval_plus",
            dataset_version="v0.1.10",
            dataset_sha256="a" * 64,
            repository_commit="deadbeef",
            task_ids=("HumanEval/0",),
            model_id="test-model",
            model_revision="revision",
            tokenizer_revision="revision",
            provider_name="dummy",
            prompt_style="raw",
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=256,
            pool_size=2,
            pool_seed=0,
            created_at_utc="2026-07-10T00:00:00Z",
        ),
        candidates=tuple(candidates),
    )
