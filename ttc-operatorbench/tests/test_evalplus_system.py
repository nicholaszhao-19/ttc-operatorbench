"""Tests for safe EvalPlus sample export and Docker orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttc_operatorbench.core.candidate_pool import (
    CandidatePool,
    CandidatePoolManifest,
    CandidateRecord,
    sha256_text,
)
from ttc_operatorbench.core.schema import Generation
from ttc_operatorbench.systems.evalplus import (
    EVALPLUS_DOCKER_IMAGE,
    DockerUnavailableError,
    build_evalplus_docker_command,
    load_humaneval_plus_problems,
    parse_evalplus_results,
    run_evalplus_docker,
    sanitize_evalplus_candidate,
    write_evalplus_dataset_override,
    write_evalplus_sample_index,
    write_evalplus_samples,
)
from ttc_operatorbench.tasks.evalplus import tasks_from_evalplus_problems


def one_candidate_pool() -> CandidatePool:
    prompt = "def add(a, b):"
    code = "def add(a, b):\n    return a + b"
    candidate = CandidateRecord(
        pool_id="pool",
        task_id="HumanEval/0",
        candidate_index=0,
        generation=Generation(
            prompt=prompt,
            generation_text=code,
            input_tokens=3,
            output_tokens=5,
            total_tokens=8,
            latency_seconds=0.1,
            model_name="test-model",
            provider_name="dummy",
        ),
        sanitized_code=code,
        prompt_sha256=sha256_text(prompt),
        raw_completion_sha256=sha256_text(code),
        sanitized_code_sha256=sha256_text(code),
    )
    return CandidatePool(
        manifest=CandidatePoolManifest(
            pool_id="pool",
            dataset_name="humaneval_plus",
            dataset_version="0.3.1",
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
            pool_size=1,
            pool_seed=0,
            created_at_utc="2026-07-10T00:00:00Z",
        ),
        candidates=(candidate,),
    )


def test_evalplus_export_uses_official_minimal_sample_format(tmp_path: Path) -> None:
    pool = one_candidate_pool()

    samples_path = write_evalplus_samples(tmp_path / "samples.jsonl", pool)
    index_path = write_evalplus_sample_index(tmp_path / "sample_index.jsonl", pool)

    sample = json.loads(samples_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert sample == {
        "task_id": "HumanEval/0",
        "solution": "def add(a, b):\n    return a + b",
    }
    assert index["candidate_index"] == 0
    assert index["sanitized_code_sha256"] == pool.candidates[0].sanitized_code_sha256


def test_evalplus_docker_command_is_pinned_and_resource_limited(tmp_path: Path) -> None:
    write_evalplus_samples(tmp_path / "samples.jsonl", one_candidate_pool())
    write_evalplus_dataset_override(
        tmp_path / "dataset.jsonl",
        {"HumanEval/0": {"task_id": "HumanEval/0", "canonical_solution": "secret"}},
        ("HumanEval/0",),
    )
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    command = build_evalplus_docker_command(
        tmp_path,
        "samples.jsonl",
        base_only=True,
        dataset_filename="dataset.jsonl",
        output_directory=output_directory,
    )
    joined = " ".join(command)

    assert command[0:3] == ("docker", "run", "--rm")
    assert EVALPLUS_DOCKER_IMAGE in command
    assert "--platform linux/amd64" in joined
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges:true" in command
    assert "--pids-limit 256" in joined
    assert "--memory 4g" in joined
    assert "HOME=/tmp/evalplus-home" in command
    assert "XDG_CACHE_HOME=/tmp/evalplus-cache" in command
    assert "--base-only" in command
    assert f"src={tmp_path / 'samples.jsonl'},dst=/input/samples.jsonl,readonly" in joined
    assert f"src={tmp_path / 'dataset.jsonl'},dst=/input/private_dataset.jsonl,readonly" in joined
    assert f"src={output_directory},dst=/output" in joined
    assert f"{tmp_path.resolve()}:/work:rw" not in joined


def test_evalplus_command_rejects_path_escape(tmp_path: Path) -> None:
    write_evalplus_samples(tmp_path / "samples.jsonl", one_candidate_pool())
    write_evalplus_dataset_override(
        tmp_path / "dataset.jsonl",
        {"HumanEval/0": {"task_id": "HumanEval/0"}},
        ("HumanEval/0",),
    )

    with pytest.raises(ValueError, match="basename"):
        build_evalplus_docker_command(
            tmp_path,
            "../samples.jsonl",
            base_only=False,
            dataset_filename="dataset.jsonl",
            output_directory=tmp_path,
        )


def test_evalplus_runner_fails_closed_without_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_evalplus_samples(tmp_path / "samples.jsonl", one_candidate_pool())
    monkeypatch.setattr("ttc_operatorbench.systems.evalplus.shutil.which", lambda _: None)

    with pytest.raises(DockerUnavailableError, match="must not run on the host"):
        run_evalplus_docker(
            tmp_path,
            "samples.jsonl",
            base_only=False,
            dataset_filename="dataset.jsonl",
            output_directory=tmp_path,
        )


def test_evalplus_result_parser_separates_base_and_hidden_grades(tmp_path: Path) -> None:
    pool = one_candidate_pool()
    results = {
        "hash": "official-md5",
        "eval": {
            "HumanEval/0": [
                {
                    "task_id": "HumanEval/0",
                    "solution": pool.candidates[0].sanitized_code,
                    "base_status": "pass",
                    "plus_status": "fail",
                    "base_fail_tests": [],
                    "plus_fail_tests": ["SECRET_HIDDEN_INPUT"],
                }
            ]
        },
    }
    path = tmp_path / "samples_eval_results.json"
    path.write_text(json.dumps(results), encoding="utf-8")

    bundle = parse_evalplus_results(path, pool)

    assert bundle.official_dataset_hash == "official-md5"
    assert bundle.base_grades[0].verification_passed is True
    assert bundle.plus_grades[0].verification_passed is False
    assert bundle.plus_grades[0].error_type == "evalplus_fail"
    assert "SECRET_HIDDEN_INPUT" not in bundle.plus_grades[0].model_dump_json()


def test_evalplus_result_parser_rejects_solution_mismatch(tmp_path: Path) -> None:
    pool = one_candidate_pool()
    path = tmp_path / "samples_eval_results.json"
    path.write_text(
        json.dumps(
            {
                "hash": "official-md5",
                "eval": {
                    "HumanEval/0": [
                        {
                            "solution": "def add(a, b):\n    return 0",
                            "base_status": "pass",
                            "plus_status": "pass",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="solution digest mismatch"):
        parse_evalplus_results(path, pool)


def test_pinned_evalplus_loader_and_sanitizer_are_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):",
            "entry_point": "add",
        }
    }
    modules = {
        "evalplus.data": SimpleNamespace(get_human_eval_plus=lambda: problems),
        "evalplus.sanitize": SimpleNamespace(
            sanitize=lambda code, entrypoint: f"{code.strip()}\n# {entrypoint}"
        ),
    }
    monkeypatch.setattr(
        "ttc_operatorbench.systems.evalplus.importlib.metadata.version",
        lambda package: "0.3.1" if package == "evalplus" else "unknown",
    )
    monkeypatch.setattr(
        "ttc_operatorbench.systems.evalplus.importlib.import_module",
        lambda name: modules[name],
    )
    task = tasks_from_evalplus_problems(problems)[0]

    assert load_humaneval_plus_problems() == problems
    assert sanitize_evalplus_candidate(task, "def add(a, b): return a + b") == (
        "def add(a, b): return a + b\n# add"
    )


def test_evalplus_loader_rejects_unpinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ttc_operatorbench.systems.evalplus.importlib.metadata.version",
        lambda _package: "9.9.9",
    )

    with pytest.raises(RuntimeError, match="0.3.1 is required"):
        load_humaneval_plus_problems()
