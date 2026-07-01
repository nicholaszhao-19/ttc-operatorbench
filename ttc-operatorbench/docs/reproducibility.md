# Reproducibility Guide

This guide explains how to recreate the local checks and supported experiment
artifacts without changing the project code.

## Environment

Requirements:

- Python 3.11 or 3.12
- `uv`

Install dependencies from the lockfile-backed project environment:

```bash
uv sync --all-groups
```

The project uses `pyproject.toml` and `uv.lock`. There is no
`requirements.txt` or `environment.yml` setup path.

## Local Checks

Run the default verification suite:

```bash
make check
```

This target runs:

- Ruff linting;
- mypy type checking;
- pytest.

Default checks are intended to be fast, deterministic, local, and free of model
downloads. Hugging Face model execution is gated separately.

## Smallest Local Experiment

Run the default config-driven protocol:

```bash
uv run --python 3.12 python scripts/run_experiment.py
```

Default config:

```text
configs/experiments/toy_protocol.yaml
```

The default protocol uses:

- `task_suite`: `toy_code`
- tasks: `is_even`, `factorial`, `reverse_string`, `is_prime`, `fibonacci`,
  `gcd`, `palindrome`
- model provider: deterministic `dummy`
- seed: `0`
- policies: `greedy`, `best_of_n_2`, `best_of_n_4`, `repair_only`,
  `plan_then_code`, `local_revision_basic`, `operator_bandit`
- budgets: `one_call`, `two_call`, `four_call`

Expected command output includes paths like:

```text
wrote attempts to outputs/runs/toy_protocol/attempts.jsonl
wrote search results to outputs/runs/toy_protocol/search_results.jsonl
wrote summary to outputs/runs/toy_protocol/summary.json
wrote decision to outputs/runs/toy_protocol/decision.json
wrote report to reports/runs/toy_protocol/report.md
```

## Generated Artifacts

For the default run, artifacts are written to:

```text
outputs/runs/toy_protocol/
reports/runs/toy_protocol/
```

The run writes:

- `attempts.jsonl`: every generated attempt, including failures.
- `search_results.jsonl`: task-level results with attempts and metadata.
- `summary.json`: aggregate metrics by model, policy, and budget.
- `summary.csv`: tabular version of the summary.
- `config_snapshot.yaml`: copy of the protocol config used for the run.
- `decision.json`: budget-aware decision summary.
- `report.md`: compact Markdown report.
- `success_vs_tokens.png`: success curve by token budget.
- `success_vs_verifier_calls.png`: success curve by verifier-call budget.

`outputs/` and `reports/` are ignored by git except for `.gitkeep` files.

## Other Supported Local Commands

Run the older deterministic toy baseline script:

```bash
uv run --python 3.12 python scripts/run_toy_eval.py
```

Generate a plot from that script's default JSONL output:

```bash
uv run --python 3.12 python scripts/make_plots.py
```

Run the 50-task curated deterministic local protocol:

```bash
uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/curated_protocol.yaml
```

Run scheduler ablations on the curated deterministic local suite:

```bash
uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/curated_ablation_protocol.yaml
```

Create a portfolio report from completed run IDs:

```bash
uv run --python 3.12 python scripts/make_portfolio_report.py \
  --runs toy_protocol curated_protocol
```

## Real-Model Runs

Real-model execution is opt-in. It may download model weights and can require
substantial CPU, memory, disk, or GPU resources.

Example tiny Hugging Face toy smoke:

```bash
RUN_REAL_MODEL_TESTS=1 HF_SMOKE_MODEL_ID=Qwen/Qwen3-0.6B UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_hf_toy_eval.py --max-tasks 1 --policies greedy
```

Example gated config-driven smoke:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_smoke_protocol.yaml
```

Do not treat real-model commands as part of the default verification path.

## Seeds

The committed config protocols use explicit seeds. The default and curated
deterministic protocols currently use:

```text
seeds: [0]
```

When adding or comparing larger experiments, record all seeds in the config and
keep the generated `config_snapshot.yaml` with run artifacts.

## Public And Hidden Tests

Policies may use public verifier feedback during search. Hidden verification is
attached only after policy execution finishes. Reports expose public solve rate,
hidden solve rate, public-hidden gap, and overfit rate.

This separation is important: hidden tests, labels, or benchmark answers must
not influence candidate selection, prompt construction, retry decisions, or
operator allocation.

## Matplotlib Cache Note

On machines where the default Matplotlib cache directory is not writable, set a
repo-local cache directory for commands that import plotting:

```bash
MPLCONFIGDIR=.uv-cache/matplotlib uv run --python 3.12 python scripts/run_experiment.py
```

`.uv-cache/` is already ignored by git.
