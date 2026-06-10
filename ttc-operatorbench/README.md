# TTC OperatorBench

TTC OperatorBench is an early-stage experimental harness for cost-aware adaptive
operator allocation in verifier-guided code reasoning. The repo contains typed
schemas, toy and curated local code tasks, deterministic verifier-backed
baselines, an adaptive operator-bandit scheduler, metrics, JSONL logging,
config-driven reports, and explicitly gated Hugging Face smoke runners.

## Requirements

- Python 3.11 or 3.12
- uv

## Setup

```bash
uv sync --all-groups
```

## Checks

```bash
make check
```

The initial check suite runs Ruff, mypy, and pytest. No model inference is
required for the default checks.

## Tiny Real-Model Smoke

Real-model runs are opt-in so default tests never download models:

```bash
RUN_REAL_MODEL_TESTS=1 HF_SMOKE_MODEL_ID=Qwen/Qwen3-0.6B UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_hf_toy_eval.py --max-tasks 1 --policies greedy
```

```bash
RUN_REAL_MODEL_TESTS=1 HF_SMOKE_MODEL_ID=Qwen/Qwen3-0.6B UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_hf_toy_eval.py --max-tasks 1 --policies operator_bandit
```

When `--output-dir` is omitted, the script writes to a scoped directory under
`outputs/hf_toy_eval/<model>/<policies>/` so separate policy smoke runs do not
overwrite one another. Pass `--output-dir` to choose an exact destination.

## Config-Driven Protocol

The main proof-of-life experiment is config driven:

```bash
uv run --python 3.12 python scripts/run_experiment.py
```

The default protocol lives at:

```text
configs/experiments/toy_protocol.yaml
```

It runs the toy task suite across the baseline policies and `operator_bandit`
under a small budget sweep. Outputs are written to:

```text
outputs/runs/toy_protocol/
reports/runs/toy_protocol/
```

The run writes `attempts.jsonl`, `search_results.jsonl`, `summary.json`,
`summary.csv`, `config_snapshot.yaml`, `decision.json`, success-curve plots, and
a compact Markdown report. Hugging Face models remain gated behind
`RUN_REAL_MODEL_TESTS=1`. The decision report is budget-aware: `operator_bandit`
is not considered promising unless it matches or exceeds the strongest baseline
at every compared budget point.

To run the larger local curated suite:

```bash
uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/curated_protocol.yaml
```

To run scheduler ablations over that same curated suite:

```bash
uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/curated_ablation_protocol.yaml
```

To run the tiny gated Hugging Face protocol through the same runner:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_smoke_protocol.yaml
```

## What This Shows

The current local protocols validate the full experimental pipeline: task
loading, policy execution, verifier calls, cost accounting, logs, summary
tables, plots, and budget-aware decisions. The dummy protocols are structural
controls, not model-quality claims.

## What This Does Not Show Yet

The repo does not yet establish that adaptive operator allocation beats strong
real-model baselines. That requires opt-in real-model runs, multiple seeds or
models, and a final report that treats ties, losses, and inconclusive runs as
first-class outcomes.
