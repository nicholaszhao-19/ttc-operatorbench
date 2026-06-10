# TTC OperatorBench

TTC OperatorBench is an early-stage experimental harness for cost-aware adaptive
operator allocation in verifier-guided code reasoning. The repo contains typed
schemas, toy code tasks, deterministic verifier-backed baselines, an adaptive
operator-bandit scheduler, metrics, JSONL logging, and explicitly gated
Hugging Face smoke runners.

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
