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
`RUN_REAL_MODEL_TESTS=1`. The decision report is budget-aware:
`operator_bandit` is reported as `matches_baseline` when it only ties the
strongest baseline, and `promising` only when at least one compared budget shows
a clear win without losses or inconclusive budget points.

Tasks now support public and hidden tests. Search policies receive only
policy-visible public verifier feedback; the experiment runner attaches hidden
verification results after policy execution. Reports expose public solve rate,
hidden solve rate, public-hidden gap, and overfit rate, and use hidden metrics as
the primary decision scope whenever hidden grading is available.

## Current Empirical Status

`v0.1-ttc-harness` freezes the stable pre-hidden-tests harness. The current
honest empirical status is:

- `Qwen/Qwen3-0.6B` smoke: structural pipeline validated, no scheduler win claimed.
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` curated probe: operator bandit matched the
  strongest baseline rather than beating it.
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` bounded probe: operator bandit matched the
  strongest baseline rather than beating it.
- No real-model adaptive scheduler win is established yet.

This is not a failure condition. If tasks are too easy, greedy saturates; if
tasks are too hard, every policy fails; and if public tests are weak, public
success can overstate real progress. The next research step is calibrated hidden
evaluation, then contextual operator allocation.

To run the 50-task local curated suite:

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

## Local-Modest Real-Model Ladder

The first credible real-model pass should stay local-modest and coding-focused.
These configs remain gated, so default checks never download models:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_05b_protocol.yaml \
  --run-id hf_qwen25_coder_05b_curated
```

On local CPU, run the bounded 1.5B probe before the full all-baselines protocol:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_15b_probe_protocol.yaml \
  --run-id hf_qwen25_coder_15b_probe
```

Then run the full 1.5B protocol when the probe is acceptable:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_15b_protocol.yaml \
  --run-id hf_qwen25_coder_15b_curated
```

Run the one-task 7B probe first if the model is not already downloaded:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_7b_probe_protocol.yaml \
  --run-id hf_qwen25_coder_7b_probe
```

Then run the larger 7B protocol only if the probe completes:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_7b_protocol.yaml \
  --run-id hf_qwen25_coder_7b_curated
```

`Qwen/Qwen2.5-Coder-1.5B-Instruct` is the main local-modest model for this
stage. `Qwen/Qwen2.5-Coder-7B-Instruct` is a stronger local candidate only if
the machine can complete a first task without memory or time limits. Kimi,
MiniMax, GLM, DeepSeek frontier checkpoints, Devstral 24B, Qwen3-Coder 30B, and
Gemma 31B are intentionally deferred until a cloud/API or stronger local-serving
phase.

Real-model code tasks are prompted with a code-only instruction and, when the
tokenizer supports it, its chat template. Logs preserve both the rendered model
prompt and the original public task prompt.

To aggregate completed runs into one portfolio-style Markdown report:

```bash
uv run --python 3.12 python scripts/make_portfolio_report.py \
  --runs hf_qwen3_06b_smoke hf_qwen25_coder_05b_curated hf_qwen25_coder_15b_probe
```

The report is written to `reports/portfolio_report.md` by default and summarizes
run verdicts, budget comparisons, summary rows, artifact checks, and failure
examples from the committed config protocols.

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
