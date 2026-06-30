# TTC OperatorBench

TTC OperatorBench is an early-stage experimental harness for error-aware,
cost-aware operator allocation in verifier-guided code reasoning. It is built to
study when an agent should repair, resample, plan, revise, verify, or stop under
equal multi-resource budgets. The repo contains typed schemas, introductory and
curated local code tasks, deterministic verifier-backed baselines, an adaptive
operator-bandit scheduler, metrics, JSONL logging, config-driven reports, failure
taxonomy artifacts, and explicitly gated Hugging Face validation runners.

## Requirements

- Python 3.11 or 3.12
- uv

## Setup

```bash
uv sync --group dev
```

The core install intentionally avoids model and data-frame dependencies. Add
extras only when needed:

```bash
uv sync --group dev --extra hf
uv sync --group dev --extra analysis
```

## Checks

```bash
make check
```

The initial check suite runs Ruff, mypy, and pytest. No model inference is
required for the default checks.

## Gated Real-Model Validation

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
`outputs/hf_toy_eval/<model>/<policies>/` so separate policy validation runs do
not overwrite one another. Pass `--output-dir` to choose an exact destination.

## Config-Driven Protocol

The default validation experiment is config driven and can be run through the
package CLI:

```bash
uv run --python 3.12 ttc-operatorbench run-experiment
```

The script entry point is a thin compatibility wrapper:

```bash
uv run --python 3.12 python scripts/run_experiment.py
```

The default protocol lives at:

```text
configs/experiments/toy_protocol.yaml
```

It runs the introductory task suite across the baseline policies and
`operator_bandit` under a small budget sweep. Outputs are written to:

```text
outputs/runs/toy_protocol/
reports/runs/toy_protocol/
```

The run writes `attempts.jsonl`, `search_results.jsonl`, `summary.json`,
`summary.csv`, `config_snapshot.yaml`, `run_manifest.json`, `decision.json`,
`failure_taxonomy.json`, `failure_taxonomy.csv`, `decision_log.jsonl`,
`state_action_analysis.json`, `state_action_analysis.csv`, success-curve plots,
and a compact Markdown report. YAML protocol files can use ordinary YAML or
JSON-compatible syntax, and config snapshots are written as YAML. Hugging Face
models remain gated behind
`RUN_REAL_MODEL_TESTS=1`. The decision report is budget-aware:
`operator_bandit` is reported as `matches_baseline` when it only ties the
strongest baseline, and `promising` only when at least one compared budget shows
a clear win without losses or inconclusive budget points and the paired
bootstrap comparison does not contradict the win.

Tasks support public and hidden tests. Search policies receive only
policy-visible public verifier feedback; the experiment runner attaches hidden
verification results after policy execution. Primary hidden metrics are
selected-candidate metrics: the attempt selected by the policy must pass hidden
tests. Reports render those as selected-hidden values, for example
`selected_hidden_solve_rate` and `selected_hidden_token_auc`. Oracle diagnostics
with names such as `oracle_hidden_solve_rate` answer the separate analysis
question "did any generated attempt pass hidden tests?" and are not used as
deployed policy success. For sealed evaluation, pass an external hidden-test
JSON or JSONL path in the protocol; the policy-visible task is scrubbed of
hidden tests, and logs store hidden-test counts and hashes rather than
hidden-test source.

Cost budgets are first-class. Model entries can declare input-token,
output-token, verifier-call, and fixed-attempt costs; attempts and summaries
record cumulative cost, cost AUC, selected-hidden cost AUC, and oracle hidden
cost diagnostics alongside token and verifier-call metrics.

The verifier executes trusted local benchmark code. Do not point the harness at
arbitrary untrusted task code without an external sandbox.

Adaptive scheduler runs also emit decision-state logs. Each operator decision
records the visible failure state, remaining attempts/tokens/verifier calls/cost,
valid operators, chosen operator, resource deltas, and immediate outcome. The
state-action analysis groups those rows by visible state and operator to support
success-per-cost analysis.

## Current Empirical Status

`v0.1-ttc-harness` freezes the stable pre-hidden-tests harness. The current
empirical status is:

- `Qwen/Qwen3-0.6B` validation run: structural pipeline validated, no scheduler
  win claimed.
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` curated probe: operator bandit matched the
  strongest baseline rather than beating it.
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` bounded probe: operator bandit matched the
  strongest baseline rather than beating it.
- No real-model adaptive scheduler win is established yet.

These results are calibration data rather than final model claims. Very easy
tasks allow greedy baselines to saturate, very difficult tasks can suppress all
policies, and weak public tests can overstate progress. The next research step
is sealed calibrated hidden evaluation across multiple seeds and models, followed
by explicit state-action analysis for contextual operator allocation.

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

To run the minimal gated Hugging Face protocol through the same runner:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_smoke_protocol.yaml
```

## Local Real-Model Evaluation Ladder

The first credible real-model pass should stay local-resource-conscious and
coding-focused. These configs remain gated, so default checks never download
models:

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

Then run the full 1.5B protocol when the probe is acceptable. The full protocol
uses stochastic sampling and multiple protocol seeds so best-of-N baselines do
not repeat identical generations:

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

Then run the larger 7B protocol only if the probe completes. Like the full 1.5B
protocol, this config uses stochastic sampling and multiple protocol seeds:

```bash
RUN_REAL_MODEL_TESTS=1 UV_CACHE_DIR=.uv-cache \
  uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/hf_curated_qwen25_coder_7b_protocol.yaml \
  --run-id hf_qwen25_coder_7b_curated
```

`Qwen/Qwen2.5-Coder-1.5B-Instruct` is the main local-resource-conscious model
for this stage. `Qwen/Qwen2.5-Coder-7B-Instruct` is a stronger local candidate
only if the machine can complete a first task within available memory and time
limits. Kimi, MiniMax, GLM, DeepSeek frontier checkpoints, Devstral 24B,
Qwen3-Coder 30B, and Gemma 31B are intentionally deferred until a cloud/API or
stronger local-serving phase.

Real-model code tasks are prompted with a code-only instruction and, when the
tokenizer supports it, its chat template. Logs preserve both the rendered model
prompt and the original public task prompt.

To aggregate completed runs into one portfolio-style Markdown report:

```bash
uv run --python 3.12 ttc-operatorbench portfolio-report
```

The script wrapper accepts a space-separated run list:

```bash
uv run --python 3.12 python scripts/make_portfolio_report.py \
  --runs hf_qwen3_06b_smoke hf_qwen25_coder_05b_curated hf_qwen25_coder_15b_probe
```

The report is written to `reports/portfolio_report.md` by default and summarizes
run verdicts, budget comparisons, summary rows, artifact checks, cost metrics,
and unsuccessful examples from the committed config protocols. Older runs remain
loadable but are flagged when they do not include the newer manifest, failure
taxonomy, or state-action artifacts.

## What This Shows

The current local protocols validate the experimental pipeline: task loading,
policy execution, verifier calls, cost accounting, sealed/post-hoc hidden
grading, attempt logs, decision-state logs, failure taxonomy, state-action
analysis, paired comparisons, summary tables, plots, and budget-aware decisions.
The deterministic control protocols are structural controls, not model-quality
claims.

Tracked reviewer notes live in `docs/`:

- `docs/canonical_structural_control.md`
- `docs/positioning.md`

## What This Does Not Show Yet

The repo does not yet establish that adaptive operator allocation beats strong
real-model baselines. That requires opt-in real-model runs, sealed hidden
evaluation, multiple seeds or models, and a final report that treats ties,
losses, confidence intervals, and inconclusive runs as first-class outcomes.
