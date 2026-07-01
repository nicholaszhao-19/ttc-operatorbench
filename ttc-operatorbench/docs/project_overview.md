# Project Overview

TTC OperatorBench is an early-stage research harness for studying whether
adaptive allocation of test-time compute can improve verifier-guided code
reasoning under explicit budgets.

The project is intentionally structured as an evaluation machine rather than a
single benchmark score. It defines tasks, model providers, search policies,
verifiers, metrics, logs, reports, and config-driven protocols so that policy
comparisons can be inspected after the run.

## Research Question

Given a code task, a model provider, a verifier, and a finite budget, can an
adaptive search policy allocate operators more effectively than fixed
verifier-guided baselines?

The current repository does not claim a real-model adaptive scheduler win. It
establishes a reproducible local harness and reports ties, losses, and
inconclusive outcomes explicitly.

## Motivation

Many reasoning and coding systems improve by spending more compute at test time:
sampling multiple candidates, repairing errors, revising outputs, or switching
prompting strategies. A credible evaluation must track not only final success,
but also the budget spent to get there.

This harness makes those costs auditable through:

- attempt-level JSONL logs;
- public and hidden verifier outcomes;
- token, attempt, verifier-call, and time budgets;
- policy-specific search traces;
- budget-aware summaries and plots;
- conservative decision reports.

## Implementation

- Typed schemas for tasks, generations, verifier results, attempts, budgets, and
  search results.
- Toy and curated local code tasks with public and hidden tests.
- Deterministic dummy providers for local structural controls.
- A Hugging Face provider for opt-in real-model smoke and curated runs.
- Baseline policies such as greedy, best-of-N, repair-only, plan-then-code, and
  local revision.
- An adaptive `operator_bandit` scheduler and ablation variants.
- Python unit-test verification for generated code.
- Config-driven experiment protocols.
- Metrics, JSONL logging, CSV/JSON summaries, plots, and Markdown reports.

## Repository Layout

```text
ttc-operatorbench/
|-- README.md
|-- AGENTS.md
|-- Makefile
|-- pyproject.toml
|-- uv.lock
|-- configs/
|   |-- experiments/        # Reproducible protocol configs.
|   `-- models/             # Model roster and small model configs.
|-- docs/                   # Overview and reproducibility documentation.
|-- outputs/                # Generated run artifacts, ignored by git.
|-- reports/                # Generated Markdown reports and plots, ignored by git.
|-- scripts/                # CLI scripts for runs, plots, and portfolio reports.
|-- src/ttc_operatorbench/  # Package source.
`-- tests/                  # Ruff/mypy/pytest-covered test suite.
```

## Core Concepts

- `Task`: benchmark item with prompt, metadata, and verifier-facing tests.
- `ModelProvider`: adapter that generates candidates from dummy or real model
  backends.
- `Verifier`: component that evaluates an attempt without leaking hidden
  grading information into policy selection.
- `SearchPolicy`: decision rule that spends budget across candidate attempts.
- `Operator`: prompting, repair, revision, or search mode used by a policy.
- `Budget`: explicit limits such as attempts, tokens, verifier calls, time, or
  cost.
- `AttemptLog`: append-only record of generated attempts and verifier outcomes.
- `SearchResult`: task-level result containing attempts, selection, and budget
  use.
- `EvalRunner`: orchestration for tasks, providers, policies, metrics, and
  report artifacts.

## Current Empirical Status

The current local protocols validate task loading, candidate generation,
verification, cost accounting, metrics, plots, and report generation. They do
not establish that adaptive operator allocation beats strong real-model
baselines.

Known status from the README:

- `Qwen/Qwen3-0.6B` smoke: structural pipeline validated, no scheduler win
  claimed.
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` curated probe: operator bandit matched the
  strongest baseline rather than beating it.
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` bounded probe: operator bandit matched the
  strongest baseline rather than beating it.
- Larger or stronger model studies remain future work.

## Where To Start

- Start with `README.md` for setup and quickstart.
- Read `docs/experiment_design.md` for the evaluation protocol.
- Read `docs/reproducibility.md` to rerun checks and experiments.
- Read `docs/results_guide.md` to interpret generated artifacts.
