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

The repository does not claim a real-model adaptive scheduler win. It now
establishes a reproducible local harness, a sandboxed EvalPlus pipeline, and two
real-model studies that report a useful fixed stopping result and a failed
repair confirmation.

## Motivation

Many reasoning and coding systems improve by spending more compute at test time:
sampling multiple candidates, repairing errors, revising outputs, or switching
prompting strategies. A credible evaluation must track not only final success,
but also the budget spent to get there.

This harness makes those costs auditable through:

- attempt-level JSONL logs;
- public verifier outcomes, selected-answer hidden metrics, and oracle hidden
  diagnostics;
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
- Immutable candidate-pool and public-only width-depth trajectory schemas.
- A pinned, no-network Docker adapter for HumanEval+ and MBPP+ evaluation.
- Paired task-bootstrap analysis for selection regret and policy differences.
- Hash-verified compact result bundles for public inspection.
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
|-- artifacts/results/      # Reviewed, hash-verified derived observations.
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
- `Budget`: explicit limits on attempts, tokens, verifier calls, or time.
- `AttemptLog`: append-only record of generated attempts and verifier outcomes.
- `SearchResult`: task-level result containing attempts, selection, and budget
  use.
- `EvalRunner`: orchestration for tasks, providers, policies, metrics, and
  report artifacts.

## Current Empirical Status

- The 133-task locked HumanEval+ pool shows repeated-sampling Pass@k rising
  from 47.6% to 82.7%. First-public-pass selection reaches 80.5% at `k=16`,
  while equivalent sequential stopping saves 72.5% of candidate calls.
- On the frozen 100-task MBPP+ confirmation, `16x1` stop-only sampling reaches
  73.0% hidden accuracy and `8x2` sampling-plus-repair reaches 70.0%. The paired
  difference is -3.0 points with a 95% interval of [-8.0, +1.0].
- The repair challenger failed confirmation. No adaptive policy has beaten the
  fixed real-model baseline.
- Evidence is limited to one model revision and one generation seed; it does
  not reproduce the complete S*, DiffCodeGen, or learned-verifier systems.

## Where To Start

- Start with `README.md` for setup and quickstart.
- Read `docs/experiment_design.md` for the evaluation protocol.
- Read `docs/reproducibility.md` to rerun checks and experiments.
- Read `docs/results_guide.md` to interpret generated artifacts.
- Read `docs/experiments/stop_then_escalate_runbook.md` for the staged
  public-before-hidden confirmation workflow.
- Inspect `artifacts/results/stop_then_escalate_v1/` for the committed
  machine-readable evidence.
