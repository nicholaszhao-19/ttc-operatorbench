# Research Memo

This memo states the public research position for TTC OperatorBench. It is
written for readers who want to understand the scientific intent, current
evidence, and next experiments without reading the full codebase first.

## Thesis

Modern code-reasoning systems spend substantial test-time compute on sampling,
verification, repair, revision, and prompt variation. TTC OperatorBench studies
whether that compute can be allocated more deliberately than fixed baselines
such as greedy decoding or best-of-N sampling.

The project claim is intentionally narrow:

> TTC OperatorBench is a reproducible harness for evaluating budget-aware
> verifier-guided code-generation policies.

It does not currently claim that the adaptive scheduler beats strong real-model
baselines.

## Core Question

Given a task, model provider, verifier, policy, and finite budget, can an
adaptive policy allocate operators more efficiently than fixed verifier-guided
baselines?

The budget can include:

- attempts;
- tokens;
- verifier calls;
- wall-clock seconds;
- cost.

## Motivation

Test-time compute is now a central axis for coding and reasoning systems. A
system can spend extra compute by drawing more samples, running tests, repairing
failures, generating plans, or revising a candidate. Those choices are not free,
and they can trade off solve rate, latency, tokens, and verifier calls.

This repo treats those tradeoffs as first-class experimental objects:

- every attempt is logged;
- public and hidden verification are separated;
- hidden performance is computed on the selected answer, with any-attempt
  hidden success reported only as an oracle diagnostic;
- success is reported over explicit budget curves;
- ties, losses, and inconclusive outcomes are preserved;
- experiment configs are saved with generated artifacts.

## Current Implementation

The harness includes:

- typed Pydantic schemas for tasks, generations, verifications, attempts,
  budgets, and search results;
- deterministic toy and curated code-task suites;
- dummy and Hugging Face model providers;
- public Python unit-test verification and post-run hidden grading;
- fixed baselines such as greedy, best-of-N, repair-only, plan-then-code, and
  local revision;
- a lightweight DiffCodeGen-style `diffcodegen_select` baseline that selects by
  behavior-trace clustering over deterministic probe inputs;
- adaptive `operator_bandit` schedulers and ablations;
- a first rule-based `bottleneck_controller` that classifies coverage failure,
  selection failure, and early-stop states before spending additional compute;
- JSONL logs, CSV/JSON summaries, plots, and Markdown reports;
- Ruff, mypy, and pytest coverage for the default local path.

## Current Evidence

The default deterministic toy protocol validates the machinery rather than
proving an adaptive scheduling win. In the current generated report:

- the verdict is `needs_analysis`;
- the decision metric scope is hidden-test success;
- `operator_bandit` does not dominate the strongest baseline;
- `best_of_n_2` is the strongest baseline at the two-call and four-call budget
  points.

This is useful evidence because it shows the decision logic is conservative: the
repo reports the negative result instead of overstating the scheduler.

## Main Limitations

- The default protocol uses a deterministic dummy provider.
- Current local tasks are small function-level tasks, not real repository
  repair.
- Real-model results are opt-in, resource-dependent, and preliminary.
- Most committed configs use a single seed.
- Gated real-model protocols that include Best-of-N now use stochastic decoding,
  but they should be rerun before being cited as evidence.
- The adaptive policy can run with per-run scheduler state, but it is not yet a
  contextual allocator over calibrated task features.
- The differential-selection baseline is probe-based, not coverage-guided
  fuzzing. It should be read as a local milestone toward DiffCodeGen-style
  comparison, not as a faithful reproduction.

## Next Research Steps

1. Rerun the gated stochastic real-model comparisons after the validity fixes.
2. Measure the public-to-hidden gap as search intensity increases.
3. Add paired confidence intervals around policy and budget comparisons.
4. Add an external benchmark adapter, starting with EvalPlus/HumanEval+ or
   MBPP+ before attempting a heavier SWE-bench-style workflow.
5. Replace deterministic probe mutation with coverage-guided or property-based
   differential input generation.
6. Extend `bottleneck_controller` into a contextual allocator using task
   features, prompt length, first-attempt errors, behavior-cluster margins, and
   calibration-set priors.
7. Harden execution isolation before running untrusted third-party benchmark
   code.

## Related Work Anchors

The current portfolio framing should acknowledge nearby test-time compute and
evaluation methodology work, including strategic bandit allocation for
test-time compute, compute-optimal inference scaling, and error bars for model
evals. The strongest near-term contribution for this repo is not a broad
state-of-the-art claim; it is a careful reproduction-style harness plus one
statistically defended finding about verifier over-optimization.

## Hiring-Signal Framing

For public portfolio use, the strongest framing is:

> This project demonstrates research engineering for code-agent evaluations:
> typed experiment infrastructure, verifier-guided search policies, budget-aware
> metrics, reproducible artifacts, and honest analysis of negative results.

That framing is stronger and more accurate than claiming a premature
state-of-the-art scheduler.
