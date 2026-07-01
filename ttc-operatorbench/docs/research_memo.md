# Research Memo

This memo states the public research position for TTC OperatorBench. It is
written for reviewers who want to understand the scientific intent, current
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

## Why This Is Worth Studying

Test-time compute is now a central axis for coding and reasoning systems. A
system can spend extra compute by drawing more samples, running tests, repairing
failures, generating plans, or revising a candidate. Those choices are not free,
and they can trade off solve rate, latency, tokens, and verifier calls.

This repo treats those tradeoffs as first-class experimental objects:

- every attempt is logged;
- public and hidden verification are separated;
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
- adaptive `operator_bandit` schedulers and ablations;
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
- Stochastic best-of-N and larger sample budgets are important baselines and
  should be included before making scheduler claims.
- The current adaptive policy is short-horizon and per-task; it does not yet
  learn contextual allocation across a calibrated task distribution.

## Next Research Steps

1. Run the stronger fixed-baseline protocol with `best_of_n_8` and
   `best_of_n_16`.
2. Run stochastic real-model comparisons with multiple seeds.
3. Add an external benchmark adapter, starting with EvalPlus/HumanEval+ or
   MBPP+ before attempting a heavier SWE-bench-style workflow.
4. Add confidence intervals and paired comparisons across budgets.
5. Extend `operator_bandit` into a contextual allocator using task features,
   prompt length, first-attempt errors, and calibration-set priors.
6. Harden execution isolation before running untrusted third-party benchmark
   code.

## Hiring-Signal Framing

For public portfolio use, the strongest framing is:

> This project demonstrates research engineering for code-agent evaluations:
> typed experiment infrastructure, verifier-guided search policies, budget-aware
> metrics, reproducible artifacts, and honest analysis of negative results.

That framing is stronger and more accurate than claiming a premature
state-of-the-art scheduler.
