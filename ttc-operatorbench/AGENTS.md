# Agent Instructions

## Mission

This repo studies cost-aware adaptive operator allocation for verifier-guided code
reasoning.

The goal is to build a reliable experimental machine: explicit contracts, typed
objects, fast checks, complete attempt logs, and reproducible evaluation flows.
Research experiments should be added only after the harness can make leakage,
cost, budget, verifier behavior, and candidate selection auditable.

## Non-Negotiable Rules

1. Implement one conceptual module per patch.
2. Add tests for every module.
3. Do not use hidden tests or benchmark answers for candidate selection.
4. Log every generated attempt, not only winners.
5. Keep default tests fast.
6. Do not add heavyweight dependencies unless explicitly requested.
7. Run `make check` before finishing.
8. Do not refactor unrelated files.

## Main Objects

The core research vocabulary is:

- `Task`: a benchmark item with public prompt, metadata, and allowed evaluation inputs.
- `ModelProvider`: an adapter that generates candidate attempts from model backends.
- `Verifier`: a component that scores, checks, or validates an attempt without leaking
  hidden answers into candidate selection.
- `SearchPolicy`: the decision rule that allocates operators and budget across attempts.
- `Operator`: a transformation, prompting strategy, repair action, or reasoning mode used
  by a search policy.
- `Budget`: explicit limits for cost, calls, tokens, time, retries, and other scarce
  resources.
- `AttemptLog`: an append-only record of each generated attempt, including failures,
  verifier outcomes, costs, timing, operator identity, and selection status.
- `SearchResult`: the final output of a search run, including the selected candidate,
  full attempt references, budget usage, and verifier summary.
- `EvalRunner`: the orchestration layer for running tasks, policies, verifiers, logging,
  metrics, and reports.

Prefer these names for public types once schemas are added. If a different name is
needed, document why in the module that introduces it.

## Module Boundaries

- `core/`: shared primitive types and cross-cutting contracts.
- `tasks/`: task definitions, task loading, and benchmark-facing metadata.
- `models/`: model provider interfaces and backend adapters.
- `verifiers/`: verification interfaces and verifier implementations.
- `search/`: search policies, operators, budget allocation, and search results.
- `schedulers/`: run scheduling and execution planning across tasks or policies.
- `evals/`: evaluation runners, aggregate metrics, and report inputs.
- `logging/`: attempt logs, run logs, serialization, and audit records.
- `systems/`: external system adapters that do not belong to model providers.

Keep module ownership crisp. A patch may update tests, configuration, or direct
callers needed for the module being introduced, but it should not redesign adjacent
areas opportunistically.

## Research Integrity

Candidate selection may use only information that would be available in a real run:
the public task input, declared training or calibration data, model outputs, verifier
signals that are allowed by the experiment design, and the current budget state.

Hidden tests, benchmark answers, final labels, or post-hoc evaluation artifacts must
never influence search policy choices, operator allocation, prompt construction,
retry decisions, or candidate ranking. If an experiment needs oracle information for
analysis, keep it in evaluation-only code paths and make that separation explicit.

## Logging

Every generated attempt must be logged, including attempts that fail parsing,
timeout, exceed budget, are rejected by a verifier, or are not selected. Logs should
be rich enough to reconstruct what happened without rerunning model inference.

At minimum, attempt logs should preserve task identity, run identity, policy identity,
operator identity, provider identity, budget snapshot, timing, cost counters, prompt
or prompt hash, output or output reference, verifier results, error state, and whether
the attempt was selected.

## Testing

Default tests must be fast, deterministic, local, and suitable for `make check`.
They should not require network access, model inference, large dataset downloads, or
GPU availability.

Use focused unit tests for each module. Add integration tests only when a contract
crosses module boundaries. Put slow or heavyweight experiments behind explicit opt-in
commands or markers rather than the default check path.

## Dependencies

Use the dependencies already declared in `pyproject.toml` when they are appropriate.
Do not add new heavyweight dependencies without an explicit user request. Prefer the
standard library or a small existing dependency for simple parsing, data structures,
and serialization.

## Development Workflow

- Use Python 3.11 or 3.12.
- Use `uv` for dependency and command execution.
- Keep the package in `src/ttc_operatorbench`.
- Keep generated outputs in `outputs/` and reports in `reports/`.
- Run `make check` before handing off changes.
- Leave unrelated files untouched.
