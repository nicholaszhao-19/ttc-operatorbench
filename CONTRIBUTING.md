# Contributing

TTC OperatorBench accepts focused improvements to experiment validity,
reproducibility, evaluation coverage, and documentation. Open an issue before a
large benchmark adapter, dependency, or policy redesign so its evidence and
cost requirements can be agreed first.

## Local Setup

```bash
cd ttc-operatorbench
uv sync --frozen --group dev
uv run ttc-operatorbench doctor
make check
```

The default checks must remain deterministic, local, and free of model or
dataset downloads.

## Change Workflow

1. Keep each change scoped to one research or engineering contract.
2. Add focused tests for new behavior and failure paths.
3. Run `make check` before opening a pull request.
4. Update the relevant protocol, runbook, or limitation when behavior changes.
5. Do not commit model caches, generated candidate pools, raw grades, or normal
   `outputs/` and `reports/` directories.

## Research Integrity

- Hidden tests, plus-test labels, canonical answers, and post-hoc outcomes must
  never influence candidate generation, ranking, repair, routing, or stopping.
- Freeze confirmatory task lists, metrics, gates, and tie-break rules before
  opening hidden labels.
- Compare policies with the same model revision, task set, sampling settings,
  and maximum budget unless the difference is the declared intervention.
- Log every generated attempt, including failures and unselected candidates.
- Report selected accuracy separately from oracle coverage and include
  uncertainty at the task level.
- Label development, exploratory, and confirmatory results explicitly.

## Generated-Code Safety

The local Python verifier is not a security sandbox. Use it only with trusted
toy fixtures. Untrusted generated programs must use the pinned, no-network
EvalPlus container path. Do not weaken container restrictions or add a host-code
execution path without an explicit threat-model review and tests.

## Pull Request Checklist

- [ ] Ruff, mypy, and pytest pass through `make check`.
- [ ] New behavior has tests, including invalid or fail-closed cases.
- [ ] Public and hidden information flows remain separated.
- [ ] Cost and provenance fields remain complete.
- [ ] Claims match the evidence and limitations.
- [ ] Generated or heavyweight files are excluded unless they form a reviewed,
      compact result bundle with a verified manifest.
