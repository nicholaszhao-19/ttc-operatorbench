# Canonical Structural-Control Report

This note is the tracked, reviewer-facing summary of the canonical local
structural-control run. The run is meant to validate the experiment machinery,
not to make a model-quality claim.

## Canonical Command

```bash
uv run --python 3.12 ttc-operatorbench run-experiment \
  --config configs/experiments/toy_protocol.yaml \
  --run-id toy_protocol
```

Equivalent script wrapper:

```bash
uv run --python 3.12 python scripts/run_experiment.py \
  --config configs/experiments/toy_protocol.yaml \
  --run-id toy_protocol
```

## Expected Artifacts

The run writes these artifacts under `outputs/runs/toy_protocol/` and
`reports/runs/toy_protocol/`:

- `attempts.jsonl`
- `search_results.jsonl`
- `summary.json`
- `summary.csv`
- `config_snapshot.yaml`
- `run_manifest.json`
- `decision.json`
- `failure_taxonomy.json`
- `failure_taxonomy.csv`
- `decision_log.jsonl`
- `state_action_analysis.json`
- `state_action_analysis.csv`
- `success_vs_tokens.png`
- `success_vs_verifier_calls.png`
- `report.md`

## Result Boundary

The canonical structural-control verdict is `needs_analysis`. This is expected:
the deterministic control validates logging, budget accounting, hidden-test
plumbing, report generation, and decision logic. It is not evidence that
adaptive operator allocation beats real-model baselines.

Primary hidden metrics are selected-candidate metrics. A policy receives credit
only when the candidate it selected for deployment passes hidden tests. Oracle
diagnostics with names beginning `oracle_hidden_*` answer a different analysis
question: whether any generated attempt passed hidden tests.

## Reviewer Checklist

- `decision.json` uses hidden metrics as the primary decision scope whenever
  hidden grading is available.
- Rendered reports label selected-hidden metrics separately from
  `oracle_hidden_*` diagnostics.
- `run_manifest.json` records the command, config hash, git status, platform,
  Python version, and package version.
- `state_action_analysis.*` is present for adaptive scheduler runs.
- Stochastic real-model protocols derive attempt seeds from stable run context;
  this structural control remains deterministic.
- The report story remains conservative unless selected-candidate hidden
  metrics support a stronger claim.
