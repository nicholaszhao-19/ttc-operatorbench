# TTC OperatorBench

TTC OperatorBench is a research harness for studying test-time compute
operator allocation in verifier-guided code reasoning. It asks a practical
question for coding agents: under a fixed budget, when should an agent sample,
repair, plan, revise, verify, or stop?

The implementation lives in [`ttc-operatorbench/`](ttc-operatorbench/). The
repo is organized to be auditable: typed experiment schemas, public and hidden
test handling, cost and token budgets, attempt logs, decision-state logs,
failure taxonomy artifacts, plots, reports, and reproducible config snapshots.

## Claim Boundary

The harness separates deployed-policy metrics from oracle diagnostics. It does
not claim an adaptive scheduler win. Hidden-test metrics mean selected-candidate
hidden success: the attempt the policy would actually deploy must pass hidden
tests. Oracle diagnostics are reported separately as `oracle_hidden_*` and mean
that some generated attempt passed hidden tests, even if the policy did not
select it.

Current empirical posture:

- Local deterministic protocols validate the experiment pipeline and artifacts.
- Bounded real-model probes are calibration runs, not final claims.
- Default CI covers local checks and mocked/gated provider behavior, not live
  Hugging Face inference.
- The adaptive scheduler has not yet been shown to beat strong real-model
  baselines under sealed hidden evaluation.

## Quickstart

```bash
cd ttc-operatorbench
uv sync --group dev
make check
```

Run the default structural protocol:

```bash
uv run --python 3.12 ttc-operatorbench run-experiment \
  --config configs/experiments/toy_protocol.yaml
```

Build a portfolio report from completed runs:

```bash
uv run --python 3.12 ttc-operatorbench portfolio-report
```

The legacy scripts in `ttc-operatorbench/scripts/` are thin wrappers around the
package CLI.

## What Reviewers Should Inspect

- Project README: [`ttc-operatorbench/README.md`](ttc-operatorbench/README.md)
- Canonical structural-control note:
  [`ttc-operatorbench/docs/canonical_structural_control.md`](ttc-operatorbench/docs/canonical_structural_control.md)
- Related-work positioning:
  [`ttc-operatorbench/docs/positioning.md`](ttc-operatorbench/docs/positioning.md)
- Default protocol:
  [`ttc-operatorbench/configs/experiments/toy_protocol.yaml`](ttc-operatorbench/configs/experiments/toy_protocol.yaml)
- Main runner:
  [`ttc-operatorbench/src/ttc_operatorbench/evals/experiment.py`](ttc-operatorbench/src/ttc_operatorbench/evals/experiment.py)

## Trust Boundary

The verifier is intended for trusted local benchmark code. Do not run arbitrary
untrusted task code through this harness without an external sandbox.
