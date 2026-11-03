# Stop-Then-Escalate Result Bundle

This directory is the compact, machine-readable evidence bundle for the
HumanEval+ coverage/selection study and the MBPP+ stop-then-escalate
confirmation. It lets a reviewer inspect aggregate and task-level outcomes
without downloading model weights or rerunning 2,128+ generations.

## Included Evidence

| Files | Statistical unit | Purpose |
|---|---|---|
| `humaneval_selection_*` | 133 tasks | Repeated-sampling coverage, first-public-pass selection, selection regret, and stopping efficiency |
| `humaneval_development_*` | 26 paired tasks | Width-depth development comparison that froze `8x2` as the confirmation challenger |
| `mbpp_confirmation_*` | 100 paired tasks | Frozen `16x1` versus `8x2` confirmation and its negative result |

The JSON files contain aggregate estimates, paired bootstrap intervals, input
hashes, and protocol identifiers. The JSONL files contain only derived numeric
or Boolean task observations. They do not contain prompts, candidate programs,
canonical answers, hidden tests, or raw evaluator output.

## Verify Integrity

From the Python project directory:

```bash
ttc-operatorbench verify-results
```

or, without installing the console script:

```bash
PYTHONPATH=src python -m ttc_operatorbench verify-results
```

The command validates every file's SHA-256, byte size, JSON/JSONL syntax, and
record count against `manifest.json`.

## Interpretation

- HumanEval+: first-public-pass selection reached 80.5% hidden accuracy at
  `k=16`, 2.3 points below realized oracle coverage, while a sequential stopping
  replay preserved that selected answer and reduced candidate calls by 72.5%.
- MBPP+: `8x2` sampling-plus-repair reached 70.0% hidden accuracy versus 73.0%
  for `16x1` stop-only sampling. The paired difference was -3.0 points with a
  95% task-bootstrap interval of [-8.0, +1.0].

These results support stop-on-public-pass sampling as the current fixed
baseline. They do not establish a state-of-the-art adaptive controller.

Full narrative reports:

- [`docs/results/evalplus_selection_regret_locked.md`](../../../docs/results/evalplus_selection_regret_locked.md)
- [`docs/results/stop_then_escalate_development.md`](../../../docs/results/stop_then_escalate_development.md)
- [`docs/results/stop_then_escalate_confirmation.md`](../../../docs/results/stop_then_escalate_confirmation.md)
