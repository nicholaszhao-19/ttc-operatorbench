# Demo Result: Toy Protocol

This page is a static, reviewer-friendly summary of the deterministic toy
protocol. It is not a real-model benchmark claim. It exists so a reader can
inspect the shape of the harness output without running the project locally.

## Protocol

- Config: `configs/experiments/toy_protocol.yaml`
- Task suite: `toy_code`
- Tasks: `is_even`, `factorial`, `reverse_string`, `is_prime`, `fibonacci`,
  `gcd`, `palindrome`
- Provider: deterministic `dummy_control`
- Policies: `greedy`, `best_of_n_2`, `best_of_n_4`, `repair_only`,
  `plan_then_code`, `local_revision_basic`, `operator_bandit`
- Budgets: `one_call`, `two_call`, `four_call`
- Seed: `0`
- Decision scope: hidden-test success

## Verdict

`needs_analysis`

The adaptive policy does not dominate the strongest configured baseline across
the compared budgets.

## Summary Table

| Policy | Budget | Hidden solve rate | Median tokens to hidden solution | Hidden token AUC |
| --- | --- | ---: | ---: | ---: |
| `greedy` | `one_call` | 0.429 | 21 | 4.500 |
| `best_of_n_2` | `two_call` | 1.000 | 42 | 53.143 |
| `best_of_n_2` | `four_call` | 1.000 | 42 | 53.143 |
| `repair_only` | `two_call` | 1.000 | 79 | 17.286 |
| `local_revision_basic` | `two_call` | 1.000 | 79 | 17.286 |
| `operator_bandit` | `two_call` | 1.000 | 79 | 17.286 |
| `operator_bandit` | `four_call` | 1.000 | 79 | 17.286 |

## Budget Comparison

| Budget | Strongest baseline | Adaptive status | Interpretation |
| --- | --- | --- | --- |
| `one_call` | `greedy` | loss | `operator_bandit` spends its first move on an operator that does not solve. |
| `two_call` | `best_of_n_2` | loss | Both solve all tasks, but best-of-N reaches solutions with fewer tokens. |
| `four_call` | `best_of_n_2` | loss | Extra available budget does not help the adaptive policy beat best-of-N. |

## What This Demonstrates

The demo validates the experiment machinery:

- task loading;
- policy execution;
- public verifier calls;
- hidden grading after search;
- append-only attempt logs;
- budget accounting;
- aggregate metrics;
- decision reports that can report losses.

The most important scientific detail is that the harness does not convert a
structural control into a positive claim. It preserves the negative result.

## Where To Inspect The Full Artifacts

After running `uv run --python 3.12 python scripts/run_experiment.py`, inspect:

- `outputs/runs/toy_protocol/attempts.jsonl`
- `outputs/runs/toy_protocol/search_results.jsonl`
- `outputs/runs/toy_protocol/summary.csv`
- `outputs/runs/toy_protocol/decision.json`
- `reports/runs/toy_protocol/report.md`
- `reports/runs/toy_protocol/success_vs_tokens.png`
- `reports/runs/toy_protocol/success_vs_verifier_calls.png`

Generated outputs are ignored by git by default, so a fresh clone will not
include those files until the protocol is run. This static page captures the
main interpretation for repository reviewers.
