# Experiment Design

TTC OperatorBench evaluates verifier-guided code reasoning policies under
explicit test-time compute budgets.

The design goal is not only to record whether a task is solved, but to preserve
enough evidence to reconstruct how a policy spent its budget and why a result
was selected.

## Protocol Flow

1. Load an experiment config from `configs/experiments/`.
2. Validate task IDs, model entries, policies, budgets, and seeds.
3. Load tasks from the requested task suite.
4. Build a model provider for each model entry.
5. Build each search policy.
6. Run each policy over each task and budget.
7. Log every generated attempt.
8. Attach hidden verification after policy execution.
9. Write JSONL logs, aggregate metrics, plots, decision JSON, and a Markdown
   report.

## Task Suites

The repo currently has two local task suites:

- `toy_code`: small deterministic Python function tasks used for fast structural
  checks.
- `curated_code`: a larger local curated suite used for deterministic policy
  comparisons.

Tasks include public tests and, where available, hidden tests. Public tests can
inform search. Hidden tests are evaluation-only and must not influence candidate
selection.

## Model Providers

Supported provider kinds are:

- `dummy`: deterministic local provider used for structural controls and fast
  tests.
- `huggingface`: opt-in provider for local Hugging Face model execution.

Real-model configs are gated by `RUN_REAL_MODEL_TESTS=1` so default tests do not
download or execute model weights.

## Search Policies

Configured policies include:

- `greedy`
- `best_of_n_2`
- `best_of_n_4`
- `repair_only`
- `plan_then_code`
- `local_revision_basic`
- `operator_bandit`
- `operator_bandit_no_error_bonus`
- `operator_bandit_unit_cost`
- `fixed_operator_order`

The `operator_bandit` family represents the adaptive operator-allocation path.
The baseline policies provide fixed or simpler verifier-guided comparisons.

## Budgets

Budgets can constrain:

- attempts;
- tokens;
- verifier calls;
- seconds;
- cost.

The committed local protocols primarily compare policies over named call/token
budget points such as `one_call`, `two_call`, `four_call`, and larger curated
budget sweeps.

## Metrics

The report path tracks both public and hidden views when hidden grading is
available:

- public solve rate;
- selected-answer hidden solve rate;
- oracle hidden solve rate for any-attempt diagnostic coverage;
- public-hidden gap;
- overfit rate;
- token-budget success curves;
- verifier-call success curves;
- median tokens to solution;
- median verifier calls to solution;
- total attempts;
- total tokens;
- total verifier calls.

The decision report uses hidden metrics as the primary scope whenever hidden
grading is available. These primary hidden metrics grade the selected answer
returned by the policy, not any hidden-passing candidate generated along the
way.

## Research Integrity Rules

Candidate selection may use only information available during a real run:

- public task input;
- declared calibration data;
- model outputs;
- allowed verifier signals;
- current budget state.

Hidden tests, benchmark answers, final labels, and post-hoc evaluation artifacts
must remain outside search policy decisions. If oracle information is needed for
analysis, it should live only in evaluation code paths and be labeled as such.

## Current Decision Semantics

The decision report is budget-aware. It avoids claiming an adaptive win unless
the configured decision policy clearly improves over the strongest baseline
without losses or inconclusive budget points.

Possible outcomes include:

- `promising`: evidence favors the decision policy across compared budgets.
- `matches_baseline`: the decision policy ties the strongest baseline without a
  clear win.
- `needs_analysis`: at least one compared budget is a loss or inconclusive.

The current local/toy status should be read as pipeline validation, not as a
claim that adaptive operator allocation beats strong real-model baselines.
