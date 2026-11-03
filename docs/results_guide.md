# Results Guide

This guide explains the files produced by the experiment runner and how to read
the default local report.

## Default Artifact Locations

The default experiment command:

```bash
uv run ttc-operatorbench run
```

uses:

```text
configs/experiments/toy_protocol.yaml
```

and writes:

```text
outputs/runs/toy_protocol/
reports/runs/toy_protocol/
```

Pass `--run-id`, `--output-root`, or `--report-root` to choose different
locations:

```bash
uv run ttc-operatorbench run \
  --run-id my_run \
  --output-root outputs/runs \
  --report-root reports/runs
```

## Output Files

### `attempts.jsonl`

Append-only attempt log. Use this file when you need to inspect individual
candidate generations, verifier outcomes, cumulative budget use, selected flags,
and failure states.

### `search_results.jsonl`

Task-level results. Each row groups the attempts for one task/policy/budget run
and records final success, selected attempt, total tokens, verifier calls, and
metadata such as budget name.

### `summary.json`

Aggregate metrics by model, policy, and budget. This is the best machine-readable
file for comparing solve rates, selected-answer hidden solve rates, oracle
hidden diagnostics, token curves, verifier-call curves, overfit rate, and
aggregate budget use.

### `summary.csv`

Spreadsheet-friendly version of `summary.json`.

### `config_snapshot.yaml`

Copy of the config used for the run. Keep this with results to make later
comparisons auditable.

### `decision.json`

Budget-aware comparison of the configured decision policy against baseline
policies. This is the most compact machine-readable verdict file.

### `report.md`

Human-readable report with protocol metadata, summary rows, budget comparisons,
and failure examples.

### `success_vs_tokens.png`

Plot of solved fraction over token budget.

### `success_vs_verifier_calls.png`

Plot of solved fraction over verifier-call budget.

## Committed Real-Model Evidence

Normal run outputs remain ignored because candidate pools and evaluator records
are large. A reviewed subset of derived evidence is committed at:

```text
artifacts/results/stop_then_escalate_v1/
```

It contains aggregate JSON summaries and task-level JSONL observations for the
locked HumanEval+ study, HumanEval+ development comparison, and frozen MBPP+
confirmation. It contains no prompts, candidate code, answers, hidden tests, or
raw evaluator payloads.

Verify its hashes, byte sizes, syntax, and record counts with:

```bash
uv run ttc-operatorbench verify-results
```

The narrative reports under `docs/results/` remain the authoritative
interpretation and limitations.

## Reading The Default Toy Report

The default toy protocol is a deterministic structural control. It is useful for
checking that the full pipeline works:

- tasks load;
- policies execute;
- verifier calls run;
- public and hidden results are attached;
- attempt logs are written;
- summaries and plots are generated;
- budget-aware decision logic runs.

It should not be interpreted as a real-model research claim.

In the current local run, the report verdict is:

```text
needs_analysis
```

That means the adaptive decision policy does not dominate the strongest
configured baseline across all compared budgets.

## Comparing Policies

For quick inspection, open:

```text
reports/runs/<run_id>/report.md
```

For programmatic comparison, read:

```text
outputs/runs/<run_id>/summary.json
outputs/runs/<run_id>/decision.json
```

Compare policies within the same:

- task suite;
- model provider;
- seed set;
- budget names;
- metric scope.

When hidden grading is available, prefer hidden metrics for decision summaries.
Primary hidden metrics grade the selected answer returned by the policy. Use
`oracle_hidden_solve_rate` only as a diagnostic for whether any generated
candidate could have passed hidden tests. Use public-hidden gap and overfit rate
to identify cases where public tests may overstate progress.

For adaptive policies, check `policy_state_scope`. `per_task` means each task
gets a fresh scheduler and should be read as a short-horizon structural control.
`per_run` means scheduler statistics carry across tasks within a model, seed,
budget, and policy group.

## Regenerating Plots

The config-driven runner writes success-curve plots automatically.

The older toy JSONL plotting path is:

```bash
uv run --python 3.12 python scripts/run_toy_eval.py
uv run --python 3.12 python scripts/make_plots.py
```

By default, those commands use:

```text
outputs/toy_eval.jsonl
reports/greedy_vs_best_of_n.png
```

## Portfolio Reports

After multiple config-driven runs are complete, aggregate them with:

```bash
uv run --python 3.12 python scripts/make_portfolio_report.py \
  --runs toy_protocol curated_protocol
```

The default output is:

```text
reports/portfolio_report.md
```

Only include run IDs that already have generated artifacts under
`outputs/runs/` and `reports/runs/`.
