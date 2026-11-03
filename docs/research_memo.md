# Research Memo

This memo states the public research position of TTC OperatorBench after its
locked HumanEval+ study and frozen MBPP+ confirmation.

## Thesis

Code-generation systems can spend test-time compute on independent sampling,
verification, repair, planning, or revision. The useful research question is
not only whether more samples increase oracle coverage, but whether a policy can
turn that coverage into a better selected answer at lower cost.

The current project claim is deliberately narrow:

> TTC OperatorBench is a reproducible harness for measuring coverage,
> selection, and cost in verifier-guided code-generation search.

It does not claim a state-of-the-art adaptive scheduler.

## Evaluation Principle

For a task, model, verifier, and budget, distinguish:

- **coverage failure:** no generated candidate is hidden-correct;
- **selection failure:** a hidden-correct candidate exists but is not selected;
- **false acceptance:** a public-passing selected candidate fails hidden tests;
- **stopping inefficiency:** the selected answer was available before the fixed
  generation budget was exhausted.

Every policy decision must use public evidence only. Hidden labels are joined
after the search trajectory is complete and are used only for analysis.

## Implemented System

The harness includes:

- strict typed schemas for tasks, candidates, grades, trajectories, budgets,
  and attempt logs;
- greedy, best-of-N, fixed-sample, repair, revision, lightweight differential,
  and adaptive toy policies;
- exact token, call, verifier, and latency accounting;
- immutable candidate pools and public-only width-depth trajectories;
- pinned model, tokenizer, dataset, Git, and dependency provenance;
- a no-network, resource-limited EvalPlus Docker evaluator;
- task-level paired bootstrap intervals and preregistered decision gates;
- a compact, hash-verified public bundle of derived task observations;
- fast Ruff, strict mypy, and pytest checks for the default local path.

## Locked HumanEval+ Result

The locked study generated 16 candidates for each of 133 HumanEval+ tasks using
one pinned Qwen2.5-Coder-1.5B revision and seed `0`.

- unbiased Pass@k rose from 47.6% at `k=1` to 82.7% at `k=16`;
- first-public-pass selected accuracy reached 80.5% at `k=16`;
- realized selection regret was 2.3 points;
- the exact decomposition was 23 coverage failures and 3 selection failures;
- replaying sequential stop-on-first-public-pass preserved the selected answer
  while saving 72.5% of candidate calls and 69.9% of generation tokens.

The result identified coverage and stopping as the immediate bottlenecks. It did
not justify prioritizing behavior clustering or a learned judge.

## Frozen MBPP+ Confirmation

Development compared fixed width-depth policies under a maximum 16 calls per
task. `8x2` sampling-plus-repair won the 26-task HumanEval+ development
tie-break and was frozen before confirmation.

On an untouched, label-free selected 100-task MBPP+ subset:

- `16x1` stop-only sampling reached 73.0% hidden accuracy;
- `8x2` sampling-plus-repair reached 70.0%;
- paired `8x2 - 16x1` accuracy was -3.0 points with a 95% interval of
  [-8.0, +1.0];
- the challenger saved 13 calls but used 8,637 more tokens;
- only 2 of 6 selected public-passing repairs were hidden-correct.

The preregistered result is `failed_confirmation`. Repair can help individual
tasks, but this repair policy should not be promoted into an adaptive
sample-versus-repair controller.

## Current Baseline

The empirical fixed baseline is:

> Sample independently up to 16 candidates and stop on the first public-test
> pass.

This is not a universal SOTA claim. It is the strongest matched baseline
established inside this repository for the current model and protocols.

## Main Limitations

- one model revision and one generation seed;
- function-level HumanEval+ and MBPP+, not repository-level software work;
- public tests are available during selection;
- no time-filtered benchmark slice;
- no faithful matched implementation of complete S*, DiffCodeGen, learned
  verifier, or agentic-verifier systems;
- no demonstrated real-model gain from the adaptive policies;
- model and Docker runs are too expensive for default CI.

## Next Decision

The next experiment should remain fixed and narrow:

1. replicate the stopping baseline with additional seeds or a second model size;
2. implement one plan-before-regenerate escalation for unresolved tasks;
3. compare it with `16x1` under the same model, tasks, and maximum budget;
4. measure hidden accuracy, false acceptance, calls, and tokens with paired
   task-level uncertainty;
5. use an untouched time-filtered benchmark before any new confirmatory claim.

Only after a fixed escalation produces repeatable signal should the project
learn a bottleneck-aware controller.

## Public Positioning

The strongest accurate portfolio framing is:

> A research-engineering project for code-agent evaluation that combines typed
> experiment infrastructure, sandboxed execution, provenance, matched-budget
> comparisons, and honest positive and negative results.

That is a stronger claim than a speculative scheduler win because it is already
supported by inspectable code and data.
