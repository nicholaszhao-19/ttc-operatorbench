# EvalPlus Selection-Regret Preregistration

Status: protocol frozen before any EvalPlus model result is generated, with the
pre-result safety amendment below.

## Amendment 1: Sandboxed Behavior Traces

Recorded on 2026-07-10 before any EvalPlus candidate pool or model result was
generated.

The repository's existing `behavior_cluster` implementation obtains behavior
traces by running candidates in a host subprocess. That violates this
experiment's container-only safety contract. Therefore:

- Phase A includes `first_sample`, `first_base_pass`, and the diagnostic oracle;
- the Phase A primary paired comparison is `first_base_pass` minus
  `first_sample` at `k = 16`;
- H3 and `behavior_cluster` move to Phase B and may run only after behavior
  probes execute in the same no-network, resource-limited container boundary;
- an unrun H3 must be reported as deferred, never as a null or negative result.

No other hypothesis, task split, candidate-generation setting, or decision
threshold changes in this amendment.

## Amendment 2: Coverage Versus Realized Prefix

Recorded on 2026-07-10 after an engineering-only development smoke test, but
before a provenance-valid pilot or any locked evaluation run. The smoke pool is
ineligible because its manifest did not hash the dirty repository state.

The original notation used the unbiased Pass@k estimate directly inside
selection regret. That estimate describes a random size-k subset of the full
pool; it does not state whether a correct candidate actually appeared in the
specific first-k prefix shown to a selector. The corrected analysis therefore:

- reports unbiased Pass@k as the repeated-sampling coverage estimand;
- separately reports realized prefix-oracle success;
- defines selection regret as prefix-oracle success minus selected success.

This makes per-task selection regret exactly zero or one and prevents the
analysis from attributing unavailable candidates to a selector. No selector,
generation setting, task split, or decision threshold changed.

## Question

When repeated sampling produces a correct program, how often can a selector
recover it without access to held-out EvalPlus tests?

This experiment measures the gap between candidate coverage and selected-answer
correctness. It does not test the adaptive bottleneck controller.

## Derived Protocol

The experiment uses HumanEval+ as a derived test-time-compute protocol:

- the task prompt and entry point are model-visible;
- original HumanEval base-test outcomes are policy-visible verifier signals;
- EvalPlus base-plus-extra outcomes are evaluation-only labels;
- canonical solutions, test inputs, expected outputs, and hidden labels never
  enter model prompts or selector state.

Because base-test outcomes influence selection, these results are not official
EvalPlus leaderboard scores.

## Candidate Pool

For every task and pool seed, generate one immutable ordered pool of 16
candidates. All selectors replay exactly the same candidates in the same order.
No selector may trigger additional model generation in this study.

Initial generation settings:

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` engineering pilot;
- temperature: `0.7`;
- top-p: `0.95`;
- maximum output tokens: `256`;
- candidate indices: `0` through `15`;
- primary pool seed: `0`.

The model may be changed after the five-task engineering pilot only if the
predeclared pilot gate fails. The final model and revision must be frozen before
evaluation-split generation begins.

## Task Split

Assign each task using the first eight bytes of:

```text
sha256("ttc-operatorbench-evalplus-v1:" + task_id)
```

Interpret the bytes as an unsigned integer. Values modulo 5 equal to 0 form the
development split; all other values form the locked evaluation split.

Development tasks may be used to debug parsing and freeze selector
hyperparameters. Evaluation plus-test labels may be read only after the code,
configuration, and candidate-pool manifest are frozen.

## Policies

Phase A compares only:

1. `first_sample`: select candidate 0.
2. `first_base_pass`: select the first candidate passing base tests; fall back
   to candidate 0 if none passes.
3. `oracle`: diagnostic only; records whether any candidate is correct and is
   never presented as a deployable selector.

Phase B adds `behavior_cluster` only after its behavior traces are produced in
the approved container boundary and use no plus-test information.

Repair, active test generation, LLM judges, and adaptive routing are excluded.

## Estimands

For `k` in `{1, 2, 4, 8, 16}` define:

```text
P_k       = unbiased Pass@k under base-plus-extra correctness
C_k       = realized hidden oracle success in candidates 0..k-1
S_k(pi)   = selected hidden Pass@1 for selector pi using candidates 0..k-1
R_k(pi)   = C_k - S_k(pi)
```

The Phase A primary comparison is the paired evaluation-task difference in
`S_16` between `first_base_pass` and `first_sample`. The conditional Phase B
primary comparison is `behavior_cluster` minus `first_base_pass`.

Secondary outcomes are:

- `P_k - P_1`, repeated-sampling coverage gain;
- `C_k`, realized first-k prefix coverage;
- `R_k(pi)`, selection regret;
- base-pass minus base-plus-extra-pass rate;
- false acceptance among selected base-passing candidates;
- token, verifier-call, and wall-clock cost;
- normalized area under each selected-correctness budget curve.

Pass@k uses the unbiased estimator:

```text
1 - choose(n - c, k) / choose(n, k)
```

where `n` is the complete candidate-pool size and `c` is the number of
base-plus-extra-correct candidates. Truncated pools are ineligible.

## Hypotheses

- H1: `P_16 > P_1`; repeated sampling increases hidden candidate coverage.
- H2: `R_16(first_base_pass) >= 0.05`; public verification leaves practically
  meaningful selection regret.
- H3 (Phase B only): `S_16(behavior_cluster) > S_16(first_base_pass)` at matched
  candidate generation cost.

The five-percentage-point H2 threshold is a project decision rule, not a
universal significance threshold.

## Uncertainty

The task is the statistical unit. Report 95% percentile intervals from 10,000
paired bootstrap resamples of evaluation tasks. Candidate samples from one task
must not be treated as independent observations.

If additional pool seeds are affordable, average within task before the task
bootstrap and report between-pool variation separately.

## Pilot Gate

Run four candidates on five development tasks before the main pool. Continue
only if:

- sandboxed base and plus evaluation completes without host execution;
- generation and code extraction failure rates are each below 5%;
- truncation is below 5%;
- estimated full-pool runtime is acceptable before launch;
- performance is not effectively zero or saturated on both base and plus
  outcomes.

If the performance gate fails, change model capacity or task difficulty. Do not
change a selector to rescue a degenerate pilot.

## Decision Rule

- If H1 fails, prioritize generation, model calibration, or repair.
- If H1 holds and H2 fails, prioritize generation or stopping rather than a
  more complex selector.
- If H1 and H2 hold, a faithful DiffCodeGen baseline and active differential
  testing are justified next.
- If base success rises while plus success does not, verifier overfitting becomes
  the primary finding.
- The bottleneck controller is evaluated only after this study identifies a
  measurable bottleneck.

## Reproducibility Contract

The frozen run must record:

- repository commit;
- EvalPlus package version and dataset hash;
- model and tokenizer revisions;
- prompt text and hash;
- sampling settings and effective seed per candidate;
- hardware and dependency versions;
- raw and sanitized candidate hashes;
- sandbox image identity and command;
- complete base and plus grading artifacts;
- all exclusions, crashes, and timeouts.

Generated code must be evaluated in a no-network resource-limited container.
The host-subprocess toy verifier is not allowed for this experiment.
