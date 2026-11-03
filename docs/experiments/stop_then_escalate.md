# Stop-Then-Escalate Preregistration

Status: protocol frozen before any stop-then-escalate model result. The completed
confirmation outcome is recorded separately in
[the confirmation report](../results/stop_then_escalate_confirmation.md).

## Objective

Determine whether compute saved by public-verifier stopping can convert
candidate-coverage failures when it is reallocated from fresh sampling to
sequential repair.

The immediate goal is a fixed hybrid baseline, not a learned controller:

> Improve selected hidden correctness over stop-only sampling while retaining a
> substantial compute advantage over fixed N=16 sampling.

## Motivation

The locked HumanEval+ study at commit
`b19fdb3c61383f9f95d028ff380ccf914ea1c66a` found:

- 23 coverage failures and 3 selection failures across 133 tasks;
- 80.5% hidden selected accuracy from first-base-pass selection;
- 72.5% fewer candidate calls and 69.9% fewer generation tokens when replayed
  with stop-on-first-base-pass behavior.

Those results make generation depth the next empirical question. They do not
justify learned routing, behavior clustering, or an LLM judge yet.

## Fixed Policies

Let width `w` be the number of independent root candidates and depth `d` include
the root plus its repair descendants. Every policy has `w * d = 16` maximum
model calls:

| Policy | Roots | Maximum depth | Interpretation |
|---|---:|---:|---|
| `width_16_depth_1` | 16 | 1 | Fresh-sampling control |
| `width_8_depth_2` | 8 | 2 | One repair opportunity per root |
| `width_4_depth_4` | 4 | 4 | Balanced parallel/sequential search |
| `width_2_depth_8` | 2 | 8 | Deep local repair |

Roots are generated in deterministic root order. If no root passes public base
tests, repair rounds visit roots in deterministic round-major then root-major
order. A task stops immediately on its first base-test pass. A repair is always
conditioned on its direct parent candidate and that candidate's public feedback.

## Feedback Boundary

Policy-visible feedback may contain only:

- base-test pass/fail status;
- normalized error type;
- failing base-test inputs returned by EvalPlus;
- bounded public stdout/stderr if explicitly enabled.

It must never contain plus-test inputs, plus-test status, expected hidden
outputs, canonical solutions, or hidden labels. Public feedback and candidate
lineage are serialized in typed schemas and covered by leakage tests.

All generated code executes only in the pinned no-network container. The
host-subprocess toy verifier and behavior-clustering runner are prohibited.

## Model And Sampling

Initial engineering model and tokenizer:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
2e1fd397ee46e1388853d2af2c993145b0f1098a
```

Frozen defaults:

- temperature: 0.7;
- top-p: 0.95;
- maximum output tokens: 256;
- maximum model calls per task: 16;
- base pool seed: 0;
- root and repair seed offsets: global per-task call index.

## Development And Confirmation

The previous 133-task HumanEval+ evaluation split is consumed. It may be used
only for clearly labeled exploratory diagnostics.

Development proceeds on the 26 HumanEval+ development tasks not used in the
five-task engineering pilot. Policy choice is frozen before confirmation.

The development winner is chosen deterministically by highest selected hidden
accuracy, then lower total generation tokens, then fewer model calls, then
greater width. This rule is frozen before development hidden labels are opened.

The preferred untouched confirmation target is MBPP+ through the same pinned
EvalPlus package and container boundary. LiveCodeBench is a later alternative
if a faithful adapter is available before confirmation.

Confirmation uses a frozen 100-task MBPP+ v0.2.0 subset. The subset is selected
without prompts or labels by sorting all task IDs on
`SHA256("ttc-operatorbench-mbpp-confirmation-v1:" + task_id)`, taking the first
100, and then storing them in canonical task-ID order in
`configs/experiments/stop_then_escalate_mbpp_confirmation_tasks.json`.

Frozen dataset provenance:

- official raw JSONL SHA-256:
  `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`;
- canonical loaded-record SHA-256:
  `a72b025c91ae0db75667474895188d7300e0bd0995ed525c7e6657cff5f15e3c`;
- confirmation task-list SHA-256:
  `a934e831be691c850db391b58e8f20038a61490836a5ce945a0c9de7c3f94a63`.

Only `16x1` and the development-frozen `8x2` policy are run on confirmation.
Both public-only trajectories must finish before either receives hidden grades.
Strong confirmation requires the paired 95% lower confidence bound for
`8x2 - 16x1` hidden accuracy to exceed zero. A positive point estimate whose
interval includes zero is suggestive only. A zero or negative point estimate
fails to replicate the development gain.

Infrastructure amendment before any MBPP+ hidden grade was produced: EvalPlus
0.3.1 exceeded 4 GB and 5 GB container limits on larger task groups. Confirmation
therefore preserves each logical policy round but grades it in deterministic,
task-preserving shards of at most 10 tasks, using one evaluator worker and an
8 GB limit per container inside a 12 GB Colima VM. The previously failing heavy
10-task shard completed under this envelope. Failed engineering attempts
produced no complete trajectory or hidden grade; their public artifacts were
not inspected for policy outcomes and are excluded from analysis.

## Metrics

Primary:

```text
selected hidden Pass@1
```

Secondary:

- candidate calls per task;
- generation tokens and recorded generation latency;
- base-pass and hidden-pass rates;
- realized hidden coverage;
- false acceptance;
- repair conversion rate;
- accuracy-versus-cost area under the curve.

The task is the statistical unit. Report paired 95% percentile intervals from
10,000 task bootstrap resamples. Do not treat candidates as independent rows.

## Decision Gates

Engineering success requires both:

1. at least a three-point selected-hidden-accuracy gain over stop-only sampling;
2. no more than eight mean model calls per task, or no more than 50% of fixed
   N=16 generation tokens.

A publishable Pareto improvement requires a positive paired accuracy difference
with its 95% interval reported and lower total generation cost than fixed N=16.

Controller gate:

- If one fixed width-depth schedule dominates across tasks, use it and do not
  build a learned controller.
- If different schedules win in measurably different public states, a
  bottleneck-aware controller is justified.
- If repair does not beat fresh sampling, test planning-after-failure as the
  next single escalation operator.
- Revisit differential selection only in a no-public-test protocol or after
  selection regret becomes material.

## Engineering Pilot Gate

Before development evaluation, run at most five development tasks with a
maximum four calls per task. Continue only if:

- every candidate and grade has complete lineage and content hashes;
- base-only container rounds complete with no host execution;
- no plus field appears in policy-visible serialized state;
- stopping and maximum-call accounting are exact;
- empty extraction and truncation are each below 5%;
- one deterministic rerun reproduces candidate text and routing.

## Claim Boundary

This milestone can establish whether sequential repair is a useful operator in
this harness. It cannot establish a state-of-the-art adaptive controller, a
faithful full S* reproduction, or superiority over DiffCodeGen without matched
same-model, same-budget comparisons on untouched data.
