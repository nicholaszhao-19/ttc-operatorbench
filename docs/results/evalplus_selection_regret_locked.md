# Locked EvalPlus Coverage and Selection Result

Status: completed real-model locked study.

This is a derived HumanEval+ test-time-compute protocol, not an official
EvalPlus leaderboard score. Original HumanEval base-test outcomes are visible
to `first_base_pass`; EvalPlus extra tests remain evaluation-only labels.

## Decision

The immediate bottleneck is candidate coverage, not candidate discrimination.

- Repeated-sampling Pass@k rises from 47.6% at k=1 to 82.7% at k=16, a
  35.2-point paired gain (95% bootstrap interval: 29.3 to 41.2 points).
- At k=16, realized prefix coverage is 82.7% and `first_base_pass` selected
  accuracy is 80.5% (95% interval: 73.7% to 86.5%).
- Selection regret is 2.3 points (95% interval: 0.0 to 5.3 points), below the
  preregistered five-point threshold.
- The exact decomposition is 107 selected-correct tasks, 3 selection failures,
  and 23 coverage failures across 133 tasks.

H1 is supported. H2 does not meet its preregistered point-estimate threshold.
H3 remains deferred because behavior traces do not yet have the approved
containerized implementation.

The evidence therefore does not support making DiffCodeGen-style clustering the
next primary milestone for this protocol. The next comparison should be
stop-on-pass sampling versus stop-on-pass plus targeted repair under the same
maximum model-call budget.

## Frozen Setup

- implementation commit: `b19fdb3c61383f9f95d028ff380ccf914ea1c66a`;
- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`;
- model/tokenizer revision:
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- EvalPlus package: `0.3.1`;
- HumanEval+ version: `v0.1.10`;
- dataset SHA-256:
  `7fbb45cf215ee4b6179aedc6c655f85e76a1179182f7614e073dae65983104e1`;
- locked evaluation tasks: 133;
- candidates per task: 16;
- temperature/top-p: `0.7` / `0.95`;
- maximum output tokens: 256;
- pool seed: 0;
- bootstrap: 10,000 paired task-level resamples, seed 0.

The complete pool contains 2,128 candidates. There are no empty raw or
sanitized completions; 37 candidates (1.7%) reached the output-token cap; and
1,476 sanitized programs are unique by SHA-256.

## Main Results

| k | Pass@k | Prefix oracle | First base pass | Selection regret |
|---:|---:|---:|---:|---:|
| 1 | 47.6% | 41.4% | 41.4% | 0.0% |
| 2 | 58.8% | 57.9% | 57.9% | 0.0% |
| 4 | 69.2% | 69.9% | 69.2% | 0.8% |
| 8 | 77.1% | 76.7% | 75.2% | 1.5% |
| 16 | 82.7% | 82.7% | 80.5% | 2.3% |

Pass@k is the unbiased estimator over each complete 16-candidate pool. Prefix
oracle is the realized fraction of tasks with a hidden-correct candidate among
indices 0 through k-1. Selection regret uses realized prefix oracle minus
selected hidden correctness.

At k=16, 116 tasks have at least one base-passing candidate. Nine selected
base-passing candidates fail the extra tests, a conditional false-acceptance
rate of 7.8%. Only three are selection failures where another hidden-correct
candidate was available; the other false accepts occur without a correct
candidate in the pool and therefore belong to the coverage/generalization
bottleneck.

## Stopping Result

Generating candidates sequentially and stopping at the first base-test pass
selects exactly the same answer as `first_base_pass` over the full frozen pool.
It therefore preserves the 80.5% hidden accuracy while reducing compute:

| Cost | Fixed k=16 | Stop on first base pass | Saving |
|---|---:|---:|---:|
| Candidate calls | 2,128 | 585 | 72.5% |
| Generation tokens | 584,496 | 175,857 | 69.9% |
| Recorded generation latency | 4,305.5 s | 1,461.1 s | 66.1% |

The stopping policy uses 4.40 candidate calls per task on average, with median
2. Nineteen tasks consume the full 16-call budget; 17 of those never produce a
base-passing candidate.

This is the strongest current result: not a higher accuracy claim, but the same
selected accuracy at much lower generation cost. A fair next experiment should
spend part of the saved compute on repair only for unresolved tasks.

## Next Objective

Freeze a simple, non-learned `stop_then_repair` baseline on the development
split:

1. Sample sequentially and stop immediately on the first base pass.
2. At a fixed routing checkpoint, classify tasks with no base pass as unresolved.
3. Spend the remaining per-task call budget repairing the most promising failed
   candidate, using public base-test feedback only.
4. Stop on the first repaired base pass or at the same maximum 16 model calls.
5. Compare hidden accuracy, coverage, calls, tokens, and wall time against
   stop-only sampling and fixed k sampling.

The first implementation should be rule-based. A learned bottleneck controller
is justified only if this fixed routing baseline produces measurable gains.

Because locked labels have now been observed, any follow-up result on this same
split must be labeled exploratory. A confirmatory claim requires a fresh
benchmark slice, dataset, or otherwise untouched evaluation set.

## Reproducibility

Evaluator image:

```text
ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740
```

The evaluator ran with no network, a read-only root, dropped capabilities,
`no-new-privileges`, CPU/memory/PID limits, read-only sample and dataset mounts,
and a separate temporary writable output mount.

Artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Manifest | `52f3779f2c136ac2afa798ec891db783cec9598f72147af83a29b706ec6fe5f9` |
| Candidates | `698054c6885131e62009ebfde9313ca886f1766f1f2f84fb740a81c6023bd24b` |
| Base grades | `40fd815f8e2a5fc778abf8f5236abe47c1404141dbd20758daba1b2ff86d3c84` |
| Hidden-plus grades | `d162491072bc6ade6fed95609f47b2b4f0d08b31e80c0535ffd32cac71f37f86` |

The final local machine-readable result is
`selection_v3_summary.json`. Earlier `selection` and `selection_v2` files are
preserved locally. After grading, the analysis code added only two
preregistered reporting outputs: the paired Pass@k gain interval and exact
stop-on-pass cost accounting. Candidate generation, grades, selectors,
hypotheses, and thresholds did not change.

## Limitations

- One model and one pool seed were evaluated.
- Base tests are available to the selector, so the result does not characterize
  settings without public verification.
- The 2.3-point selection-regret estimate is uncertain; its interval reaches
  5.3 points even though the preregistered point threshold is not met.
- The study does not reproduce S*, DiffCodeGen, or a trained verifier.
- Generated pool and grade files remain local and ignored by Git because they
  contain large candidate data and evaluator-only records.
