# Stop-Then-Escalate MBPP+ Confirmation

Status: confirmation complete; the HumanEval+ development gain did not
replicate.

## Protocol Integrity

Both public-only trajectories completed at repository commit `86feb3d` before
either received hidden grades. They used the same:

- frozen 100-task MBPP+ v0.2.0 task list;
- canonical dataset SHA-256
  `a72b025c91ae0db75667474895188d7300e0bd0995ed525c7e6657cff5f15e3c`;
- task-list SHA-256
  `a934e831be691c850db391b58e8f20038a61490836a5ce945a0c9de7c3f94a63`;
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` model and tokenizer revision
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- temperature 0.7, top-p 0.95, 256-token output cap, seed 0, and 16-call cap;
- pinned no-network EvalPlus 0.3.1 evaluator image
  `ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`.

All 265 root records shared by `16x1` and `8x2` matched exactly after excluding
run identifiers and latency. Their canonical shared-root SHA-256 was
`222afdf53583365223d30c5721da5690b21c07c30a1f64479a7eb903e318353e`.
Every post-search hidden evaluation reproduced every public base grade used for
routing. Comparisons use 10,000 paired task bootstrap resamples with seed 0.

The exact analyzer inputs were:

| Policy | Trajectory manifest SHA-256 | Trajectory steps SHA-256 | Hidden grades SHA-256 |
|---|---|---|---|
| `16x1` | `e542e67b2a6252cf82a9ca0adbedbacd7145bb4bd6df6b9b5df77c860ce41812` | `4b7654100452c39d059308c604db186e77a898a92e791ff8ec9dde3b92638944` | `acd0ca9d33c3d8af89ba19ba7734499c4d4639369361623a64d359c63b49d319` |
| `8x2` | `1ff6d1d3f9247591b95e6806b4769e3bc756ed03385d4473f92fb5d53554a1af` | `4e0438bfb48e6bd256353d062bfedb5568f22eeea6b54b8ed19025135df13d14` | `fd71b01b5de3755998b1383fca95974aa94fcef2c50a6b650cfe21fc801f84be` |

## Results

| Policy | Hidden pass | 95% interval | Public pass | False accept | Mean calls | Total calls | Generation tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `16x1` | **73.0%** | [64.0%, 81.0%] | 89.0% | 18.0% | 3.71 | 371 | **59,365** |
| `8x2` | 70.0% | [61.0%, 79.0%] | 90.0% | 22.2% | **3.58** | **358** | 68,002 |

Paired `8x2 - 16x1` results:

| Hidden difference | 95% interval | Win/loss/tie | Mean-call difference | 95% interval | Mean-token difference | 95% interval |
|---:|---:|---:|---:|---:|---:|---:|
| **-3.0 points** | [-8.0, +1.0] | 1 / 4 / 95 | -0.13 | [-0.36, +0.04] | +86.4 | [+15.7, +162.0] |

The challenger saved 13 model calls (3.5%) but used 8,637 more generation
tokens (14.5%) and 31.8 more recorded generation seconds. It improved public
resolution by one task while reducing hidden correctness by three tasks.

## Discordant Tasks

The one challenger win was `Mbpp/769`: its first repair was hidden-correct,
while all 16 fresh samples remained unresolved.

The four challenger losses were:

- `Mbpp/268` and `Mbpp/580`: later fresh roots solved the task, while all eight
  repairs failed public tests;
- `Mbpp/294` and `Mbpp/410`: a repair passed the public tests but failed hidden
  tests, causing an early false acceptance before fresh sampling found a
  hidden-correct candidate.

Six tasks selected a repair. Only `Mbpp/264` and `Mbpp/769` were hidden-correct;
`Mbpp/255`, `Mbpp/294`, `Mbpp/300`, and `Mbpp/410` were false accepts. The
observed hidden precision of a public-passing repair was therefore 2/6, or
33.3%, on this frozen subset.

## Decision

The preregistered outcome is `failed_confirmation`: the challenger point
estimate is negative, so the HumanEval+ development gain did not replicate.
This is not evidence that repair never helps, but it is evidence against
promoting this repair policy or learning a sample-versus-repair controller from
the current data.

`16x1` stop-on-first-public-pass sampling remains the current empirical fixed
baseline. The next operator study should remain fixed and narrow: test a
plan-before-regenerate escalation against `16x1`, with explicit attention to
repair-style false acceptance. The remaining MBPP+ tasks may be used for
development, but a later claim requires a newly untouched benchmark such as a
time-filtered LiveCodeBench slice.

## Local Artifacts

The gitignored run directories are:

```text
outputs/width_depth/evalplus_mbpp100_w16_d1_seed0_86feb3d
outputs/width_depth/evalplus_mbpp100_w8_d2_seed0_86feb3d
outputs/width_depth/evalplus_mbpp100_confirmation_86feb3d
```

They contain trajectory manifests, public-only routing traces, task-preserving
container shards, base rechecks, hidden grades, task-level observations, input
hashes, and the complete machine-readable paired summary.
