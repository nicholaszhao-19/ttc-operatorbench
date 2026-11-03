# Stop-Then-Escalate Development Comparison

Status: development winner frozen; untouched confirmation still required.

## Protocol

All four public-only trajectories were generated at repository commit
`e17310a` before any development hidden label was opened. They used the same:

- 26-task list in
  `configs/experiments/stop_then_escalate_dev_tasks.json`;
- HumanEval+ dataset SHA-256
  `7fbb45cf215ee4b6179aedc6c655f85e76a1179182f7614e073dae65983104e1`;
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` model and tokenizer revision
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- temperature 0.7, top-p 0.95, 256-token output cap, and seed 0;
- pinned no-network EvalPlus container and stop-on-first-base-pass rule.

The full hidden evaluator was run only after all four trajectories completed.
Every full evaluation reproduced every base outcome used during search.
Comparisons use 10,000 paired task bootstrap resamples with seed 0.

## Results

| Policy | Hidden pass | 95% interval | Public pass | Mean calls | Total calls | Generation tokens |
|---|---:|---:|---:|---:|---:|---:|
| `16x1` | 88.5% | [76.9%, 100.0%] | 96.2% | 2.77 | 72 | 17,936 |
| `8x2` | **92.3%** | [80.8%, 100.0%] | 100.0% | **2.73** | **71** | 18,380 |
| `4x4` | 80.8% | [65.4%, 96.2%] | 84.6% | 4.23 | 110 | 38,063 |
| `2x8` | 80.8% | [65.4%, 96.2%] | 92.3% | 3.19 | 83 | 26,927 |

Paired against `16x1`:

| Challenger | Hidden difference | 95% interval | Win/loss/tie | Mean-call difference | Mean-token difference |
|---|---:|---:|---:|---:|---:|
| `8x2` | **+3.8 points** | [0.0, 11.5] | 1 / 0 / 25 | -0.04 | +17.1 |
| `4x4` | -7.7 points | [-23.1, 7.7] | 1 / 3 / 22 | +1.46 | +774.1 |
| `2x8` | -7.7 points | [-23.1, 7.7] | 1 / 3 / 22 | +0.42 | +345.8 |

The `8x2` repair converted `HumanEval/5` at call 15 and that candidate passed
the hidden suite. It lost no task solved by `16x1`. Relative to `16x1`, it used
one fewer model call but 444 more total generation tokens because repair prompts
are longer.

The deeper policies also solved `HumanEval/5`, but insufficient width lost three
tasks that later fresh samples solved: `HumanEval/21`, `HumanEval/3`, and
`HumanEval/88`. This is direct evidence that depth cannot simply replace width.

## Decision

The preregistered rule freezes `8x2` as the development winner. Its point
estimate meets the engineering gate of at least three accuracy points with no
more than eight mean calls. The paired interval includes zero, however, and the
gain is one task. This result is promising confirmation evidence, not a
publishable superiority claim.

The controller gate is not met. One fixed shallow-repair schedule dominated the
fresh-sampling baseline taskwise, while deeper schedules were worse and more
expensive. The next valid experiment is therefore a matched `16x1` versus
frozen `8x2` comparison on untouched MBPP+. An adaptive controller should be
reconsidered only if confirmation reveals stable public states in which the two
fixed schedules win on different tasks.

## Local Artifacts

The gitignored run directories are:

```text
outputs/width_depth/evalplus_dev26_w16_d1_seed0_e17310a
outputs/width_depth/evalplus_dev26_w8_d2_seed0_e17310a
outputs/width_depth/evalplus_dev26_w4_d4_seed0_e17310a
outputs/width_depth/evalplus_dev26_w2_d8_seed0_e17310a
outputs/width_depth/evalplus_dev26_policy_comparison_e17310a
```

Each contains content hashes for trajectory inputs, container outputs, base
rechecks, and hidden grades. The analysis directory contains task-level rows,
the complete machine-readable summary, and the rendered comparison report.
