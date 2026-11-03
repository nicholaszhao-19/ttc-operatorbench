# Stop-Then-Escalate Engineering Pilot

Status: engineering gate passed; no hidden-test performance claim.

## Frozen Run

- repository commit: `a4ae07e76339603d8a881d8a172b10231faffdab`;
- model and tokenizer: `Qwen/Qwen2.5-Coder-1.5B-Instruct` at
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- HumanEval+ dataset SHA-256:
  `7fbb45cf215ee4b6179aedc6c655f85e76a1179182f7614e073dae65983104e1`;
- policy: width 2, depth 2, maximum four calls per task;
- tasks: `HumanEval/121`, `HumanEval/125`, `HumanEval/131`,
  `HumanEval/137`, and `HumanEval/145`;
- evaluator: pinned `ganler/evalplus` image digest, no network, base-only mode;
- generation: temperature 0.7, top-p 0.95, maximum 256 output tokens, seed 0.

The run directory was
`outputs/width_depth/evalplus_dev_offset5_w2_d2_seed0_pilot_a4ae07e`.
Outputs are intentionally gitignored; all claim-level values are recorded below.

## Public-Only Result

| Metric | Result |
|---|---:|
| Tasks | 5 |
| Actual model calls | 14 / 20 maximum |
| Mean calls per task | 2.8 |
| Publicly resolved tasks | 3 / 5 |
| Total generation tokens | 3,913 |
| Empty sanitized candidates | 0 / 14 |
| Candidates at the output-token cap | 0 / 14 |

One task, `HumanEval/121`, failed both independent roots and passed after the
first repair. This establishes that the repair path is operational; it does not
yet establish a hidden-correctness gain.

The task-level stopping traces were:

| Task | Calls | Terminal public state |
|---|---:|---|
| `HumanEval/121` | 3 | repair pass |
| `HumanEval/125` | 2 | second-root pass |
| `HumanEval/131` | 1 | first-root pass |
| `HumanEval/137` | 4 | unresolved |
| `HumanEval/145` | 4 | unresolved |

## Determinism And Leakage Audit

An independent rerun used the same configuration and produced the same 14
normalized steps. Prompt text, raw completion, sanitized code, token counts,
sampling state, public feedback, public outcome, operator, parent, root, depth,
and stopping decision all matched. The shared normalized SHA-256 was:

```text
bfedc8e266f1a81b3771a94c144c64f23d0ffb56455da510739a262c8ab10ef1
```

Timestamps, latency measurements, run identifiers, and identifiers derived from
the run ID were excluded from that comparison.

No forbidden plus-test key occurred in the run plan, trajectory manifest,
trajectory steps, public summary, or serialized base grades. EvalPlus 0.3.1
still writes `plus_status: null` and an empty `plus_fail_tests` field in its raw
base-only result file. That raw file is evaluator-private; the parser strips
both fields before returning any policy-visible grade.

## Gate Decision

Every preregistered engineering condition passed:

- complete candidate, grade, lineage, and content-hash validation;
- generated code executed only in the pinned container;
- no hidden field reached policy-visible state;
- exact stop-on-first-pass and maximum-call accounting;
- zero empty extractions and zero observed token-cap truncations;
- exact normalized deterministic replay.

The next valid step is therefore the frozen four-policy comparison on the 26
development tasks that exclude these five pilot tasks. Hidden grading must occur
only after all public-only trajectories are complete.
