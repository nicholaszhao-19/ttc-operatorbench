# Validity Fixes

This note records the validity repairs made before treating TTC OperatorBench
outputs as research evidence.

## Sampling

Budget ledgers now pass only budget constraints and a deterministic seed offset.
They do not implicitly reset decoding to `temperature=0.0` or `do_sample=false`.
The Hugging Face provider resolves unspecified sampling fields from the model
config, then applies the seed offset so repeated Best-of-N attempts are
reproducible but distinct.

The gated Hugging Face protocols that include Best-of-N use stochastic decoding.
The smoke protocol remains deterministic so local checks stay cheap and stable.

## Hidden Metrics

Primary hidden metrics now grade the selected answer returned by a policy. A
candidate generated and then rejected by the policy cannot improve
`hidden_solve_rate`, hidden success curves, public-hidden gap, or decision
reports.

The previous any-attempt behavior is preserved as an oracle diagnostic through
`oracle_hidden_solve_rate`. Use it to inspect search coverage, not to claim
policy performance.

## Bandit State

Experiment configs now include `policy_state_scope`:

- `per_task`: construct a fresh policy per task. This preserves the original
  short-horizon structural-control behavior.
- `per_run`: reuse the policy across tasks within each model, seed, budget, and
  policy group. Use this for claims that the bandit learned operator values.

Bandit results record the state scope, operator sequence, and operator
statistics before and after each task run.

## Research Position

Existing generated demo artifacts should be read as structural controls, not as
evidence that adaptive allocation beats strong baselines. The next defensible
result should measure the public-to-hidden gap as search intensity increases,
with selected-answer hidden metrics and stochastic real-model Best-of-N
baselines.

Relevant context:

- Strategic Scaling of Test-Time Compute: A Bandit Learning Approach,
  https://arxiv.org/abs/2506.12721
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling
  Model Parameters, https://arxiv.org/abs/2408.03314
- Adding Error Bars to Evals, https://arxiv.org/abs/2411.00640
