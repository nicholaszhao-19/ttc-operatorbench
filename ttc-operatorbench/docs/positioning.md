# Related Work and Positioning

TTC OperatorBench is best read as an auditable harness for test-time compute
operator allocation, not as a new coding benchmark by itself.

## Layer Map

- Benchmark layer: small local Python tasks and curated coding tasks provide a
  controllable verifier-backed substrate.
- Test-time compute layer: policies allocate attempts, tokens, verifier calls,
  and cost across direct sampling, repair, planning, and local revision.
- Verifier layer: public tests guide the policy; hidden tests are attached only
  after policy execution for evaluation.
- Scheduler layer: `operator_bandit` records visible decision states and chosen
  operators so later work can study contextual allocation.

## Nearby Work

- Code benchmarks such as HumanEval and LiveCodeBench focus on measuring code
  generation performance. TTC OperatorBench focuses on the allocation policy
  around a verifier-backed coding task.
- Best-of-N and sampling studies measure gains from repeated generation. This
  harness keeps those baselines explicit and compares them against adaptive
  policies under matched budgets.
- Repair and verifier-guided systems such as AlphaCodium-style pipelines use
  feedback to improve solutions. This harness asks which operator to spend on
  next, and logs the state-action trace needed to audit that choice.
- Test-time compute scaling work studies how extra inference-time work changes
  performance. This repo makes the budget ledger explicit across attempts,
  tokens, verifier calls, and cost.

## Honest Current Claim

The implemented contribution is infrastructure: reproducible protocols,
selected-candidate hidden metrics, oracle diagnostics, cost-aware summaries,
decision-state logs, and portfolio reports. The current public claim should stay
at "pipeline validated; scheduler win not established" until sealed
multi-seed real-model results show selected-candidate hidden improvements over
strong baselines.
