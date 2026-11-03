# Stop-Then-Escalate Reproduction Runbook

This runbook reproduces the frozen MBPP+ confirmation protocol: compare
stop-only `16x1` sampling with `8x2` sampling-plus-repair under the same
16-call maximum budget.

Read the [preregistration](stop_then_escalate.md) and the
[completed confirmation report](../results/stop_then_escalate_confirmation.md)
before running it. A new run reproduces the protocol; exact latencies and
candidate bytes are not guaranteed across hardware or dependency changes.

## Safety And Label Boundary

Generation occurs on the host. Every generated program is evaluated in the
pinned, no-network EvalPlus container. Do not replace that evaluator with the
host-subprocess toy verifier.

The required order is:

1. complete both public-only trajectories;
2. validate matched provenance and shared root samples;
3. enable hidden evaluation for both trajectories;
4. analyze the paired task outcomes.

Never inspect or generate hidden grades between the two public runs. The tools
refuse nonempty output directories and mismatched trajectory provenance.

## Frozen Inputs

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`;
- model/tokenizer revision:
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- EvalPlus: `0.3.1`;
- MBPP+: `v0.2.0`;
- frozen tasks:
  `configs/experiments/stop_then_escalate_mbpp_confirmation_tasks.json`;
- task-list SHA-256:
  `a934e831be691c850db391b58e8f20038a61490836a5ce945a0c9de7c3f94a63`;
- temperature/top-p: `0.7` / `0.95`;
- output cap: 256 tokens;
- seed: `0`;
- maximum model calls per task: 16;
- container envelope: 2 CPUs, 8 GB memory, at most 10 MBPP+ tasks per shard;
- paired task bootstrap: 10,000 resamples, seed `0`.

## 1. Prepare A Clean Environment

From this Python project directory:

```bash
git status --short
uv sync --frozen --group dev --group hf --group evalplus
uv run ttc-operatorbench doctor --evalplus
make check
```

`git status --short` must be empty. Candidate generation records the exact Git
commit and refuses a dirty checkout unless `--allow-dirty` is supplied. Do not
use that override for a confirmation run.

Use repo-local model and dataset caches:

```bash
export HOME="$PWD/.runtime-cache/home"
export HF_HOME="$PWD/.runtime-cache/huggingface"
export RUN_REAL_MODEL_TESTS=1
```

The first run may download about 3 GB of model data and the pinned evaluator
image. Ensure Docker has at least 12 GB of VM memory available.

## 2. Generate Both Public-Only Trajectories

Generate the stop-only baseline:

```bash
uv run --frozen python scripts/run_evalplus_width_depth.py \
  --run-id repro_mbpp100_w16_d1_seed0 \
  --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --tokenizer-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset mbpp \
  --task-ids-file configs/experiments/stop_then_escalate_mbpp_confirmation_tasks.json \
  --width 16 \
  --depth 1 \
  --seed 0 \
  --cpus 2 \
  --memory 8g \
  --allow-large-run
```

Generate the frozen sampling-plus-repair challenger:

```bash
uv run --frozen python scripts/run_evalplus_width_depth.py \
  --run-id repro_mbpp100_w8_d2_seed0 \
  --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --tokenizer-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset mbpp \
  --task-ids-file configs/experiments/stop_then_escalate_mbpp_confirmation_tasks.json \
  --width 8 \
  --depth 2 \
  --seed 0 \
  --cpus 2 \
  --memory 8g \
  --allow-large-run
```

Each directory must contain `trajectory_manifest.json`,
`trajectory_steps.jsonl`, and `public_summary.json`. It must not yet contain a
`hidden_evaluation/` directory.

## 3. Apply The Pre-Hidden Gate

```bash
uv run ttc-operatorbench validate-trajectories \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w16_d1_seed0 \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w8_d2_seed0
```

This command fails unless task, dataset, model, sampling, hardware, dependency,
repository, and shared-root records match. Record its shared-root count and
canonical SHA-256 before continuing.

Also confirm that neither hidden directory exists:

```bash
test ! -e outputs/width_depth/repro_mbpp100_w16_d1_seed0/hidden_evaluation
test ! -e outputs/width_depth/repro_mbpp100_w8_d2_seed0/hidden_evaluation
```

## 4. Join Hidden Labels

Only after Step 3 succeeds:

```bash
RUN_HIDDEN_EVAL=1 uv run --frozen python scripts/evaluate_evalplus_trajectory.py \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w16_d1_seed0 \
  --cpus 2 \
  --memory 8g
```

```bash
RUN_HIDDEN_EVAL=1 uv run --frozen python scripts/evaluate_evalplus_trajectory.py \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w8_d2_seed0 \
  --cpus 2 \
  --memory 8g
```

Each evaluator manifest must report:

- `search_was_complete_before_hidden_evaluation: true`;
- `base_recheck_matches_search: true`;
- the pinned `ganler/evalplus` image digest;
- complete task-preserving shard coverage.

## 5. Run The Paired Confirmation Analysis

```bash
uv run --frozen python scripts/analyze_evalplus_trajectories.py \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w16_d1_seed0 \
  --trajectory-dir outputs/width_depth/repro_mbpp100_w8_d2_seed0 \
  --output-dir outputs/width_depth/repro_mbpp100_confirmation \
  --analysis-stage confirmation \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 0
```

Inspect:

```text
outputs/width_depth/repro_mbpp100_confirmation/comparison_summary.json
outputs/width_depth/repro_mbpp100_confirmation/task_observations.jsonl
outputs/width_depth/repro_mbpp100_confirmation/comparison_report.md
```

The preregistered strong-confirmation gate requires the lower endpoint of the
paired `8x2 - 16x1` accuracy interval to exceed zero. A positive point estimate
whose interval includes zero is only suggestive; a nonpositive point estimate
is failed confirmation.

## 6. Verify The Published Evidence

The committed compact bundle is independent of local ignored outputs:

```bash
uv run ttc-operatorbench verify-results
```

Its manifest verifies hashes, sizes, syntax, and task-record counts for the
published HumanEval+ and MBPP+ derived observations.
