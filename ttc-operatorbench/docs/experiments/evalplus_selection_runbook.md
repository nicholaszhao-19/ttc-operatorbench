# EvalPlus Selection-Regret Runbook

This runbook executes the preregistered HumanEval+ Phase A experiment. Read the
[preregistration](evalplus_selection_regret.md) before changing a parameter.

## Safety Boundary

Model generation happens on the host, but generated code never does. Evaluation
must use the pinned, no-network EvalPlus container. Do not substitute the
repository's host-subprocess toy verifier.

The scripts fail closed when Docker is unavailable and refuse to overwrite an
existing candidate pool, grade file, or analysis artifact.

Candidate generation also requires a clean Git worktree. `--allow-dirty` is
available only for engineering pilots; it records a digest of the tracked diff
and every untracked non-ignored file. Never use that override for the locked
evaluation pool.

## Frozen Pilot Inputs

- model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`;
- model and tokenizer revision:
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`;
- EvalPlus package: `0.3.1`;
- HumanEval+ data bundled by that release: `v0.1.10`;
- full-data SHA-256: `7fbb45cf215ee4b6179aedc6c655f85e76a1179182f7614e073dae65983104e1`;
- deterministic split sizes: 31 development and 133 evaluation tasks;
- development tasks: first five tasks in canonical split order;
- candidates per task: 4;
- seed: 0;
- temperature/top-p: `0.7` / `0.95`;
- maximum output tokens: 256.

The model revision was resolved from the official Hugging Face repository before
the pilot. It must remain exact in the pool manifest.

## 1. Verify Prerequisites

```bash
make check
make sync-pilot
docker version
```

`docker version` must show a running server, not only a client. Pulling the
pinned evaluator image and model weights requires network access on the first
run.

## 2. Generate the Development Pool

```bash
HOME="$PWD/.runtime-cache/home" \
HF_HOME="$PWD/.runtime-cache/huggingface" \
RUN_REAL_MODEL_TESTS=1 \
.venv/bin/python scripts/generate_evalplus_pool.py \
  --pool-id evalplus_dev_qwen25_coder_15b_n4_seed0 \
  --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --tokenizer-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --split development \
  --max-tasks 5 \
  --pool-size 4 \
  --seed 0
```

The immutable pool is written under:

```text
outputs/candidate_pools/evalplus_dev_qwen25_coder_15b_n4_seed0/
```

Generation is atomic at the schema level: an incomplete task-by-candidate pool
is rejected rather than analyzed as if it were complete.

## 3. Evaluate in the Container

```bash
.venv/bin/python scripts/evaluate_evalplus_pool.py \
  --pool-dir outputs/candidate_pools/evalplus_dev_qwen25_coder_15b_n4_seed0
```

This writes separate `base_grades.jsonl` and `hidden_plus_grades.jsonl` files.
Policy code may consume only the base grades. The plus grades are joined after a
selector decision for evaluation.

## 4. Analyze Once

```bash
.venv/bin/python scripts/analyze_evalplus_pool.py \
  --pool-dir outputs/candidate_pools/evalplus_dev_qwen25_coder_15b_n4_seed0 \
  --k-values 1 2 4 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 0
```

Inspect `selection_report.md` and `selection_summary.json`. On five tasks the
intervals are diagnostic, not a research conclusion.

## 5. Apply the Pilot Gate

Proceed only when all of the following are documented:

- the evaluator manifest records the pinned container digest and successful
  no-network command;
- all 20 task/candidate slots exist and have base and plus grades;
- empty extraction and truncation are each below 5%;
- plus correctness is neither zero nor saturated across the candidate pool;
- measured generation and evaluation time make the locked run affordable.

If the gate fails, record the failure before changing model capacity or task
difficulty. Do not tune a selector against the five hidden development labels.

## 6. Generate the Locked Pool

Only after the pilot gate passes:

```bash
HOME="$PWD/.runtime-cache/home" \
HF_HOME="$PWD/.runtime-cache/huggingface" \
RUN_REAL_MODEL_TESTS=1 \
.venv/bin/python scripts/generate_evalplus_pool.py \
  --pool-id evalplus_eval_qwen25_coder_15b_n16_seed0 \
  --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --tokenizer-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --split evaluation \
  --all-tasks \
  --pool-size 16 \
  --seed 0 \
  --allow-evaluation-split
```

Then run the same evaluator and analyzer commands with the locked pool path and
`--k-values 1 2 4 8 16`. Do not regenerate candidates after viewing locked plus
labels.
