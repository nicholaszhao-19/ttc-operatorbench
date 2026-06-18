"""Run an explicitly gated real-model introductory evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.core.schema import Task  # noqa: E402
from ttc_operatorbench.evals.hf_toy_eval import (  # noqa: E402
    HFToyEvalConfig,
    default_output_dir_for_run,
    run_hf_toy_eval,
)
from ttc_operatorbench.models.hf_provider import (  # noqa: E402
    DEFAULT_HF_SMOKE_MODEL_ID,
    HuggingFaceModelProvider,
)

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("HF_SMOKE_MODEL_ID", DEFAULT_HF_SMOKE_MODEL_ID),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for attempts, search results, summary, and plot. "
            "Defaults to outputs/hf_toy_eval/<model>/<policies>."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", default="greedy,best_of_n_2,repair_only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def real_model_eval_enabled() -> bool:
    """Return whether the real-model gate is enabled."""
    return os.environ.get(REAL_MODEL_TESTS_ENV) == "1"


def main() -> None:
    """Run one bounded real-model validation if explicitly enabled."""
    args = parse_args()
    if not real_model_eval_enabled():
        print(f"skipping HF validation; set {REAL_MODEL_TESTS_ENV}=1 to run a real model")
        return

    policies = tuple(policy.strip() for policy in args.policies.split(",") if policy.strip())
    output_dir = args.output_dir or default_output_dir_for_run(args.model_id, policies)

    config = HFToyEvalConfig(
        model_id=args.model_id,
        output_dir=output_dir,
        max_tasks=args.max_tasks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        seed=args.seed,
        policies=policies,
        device=args.device,
        dtype=args.dtype,
    )
    provider = HuggingFaceModelProvider(
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        seed=config.seed,
    )

    def provider_factory(_policy_name: str, _task: Task) -> HuggingFaceModelProvider:
        return provider

    artifacts = run_hf_toy_eval(config, provider_factory)
    print(f"wrote attempts to {artifacts.attempts_path}")
    print(f"wrote summary to {artifacts.summary_path}")
    if artifacts.plot_path is not None:
        print(f"wrote plot to {artifacts.plot_path}")


if __name__ == "__main__":
    main()
