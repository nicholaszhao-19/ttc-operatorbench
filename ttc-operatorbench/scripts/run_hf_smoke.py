"""Explicit real-model validation test for the Hugging Face provider."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.core.schema import Budget  # noqa: E402
from ttc_operatorbench.logging.writer import write_search_results_jsonl  # noqa: E402
from ttc_operatorbench.models.hf_provider import (  # noqa: E402
    DEFAULT_HF_SMOKE_MODEL_ID,
    HuggingFaceModelProvider,
)
from ttc_operatorbench.search.baselines import GreedyPolicy  # noqa: E402
from ttc_operatorbench.tasks.toy_code import get_toy_task  # noqa: E402
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier  # noqa: E402

REAL_MODEL_TESTS_ENV = "RUN_REAL_MODEL_TESTS"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("HF_SMOKE_MODEL_ID", DEFAULT_HF_SMOKE_MODEL_ID),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_smoke.jsonl"))
    return parser.parse_args()


def real_model_validation_enabled() -> bool:
    """Return whether the opt-in real-model validation gate is enabled."""
    return os.environ.get(REAL_MODEL_TESTS_ENV) == "1"


def main() -> None:
    """Run one greedy introductory task against a real HF model."""
    args = parse_args()
    if not real_model_validation_enabled():
        print(f"skipping HF validation; set {REAL_MODEL_TESTS_ENV}=1 to run a real model")
        return

    task = get_toy_task("is_even")
    provider = HuggingFaceModelProvider(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    result = GreedyPolicy().run(
        task,
        provider,
        PythonUnitTestVerifier(timeout_seconds=2.0),
        Budget(max_attempts=1, max_verifier_calls=1, max_tokens=2_000),
        run_id="hf-smoke",
    )
    write_search_results_jsonl(args.output, (result,))
    print(f"wrote HF validation result to {args.output}")


if __name__ == "__main__":
    main()
