"""Run deterministic introductory evaluations for the baseline policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ttc_operatorbench.core.schema import Budget, SearchResult, Task  # noqa: E402
from ttc_operatorbench.logging.writer import write_search_results_jsonl  # noqa: E402
from ttc_operatorbench.models.dummy import DummyModelProvider  # noqa: E402
from ttc_operatorbench.search.baselines import (  # noqa: E402
    BaselinePolicy,
    BestOfNPolicy,
    GreedyPolicy,
    LocalRevisionBasicPolicy,
    PlanThenCodePolicy,
    RepairOnlyPolicy,
)
from ttc_operatorbench.tasks.toy_code import list_toy_tasks  # noqa: E402
from ttc_operatorbench.verifiers.python_unit_tests import PythonUnitTestVerifier  # noqa: E402

CORRECT_CANDIDATES = {
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "factorial": (
        "def factorial(n):\n"
        "    result = 1\n"
        "    for value in range(2, n + 1):\n"
        "        result *= value\n"
        "    return result"
    ),
    "reverse_string": "def reverse_string(s):\n    return s[::-1]",
    "is_prime": (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for divisor in range(2, int(n ** 0.5) + 1):\n"
        "        if n % divisor == 0:\n"
        "            return False\n"
        "    return True"
    ),
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a"
    ),
    "gcd": ("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)"),
    "palindrome": "def palindrome(s):\n    return s == s[::-1]",
}

GREEDY_SOLVES = {"is_even", "reverse_string", "gcd"}


def wrong_candidate(task: Task) -> str:
    """Return a simple public-test-failing candidate for a task."""
    entrypoint = task.metadata["entrypoint"]
    return f"def {entrypoint}(*args):\n    return None"


def sequence_for(policy_name: str, task: Task) -> tuple[str, ...]:
    """Return deterministic candidate sequence for a policy and task."""
    correct = CORRECT_CANDIDATES[task.task_id]
    wrong = wrong_candidate(task)
    if policy_name == "greedy":
        return (correct,) if task.task_id in GREEDY_SOLVES else (wrong,)
    if policy_name == "best_of_n":
        return (wrong, correct)
    if policy_name in {"repair_only", "local_revision_basic"}:
        return (wrong, correct)
    if policy_name == "plan_then_code":
        return ("Use the function definition requested by the task.", correct)
    raise ValueError(f"unknown policy: {policy_name}")


def baseline_policies(best_of_n: int) -> tuple[BaselinePolicy, ...]:
    """Return the deterministic baseline policy suite."""
    return (
        GreedyPolicy(),
        BestOfNPolicy(n=best_of_n),
        RepairOnlyPolicy(max_repairs=1),
        PlanThenCodePolicy(),
        LocalRevisionBasicPolicy(max_revisions=1),
    )


def run_toy_eval(best_of_n: int) -> tuple[SearchResult, ...]:
    """Run all baselines on all introductory tasks."""
    verifier = PythonUnitTestVerifier(timeout_seconds=1.0)
    budget = Budget(
        max_attempts=max(2, best_of_n),
        max_verifier_calls=max(2, best_of_n),
        max_tokens=2_000,
    )
    results: list[SearchResult] = []
    for policy in baseline_policies(best_of_n):
        for task in list_toy_tasks():
            provider = DummyModelProvider({task.task_id: sequence_for(policy.name, task)})
            results.append(
                policy.run(
                    task,
                    provider,
                    verifier,
                    budget,
                    run_id="toy-eval",
                )
            )
    return tuple(results)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/toy_eval.jsonl"))
    parser.add_argument("--best-of-n", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the introductory evaluation and write JSONL results."""
    args = parse_args()
    results = run_toy_eval(args.best_of_n)
    write_search_results_jsonl(args.output, results)
    print(f"wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
