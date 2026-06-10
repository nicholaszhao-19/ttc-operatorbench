"""Curated local Python coding tasks beyond the initial toy suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY, PUBLIC_TESTS_KEY

CuratedTaskId = Literal[
    "count_vowels",
    "sum_squares",
    "flatten_once",
    "unique_preserve_order",
    "binary_search",
    "merge_sorted",
    "balanced_parentheses",
    "transpose_matrix",
    "run_length_encode",
    "run_length_decode",
    "is_power_of_two",
    "rotate_list",
    "dot_product",
    "hamming_distance",
    "normalize_whitespace",
    "clamp",
    "fizz_buzz",
    "chunk_list",
    "second_largest",
    "count_words",
]


@dataclass(frozen=True)
class CuratedCodeTaskSpec:
    """Definition for one self-contained public-test coding task."""

    task_id: CuratedTaskId
    entrypoint: str
    prompt: str
    public_tests: tuple[str, ...]
    reference_solution: str

    def to_task(self) -> Task:
        """Convert the curated spec into the shared task schema."""
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            metadata={
                "suite": "curated_code",
                "entrypoint": self.entrypoint,
            },
            allowed_verifier_inputs={
                ENTRYPOINT_KEY: self.entrypoint,
                PUBLIC_TESTS_KEY: self.public_tests,
            },
        )


CURATED_CODE_TASK_SPECS: tuple[CuratedCodeTaskSpec, ...] = (
    CuratedCodeTaskSpec(
        task_id="count_vowels",
        entrypoint="count_vowels",
        prompt="Write a Python function count_vowels(s) that counts a, e, i, o, and u.",
        public_tests=(
            "assert count_vowels('Alphabet') == 3",
            "assert count_vowels('sky') == 0",
            "assert count_vowels('') == 0",
        ),
        reference_solution=(
            "def count_vowels(s):\n"
            "    return sum(1 for ch in s.lower() if ch in 'aeiou')"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="sum_squares",
        entrypoint="sum_squares",
        prompt="Write a Python function sum_squares(nums) that returns the sum of squares.",
        public_tests=(
            "assert sum_squares([1, 2, 3]) == 14",
            "assert sum_squares([-2, 5]) == 29",
            "assert sum_squares([]) == 0",
        ),
        reference_solution="def sum_squares(nums):\n    return sum(x * x for x in nums)",
    ),
    CuratedCodeTaskSpec(
        task_id="flatten_once",
        entrypoint="flatten_once",
        prompt="Write a Python function flatten_once(nested) that flattens one list level.",
        public_tests=(
            "assert flatten_once([[1, 2], [], [3]]) == [1, 2, 3]",
            "assert flatten_once([['a'], ['b', 'c']]) == ['a', 'b', 'c']",
            "assert flatten_once([]) == []",
        ),
        reference_solution=(
            "def flatten_once(nested):\n"
            "    out = []\n"
            "    for group in nested:\n"
            "        out.extend(group)\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="unique_preserve_order",
        entrypoint="unique_preserve_order",
        prompt=(
            "Write a Python function unique_preserve_order(items) that removes duplicates "
            "while preserving first occurrence order."
        ),
        public_tests=(
            "assert unique_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]",
            "assert unique_preserve_order(['a', 'a', 'b']) == ['a', 'b']",
            "assert unique_preserve_order([]) == []",
        ),
        reference_solution=(
            "def unique_preserve_order(items):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            out.append(item)\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="binary_search",
        entrypoint="binary_search",
        prompt=(
            "Write a Python function binary_search(nums, target) that returns the index of "
            "target in sorted nums, or -1 if missing."
        ),
        public_tests=(
            "assert binary_search([1, 3, 5, 9], 5) == 2",
            "assert binary_search([1, 3, 5, 9], 2) == -1",
            "assert binary_search([], 1) == -1",
        ),
        reference_solution=(
            "def binary_search(nums, target):\n"
            "    lo, hi = 0, len(nums) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n"
            "            return mid\n"
            "        if nums[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="merge_sorted",
        entrypoint="merge_sorted",
        prompt="Write a Python function merge_sorted(a, b) that merges two sorted lists.",
        public_tests=(
            "assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]",
            "assert merge_sorted([], [1]) == [1]",
            "assert merge_sorted([1, 1], [1, 2]) == [1, 1, 1, 2]",
        ),
        reference_solution=(
            "def merge_sorted(a, b):\n"
            "    i = j = 0\n"
            "    out = []\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            out.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            out.append(b[j])\n"
            "            j += 1\n"
            "    out.extend(a[i:])\n"
            "    out.extend(b[j:])\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="balanced_parentheses",
        entrypoint="balanced_parentheses",
        prompt=(
            "Write a Python function balanced_parentheses(s) that returns True exactly when "
            "the parentheses in s are balanced."
        ),
        public_tests=(
            "assert balanced_parentheses('(())') is True",
            "assert balanced_parentheses('(()') is False",
            "assert balanced_parentheses('a(b)c') is True",
            "assert balanced_parentheses(')(') is False",
        ),
        reference_solution=(
            "def balanced_parentheses(s):\n"
            "    balance = 0\n"
            "    for ch in s:\n"
            "        if ch == '(':\n"
            "            balance += 1\n"
            "        elif ch == ')':\n"
            "            balance -= 1\n"
            "            if balance < 0:\n"
            "                return False\n"
            "    return balance == 0"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="transpose_matrix",
        entrypoint="transpose_matrix",
        prompt="Write a Python function transpose_matrix(matrix) that transposes a matrix.",
        public_tests=(
            "assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]",
            "assert transpose_matrix([[1, 2, 3]]) == [[1], [2], [3]]",
            "assert transpose_matrix([]) == []",
        ),
        reference_solution=(
            "def transpose_matrix(matrix):\n"
            "    if not matrix:\n"
            "        return []\n"
            "    return [list(row) for row in zip(*matrix)]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="run_length_encode",
        entrypoint="run_length_encode",
        prompt=(
            "Write a Python function run_length_encode(s) that returns a list of "
            "(character, count) pairs."
        ),
        public_tests=(
            "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]",
            "assert run_length_encode('') == []",
            "assert run_length_encode('z') == [('z', 1)]",
        ),
        reference_solution=(
            "def run_length_encode(s):\n"
            "    if not s:\n"
            "        return []\n"
            "    out = []\n"
            "    current = s[0]\n"
            "    count = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == current:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append((current, count))\n"
            "            current = ch\n"
            "            count = 1\n"
            "    out.append((current, count))\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="run_length_decode",
        entrypoint="run_length_decode",
        prompt=(
            "Write a Python function run_length_decode(pairs) that decodes "
            "(character, count) pairs into a string."
        ),
        public_tests=(
            "assert run_length_decode([('a', 3), ('b', 2), ('c', 1)]) == 'aaabbc'",
            "assert run_length_decode([]) == ''",
            "assert run_length_decode([('z', 1)]) == 'z'",
        ),
        reference_solution=(
            "def run_length_decode(pairs):\n"
            "    return ''.join(ch * count for ch, count in pairs)"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="is_power_of_two",
        entrypoint="is_power_of_two",
        prompt=(
            "Write a Python function is_power_of_two(n) that returns True exactly when n "
            "is a positive power of two."
        ),
        public_tests=(
            "assert is_power_of_two(1) is True",
            "assert is_power_of_two(16) is True",
            "assert is_power_of_two(18) is False",
            "assert is_power_of_two(0) is False",
        ),
        reference_solution=(
            "def is_power_of_two(n):\n"
            "    return n > 0 and (n & (n - 1)) == 0"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="rotate_list",
        entrypoint="rotate_list",
        prompt=(
            "Write a Python function rotate_list(items, k) that rotates items right by k "
            "positions. Negative k rotates left."
        ),
        public_tests=(
            "assert rotate_list([1, 2, 3, 4], 1) == [4, 1, 2, 3]",
            "assert rotate_list([1, 2, 3, 4], -1) == [2, 3, 4, 1]",
            "assert rotate_list([], 3) == []",
        ),
        reference_solution=(
            "def rotate_list(items, k):\n"
            "    if not items:\n"
            "        return []\n"
            "    k %= len(items)\n"
            "    return items[-k:] + items[:-k]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="dot_product",
        entrypoint="dot_product",
        prompt="Write a Python function dot_product(a, b) that returns the dot product.",
        public_tests=(
            "assert dot_product([1, 2, 3], [4, 5, 6]) == 32",
            "assert dot_product([], []) == 0",
            "assert dot_product([-1, 2], [3, 4]) == 5",
        ),
        reference_solution="def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))",
    ),
    CuratedCodeTaskSpec(
        task_id="hamming_distance",
        entrypoint="hamming_distance",
        prompt=(
            "Write a Python function hamming_distance(a, b) that counts differing "
            "positions and raises ValueError for unequal lengths."
        ),
        public_tests=(
            "assert hamming_distance('karolin', 'kathrin') == 3",
            "assert hamming_distance('', '') == 0",
            "try:\n"
            "    hamming_distance('a', 'ab')\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        reference_solution=(
            "def hamming_distance(a, b):\n"
            "    if len(a) != len(b):\n"
            "        raise ValueError('inputs must have equal length')\n"
            "    return sum(x != y for x, y in zip(a, b))"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="normalize_whitespace",
        entrypoint="normalize_whitespace",
        prompt=(
            "Write a Python function normalize_whitespace(text) that trims and collapses "
            "all whitespace runs to single spaces."
        ),
        public_tests=(
            "assert normalize_whitespace('  hello   world  ') == 'hello world'",
            "assert normalize_whitespace('\\nalpha\\tbeta\\n') == 'alpha beta'",
            "assert normalize_whitespace('') == ''",
        ),
        reference_solution="def normalize_whitespace(text):\n    return ' '.join(text.split())",
    ),
    CuratedCodeTaskSpec(
        task_id="clamp",
        entrypoint="clamp",
        prompt=(
            "Write a Python function clamp(value, low, high) that limits value to the "
            "inclusive range [low, high]."
        ),
        public_tests=(
            "assert clamp(5, 0, 10) == 5",
            "assert clamp(-1, 0, 10) == 0",
            "assert clamp(11, 0, 10) == 10",
        ),
        reference_solution="def clamp(value, low, high):\n    return max(low, min(high, value))",
    ),
    CuratedCodeTaskSpec(
        task_id="fizz_buzz",
        entrypoint="fizz_buzz",
        prompt=(
            "Write a Python function fizz_buzz(n) that returns the classic FizzBuzz list "
            "for numbers 1 through n."
        ),
        public_tests=(
            "assert fizz_buzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
            "assert fizz_buzz(15)[-1] == 'FizzBuzz'",
            "assert fizz_buzz(0) == []",
        ),
        reference_solution=(
            "def fizz_buzz(n):\n"
            "    out = []\n"
            "    for value in range(1, n + 1):\n"
            "        if value % 15 == 0:\n"
            "            out.append('FizzBuzz')\n"
            "        elif value % 3 == 0:\n"
            "            out.append('Fizz')\n"
            "        elif value % 5 == 0:\n"
            "            out.append('Buzz')\n"
            "        else:\n"
            "            out.append(str(value))\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="chunk_list",
        entrypoint="chunk_list",
        prompt=(
            "Write a Python function chunk_list(items, size) that splits items into "
            "consecutive chunks of length size."
        ),
        public_tests=(
            "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
            "assert chunk_list([], 3) == []",
            "assert chunk_list(['a', 'b'], 5) == [['a', 'b']]",
        ),
        reference_solution=(
            "def chunk_list(items, size):\n"
            "    return [items[index:index + size] for index in range(0, len(items), size)]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="second_largest",
        entrypoint="second_largest",
        prompt=(
            "Write a Python function second_largest(nums) that returns the second largest "
            "distinct value and raises ValueError if unavailable."
        ),
        public_tests=(
            "assert second_largest([3, 1, 5, 5, 2]) == 3",
            "assert second_largest([-2, -5, -1]) == -2",
            "try:\n"
            "    second_largest([1, 1])\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        reference_solution=(
            "def second_largest(nums):\n"
            "    values = sorted(set(nums))\n"
            "    if len(values) < 2:\n"
            "        raise ValueError('need at least two distinct values')\n"
            "    return values[-2]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="count_words",
        entrypoint="count_words",
        prompt=(
            "Write a Python function count_words(text) that returns the number of "
            "whitespace-separated words."
        ),
        public_tests=(
            "assert count_words('hello world') == 2",
            "assert count_words('  multiple   spaces ') == 2",
            "assert count_words('') == 0",
        ),
        reference_solution="def count_words(text):\n    return len(text.split())",
    ),
)

_CURATED_CODE_TASKS_BY_ID: dict[str, CuratedCodeTaskSpec] = {
    spec.task_id: spec for spec in CURATED_CODE_TASK_SPECS
}

CURATED_REFERENCE_CANDIDATES: dict[str, str] = {
    spec.task_id: spec.reference_solution for spec in CURATED_CODE_TASK_SPECS
}


def list_curated_tasks() -> tuple[Task, ...]:
    """Return all curated code tasks as shared task schemas."""
    return tuple(spec.to_task() for spec in CURATED_CODE_TASK_SPECS)


def get_curated_task(task_id: str) -> Task:
    """Return one curated code task by identifier."""
    return _CURATED_CODE_TASKS_BY_ID[task_id].to_task()


def curated_task_ids() -> tuple[str, ...]:
    """Return the stable curated task identifiers."""
    return tuple(spec.task_id for spec in CURATED_CODE_TASK_SPECS)


__all__ = [
    "CURATED_CODE_TASK_SPECS",
    "CURATED_REFERENCE_CANDIDATES",
    "CuratedCodeTaskSpec",
    "CuratedTaskId",
    "curated_task_ids",
    "get_curated_task",
    "list_curated_tasks",
]
