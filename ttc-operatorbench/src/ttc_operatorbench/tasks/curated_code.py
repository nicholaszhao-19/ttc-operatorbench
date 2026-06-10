"""Curated local Python coding tasks beyond the initial toy suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ttc_operatorbench.core.schema import Task
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY, HIDDEN_TESTS_KEY, PUBLIC_TESTS_KEY

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
    "parse_ints",
    "camel_to_snake",
    "is_anagram",
    "most_frequent",
    "prefix_sums",
    "matrix_diagonal_sum",
    "safe_divide",
    "zip_to_dict",
    "group_by_first_letter",
    "top_k",
    "remove_none",
    "parse_csv_line",
    "title_case_words",
    "caesar_shift",
    "roman_to_int",
    "is_valid_ipv4",
    "flatten_dict",
    "dedupe_by_key",
    "merge_intervals",
    "paginate",
    "powerset",
    "balanced_brackets",
    "strip_comments",
    "levenshtein_distance",
    "topological_sort",
    "word_frequencies",
    "sliding_window_max",
    "binary_to_decimal",
    "decimal_to_binary",
    "common_prefix",
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
        hidden_tests = CURATED_HIDDEN_TESTS[self.task_id]
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            public_tests=self.public_tests,
            hidden_tests=hidden_tests,
            task_family="curated_code",
            difficulty_label="uncalibrated",
            metadata={
                "suite": "curated_code",
                "entrypoint": self.entrypoint,
                "task_family": "curated_code",
                "difficulty_label": "uncalibrated",
            },
            allowed_verifier_inputs={
                ENTRYPOINT_KEY: self.entrypoint,
                PUBLIC_TESTS_KEY: self.public_tests,
                HIDDEN_TESTS_KEY: hidden_tests,
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
    CuratedCodeTaskSpec(
        task_id="parse_ints",
        entrypoint="parse_ints",
        prompt=(
            "Write a Python function parse_ints(text) that returns all whitespace-separated "
            "integers in text."
        ),
        public_tests=(
            "assert parse_ints('1 2 3') == [1, 2, 3]",
            "assert parse_ints('-1 0 42') == [-1, 0, 42]",
            "assert parse_ints('') == []",
        ),
        reference_solution="def parse_ints(text):\n    return [int(part) for part in text.split()]",
    ),
    CuratedCodeTaskSpec(
        task_id="camel_to_snake",
        entrypoint="camel_to_snake",
        prompt=(
            "Write a Python function camel_to_snake(name) that converts CamelCase to "
            "snake_case."
        ),
        public_tests=(
            "assert camel_to_snake('CamelCase') == 'camel_case'",
            "assert camel_to_snake('HTTPRequest') == 'h_t_t_p_request'",
            "assert camel_to_snake('Name') == 'name'",
        ),
        reference_solution=(
            "def camel_to_snake(name):\n"
            "    out = []\n"
            "    for index, ch in enumerate(name):\n"
            "        if ch.isupper() and index > 0:\n"
            "            out.append('_')\n"
            "        out.append(ch.lower())\n"
            "    return ''.join(out)"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="is_anagram",
        entrypoint="is_anagram",
        prompt=(
            "Write a Python function is_anagram(a, b) that ignores spaces and case when "
            "checking whether two strings are anagrams."
        ),
        public_tests=(
            "assert is_anagram('listen', 'silent') is True",
            "assert is_anagram('Dormitory', 'dirty room') is True",
            "assert is_anagram('abc', 'abd') is False",
        ),
        reference_solution=(
            "def is_anagram(a, b):\n"
            "    left = a.replace(' ', '').lower()\n"
            "    right = b.replace(' ', '').lower()\n"
            "    return sorted(left) == sorted(right)"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="most_frequent",
        entrypoint="most_frequent",
        prompt=(
            "Write a Python function most_frequent(items) that returns the most frequent item, "
            "breaking ties by first occurrence. Return None for an empty list."
        ),
        public_tests=(
            "assert most_frequent([1, 2, 2, 3]) == 2",
            "assert most_frequent(['a', 'b', 'a', 'b']) == 'a'",
            "assert most_frequent([]) is None",
        ),
        reference_solution=(
            "def most_frequent(items):\n"
            "    counts = {}\n"
            "    for item in items:\n"
            "        counts[item] = counts.get(item, 0) + 1\n"
            "    best = None\n"
            "    for item in items:\n"
            "        if best is None or counts[item] > counts[best]:\n"
            "            best = item\n"
            "    return best"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="prefix_sums",
        entrypoint="prefix_sums",
        prompt="Write a Python function prefix_sums(nums) that returns cumulative sums.",
        public_tests=(
            "assert prefix_sums([1, 2, 3]) == [1, 3, 6]",
            "assert prefix_sums([-1, 1, 5]) == [-1, 0, 5]",
            "assert prefix_sums([]) == []",
        ),
        reference_solution=(
            "def prefix_sums(nums):\n"
            "    total = 0\n"
            "    out = []\n"
            "    for num in nums:\n"
            "        total += num\n"
            "        out.append(total)\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="matrix_diagonal_sum",
        entrypoint="matrix_diagonal_sum",
        prompt=(
            "Write a Python function matrix_diagonal_sum(matrix) that returns the main "
            "diagonal sum of a square matrix."
        ),
        public_tests=(
            "assert matrix_diagonal_sum([[1, 2], [3, 4]]) == 5",
            "assert matrix_diagonal_sum([[5]]) == 5",
            "assert matrix_diagonal_sum([]) == 0",
        ),
        reference_solution=(
            "def matrix_diagonal_sum(matrix):\n"
            "    return sum(row[index] for index, row in enumerate(matrix))"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="safe_divide",
        entrypoint="safe_divide",
        prompt="Write a Python function safe_divide(a, b) that returns None when b is zero.",
        public_tests=(
            "assert safe_divide(6, 3) == 2",
            "assert safe_divide(1, 0) is None",
            "assert safe_divide(-9, 3) == -3",
        ),
        reference_solution="def safe_divide(a, b):\n    return None if b == 0 else a / b",
    ),
    CuratedCodeTaskSpec(
        task_id="zip_to_dict",
        entrypoint="zip_to_dict",
        prompt=(
            "Write a Python function zip_to_dict(keys, values) that zips keys and values "
            "into a dictionary, stopping at the shorter input."
        ),
        public_tests=(
            "assert zip_to_dict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}",
            "assert zip_to_dict(['a', 'b'], [1]) == {'a': 1}",
            "assert zip_to_dict([], [1]) == {}",
        ),
        reference_solution="def zip_to_dict(keys, values):\n    return dict(zip(keys, values))",
    ),
    CuratedCodeTaskSpec(
        task_id="group_by_first_letter",
        entrypoint="group_by_first_letter",
        prompt=(
            "Write a Python function group_by_first_letter(words) that groups nonempty words "
            "by lowercase first letter."
        ),
        public_tests=(
            "assert group_by_first_letter(['Apple', 'ape', 'Bee']) == "
            "{'a': ['Apple', 'ape'], 'b': ['Bee']}",
            "assert group_by_first_letter(['', 'cat']) == {'c': ['cat']}",
            "assert group_by_first_letter([]) == {}",
        ),
        reference_solution=(
            "def group_by_first_letter(words):\n"
            "    groups = {}\n"
            "    for word in words:\n"
            "        if not word:\n"
            "            continue\n"
            "        key = word[0].lower()\n"
            "        groups.setdefault(key, []).append(word)\n"
            "    return groups"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="top_k",
        entrypoint="top_k",
        prompt=(
            "Write a Python function top_k(nums, k) that returns the k largest values "
            "descending."
        ),
        public_tests=(
            "assert top_k([3, 1, 5, 2], 2) == [5, 3]",
            "assert top_k([1, 1, 2], 5) == [2, 1, 1]",
            "assert top_k([1, 2], 0) == []",
        ),
        reference_solution="def top_k(nums, k):\n    return sorted(nums, reverse=True)[:max(k, 0)]",
    ),
    CuratedCodeTaskSpec(
        task_id="remove_none",
        entrypoint="remove_none",
        prompt="Write a Python function remove_none(items) that removes only None values.",
        public_tests=(
            "assert remove_none([1, None, 2]) == [1, 2]",
            "assert remove_none([None, None]) == []",
            "assert remove_none([0, False, '']) == [0, False, '']",
        ),
        reference_solution=(
            "def remove_none(items):\n"
            "    return [item for item in items if item is not None]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="parse_csv_line",
        entrypoint="parse_csv_line",
        prompt=(
            "Write a Python function parse_csv_line(line) that splits a simple comma-separated "
            "line into stripped cells. Empty input returns an empty list."
        ),
        public_tests=(
            "assert parse_csv_line('a,b,c') == ['a', 'b', 'c']",
            "assert parse_csv_line(' a, b ,c ') == ['a', 'b', 'c']",
            "assert parse_csv_line('') == []",
        ),
        reference_solution=(
            "def parse_csv_line(line):\n"
            "    if line == '':\n"
            "        return []\n"
            "    return [cell.strip() for cell in line.split(',')]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="title_case_words",
        entrypoint="title_case_words",
        prompt=(
            "Write a Python function title_case_words(text) that title-cases each "
            "whitespace-separated word."
        ),
        public_tests=(
            "assert title_case_words('hello world') == 'Hello World'",
            "assert title_case_words('mIxEd CASE') == 'Mixed Case'",
            "assert title_case_words('') == ''",
        ),
        reference_solution=(
            "def title_case_words(text):\n"
            "    return ' '.join(word[:1].upper() + word[1:].lower() for word in text.split())"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="caesar_shift",
        entrypoint="caesar_shift",
        prompt=(
            "Write a Python function caesar_shift(text, shift) that shifts ASCII letters and "
            "preserves case and nonletters."
        ),
        public_tests=(
            "assert caesar_shift('abc', 2) == 'cde'",
            "assert caesar_shift('XYZ', 3) == 'ABC'",
            "assert caesar_shift('a-b!', -1) == 'z-a!'",
        ),
        reference_solution=(
            "def caesar_shift(text, shift):\n"
            "    out = []\n"
            "    for ch in text:\n"
            "        if 'a' <= ch <= 'z':\n"
            "            out.append(chr((ord(ch) - ord('a') + shift) % 26 + ord('a')))\n"
            "        elif 'A' <= ch <= 'Z':\n"
            "            out.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))\n"
            "        else:\n"
            "            out.append(ch)\n"
            "    return ''.join(out)"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="roman_to_int",
        entrypoint="roman_to_int",
        prompt=(
            "Write a Python function roman_to_int(s) that converts a Roman numeral to an "
            "integer."
        ),
        public_tests=(
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('IV') == 4",
            "assert roman_to_int('MCMXCIV') == 1994",
        ),
        reference_solution=(
            "def roman_to_int(s):\n"
            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    previous = 0\n"
            "    for ch in reversed(s):\n"
            "        value = values[ch]\n"
            "        total += -value if value < previous else value\n"
            "        previous = max(previous, value)\n"
            "    return total"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="is_valid_ipv4",
        entrypoint="is_valid_ipv4",
        prompt=(
            "Write a Python function is_valid_ipv4(s) that validates dotted IPv4 strings "
            "with four decimal parts from 0 to 255 and no leading zeros."
        ),
        public_tests=(
            "assert is_valid_ipv4('127.0.0.1') is True",
            "assert is_valid_ipv4('256.0.0.1') is False",
            "assert is_valid_ipv4('01.2.3.4') is False",
        ),
        reference_solution=(
            "def is_valid_ipv4(s):\n"
            "    parts = s.split('.')\n"
            "    if len(parts) != 4:\n"
            "        return False\n"
            "    for part in parts:\n"
            "        if not part.isdigit():\n"
            "            return False\n"
            "        if len(part) > 1 and part[0] == '0':\n"
            "            return False\n"
            "        value = int(part)\n"
            "        if value < 0 or value > 255:\n"
            "            return False\n"
            "    return True"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="flatten_dict",
        entrypoint="flatten_dict",
        prompt=(
            "Write a Python function flatten_dict(d, sep='.') that flattens nested dictionaries "
            "using joined keys."
        ),
        public_tests=(
            "assert flatten_dict({'a': {'b': 1}, 'c': 2}) == {'a.b': 1, 'c': 2}",
            "assert flatten_dict({}) == {}",
            "assert flatten_dict({'x': {'y': {'z': 3}}}) == {'x.y.z': 3}",
        ),
        reference_solution=(
            "def flatten_dict(d, sep='.'):\n"
            "    out = {}\n"
            "    def visit(prefix, value):\n"
            "        if isinstance(value, dict):\n"
            "            for key, child in value.items():\n"
            "                visit(str(key) if not prefix else prefix + sep + str(key), child)\n"
            "        else:\n"
            "            out[prefix] = value\n"
            "    for key, value in d.items():\n"
            "        visit(str(key), value)\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="dedupe_by_key",
        entrypoint="dedupe_by_key",
        prompt=(
            "Write a Python function dedupe_by_key(records, key) that keeps the first record "
            "for each distinct key value."
        ),
        public_tests=(
            "assert dedupe_by_key([{'id': 1}, {'id': 1}, {'id': 2}], 'id') == "
            "[{'id': 1}, {'id': 2}]",
            "assert dedupe_by_key([], 'id') == []",
            "assert dedupe_by_key([{'x': 'a'}, {'x': 'b'}], 'x') == [{'x': 'a'}, {'x': 'b'}]",
        ),
        reference_solution=(
            "def dedupe_by_key(records, key):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for record in records:\n"
            "        value = record[key]\n"
            "        if value not in seen:\n"
            "            seen.add(value)\n"
            "            out.append(record)\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="merge_intervals",
        entrypoint="merge_intervals",
        prompt=(
            "Write a Python function merge_intervals(intervals) that merges overlapping "
            "inclusive intervals and returns sorted tuples."
        ),
        public_tests=(
            "assert merge_intervals([(1, 3), (2, 4), (6, 8)]) == [(1, 4), (6, 8)]",
            "assert merge_intervals([]) == []",
            "assert merge_intervals([(5, 5), (1, 2)]) == [(1, 2), (5, 5)]",
        ),
        reference_solution=(
            "def merge_intervals(intervals):\n"
            "    merged = []\n"
            "    for start, end in sorted(intervals):\n"
            "        if not merged or start > merged[-1][1]:\n"
            "            merged.append([start, end])\n"
            "        else:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "    return [tuple(interval) for interval in merged]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="paginate",
        entrypoint="paginate",
        prompt=(
            "Write a Python function paginate(items, page, size) that returns a one-indexed "
            "page and returns [] for invalid page or size."
        ),
        public_tests=(
            "assert paginate([1, 2, 3, 4, 5], 1, 2) == [1, 2]",
            "assert paginate([1, 2, 3, 4, 5], 3, 2) == [5]",
            "assert paginate([1, 2], 0, 2) == []",
        ),
        reference_solution=(
            "def paginate(items, page, size):\n"
            "    if page < 1 or size <= 0:\n"
            "        return []\n"
            "    start = (page - 1) * size\n"
            "    return items[start:start + size]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="powerset",
        entrypoint="powerset",
        prompt=(
            "Write a Python function powerset(items) that returns all subsets as tuples in "
            "iterative order."
        ),
        public_tests=(
            "assert powerset([]) == [()]",
            "assert powerset([1]) == [(), (1,)]",
            "assert powerset([1, 2]) == [(), (1,), (2,), (1, 2)]",
        ),
        reference_solution=(
            "def powerset(items):\n"
            "    out = [()]\n"
            "    for item in items:\n"
            "        out += [subset + (item,) for subset in out]\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="balanced_brackets",
        entrypoint="balanced_brackets",
        prompt=(
            "Write a Python function balanced_brackets(s) that validates (), [], and {} "
            "brackets while ignoring other characters."
        ),
        public_tests=(
            "assert balanced_brackets('([])') is True",
            "assert balanced_brackets('([)]') is False",
            "assert balanced_brackets('a{b[c]}') is True",
        ),
        reference_solution=(
            "def balanced_brackets(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    openings = set(pairs.values())\n"
            "    stack = []\n"
            "    for ch in s:\n"
            "        if ch in openings:\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="strip_comments",
        entrypoint="strip_comments",
        prompt=(
            "Write a Python function strip_comments(lines, marker='#') that removes text "
            "after the marker from each line and strips trailing whitespace."
        ),
        public_tests=(
            "assert strip_comments(['a # note', 'b']) == ['a', 'b']",
            "assert strip_comments(['x//y'], marker='//') == ['x']",
            "assert strip_comments([]) == []",
        ),
        reference_solution=(
            "def strip_comments(lines, marker='#'):\n"
            "    return [line.split(marker, 1)[0].rstrip() for line in lines]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="levenshtein_distance",
        entrypoint="levenshtein_distance",
        prompt=(
            "Write a Python function levenshtein_distance(a, b) that returns edit distance "
            "with insertions, deletions, and substitutions."
        ),
        public_tests=(
            "assert levenshtein_distance('kitten', 'sitting') == 3",
            "assert levenshtein_distance('', 'abc') == 3",
            "assert levenshtein_distance('same', 'same') == 0",
        ),
        reference_solution=(
            "def levenshtein_distance(a, b):\n"
            "    previous = list(range(len(b) + 1))\n"
            "    for i, ch_a in enumerate(a, 1):\n"
            "        current = [i]\n"
            "        for j, ch_b in enumerate(b, 1):\n"
            "            cost = 0 if ch_a == ch_b else 1\n"
            "            current.append(\n"
            "                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)\n"
            "            )\n"
            "        previous = current\n"
            "    return previous[-1]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="topological_sort",
        entrypoint="topological_sort",
        prompt=(
            "Write a Python function topological_sort(edges) that returns a stable sorted "
            "topological order for directed acyclic edges and raises ValueError on cycles."
        ),
        public_tests=(
            "assert topological_sort([('a', 'b'), ('b', 'c')]) == ['a', 'b', 'c']",
            "assert topological_sort([('a', 'c'), ('b', 'c')]) == ['a', 'b', 'c']",
            "try:\n"
            "    topological_sort([('a', 'b'), ('b', 'a')])\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        reference_solution=(
            "def topological_sort(edges):\n"
            "    nodes = set()\n"
            "    graph = {}\n"
            "    indegree = {}\n"
            "    for left, right in edges:\n"
            "        nodes.update((left, right))\n"
            "        graph.setdefault(left, []).append(right)\n"
            "        indegree[right] = indegree.get(right, 0) + 1\n"
            "        indegree.setdefault(left, indegree.get(left, 0))\n"
            "    ready = sorted(node for node in nodes if indegree.get(node, 0) == 0)\n"
            "    out = []\n"
            "    while ready:\n"
            "        node = ready.pop(0)\n"
            "        out.append(node)\n"
            "        for child in sorted(graph.get(node, [])):\n"
            "            indegree[child] -= 1\n"
            "            if indegree[child] == 0:\n"
            "                ready.append(child)\n"
            "                ready.sort()\n"
            "    if len(out) != len(nodes):\n"
            "        raise ValueError('cycle detected')\n"
            "    return out"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="word_frequencies",
        entrypoint="word_frequencies",
        prompt=(
            "Write a Python function word_frequencies(text) that lowercases words split by "
            "whitespace and returns frequency counts."
        ),
        public_tests=(
            "assert word_frequencies('One one two') == {'one': 2, 'two': 1}",
            "assert word_frequencies('') == {}",
            "assert word_frequencies('A a A') == {'a': 3}",
        ),
        reference_solution=(
            "def word_frequencies(text):\n"
            "    counts = {}\n"
            "    for word in text.lower().split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="sliding_window_max",
        entrypoint="sliding_window_max",
        prompt=(
            "Write a Python function sliding_window_max(nums, k) that returns the maximum "
            "for each consecutive window, or [] for invalid k."
        ),
        public_tests=(
            "assert sliding_window_max([1, 3, 2, 5], 2) == [3, 3, 5]",
            "assert sliding_window_max([4, 1], 3) == []",
            "assert sliding_window_max([2, 1], 0) == []",
        ),
        reference_solution=(
            "def sliding_window_max(nums, k):\n"
            "    if k <= 0 or k > len(nums):\n"
            "        return []\n"
            "    return [max(nums[index:index + k]) for index in range(len(nums) - k + 1)]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="binary_to_decimal",
        entrypoint="binary_to_decimal",
        prompt=(
            "Write a Python function binary_to_decimal(bits) that converts a nonempty binary "
            "string to an integer and raises ValueError for invalid input."
        ),
        public_tests=(
            "assert binary_to_decimal('101') == 5",
            "assert binary_to_decimal('0') == 0",
            "try:\n"
            "    binary_to_decimal('102')\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        reference_solution=(
            "def binary_to_decimal(bits):\n"
            "    if not bits or any(ch not in '01' for ch in bits):\n"
            "        raise ValueError('invalid binary string')\n"
            "    return int(bits, 2)"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="decimal_to_binary",
        entrypoint="decimal_to_binary",
        prompt=(
            "Write a Python function decimal_to_binary(n) that converts a nonnegative integer "
            "to a binary string and raises ValueError for negatives."
        ),
        public_tests=(
            "assert decimal_to_binary(5) == '101'",
            "assert decimal_to_binary(0) == '0'",
            "try:\n"
            "    decimal_to_binary(-1)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError')",
        ),
        reference_solution=(
            "def decimal_to_binary(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be nonnegative')\n"
            "    return bin(n)[2:]"
        ),
    ),
    CuratedCodeTaskSpec(
        task_id="common_prefix",
        entrypoint="common_prefix",
        prompt=(
            "Write a Python function common_prefix(strings) that returns the longest common "
            "prefix of all strings."
        ),
        public_tests=(
            "assert common_prefix(['flower', 'flow', 'flight']) == 'fl'",
            "assert common_prefix(['dog', 'racecar']) == ''",
            "assert common_prefix([]) == ''",
        ),
        reference_solution=(
            "def common_prefix(strings):\n"
            "    if not strings:\n"
            "        return ''\n"
            "    prefix = strings[0]\n"
            "    for value in strings[1:]:\n"
            "        while not value.startswith(prefix):\n"
            "            prefix = prefix[:-1]\n"
            "            if not prefix:\n"
            "                return ''\n"
            "    return prefix"
        ),
    ),
)

_CURATED_CODE_TASKS_BY_ID: dict[str, CuratedCodeTaskSpec] = {
    spec.task_id: spec for spec in CURATED_CODE_TASK_SPECS
}

CURATED_REFERENCE_CANDIDATES: dict[str, str] = {
    spec.task_id: spec.reference_solution for spec in CURATED_CODE_TASK_SPECS
}

CURATED_HIDDEN_TESTS: dict[str, tuple[str, ...]] = {
    "count_vowels": (
        "assert count_vowels('AEIOUxyz') == 5",
        "assert count_vowels('queueing') == 5",
    ),
    "sum_squares": (
        "assert sum_squares([0, 10, -10]) == 200",
        "assert sum_squares([4]) == 16",
    ),
    "flatten_once": (
        "assert flatten_once([[[1]], [2, 3]]) == [[1], 2, 3]",
        "assert flatten_once([[None], [True, False]]) == [None, True, False]",
    ),
    "unique_preserve_order": (
        "assert unique_preserve_order([1, 2, 1, 3, 2, 4]) == [1, 2, 3, 4]",
        "assert unique_preserve_order([(1, 2), (1, 2), (2, 3)]) == [(1, 2), (2, 3)]",
    ),
    "binary_search": (
        "assert binary_search([-5, -1, 0, 2, 9], -5) == 0",
        "assert binary_search([-5, -1, 0, 2, 9], 9) == 4",
        "assert binary_search([-5, -1, 0, 2, 9], 8) == -1",
    ),
    "merge_sorted": (
        "assert merge_sorted([-3, 0, 10], [-4, -3, 11]) == [-4, -3, -3, 0, 10, 11]",
        "assert merge_sorted([1, 2], []) == [1, 2]",
    ),
    "balanced_parentheses": (
        "assert balanced_parentheses('((a)(b))') is True",
        "assert balanced_parentheses('(()))(') is False",
    ),
    "transpose_matrix": (
        "assert transpose_matrix([[1], [2], [3]]) == [[1, 2, 3]]",
        "assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]",
    ),
    "run_length_encode": (
        "assert run_length_encode('aabccccaaa') == [('a', 2), ('b', 1), ('c', 4), ('a', 3)]",
        "assert run_length_encode('112') == [('1', 2), ('2', 1)]",
    ),
    "run_length_decode": (
        "assert run_length_decode([('x', 0), ('y', 2)]) == 'yy'",
        "assert run_length_decode([('ab', 2)]) == 'abab'",
    ),
    "is_power_of_two": (
        "assert is_power_of_two(1024) is True",
        "assert is_power_of_two(1023) is False",
        "assert is_power_of_two(-8) is False",
    ),
    "rotate_list": (
        "assert rotate_list([1, 2, 3], 3) == [1, 2, 3]",
        "assert rotate_list([1, 2, 3], 4) == [3, 1, 2]",
        "assert rotate_list(['a', 'b', 'c'], -4) == ['b', 'c', 'a']",
    ),
    "dot_product": (
        "assert dot_product([2, 0, -1], [3, 5, 4]) == 2",
        "assert dot_product([1, 2], [10]) == 10",
    ),
    "hamming_distance": (
        "assert hamming_distance('10101', '10011') == 2",
        "try:\n"
        "    hamming_distance([1, 2], [1])\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ValueError')",
    ),
    "normalize_whitespace": (
        "assert normalize_whitespace('a\\r\\n b\\t\\t c') == 'a b c'",
        "assert normalize_whitespace('   ') == ''",
    ),
    "clamp": (
        "assert clamp(0, 0, 0) == 0",
        "assert clamp(3.5, 1.0, 3.0) == 3.0",
    ),
    "fizz_buzz": (
        "assert fizz_buzz(1) == ['1']",
        "assert fizz_buzz(16)[14:] == ['FizzBuzz', '16']",
    ),
    "chunk_list": (
        "assert chunk_list([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]",
        "assert chunk_list([1, 2, 3, 4], 1) == [[1], [2], [3], [4]]",
    ),
    "second_largest": (
        "assert second_largest([10, 9, 8, 10]) == 9",
        "try:\n"
        "    second_largest([])\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ValueError')",
    ),
    "count_words": (
        "assert count_words('one\\ntwo\\tthree') == 3",
        "assert count_words(' punctuation, stays attached ') == 3",
    ),
    "parse_ints": (
        "assert parse_ints('  10\\n-20\\t30 ') == [10, -20, 30]",
        "assert parse_ints('+7 -0') == [7, 0]",
    ),
    "camel_to_snake": (
        "assert camel_to_snake('AlreadyCamel') == 'already_camel'",
        "assert camel_to_snake('XYPoint') == 'x_y_point'",
    ),
    "is_anagram": (
        "assert is_anagram('The eyes', 'they see') is True",
        "assert is_anagram('conversation', 'voices rant on') is True",
        "assert is_anagram('abc', 'abcc') is False",
    ),
    "most_frequent": (
        "assert most_frequent([3, 3, 2, 2, 2, 3]) == 3",
        "assert most_frequent([None, 1, None]) is None",
    ),
    "prefix_sums": (
        "assert prefix_sums([0, 0, 0]) == [0, 0, 0]",
        "assert prefix_sums([5, -2, -3, 10]) == [5, 3, 0, 10]",
    ),
    "matrix_diagonal_sum": (
        "assert matrix_diagonal_sum([[1, 0, 0], [0, -2, 0], [0, 0, 3]]) == 2",
        "assert matrix_diagonal_sum([[0, 9], [8, 0]]) == 0",
    ),
    "safe_divide": (
        "assert safe_divide(5, 2) == 2.5",
        "assert safe_divide(0, 0) is None",
    ),
    "zip_to_dict": (
        "assert zip_to_dict(('x', 'y', 'z'), (1, 2)) == {'x': 1, 'y': 2}",
        "assert zip_to_dict(['a', 'a'], [1, 2]) == {'a': 2}",
    ),
    "group_by_first_letter": (
        "assert group_by_first_letter(['Alpha', 'beta', 'atom']) == "
        "{'a': ['Alpha', 'atom'], 'b': ['beta']}",
        "assert group_by_first_letter(['', '', 'Zoo']) == {'z': ['Zoo']}",
    ),
    "top_k": (
        "assert top_k([-1, -5, 0], 2) == [0, -1]",
        "assert top_k([4, 4, 3], -1) == []",
    ),
    "remove_none": (
        "assert remove_none([None, 'x', None, 0]) == ['x', 0]",
        "assert remove_none([]) == []",
    ),
    "parse_csv_line": (
        "assert parse_csv_line(' one , two ,, three ') == ['one', 'two', '', 'three']",
        "assert parse_csv_line(',') == ['', '']",
    ),
    "title_case_words": (
        "assert title_case_words('  many\\tspaces ') == 'Many Spaces'",
        "assert title_case_words('a B c') == 'A B C'",
    ),
    "caesar_shift": (
        "assert caesar_shift('Hello, Zed!', 5) == 'Mjqqt, Eji!'",
        "assert caesar_shift('abcXYZ', 26) == 'abcXYZ'",
    ),
    "roman_to_int": (
        "assert roman_to_int('LVIII') == 58",
        "assert roman_to_int('CMXLIV') == 944",
    ),
    "is_valid_ipv4": (
        "assert is_valid_ipv4('0.0.0.0') is True",
        "assert is_valid_ipv4('1.2.3') is False",
        "assert is_valid_ipv4('1.2.3.-4') is False",
    ),
    "flatten_dict": (
        "assert flatten_dict({'a': 1, 'b': {'c': 2, 'd': 3}}) == {'a': 1, 'b.c': 2, 'b.d': 3}",
        "assert flatten_dict({1: {'x': 2}}) == {'1.x': 2}",
    ),
    "dedupe_by_key": (
        "assert dedupe_by_key([{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}, "
        "{'id': 1, 'v': 'c'}], 'id') == [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]",
        "assert dedupe_by_key([{'k': None}, {'k': None}], 'k') == [{'k': None}]",
    ),
    "merge_intervals": (
        "assert merge_intervals([(5, 7), (1, 3), (3, 4)]) == [(1, 4), (5, 7)]",
        "assert merge_intervals([(1, 10), (2, 3), (11, 12)]) == [(1, 10), (11, 12)]",
    ),
    "paginate": (
        "assert paginate([1, 2, 3], 2, 5) == []",
        "assert paginate([1, 2, 3, 4], 2, 2) == [3, 4]",
        "assert paginate([1, 2, 3], 1, 0) == []",
    ),
    "powerset": (
        "assert powerset(['a', 'b']) == [(), ('a',), ('b',), ('a', 'b')]",
        "assert powerset([1, 2, 3])[-1] == (1, 2, 3)",
    ),
    "balanced_brackets": (
        "assert balanced_brackets('{[()()]}') is True",
        "assert balanced_brackets('((missing]') is False",
        "assert balanced_brackets('no brackets') is True",
    ),
    "strip_comments": (
        "assert strip_comments(['  a   # b', '# full']) == ['  a', '']",
        "assert strip_comments(['a;b;c'], marker=';') == ['a']",
    ),
    "levenshtein_distance": (
        "assert levenshtein_distance('flaw', 'lawn') == 2",
        "assert levenshtein_distance('abc', '') == 3",
    ),
    "topological_sort": (
        "assert topological_sort([('cook', 'eat'), ('shop', 'cook')]) == ['shop', 'cook', 'eat']",
        "assert topological_sort([]) == []",
    ),
    "word_frequencies": (
        "assert word_frequencies('Repeat repeat REPEAT') == {'repeat': 3}",
        "assert word_frequencies(' spaced\\nwords spaced ') == {'spaced': 2, 'words': 1}",
    ),
    "sliding_window_max": (
        "assert sliding_window_max([9, 8, 7], 1) == [9, 8, 7]",
        "assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]",
    ),
    "binary_to_decimal": (
        "assert binary_to_decimal('11111111') == 255",
        "try:\n"
        "    binary_to_decimal('')\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ValueError')",
    ),
    "decimal_to_binary": (
        "assert decimal_to_binary(255) == '11111111'",
        "assert decimal_to_binary(1) == '1'",
    ),
    "common_prefix": (
        "assert common_prefix(['interview', 'internet', 'internal']) == 'inter'",
        "assert common_prefix(['same']) == 'same'",
    ),
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
    "CURATED_HIDDEN_TESTS",
    "CURATED_REFERENCE_CANDIDATES",
    "CuratedCodeTaskSpec",
    "CuratedTaskId",
    "curated_task_ids",
    "get_curated_task",
    "list_curated_tasks",
]
