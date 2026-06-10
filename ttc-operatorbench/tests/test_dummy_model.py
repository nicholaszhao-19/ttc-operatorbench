"""Tests for the deterministic dummy model provider."""

from ttc_operatorbench.core.schema import SamplingConfig
from ttc_operatorbench.models.dummy import DummyModelProvider, count_tokens
from ttc_operatorbench.tasks.toy_code import get_toy_task


def test_dummy_provider_returns_configured_generation() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: "def is_even(n):\n    return n % 2 == 0"})

    generation = provider.generate(task, SamplingConfig(seed=1))

    assert generation.prompt == task.prompt
    assert generation.generation_text.startswith("def is_even")
    assert generation.input_tokens == count_tokens(task.prompt)
    assert generation.total_tokens == generation.input_tokens + generation.output_tokens
    assert generation.provider_name == "dummy"


def test_dummy_provider_defaults_to_empty_generation() -> None:
    task = get_toy_task("factorial")
    provider = DummyModelProvider()

    generation = provider.generate(task)

    assert generation.generation_text == ""
    assert generation.output_tokens == 0


def test_dummy_provider_returns_scripted_sequence() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: ("first", "second")})

    first = provider.generate(task)
    second = provider.generate(task)
    third = provider.generate(task)

    assert first.generation_text == "first"
    assert second.generation_text == "second"
    assert third.generation_text == "second"


def test_dummy_provider_honors_max_output_tokens() -> None:
    task = get_toy_task("is_even")
    provider = DummyModelProvider({task.task_id: "one two three"})

    generation = provider.generate(task, SamplingConfig(max_output_tokens=2))

    assert generation.generation_text == "one two"
    assert generation.output_tokens == 2
