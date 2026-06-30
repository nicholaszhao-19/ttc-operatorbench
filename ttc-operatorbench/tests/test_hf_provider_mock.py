"""Mocked tests for the Hugging Face provider."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task
from ttc_operatorbench.tasks.toy_code import get_toy_task


class FakeBatch(dict[str, Any]):
    """Small tokenizer batch with a ``to`` method like HF BatchEncoding."""

    def to(self, device: str) -> FakeBatch:
        self["device"] = device
        return self


class FakeTokenizer:
    """Tiny tokenizer mock with whitespace token accounting."""

    eos_token_id = 0

    def __init__(self, calls: dict[str, Any], completion_text: str, use_chat_template: bool):
        self.calls = calls
        self.completion_text = completion_text
        self.use_chat_template = use_chat_template

    def __call__(self, text: str, return_tensors: str) -> FakeBatch:
        self.calls["prompt_text"] = text
        return FakeBatch({"input_ids": [self.encode(text)], "return_tensors": return_tensors})

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text.strip():
            return []
        return list(range(len(text.split())))

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return self.completion_text.strip()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if not self.use_chat_template:
            raise ValueError("chat template unavailable")
        self.calls["chat_template_messages"] = messages
        self.calls["chat_template_tokenize"] = tokenize
        self.calls["chat_template_add_generation_prompt"] = add_generation_prompt
        return f"<chat>{messages[0]['content']}</chat><assistant>"


class FakeModel:
    """Tiny causal LM mock."""

    def __init__(self, calls: dict[str, Any]):
        self.calls = calls

    def to(self, device: str) -> FakeModel:
        self.calls["model_to_device"] = device
        return self

    def eval(self) -> None:
        self.calls["eval_called"] = True

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls["generate_kwargs"] = kwargs
        input_ids = kwargs.get("input_ids", [[0]])
        return [input_ids[0] + [1, 2, 3]]


class FakeAutoTokenizer:
    """Mock AutoTokenizer class."""

    def __init__(self, calls: dict[str, Any], completion_text: str, use_chat_template: bool):
        self.calls = calls
        self.completion_text = completion_text
        self.use_chat_template = use_chat_template

    def from_pretrained(self, model_id: str, **kwargs: Any) -> FakeTokenizer:
        self.calls["tokenizer_loads"] += 1
        self.calls["tokenizer_model_id"] = model_id
        self.calls["tokenizer_kwargs"] = kwargs
        return FakeTokenizer(self.calls, self.completion_text, self.use_chat_template)


class FakeAutoModelForCausalLM:
    """Mock AutoModelForCausalLM class."""

    def __init__(self, calls: dict[str, Any]):
        self.calls = calls

    def from_pretrained(self, model_id: str, **kwargs: Any) -> FakeModel:
        self.calls["model_loads"] += 1
        self.calls["model_model_id"] = model_id
        self.calls["model_kwargs"] = kwargs
        return FakeModel(self.calls)


class FakeTransformers(ModuleType):
    """Typed fake transformers module."""

    AutoTokenizer: FakeAutoTokenizer
    AutoModelForCausalLM: FakeAutoModelForCausalLM
    set_seed: Any


def install_fake_transformers(
    monkeypatch: Any,
    *,
    completion_text: str,
    use_chat_template: bool = False,
) -> dict[str, Any]:
    """Install a fake transformers module into sys.modules."""
    calls: dict[str, Any] = {"tokenizer_loads": 0, "model_loads": 0, "seeds": []}
    fake_transformers = FakeTransformers("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer(calls, completion_text, use_chat_template)
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM(calls)
    fake_transformers.set_seed = lambda seed: calls["seeds"].append(seed)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls


def test_importing_hf_provider_does_not_load_model(monkeypatch: Any) -> None:
    calls = install_fake_transformers(monkeypatch, completion_text="unused")

    module = importlib.import_module("ttc_operatorbench.models.hf_provider")

    assert module.HuggingFaceModelProvider
    assert calls["tokenizer_loads"] == 0
    assert calls["model_loads"] == 0


def test_generate_loads_lazily_and_returns_generation(monkeypatch: Any) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(
        monkeypatch,
        completion_text="\ndef is_even(n): return n % 2 == 0",
    )
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider(
        model_id="Qwen/Qwen3-0.6B",
        device="cpu",
        dtype="auto",
        max_new_tokens=12,
        temperature=0.3,
        top_p=0.8,
        do_sample=True,
        seed=123,
    )

    generation = provider.generate(task)

    assert isinstance(generation, Generation)
    assert calls["tokenizer_loads"] == 1
    assert calls["model_loads"] == 1
    assert task.prompt in calls["prompt_text"]
    assert "Return only valid Python code." in calls["prompt_text"]
    assert "Define the function `is_even` exactly as requested." in calls["prompt_text"]
    assert "Do not include Markdown fences" in calls["prompt_text"]
    assert generation.prompt == calls["prompt_text"]
    assert generation.generation_text == "def is_even(n): return n % 2 == 0"
    assert generation.input_tokens > 0
    assert generation.output_tokens > 0
    assert generation.total_tokens == generation.input_tokens + generation.output_tokens
    assert generation.latency_seconds >= 0.0
    assert generation.model_name == "Qwen/Qwen3-0.6B"
    assert generation.provider_name == "huggingface"
    assert generation.metadata["model_id"] == "Qwen/Qwen3-0.6B"
    assert generation.metadata["device"] == "cpu"
    assert generation.metadata["dtype"] == "auto"
    assert generation.metadata["max_new_tokens"] == 12
    assert generation.metadata["temperature"] == 0.3
    assert generation.metadata["top_p"] == 0.8
    assert generation.metadata["do_sample"] is True
    assert generation.metadata["seed"] == 123
    assert generation.metadata["task_prompt"] == task.prompt
    assert generation.metadata["instruction_prompt"] != task.prompt
    assert generation.metadata["prompt_style"] == "code_only"
    assert generation.metadata["prompt_format"] == "plain"
    assert generation.sampling.max_output_tokens == 12
    assert generation.sampling.temperature == 0.3
    assert generation.sampling.top_p == 0.8
    assert generation.sampling.do_sample is True
    assert generation.sampling.seed == 123
    assert calls["seeds"] == [123]


def test_generate_accepts_sampling_override(monkeypatch: Any) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(monkeypatch, completion_text=" completion")
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider(max_new_tokens=100, temperature=0.0, top_p=1.0)
    generation = provider.generate(
        task,
        SamplingConfig(
            max_output_tokens=7,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            seed=9,
        ),
    )

    generate_kwargs = calls["generate_kwargs"]
    assert generate_kwargs["max_new_tokens"] == 7
    assert generate_kwargs["temperature"] == 0.5
    assert generate_kwargs["top_p"] == 0.9
    assert generate_kwargs["do_sample"] is True
    assert generation.metadata["max_new_tokens"] == 7
    assert generation.metadata["seed"] == 9


def test_generate_preserves_provider_sampling_when_budget_only_sets_tokens(
    monkeypatch: Any,
) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(monkeypatch, completion_text=" completion")
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
        seed=41,
    )

    generation = provider.generate(task, SamplingConfig(max_output_tokens=7))

    generate_kwargs = calls["generate_kwargs"]
    assert generate_kwargs["max_new_tokens"] == 7
    assert generate_kwargs["temperature"] == 0.7
    assert generate_kwargs["top_p"] == 0.8
    assert generate_kwargs["do_sample"] is True
    assert generation.metadata["seed"] == 41
    assert calls["seeds"] == [41]


def test_stochastic_generation_uses_distinct_deterministic_attempt_seeds(
    monkeypatch: Any,
) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(monkeypatch, completion_text=" completion")
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider(do_sample=True, seed=123)

    first = provider.generate(task)
    second = provider.generate(task)

    assert first.metadata["seed"] == 123
    assert second.metadata["seed"] == 124
    assert calls["seeds"] == [123, 124]


def test_second_generate_reuses_loaded_tokenizer_and_model(monkeypatch: Any) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(monkeypatch, completion_text=" completion")
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider()

    provider.generate(task)
    provider.generate(task)

    assert calls["tokenizer_loads"] == 1
    assert calls["model_loads"] == 1


def test_generate_uses_raw_prompt_for_non_code_tasks(monkeypatch: Any) -> None:
    task = Task(task_id="plain_task", prompt="Say hello.")
    calls = install_fake_transformers(monkeypatch, completion_text=" hello")
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider()

    generation = provider.generate(task)

    assert calls["prompt_text"] == "Say hello."
    assert generation.prompt == "Say hello."
    assert generation.generation_text == "hello"
    assert generation.metadata["task_prompt"] == "Say hello."
    assert generation.metadata["instruction_prompt"] == "Say hello."
    assert generation.metadata["prompt_style"] == "raw"


def test_generate_uses_chat_template_when_available(monkeypatch: Any) -> None:
    task = get_toy_task("is_even")
    calls = install_fake_transformers(
        monkeypatch,
        completion_text="def is_even(n): return n % 2 == 0",
        use_chat_template=True,
    )
    from ttc_operatorbench.models.hf_provider import HuggingFaceModelProvider

    provider = HuggingFaceModelProvider()

    generation = provider.generate(task)

    assert generation.prompt.startswith("<chat>")
    assert generation.prompt.endswith("<assistant>")
    assert calls["prompt_text"] == generation.prompt
    assert calls["chat_template_add_generation_prompt"] is True
    assert generation.generation_text == "def is_even(n): return n % 2 == 0"
    assert "Return only valid Python code." in calls["chat_template_messages"][0]["content"]
    assert generation.metadata["prompt_format"] == "chat_template"
