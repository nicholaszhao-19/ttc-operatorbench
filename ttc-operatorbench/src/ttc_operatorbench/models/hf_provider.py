"""Hugging Face model provider for real prompt-in/text-out generations."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task

DEFAULT_HF_SMOKE_MODEL_ID = "Qwen/Qwen3-0.6B"
CODE_ONLY_PROMPT_STYLE = "code_only"
RAW_PROMPT_STYLE = "raw"


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def _strip_prompt_prefix(decoded_text: str, prompt: str) -> str:
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt) :].lstrip()
    return decoded_text.strip()


def _input_token_length(model_inputs: Any) -> int:
    """Return the prompt token length from a tokenizer batch."""
    input_ids = model_inputs.get("input_ids") if isinstance(model_inputs, dict) else None
    if input_ids is None:
        input_ids = getattr(model_inputs, "input_ids", None)
    if input_ids is None:
        return 0
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    try:
        return _sequence_length(input_ids[0])
    except (IndexError, KeyError, TypeError):
        return 0


def _generated_token_ids(output_ids: Any, *, input_length: int) -> Any:
    """Return only generated token ids when output includes the prompt prefix."""
    try:
        sequence = output_ids[0]
    except (IndexError, KeyError, TypeError):
        return output_ids
    if input_length > 0 and _sequence_length(sequence) > input_length:
        return sequence[input_length:]
    return sequence


def _sequence_length(sequence: Any) -> int:
    shape = getattr(sequence, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    return len(sequence)


def _instruction_prompt_for_task(task: Task) -> tuple[str, str]:
    """Return the user-facing instruction prompt and its prompt style."""
    entrypoint = task.metadata.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        return task.prompt, RAW_PROMPT_STYLE
    instruction = (
        f"{task.prompt}\n\n"
        "Return only valid Python code.\n"
        f"Define the function `{entrypoint}` exactly as requested.\n"
        "Do not include Markdown fences, explanations, examples, print calls, or tests."
    )
    return instruction, CODE_ONLY_PROMPT_STYLE


def _model_prompt_for_tokenizer(tokenizer: Any, instruction_prompt: str) -> tuple[str, str]:
    """Render a tokenizer-specific prompt when a chat template is available."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered_prompt = apply_chat_template(
                [{"role": "user", "content": instruction_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            rendered_prompt = None
        if isinstance(rendered_prompt, str) and rendered_prompt.strip():
            return rendered_prompt, "chat_template"
    return instruction_prompt, "plain"


def _torch_dtype(dtype: str | None) -> Any:
    if dtype is None or dtype == "auto":
        return "auto"
    torch = importlib.import_module("torch")
    dtype_aliases = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    dtype_name = dtype_aliases.get(dtype, dtype)
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return getattr(torch, dtype_name)


@dataclass
class HuggingFaceModelProvider:
    """Lazy-loading Hugging Face causal language model provider."""

    model_id: str = DEFAULT_HF_SMOKE_MODEL_ID
    device: str = "cpu"
    dtype: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int | None = None
    trust_remote_code: bool = False
    provider_name: str = "huggingface"
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        """Generate one text completion for a task."""
        tokenizer, model = self._load()
        sampling_config = self._resolve_sampling(sampling)
        self._set_seed(sampling_config.seed)
        instruction_prompt, prompt_style = _instruction_prompt_for_task(task)
        model_prompt, prompt_format = _model_prompt_for_tokenizer(tokenizer, instruction_prompt)

        model_inputs = tokenizer(model_prompt, return_tensors="pt")
        if self.device != "auto" and hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": sampling_config.max_output_tokens,
            "do_sample": sampling_config.do_sample,
        }
        if sampling_config.do_sample:
            generation_kwargs["temperature"] = sampling_config.temperature
            generation_kwargs["top_p"] = sampling_config.top_p
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        started_at = time.perf_counter()
        input_length = _input_token_length(model_inputs)
        output_ids = model.generate(**model_inputs, **generation_kwargs)
        latency_seconds = time.perf_counter() - started_at

        generated_ids = _generated_token_ids(output_ids, input_length=input_length)
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generation_text = _strip_prompt_prefix(decoded_text, model_prompt)
        input_tokens = _token_count(tokenizer, model_prompt)
        output_tokens = _token_count(tokenizer, generation_text)

        return Generation(
            prompt=model_prompt,
            generation_text=generation_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=latency_seconds,
            sampling=sampling_config,
            model_name=self.model_id,
            provider_name=self.provider_name,
            metadata={
                "model_id": self.model_id,
                "device": self.device,
                "dtype": self.dtype,
                "max_new_tokens": sampling_config.max_output_tokens,
                "temperature": sampling_config.temperature,
                "top_p": sampling_config.top_p,
                "do_sample": sampling_config.do_sample,
                "seed": sampling_config.seed,
                "trust_remote_code": self.trust_remote_code,
                "task_prompt": task.prompt,
                "instruction_prompt": instruction_prompt,
                "prompt_style": prompt_style,
                "prompt_format": prompt_format,
            },
        )

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        transformers = importlib.import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": _torch_dtype(self.dtype),
        }
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.device != "auto" and hasattr(self._model, "to"):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._tokenizer, self._model

    def _resolve_sampling(self, sampling: SamplingConfig | None) -> SamplingConfig:
        if sampling is None:
            return SamplingConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                seed=self.seed,
            )
        return SamplingConfig(
            max_output_tokens=self._bounded_max_new_tokens(sampling.max_output_tokens),
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            do_sample=sampling.do_sample,
            seed=sampling.seed if sampling.seed is not None else self.seed,
            stop_sequences=sampling.stop_sequences,
        )

    def _bounded_max_new_tokens(self, requested_tokens: int | None) -> int:
        if requested_tokens is None:
            return self.max_new_tokens
        return min(requested_tokens, self.max_new_tokens)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        transformers = importlib.import_module("transformers")
        set_seed = getattr(transformers, "set_seed", None)
        if set_seed is not None:
            set_seed(seed)


__all__ = [
    "CODE_ONLY_PROMPT_STYLE",
    "DEFAULT_HF_SMOKE_MODEL_ID",
    "HuggingFaceModelProvider",
    "RAW_PROMPT_STYLE",
]
