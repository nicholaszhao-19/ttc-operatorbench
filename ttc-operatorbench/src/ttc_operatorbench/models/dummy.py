"""Deterministic dummy model provider for verifier-first tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ttc_operatorbench.core.schema import Generation, SamplingConfig, Task

GenerationScript = str | Sequence[str]


def count_tokens(text: str) -> int:
    """Count simple whitespace-delimited tokens for deterministic toy accounting."""
    return len(text.split())


def _truncate_to_token_budget(text: str, max_output_tokens: int | None) -> str:
    if max_output_tokens is None:
        return text
    tokens = text.split()
    if len(tokens) <= max_output_tokens:
        return text
    return " ".join(tokens[:max_output_tokens])


@dataclass
class DummyModelProvider:
    """Static provider that returns configured generations without model inference."""

    generations_by_task_id: Mapping[str, GenerationScript] = field(default_factory=dict)
    provider_name: str = "dummy"
    model_name: str = "dummy-static"
    _calls_by_task_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def _next_generation_text(self, task_id: str) -> str:
        script = self.generations_by_task_id.get(task_id, "")
        if isinstance(script, str):
            return script

        call_index = self._calls_by_task_id.get(task_id, 0)
        self._calls_by_task_id[task_id] = call_index + 1
        if not script:
            return ""
        return script[min(call_index, len(script) - 1)]

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        """Return a deterministic generation for a task."""
        sampling_config = sampling or SamplingConfig()
        generation_text = _truncate_to_token_budget(
            self._next_generation_text(task.task_id),
            sampling_config.max_output_tokens,
        )
        input_tokens = count_tokens(task.prompt)
        output_tokens = count_tokens(generation_text)
        return Generation(
            prompt=task.prompt,
            generation_text=generation_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=0.0,
            sampling=sampling_config,
            model_name=self.model_name,
            provider_name=self.provider_name,
        )


__all__ = ["DummyModelProvider", "GenerationScript", "count_tokens"]
