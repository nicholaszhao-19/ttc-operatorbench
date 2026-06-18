"""Shared cost accounting helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostRates:
    """Per-resource cost rates used for budget enforcement and reporting."""

    input_token_cost: float = 0.0
    output_token_cost: float = 0.0
    verifier_call_cost: float = 0.0
    fixed_attempt_cost: float = 0.0

    def generation_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        verifier_called: bool,
    ) -> float:
        """Return the realized cost for one generated attempt."""
        verifier_cost = self.verifier_call_cost if verifier_called else 0.0
        return (
            input_tokens * self.input_token_cost
            + output_tokens * self.output_token_cost
            + verifier_cost
            + self.fixed_attempt_cost
        )

    def estimated_attempt_cost(
        self,
        *,
        prompt_tokens: int,
        max_output_tokens: int | None,
        verifier_called: bool,
    ) -> float:
        """Return a conservative pre-generation cost estimate."""
        output_tokens = max_output_tokens or 0
        return self.generation_cost(
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            verifier_called=verifier_called,
        )

    def as_metadata(self) -> dict[str, float]:
        """Return JSON-serializable cost metadata."""
        return {
            "input_token_cost": self.input_token_cost,
            "output_token_cost": self.output_token_cost,
            "verifier_call_cost": self.verifier_call_cost,
            "fixed_attempt_cost": self.fixed_attempt_cost,
        }


def cost_rates_from_provider(provider: object) -> CostRates:
    """Read optional cost rates from a model provider."""
    return CostRates(
        input_token_cost=_nonnegative_float_attr(provider, "input_token_cost"),
        output_token_cost=_nonnegative_float_attr(provider, "output_token_cost"),
        verifier_call_cost=_nonnegative_float_attr(provider, "verifier_call_cost"),
        fixed_attempt_cost=_nonnegative_float_attr(provider, "fixed_attempt_cost"),
    )


def cost_rates_from_metadata(metadata: dict[str, Any]) -> CostRates:
    """Read optional cost rates from generation metadata."""
    return CostRates(
        input_token_cost=_nonnegative_float_value(metadata.get("input_token_cost")),
        output_token_cost=_nonnegative_float_value(metadata.get("output_token_cost")),
        verifier_call_cost=_nonnegative_float_value(metadata.get("verifier_call_cost")),
        fixed_attempt_cost=_nonnegative_float_value(metadata.get("fixed_attempt_cost")),
    )


def _nonnegative_float_attr(obj: object, name: str) -> float:
    return _nonnegative_float_value(getattr(obj, name, 0.0))


def _nonnegative_float_value(value: object) -> float:
    if isinstance(value, int | float):
        return max(float(value), 0.0)
    return 0.0


__all__ = ["CostRates", "cost_rates_from_metadata", "cost_rates_from_provider"]
