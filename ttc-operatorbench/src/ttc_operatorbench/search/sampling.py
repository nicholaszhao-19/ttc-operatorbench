"""Sampling helpers shared by search policies."""

from __future__ import annotations

import hashlib
from typing import Any

from ttc_operatorbench.core.schema import SamplingConfig

_MAX_SEED = 2_147_483_647


def stable_attempt_seed(base_seed: int | None, *parts: object) -> int | None:
    """Derive a deterministic seed from protocol seed and attempt context."""
    if base_seed is None:
        return None
    payload = "\x1f".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % _MAX_SEED
    return (base_seed + offset) % _MAX_SEED


def sampling_with_attempt_seed(
    sampling: SamplingConfig,
    provider: Any,
    *,
    run_id: str,
    task_id: str,
    policy_name: str,
    operator_name: str,
    attempt_number: int,
) -> SamplingConfig:
    """Add a stable attempt seed when the provider exposes a protocol seed."""
    if "seed" in sampling.model_fields_set:
        return sampling
    do_sample = (
        sampling.do_sample
        if "do_sample" in sampling.model_fields_set
        else getattr(provider, "do_sample", True)
    )
    if do_sample is False:
        return sampling
    base_seed = getattr(provider, "seed", None)
    if not isinstance(base_seed, int):
        return sampling
    seed = stable_attempt_seed(
        base_seed,
        run_id,
        task_id,
        policy_name,
        operator_name,
        attempt_number,
    )
    explicit_values = {
        field: getattr(sampling, field) for field in sampling.model_fields_set
    }
    explicit_values["seed"] = seed
    return SamplingConfig(**explicit_values)


__all__ = ["sampling_with_attempt_seed", "stable_attempt_seed"]
