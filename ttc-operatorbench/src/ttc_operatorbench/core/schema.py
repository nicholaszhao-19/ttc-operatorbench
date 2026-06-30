"""Core schemas for auditable operator search traces."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
VerificationScope = Literal["public", "hidden"]


class SchemaModel(BaseModel):
    """Shared behavior for immutable, strict benchmark schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingConfig(SchemaModel):
    """Sampling parameters used to generate a candidate attempt."""

    temperature: NonNegativeFloat = Field(default=0.0, le=2.0)
    top_p: PositiveFloat = Field(default=1.0, le=1.0)
    do_sample: bool = False
    max_output_tokens: PositiveInt | None = None
    seed: NonNegativeInt | None = None
    stop_sequences: tuple[str, ...] = ()


class Generation(SchemaModel):
    """Raw model generation and generation-side accounting."""

    prompt: NonEmptyStr
    generation_text: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    latency_seconds: NonNegativeFloat
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model_name: str | None = None
    provider_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class Task(SchemaModel):
    """Benchmark task inputs available before candidate generation."""

    task_id: NonEmptyStr
    prompt: NonEmptyStr
    public_tests: tuple[str, ...] = ()
    hidden_tests: tuple[str, ...] = ()
    task_family: str | None = None
    difficulty_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_verifier_inputs: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(SchemaModel):
    """Verifier output allowed to influence search decisions."""

    verification_passed: bool
    verification_score: Score
    scope: VerificationScope | None = None
    verifier_name: str | None = None
    latency_seconds: NonNegativeFloat = 0.0
    failure_category: str | None = None
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None


class Budget(SchemaModel):
    """Explicit resource limits for cost-aware search."""

    max_attempts: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    max_verifier_calls: PositiveInt | None = None
    max_seconds: PositiveFloat | None = None
    max_cost: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_has_limit(self) -> Self:
        limits = (
            self.max_attempts,
            self.max_tokens,
            self.max_verifier_calls,
            self.max_seconds,
            self.max_cost,
        )
        if all(limit is None for limit in limits):
            raise ValueError("at least one budget limit must be set")
        return self


class AttemptLog(SchemaModel):
    """Append-only record for one generated attempt and its observed costs."""

    attempt_id: NonEmptyStr
    task_id: NonEmptyStr
    model_id: str | None = None
    operator_name: NonEmptyStr
    prompt: NonEmptyStr
    generation_text: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    latency_seconds: NonNegativeFloat
    verification_passed: bool
    verification_score: Score
    public_verification: VerificationResult | None = None
    hidden_verification: VerificationResult | None = None
    error_type: str | None = None
    stdout: str = ""
    stderr: str = ""
    cumulative_tokens: NonNegativeInt
    cumulative_verifier_calls: NonNegativeInt
    cumulative_seconds: NonNegativeFloat
    cumulative_cost: NonNegativeFloat = 0.0
    verifier_seconds: NonNegativeFloat = 0.0
    failure_category: str | None = None
    selected: bool = False
    run_id: str | None = None
    policy_name: str | None = None
    provider_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trace_accounting(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.cumulative_tokens < self.total_tokens:
            raise ValueError("cumulative_tokens must be at least total_tokens")
        if self.cumulative_seconds < self.latency_seconds:
            raise ValueError("cumulative_seconds must be at least latency_seconds")
        if self.public_verification is not None:
            if self.public_verification.scope not in (None, "public"):
                raise ValueError("public_verification scope must be public")
            if self.public_verification.verification_passed != self.verification_passed:
                raise ValueError("public_verification must match verification_passed")
            if self.public_verification.verification_score != self.verification_score:
                raise ValueError("public_verification must match verification_score")
        if self.hidden_verification is not None and self.hidden_verification.scope not in (
            None,
            "hidden",
        ):
            raise ValueError("hidden_verification scope must be hidden")
        return self


class OperatorResult(SchemaModel):
    """Result returned by applying one operator to one task."""

    operator_name: NonEmptyStr
    generation: Generation
    verification: VerificationResult
    attempt_log: AttemptLog

    @model_validator(mode="after")
    def validate_operator_consistency(self) -> Self:
        if self.operator_name != self.attempt_log.operator_name:
            raise ValueError("operator_name must match attempt_log.operator_name")
        if self.generation.prompt != self.attempt_log.prompt:
            raise ValueError("generation.prompt must match attempt_log.prompt")
        if self.generation.generation_text != self.attempt_log.generation_text:
            raise ValueError("generation_text must match attempt_log.generation_text")
        if self.verification.verification_passed != self.attempt_log.verification_passed:
            raise ValueError("verification_passed must match attempt_log")
        if self.verification.verification_score != self.attempt_log.verification_score:
            raise ValueError("verification_score must match attempt_log")
        return self


class DecisionLog(SchemaModel):
    """State-action record for one operator-selection decision."""

    decision_id: NonEmptyStr
    task_id: NonEmptyStr
    policy_name: NonEmptyStr
    run_id: str | None = None
    step_index: PositiveInt
    chosen_operator_name: NonEmptyStr
    valid_operator_names: tuple[NonEmptyStr, ...]
    previous_operator_name: str | None = None
    previous_error_type: str | None = None
    previous_failure_category: str | None = None
    repeated_error_count: NonNegativeInt = 0
    state_attempts: NonNegativeInt
    state_tokens: NonNegativeInt
    state_verifier_calls: NonNegativeInt
    state_seconds: NonNegativeFloat
    state_cost: NonNegativeFloat = 0.0
    remaining_attempts: NonNegativeInt | None = None
    remaining_tokens: NonNegativeInt | None = None
    remaining_verifier_calls: NonNegativeInt | None = None
    remaining_seconds: NonNegativeFloat | None = None
    remaining_cost: NonNegativeFloat | None = None
    operator_scores: dict[str, float] = Field(default_factory=dict)
    produced_attempt_ids: tuple[str, ...] = ()
    produced_attempt_count: NonNegativeInt = 0
    delta_tokens: NonNegativeInt = 0
    delta_verifier_calls: NonNegativeInt = 0
    delta_seconds: NonNegativeFloat = 0.0
    delta_cost: NonNegativeFloat = 0.0
    outcome_success: bool = False
    outcome_error_type: str | None = None
    outcome_failure_category: str | None = None
    budget_exhausted_after: bool = False


class SearchResult(SchemaModel):
    """Final result for a task search, preserving every attempted trace."""

    task_id: NonEmptyStr
    policy_name: NonEmptyStr
    budget: Budget
    attempts: tuple[AttemptLog, ...] = ()
    decision_log: tuple[DecisionLog, ...] = ()
    selected_attempt_id: str | None = None
    success: bool = False
    total_tokens: NonNegativeInt = 0
    total_verifier_calls: NonNegativeInt = 0
    total_seconds: NonNegativeFloat = 0.0
    total_cost: NonNegativeFloat = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selected_attempt(self) -> Self:
        attempts_by_id = {attempt.attempt_id: attempt for attempt in self.attempts}
        if len(attempts_by_id) != len(self.attempts):
            raise ValueError("attempt_id values must be unique")
        if self.selected_attempt_id is None:
            if self.success:
                raise ValueError("successful search results require a selected_attempt_id")
            return self
        selected = attempts_by_id.get(self.selected_attempt_id)
        if selected is None:
            raise ValueError("selected_attempt_id must reference an attempt")
        if selected.task_id != self.task_id:
            raise ValueError("selected attempt task_id must match SearchResult.task_id")
        if not selected.selected:
            raise ValueError("selected attempt must have selected=True")
        if self.success and not selected.verification_passed:
            raise ValueError("successful search results require a passing selected attempt")
        return self


__all__ = [
    "AttemptLog",
    "Budget",
    "DecisionLog",
    "Generation",
    "OperatorResult",
    "SamplingConfig",
    "SearchResult",
    "Task",
    "VerificationScope",
    "VerificationResult",
]
