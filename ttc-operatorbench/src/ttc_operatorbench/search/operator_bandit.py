"""Cost-normalized UCB scheduler over verifier-guided operators."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ttc_operatorbench.core.costing import (
    CostRates,
    cost_rates_from_metadata,
    cost_rates_from_provider,
)
from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    DecisionLog,
    Generation,
    SamplingConfig,
    SearchResult,
    Task,
    VerificationResult,
)
from ttc_operatorbench.models.dummy import count_tokens
from ttc_operatorbench.search.baselines import ModelProvider, Verifier

CostMetric = Literal["tokens", "verifier_calls", "wall_clock_seconds", "cost", "unit"]

DEFAULT_ERROR_TYPE_BONUSES: dict[str, dict[str, float]] = {
    "syntax_error": {"repair_from_error": 0.25},
    "wrong_answer": {"repair_from_error": 0.15, "plan_then_code": 0.10},
    "test_failure": {"repair_from_error": 0.15, "plan_then_code": 0.10},
    "timeout": {"local_revision": 0.15, "plan_then_code": 0.10},
    "duplicate": {"direct_sample": 0.10, "plan_then_code": 0.10},
    "no_progress": {"direct_sample": 0.10, "plan_then_code": 0.10},
}


@dataclass
class OperatorStatistics:
    """Running statistics for one operator arm."""

    operator_name: str
    n: int = 0
    successes: int = 0
    total_cost: float = 0.0
    last_error_type: str | None = None

    @property
    def mean_success(self) -> float:
        if self.n == 0:
            return 0.0
        return self.successes / self.n

    @property
    def mean_cost(self) -> float:
        if self.n == 0:
            return 0.0
        return self.total_cost / self.n

    def update(self, *, success: bool, cost: float, error_type: str | None) -> None:
        self.n += 1
        self.successes += int(success)
        self.total_cost += cost
        self.last_error_type = error_type

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "operator_name": self.operator_name,
            "n": self.n,
            "successes": self.successes,
            "total_cost": self.total_cost,
            "mean_success": self.mean_success,
            "mean_cost": self.mean_cost,
            "last_error_type": self.last_error_type,
        }


@dataclass(frozen=True)
class OperatorStepResult:
    """Attempts produced by one operator application."""

    attempts: tuple[AttemptLog, ...]
    success: bool
    error_type: str | None


class SearchOperator(Protocol):
    """One selectable operator arm."""

    name: str

    def can_run(self, context: OperatorContext) -> bool:
        """Return whether the operator can spend budget in the current state."""

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        """Apply the operator and return generated attempts."""


@dataclass
class _BanditLedger:
    budget: Budget
    cost_rates: CostRates = CostRates()
    attempts: int = 0
    tokens: int = 0
    verifier_calls: int = 0
    seconds: float = 0.0
    cost: float = 0.0

    def has_capacity(
        self,
        *,
        prompt: str,
        requires_verifier: bool,
        min_attempts: int = 1,
    ) -> bool:
        if (
            self.budget.max_attempts is not None
            and self.attempts + min_attempts > self.budget.max_attempts
        ):
            return False
        if (
            requires_verifier
            and self.budget.max_verifier_calls is not None
            and self.verifier_calls >= self.budget.max_verifier_calls
        ):
            return False
        if self.budget.max_seconds is not None and self.seconds >= self.budget.max_seconds:
            return False
        if self.budget.max_tokens is not None and not (
            self.tokens + count_tokens(prompt) < self.budget.max_tokens
        ):
            return False
        if self.budget.max_cost is None:
            return True
        sampling = self.sampling_for(prompt)
        estimated_cost = self.cost_rates.estimated_attempt_cost(
            prompt_tokens=count_tokens(prompt),
            max_output_tokens=sampling.max_output_tokens,
            verifier_called=requires_verifier,
        )
        return self.cost + estimated_cost <= self.budget.max_cost

    def sampling_for(self, prompt: str) -> SamplingConfig:
        if self.budget.max_tokens is None:
            return SamplingConfig()
        remaining_output_tokens = self.budget.max_tokens - self.tokens - count_tokens(prompt)
        return SamplingConfig(max_output_tokens=max(1, remaining_output_tokens))

    def can_record_synthetic(self, *, total_tokens: int, verifier_called: bool) -> bool:
        if self.budget.max_attempts is not None and self.attempts >= self.budget.max_attempts:
            return False
        if (
            verifier_called
            and self.budget.max_verifier_calls is not None
            and self.verifier_calls >= self.budget.max_verifier_calls
        ):
            return False
        if (
            self.budget.max_tokens is not None
            and self.tokens + total_tokens > self.budget.max_tokens
        ):
            return False
        if self.budget.max_seconds is not None and self.seconds >= self.budget.max_seconds:
            return False
        if self.budget.max_cost is not None:
            synthetic_cost = self.cost_rates.generation_cost(
                input_tokens=total_tokens,
                output_tokens=0,
                verifier_called=verifier_called,
            )
            if self.cost + synthetic_cost > self.budget.max_cost:
                return False
        return True

    def record(
        self,
        generation: Generation,
        *,
        verifier_elapsed: float,
        verifier_called: bool,
    ) -> None:
        self.attempts += 1
        self.tokens += generation.total_tokens
        if verifier_called:
            self.verifier_calls += 1
        self.seconds += generation.latency_seconds + verifier_elapsed
        cost_rates = cost_rates_from_metadata(generation.metadata)
        if cost_rates == CostRates():
            cost_rates = self.cost_rates
        self.cost += cost_rates.generation_cost(
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            verifier_called=verifier_called,
        )

    def record_synthetic(self, *, total_tokens: int, verifier_called: bool) -> None:
        self.attempts += 1
        self.tokens += total_tokens
        if verifier_called:
            self.verifier_calls += 1
        self.cost += self.cost_rates.generation_cost(
            input_tokens=total_tokens,
            output_tokens=0,
            verifier_called=verifier_called,
        )


@dataclass
class OperatorContext:
    """Mutable state shared by operators during one scheduler run."""

    task: Task
    provider: ModelProvider
    verifier: Verifier
    budget: Budget
    run_id: str
    policy_name: str
    ledger: _BanditLedger
    attempts: list[AttemptLog] = field(default_factory=list)
    decision_log: list[DecisionLog] = field(default_factory=list)
    last_attempt: AttemptLog | None = None
    last_error_type: str | None = None

    def can_continue(self) -> bool:
        if (
            self.budget.max_attempts is not None
            and self.ledger.attempts >= self.budget.max_attempts
        ):
            return False
        if (
            self.budget.max_verifier_calls is not None
            and self.ledger.verifier_calls >= self.budget.max_verifier_calls
        ):
            return False
        if self.budget.max_tokens is not None and self.ledger.tokens >= self.budget.max_tokens:
            return False
        if self.budget.max_seconds is not None and self.ledger.seconds >= self.budget.max_seconds:
            return False
        if self.budget.max_cost is not None and self.ledger.cost >= self.budget.max_cost:
            return False
        return True

    def can_generate(self, prompt: str, *, requires_verifier: bool) -> bool:
        return self.ledger.has_capacity(prompt=prompt, requires_verifier=requires_verifier)

    def task_with_prompt(self, prompt: str) -> Task:
        return self.task.model_copy(update={"prompt": prompt})

    def step_since(self, start_index: int) -> OperatorStepResult:
        attempts = tuple(self.attempts[start_index:])
        return OperatorStepResult(
            attempts=attempts,
            success=any(attempt.verification_passed for attempt in attempts),
            error_type=attempts[-1].error_type if attempts else None,
        )

    def generate_attempt(
        self,
        *,
        prompt: str,
        operator_name: str,
        verify: bool,
        unverified_error_type: str | None = None,
    ) -> AttemptLog | None:
        if not self.can_generate(prompt, requires_verifier=verify):
            return None

        task_for_prompt = self.task_with_prompt(prompt)
        generation = self.provider.generate(task_for_prompt, self.ledger.sampling_for(prompt))
        if (
            self.budget.max_tokens is not None
            and self.ledger.tokens + generation.total_tokens > self.budget.max_tokens
        ):
            return None

        verifier_elapsed = 0.0
        verifier_called = verify
        if verify:
            started_at = time.perf_counter()
            verification = self.verifier.verify_generation(self.task, generation)
            verifier_elapsed = time.perf_counter() - started_at
            verification = verification.model_copy(update={"latency_seconds": verifier_elapsed})
        else:
            verification = VerificationResult(
                verification_passed=False,
                verification_score=0.0,
                error_type=unverified_error_type,
                failure_category="unverified_plan"
                if unverified_error_type == "not_verified_plan"
                else None,
            )

        self.ledger.record(
            generation,
            verifier_elapsed=verifier_elapsed,
            verifier_called=verifier_called,
        )
        attempt_id = (
            f"{self.run_id}:{self.task.task_id}:{self.policy_name}:{len(self.attempts) + 1}"
        )
        attempt = AttemptLog(
            attempt_id=attempt_id,
            task_id=self.task.task_id,
            model_id=generation.model_name,
            operator_name=operator_name,
            prompt=generation.prompt,
            generation_text=generation.generation_text,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            total_tokens=generation.total_tokens,
            latency_seconds=generation.latency_seconds,
            verification_passed=verification.verification_passed,
            verification_score=verification.verification_score,
            public_verification=verification,
            error_type=verification.error_type,
            stdout=verification.stdout,
            stderr=verification.stderr,
            cumulative_tokens=self.ledger.tokens,
            cumulative_verifier_calls=self.ledger.verifier_calls,
            cumulative_seconds=self.ledger.seconds,
            cumulative_cost=self.ledger.cost,
            verifier_seconds=verifier_elapsed,
            selected=verification.verification_passed,
            run_id=self.run_id,
            policy_name=self.policy_name,
            provider_name=generation.provider_name,
            metadata=generation.metadata,
        )
        self._append_attempt(attempt)
        return attempt

    def record_synthetic_attempt(
        self,
        *,
        operator_name: str,
        total_tokens: int,
        verification_passed: bool,
        error_type: str | None = None,
        verifier_called: bool = True,
    ) -> AttemptLog | None:
        if not self.ledger.can_record_synthetic(
            total_tokens=total_tokens,
            verifier_called=verifier_called,
        ):
            return None
        self.ledger.record_synthetic(total_tokens=total_tokens, verifier_called=verifier_called)
        attempt_id = (
            f"{self.run_id}:{self.task.task_id}:{self.policy_name}:{len(self.attempts) + 1}"
        )
        attempt = AttemptLog(
            attempt_id=attempt_id,
            task_id=self.task.task_id,
            model_id="synthetic",
            operator_name=operator_name,
            prompt=self.task.prompt,
            generation_text=f"synthetic attempt from {operator_name}",
            input_tokens=total_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
            latency_seconds=0.0,
            verification_passed=verification_passed,
            verification_score=1.0 if verification_passed else 0.0,
            public_verification=VerificationResult(
                verification_passed=verification_passed,
                verification_score=1.0 if verification_passed else 0.0,
                scope="public",
                error_type=error_type,
                failure_category="success" if verification_passed else error_type,
            ),
            error_type=error_type,
            failure_category="success" if verification_passed else error_type,
            cumulative_tokens=self.ledger.tokens,
            cumulative_verifier_calls=self.ledger.verifier_calls,
            cumulative_seconds=self.ledger.seconds,
            cumulative_cost=self.ledger.cost,
            verifier_seconds=0.0,
            selected=verification_passed,
            run_id=self.run_id,
            policy_name=self.policy_name,
            provider_name="synthetic",
        )
        self._append_attempt(attempt)
        return attempt

    def _append_attempt(self, attempt: AttemptLog) -> None:
        self.attempts.append(attempt)
        self.last_attempt = attempt
        self.last_error_type = attempt.error_type


class DirectSampleOperator:
    """Generate and verify one direct candidate."""

    name = "direct_sample"

    def can_run(self, context: OperatorContext) -> bool:
        return context.can_generate(context.task.prompt, requires_verifier=True)

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        start_index = len(context.attempts)
        context.generate_attempt(prompt=context.task.prompt, operator_name=self.name, verify=True)
        return context.step_since(start_index)


class RepairFromErrorOperator:
    """Repair the latest candidate using verifier error text."""

    name = "repair_from_error"

    def can_run(self, context: OperatorContext) -> bool:
        if context.last_attempt is None:
            return False
        return context.can_generate(self._prompt(context), requires_verifier=True)

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        start_index = len(context.attempts)
        context.generate_attempt(prompt=self._prompt(context), operator_name=self.name, verify=True)
        return context.step_since(start_index)

    def _prompt(self, context: OperatorContext) -> str:
        assert context.last_attempt is not None
        return (
            f"{context.task.prompt}\n\nPrevious candidate:\n```python\n"
            f"{context.last_attempt.generation_text}\n```\n\n"
            f"Verifier error type: {context.last_attempt.error_type}\n"
            f"Verifier stderr:\n{context.last_attempt.stderr}\n\n"
            "Return repaired Python code only."
        )


class PlanThenCodeOperator:
    """Generate an unverified plan, then generate and verify code from it."""

    name = "plan_then_code"

    def can_run(self, context: OperatorContext) -> bool:
        return context.can_generate(self._plan_prompt(context), requires_verifier=False)

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        start_index = len(context.attempts)
        plan_attempt = context.generate_attempt(
            prompt=self._plan_prompt(context),
            operator_name="plan_then_code/plan",
            verify=False,
            unverified_error_type="not_verified_plan",
        )
        if plan_attempt is not None:
            code_prompt = (
                f"{context.task.prompt}\n\nPlan:\n{plan_attempt.generation_text}\n\n"
                "Return Python code only."
            )
            context.generate_attempt(
                prompt=code_prompt,
                operator_name="plan_then_code/code",
                verify=True,
            )
        return context.step_since(start_index)

    def _plan_prompt(self, context: OperatorContext) -> str:
        return f"{context.task.prompt}\n\nFirst produce a concise implementation plan."


class LocalRevisionOperator:
    """Revise the current local candidate using verifier feedback."""

    name = "local_revision"

    def can_run(self, context: OperatorContext) -> bool:
        if context.last_attempt is None:
            return False
        return context.can_generate(self._prompt(context), requires_verifier=True)

    def apply(self, context: OperatorContext) -> OperatorStepResult:
        start_index = len(context.attempts)
        context.generate_attempt(prompt=self._prompt(context), operator_name=self.name, verify=True)
        return context.step_since(start_index)

    def _prompt(self, context: OperatorContext) -> str:
        assert context.last_attempt is not None
        return (
            f"{context.task.prompt}\n\nCurrent local candidate:\n```python\n"
            f"{context.last_attempt.generation_text}\n```\n\n"
            f"Verifier feedback:\n{context.last_attempt.stderr}\n\n"
            "Revise the local candidate. Return Python code only."
        )


class OperatorBanditScheduler:
    """Cost-normalized UCB scheduler over verifier-guided operators."""

    def __init__(
        self,
        operators: tuple[SearchOperator, ...] | None = None,
        *,
        exploration_weight: float = 1.0,
        epsilon: float = 1e-6,
        cost_metric: CostMetric = "tokens",
        error_type_bonuses: dict[str, dict[str, float]] | None = None,
        policy_name: str = "operator_bandit",
    ):
        if exploration_weight < 0:
            raise ValueError("exploration_weight must be nonnegative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.operators = operators or (
            DirectSampleOperator(),
            RepairFromErrorOperator(),
            PlanThenCodeOperator(),
            LocalRevisionOperator(),
        )
        if not self.operators:
            raise ValueError("at least one operator is required")
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon
        self.cost_metric = cost_metric
        self.error_type_bonuses = (
            DEFAULT_ERROR_TYPE_BONUSES if error_type_bonuses is None else error_type_bonuses
        )
        self.policy_name = policy_name
        self.operator_statistics = {
            operator.name: OperatorStatistics(operator.name) for operator in self.operators
        }

    def reset_statistics(self) -> None:
        """Reset learned operator statistics."""
        self.operator_statistics = {
            operator.name: OperatorStatistics(operator.name) for operator in self.operators
        }

    def select_operator(self, context: OperatorContext) -> SearchOperator | None:
        """Select a valid operator using cost-normalized UCB."""
        valid_operators = self._valid_operators(context)
        return self._choose_operator(valid_operators, context)

    def _valid_operators(self, context: OperatorContext) -> list[SearchOperator]:
        return [operator for operator in self.operators if operator.can_run(context)]

    def _choose_operator(
        self,
        valid_operators: list[SearchOperator],
        context: OperatorContext,
    ) -> SearchOperator | None:
        if not valid_operators:
            return None

        if self.exploration_weight > 0:
            untried = [
                operator
                for operator in valid_operators
                if self.operator_statistics[operator.name].n == 0
            ]
            if untried:
                return max(
                    untried,
                    key=lambda operator: self._error_bonus(operator.name, context.last_error_type),
                )

        decision_step = max(1, sum(stat.n for stat in self.operator_statistics.values()) + 1)
        return max(
            valid_operators,
            key=lambda operator: self.operator_score(
                operator.name,
                decision_step,
                context.last_error_type,
            ),
        )

    def operator_scores_for_context(
        self,
        valid_operators: list[SearchOperator],
        context: OperatorContext,
    ) -> dict[str, float]:
        """Return selection scores for valid operators in the current context."""
        decision_step = max(1, sum(stat.n for stat in self.operator_statistics.values()) + 1)
        return {
            operator.name: self.operator_score(
                operator.name,
                decision_step,
                context.last_error_type,
            )
            for operator in valid_operators
        }

    def operator_score(
        self,
        operator_name: str,
        decision_step: int,
        last_error_type: str | None = None,
    ) -> float:
        """Return cost-normalized UCB score for one operator."""
        stats = self.operator_statistics[operator_name]
        exploitation = stats.mean_success / max(stats.mean_cost, self.epsilon)
        exploration = self.exploration_weight * math.sqrt(
            math.log(decision_step + 1) / (1 + stats.n)
        )
        return exploitation + exploration + self._error_bonus(operator_name, last_error_type)

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "operator-bandit-run",
    ) -> SearchResult:
        """Run the adaptive operator scheduler for one task."""
        context = OperatorContext(
            task=task,
            provider=provider,
            verifier=verifier,
            budget=budget,
            run_id=run_id,
            policy_name=self.policy_name,
            ledger=_BanditLedger(budget=budget, cost_rates=cost_rates_from_provider(provider)),
        )

        while context.can_continue():
            valid_operators = self._valid_operators(context)
            operator = self._choose_operator(valid_operators, context)
            if operator is None:
                break
            decision_start = _decision_log_start(
                context=context,
                step_index=len(context.decision_log) + 1,
                chosen_operator_name=operator.name,
                valid_operator_names=tuple(
                    valid_operator.name for valid_operator in valid_operators
                ),
                operator_scores=self.operator_scores_for_context(valid_operators, context),
            )

            before_tokens = context.ledger.tokens
            before_calls = context.ledger.verifier_calls
            before_seconds = context.ledger.seconds
            before_cost = context.ledger.cost
            step = operator.apply(context)
            if not step.attempts:
                break

            cost = self._step_cost(
                before_tokens=before_tokens,
                before_calls=before_calls,
                before_seconds=before_seconds,
                before_cost=before_cost,
                context=context,
            )
            self.operator_statistics[operator.name].update(
                success=step.success,
                cost=cost,
                error_type=step.error_type,
            )
            context.decision_log.append(
                _decision_log_complete(
                    decision_start,
                    context=context,
                    produced_attempts=step.attempts,
                    before_tokens=before_tokens,
                    before_calls=before_calls,
                    before_seconds=before_seconds,
                    before_cost=before_cost,
                    outcome_success=step.success,
                    outcome_error_type=step.error_type,
                )
            )
            if step.success:
                break

        selected_attempt_id = next(
            (attempt.attempt_id for attempt in context.attempts if attempt.selected),
            None,
        )
        return SearchResult(
            task_id=task.task_id,
            policy_name=self.policy_name,
            budget=budget,
            attempts=tuple(context.attempts),
            decision_log=tuple(context.decision_log),
            selected_attempt_id=selected_attempt_id,
            success=selected_attempt_id is not None,
            total_tokens=context.ledger.tokens,
            total_verifier_calls=context.ledger.verifier_calls,
            total_seconds=context.ledger.seconds,
            total_cost=context.ledger.cost,
            metadata={
                "cost_metric": self.cost_metric,
                "exploration_weight": self.exploration_weight,
                "epsilon": self.epsilon,
                "operator_statistics": self.operator_statistics_as_dict(),
            },
        )

    def operator_statistics_as_dict(self) -> dict[str, dict[str, int | float | str | None]]:
        """Return JSON-serializable operator statistics."""
        return {name: stats.as_dict() for name, stats in self.operator_statistics.items()}

    def _step_cost(
        self,
        *,
        before_tokens: int,
        before_calls: int,
        before_seconds: float,
        before_cost: float,
        context: OperatorContext,
    ) -> float:
        if self.cost_metric == "tokens":
            token_cost = context.ledger.tokens - before_tokens
            if token_cost > 0:
                return float(token_cost)
            call_cost = context.ledger.verifier_calls - before_calls
            return float(max(call_cost, 1))
        if self.cost_metric == "verifier_calls":
            return float(max(context.ledger.verifier_calls - before_calls, 1))
        if self.cost_metric == "cost":
            return max(context.ledger.cost - before_cost, self.epsilon)
        if self.cost_metric == "unit":
            return 1.0
        return max(context.ledger.seconds - before_seconds, self.epsilon)

    def _error_bonus(self, operator_name: str, last_error_type: str | None) -> float:
        if last_error_type is None:
            return 0.0
        return self.error_type_bonuses.get(last_error_type, {}).get(operator_name, 0.0)


class FixedOperatorOrderScheduler:
    """Ablation that cycles through operators in a fixed order without learning."""

    def __init__(
        self,
        operators: tuple[SearchOperator, ...] | None = None,
        *,
        policy_name: str = "fixed_operator_order",
    ):
        self.operators = operators or (
            DirectSampleOperator(),
            RepairFromErrorOperator(),
            PlanThenCodeOperator(),
            LocalRevisionOperator(),
        )
        if not self.operators:
            raise ValueError("at least one operator is required")
        self.policy_name = policy_name
        self._next_index = 0

    def select_operator(self, context: OperatorContext) -> SearchOperator | None:
        """Return the next valid operator in cyclic fixed order."""
        for offset in range(len(self.operators)):
            index = (self._next_index + offset) % len(self.operators)
            operator = self.operators[index]
            if operator.can_run(context):
                self._next_index = (index + 1) % len(self.operators)
                return operator
        return None

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "fixed-order-run",
    ) -> SearchResult:
        """Run the fixed operator-order ablation for one task."""
        self._next_index = 0
        context = OperatorContext(
            task=task,
            provider=provider,
            verifier=verifier,
            budget=budget,
            run_id=run_id,
            policy_name=self.policy_name,
            ledger=_BanditLedger(budget=budget, cost_rates=cost_rates_from_provider(provider)),
        )

        while context.can_continue():
            valid_operators = [operator for operator in self.operators if operator.can_run(context)]
            operator = self.select_operator(context)
            if operator is None:
                break
            decision_start = _decision_log_start(
                context=context,
                step_index=len(context.decision_log) + 1,
                chosen_operator_name=operator.name,
                valid_operator_names=tuple(
                    valid_operator.name for valid_operator in valid_operators
                ),
                operator_scores={},
            )
            before_tokens = context.ledger.tokens
            before_calls = context.ledger.verifier_calls
            before_seconds = context.ledger.seconds
            before_cost = context.ledger.cost
            step = operator.apply(context)
            if not step.attempts:
                break
            context.decision_log.append(
                _decision_log_complete(
                    decision_start,
                    context=context,
                    produced_attempts=step.attempts,
                    before_tokens=before_tokens,
                    before_calls=before_calls,
                    before_seconds=before_seconds,
                    before_cost=before_cost,
                    outcome_success=step.success,
                    outcome_error_type=step.error_type,
                )
            )
            if step.success:
                break

        selected_attempt_id = next(
            (attempt.attempt_id for attempt in context.attempts if attempt.selected),
            None,
        )
        return SearchResult(
            task_id=task.task_id,
            policy_name=self.policy_name,
            budget=budget,
            attempts=tuple(context.attempts),
            decision_log=tuple(context.decision_log),
            selected_attempt_id=selected_attempt_id,
            success=selected_attempt_id is not None,
            total_tokens=context.ledger.tokens,
            total_verifier_calls=context.ledger.verifier_calls,
            total_seconds=context.ledger.seconds,
            total_cost=context.ledger.cost,
            metadata={
                "operator_order": tuple(operator.name for operator in self.operators),
                "ablation": "fixed_operator_order",
            },
        )


def _decision_log_start(
    *,
    context: OperatorContext,
    step_index: int,
    chosen_operator_name: str,
    valid_operator_names: tuple[str, ...],
    operator_scores: dict[str, float],
) -> DecisionLog:
    last_attempt = context.last_attempt
    previous_error_type = context.last_error_type
    previous_failure_category = (
        _failure_category_for_attempt(last_attempt) if last_attempt is not None else "no_attempt"
    )
    return DecisionLog(
        decision_id=f"{context.run_id}:{context.task.task_id}:{context.policy_name}:decision:{step_index}",
        task_id=context.task.task_id,
        policy_name=context.policy_name,
        run_id=context.run_id,
        step_index=step_index,
        chosen_operator_name=chosen_operator_name,
        valid_operator_names=valid_operator_names,
        previous_operator_name=last_attempt.operator_name if last_attempt is not None else None,
        previous_error_type=previous_error_type,
        previous_failure_category=previous_failure_category,
        repeated_error_count=_trailing_error_count(context.attempts, previous_error_type),
        state_attempts=context.ledger.attempts,
        state_tokens=context.ledger.tokens,
        state_verifier_calls=context.ledger.verifier_calls,
        state_seconds=context.ledger.seconds,
        state_cost=context.ledger.cost,
        remaining_attempts=_remaining_int(context.budget.max_attempts, context.ledger.attempts),
        remaining_tokens=_remaining_int(context.budget.max_tokens, context.ledger.tokens),
        remaining_verifier_calls=_remaining_int(
            context.budget.max_verifier_calls,
            context.ledger.verifier_calls,
        ),
        remaining_seconds=_remaining_float(context.budget.max_seconds, context.ledger.seconds),
        remaining_cost=_remaining_float(context.budget.max_cost, context.ledger.cost),
        operator_scores=operator_scores,
    )


def _decision_log_complete(
    decision: DecisionLog,
    *,
    context: OperatorContext,
    produced_attempts: tuple[AttemptLog, ...],
    before_tokens: int,
    before_calls: int,
    before_seconds: float,
    before_cost: float,
    outcome_success: bool,
    outcome_error_type: str | None,
) -> DecisionLog:
    return decision.model_copy(
        update={
            "produced_attempt_ids": tuple(attempt.attempt_id for attempt in produced_attempts),
            "produced_attempt_count": len(produced_attempts),
            "delta_tokens": context.ledger.tokens - before_tokens,
            "delta_verifier_calls": context.ledger.verifier_calls - before_calls,
            "delta_seconds": max(context.ledger.seconds - before_seconds, 0.0),
            "delta_cost": max(context.ledger.cost - before_cost, 0.0),
            "outcome_success": outcome_success,
            "outcome_error_type": outcome_error_type,
            "outcome_failure_category": _failure_category_for_error(outcome_error_type),
            "budget_exhausted_after": not context.can_continue(),
        }
    )


def _remaining_int(limit: int | None, used: int) -> int | None:
    if limit is None:
        return None
    return max(limit - used, 0)


def _remaining_float(limit: float | None, used: float) -> float | None:
    if limit is None:
        return None
    return max(limit - used, 0.0)


def _trailing_error_count(attempts: list[AttemptLog], error_type: str | None) -> int:
    if error_type is None:
        return 0
    count = 0
    for attempt in reversed(attempts):
        if attempt.error_type != error_type:
            break
        count += 1
    return count


def _failure_category_for_attempt(attempt: AttemptLog) -> str:
    if attempt.failure_category is not None:
        return attempt.failure_category
    if attempt.verification_passed:
        return "success"
    return _failure_category_for_error(attempt.error_type)


def _failure_category_for_error(error_type: str | None) -> str:
    if error_type is None:
        return "success"
    if error_type == "not_verified_plan":
        return "unverified_plan"
    if error_type in {"syntax_error", "parse_error"}:
        return "syntax_or_parse_error"
    if error_type == "runtime_error":
        return "runtime_error"
    if error_type == "timeout":
        return "timeout"
    if error_type in {"empty_code", "empty_generation", "no_code"}:
        return "empty_or_non_code"
    if error_type.startswith("missing_"):
        return "missing_tests"
    return "public_test_failure"


__all__ = [
    "CostMetric",
    "DEFAULT_ERROR_TYPE_BONUSES",
    "DirectSampleOperator",
    "FixedOperatorOrderScheduler",
    "LocalRevisionOperator",
    "OperatorBanditScheduler",
    "OperatorContext",
    "OperatorStatistics",
    "OperatorStepResult",
    "PlanThenCodeOperator",
    "RepairFromErrorOperator",
    "SearchOperator",
]
