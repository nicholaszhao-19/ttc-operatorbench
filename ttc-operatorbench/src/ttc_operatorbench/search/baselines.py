"""Verifier-guided baseline search policies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ttc_operatorbench.core.costing import (
    CostRates,
    cost_rates_from_metadata,
    cost_rates_from_provider,
)
from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    Generation,
    SamplingConfig,
    SearchResult,
    Task,
    VerificationResult,
)
from ttc_operatorbench.models.dummy import count_tokens


class ModelProvider(Protocol):
    """Minimal model provider contract used by baseline policies."""

    def generate(self, task: Task, sampling: SamplingConfig | None = None) -> Generation:
        """Generate one candidate for a task."""


class Verifier(Protocol):
    """Minimal verifier contract used by baseline policies."""

    def verify_generation(self, task: Task, generation: Generation) -> VerificationResult:
        """Verify one structured generation for a task."""


def best_of_n_success_probability(one_sample_success_probability: float, n: int) -> float:
    """Return ``1 - (1 - p) ** n`` for independent best-of-N sampling."""
    if not 0.0 <= one_sample_success_probability <= 1.0:
        raise ValueError("one_sample_success_probability must be in [0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - (1.0 - one_sample_success_probability) ** n


@dataclass
class _BudgetLedger:
    budget: Budget
    cost_rates: CostRates = CostRates()
    attempts: int = 0
    tokens: int = 0
    verifier_calls: int = 0
    seconds: float = 0.0
    cost: float = 0.0

    def can_generate(self, prompt: str, *, requires_verifier: bool) -> bool:
        if self.budget.max_attempts is not None and self.attempts >= self.budget.max_attempts:
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
        if remaining_output_tokens <= 0:
            return SamplingConfig(max_output_tokens=1)
        return SamplingConfig(max_output_tokens=remaining_output_tokens)

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


@dataclass(frozen=True)
class _AttemptRecord:
    generation: Generation
    verification: VerificationResult
    attempt_log: AttemptLog


class BaselinePolicy:
    """Base class for deterministic verifier-guided baseline policies."""

    name: str

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        """Run the policy for one task."""
        raise NotImplementedError

    def _task_with_prompt(self, task: Task, prompt: str) -> Task:
        return task.model_copy(update={"prompt": prompt})

    def _ledger_for(self, budget: Budget, provider: ModelProvider) -> _BudgetLedger:
        return _BudgetLedger(budget=budget, cost_rates=cost_rates_from_provider(provider))

    def _generate(
        self,
        task: Task,
        provider: ModelProvider,
        ledger: _BudgetLedger,
        *,
        operator_name: str,
        requires_verifier: bool,
    ) -> Generation | None:
        if not ledger.can_generate(task.prompt, requires_verifier=requires_verifier):
            return None
        return provider.generate(task, ledger.sampling_for(task.prompt))

    def _verify(
        self,
        task: Task,
        verifier: Verifier,
        generation: Generation,
    ) -> tuple[VerificationResult, float]:
        started_at = time.perf_counter()
        verification = verifier.verify_generation(task, generation)
        verifier_elapsed = time.perf_counter() - started_at
        verification = verification.model_copy(update={"latency_seconds": verifier_elapsed})
        return verification, verifier_elapsed

    def _record_attempt(
        self,
        task: Task,
        ledger: _BudgetLedger,
        generation: Generation,
        verification: VerificationResult,
        *,
        operator_name: str,
        attempt_number: int,
        run_id: str,
        verifier_elapsed: float,
        verifier_called: bool,
        selected: bool,
    ) -> _AttemptRecord:
        ledger.record(
            generation,
            verifier_elapsed=verifier_elapsed,
            verifier_called=verifier_called,
        )
        attempt_log = AttemptLog(
            attempt_id=f"{run_id}:{task.task_id}:{self.name}:{attempt_number}",
            task_id=task.task_id,
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
            cumulative_tokens=ledger.tokens,
            cumulative_verifier_calls=ledger.verifier_calls,
            cumulative_seconds=ledger.seconds,
            cumulative_cost=ledger.cost,
            verifier_seconds=verifier_elapsed,
            selected=selected,
            run_id=run_id,
            policy_name=self.name,
            provider_name=generation.provider_name,
            metadata=generation.metadata,
        )
        return _AttemptRecord(generation, verification, attempt_log)

    def _verified_attempt(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        ledger: _BudgetLedger,
        *,
        operator_name: str,
        attempt_number: int,
        run_id: str,
    ) -> _AttemptRecord | None:
        generation = self._generate(
            task,
            provider,
            ledger,
            operator_name=operator_name,
            requires_verifier=True,
        )
        if generation is None:
            return None
        verification, verifier_elapsed = self._verify(task, verifier, generation)
        return self._record_attempt(
            task,
            ledger,
            generation,
            verification,
            operator_name=operator_name,
            attempt_number=attempt_number,
            run_id=run_id,
            verifier_elapsed=verifier_elapsed,
            verifier_called=True,
            selected=verification.verification_passed,
        )

    def _unverified_attempt(
        self,
        task: Task,
        provider: ModelProvider,
        ledger: _BudgetLedger,
        *,
        operator_name: str,
        attempt_number: int,
        run_id: str,
        error_type: str,
    ) -> _AttemptRecord | None:
        generation = self._generate(
            task,
            provider,
            ledger,
            operator_name=operator_name,
            requires_verifier=False,
        )
        if generation is None:
            return None
        verification = VerificationResult(
            verification_passed=False,
            verification_score=0.0,
            error_type=error_type,
            failure_category="unverified_plan" if error_type == "not_verified_plan" else None,
        )
        return self._record_attempt(
            task,
            ledger,
            generation,
            verification,
            operator_name=operator_name,
            attempt_number=attempt_number,
            run_id=run_id,
            verifier_elapsed=0.0,
            verifier_called=False,
            selected=False,
        )

    def _result(
        self,
        task: Task,
        budget: Budget,
        attempts: list[AttemptLog],
    ) -> SearchResult:
        selected_attempt_id = next(
            (attempt.attempt_id for attempt in attempts if attempt.selected),
            None,
        )
        final_attempt = attempts[-1] if attempts else None
        return SearchResult(
            task_id=task.task_id,
            policy_name=self.name,
            budget=budget,
            attempts=tuple(attempts),
            selected_attempt_id=selected_attempt_id,
            success=selected_attempt_id is not None,
            total_tokens=final_attempt.cumulative_tokens if final_attempt is not None else 0,
            total_verifier_calls=(
                final_attempt.cumulative_verifier_calls if final_attempt is not None else 0
            ),
            total_seconds=final_attempt.cumulative_seconds if final_attempt is not None else 0.0,
            total_cost=final_attempt.cumulative_cost if final_attempt is not None else 0.0,
        )


class GreedyPolicy(BaselinePolicy):
    """One generation, one verification."""

    name = "greedy"

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        ledger = self._ledger_for(budget, provider)
        attempts: list[AttemptLog] = []
        record = self._verified_attempt(
            task,
            provider,
            verifier,
            ledger,
            operator_name="greedy",
            attempt_number=1,
            run_id=run_id,
        )
        if record is not None:
            attempts.append(record.attempt_log)
        return self._result(task, budget, attempts)


class BestOfNPolicy(BaselinePolicy):
    """Repeated independent samples, stopping at the first verified success."""

    name = "best_of_n"

    def __init__(self, n: int = 4):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        ledger = self._ledger_for(budget, provider)
        attempts: list[AttemptLog] = []
        for attempt_number in range(1, self.n + 1):
            record = self._verified_attempt(
                task,
                provider,
                verifier,
                ledger,
                operator_name="best_of_n",
                attempt_number=attempt_number,
                run_id=run_id,
            )
            if record is None:
                break
            attempts.append(record.attempt_log)
            if record.verification.verification_passed:
                break
        return self._result(task, budget, attempts)


class RepairOnlyPolicy(BaselinePolicy):
    """Draft code, then repair using public verifier feedback."""

    name = "repair_only"

    def __init__(self, max_repairs: int = 1):
        if max_repairs < 0:
            raise ValueError("max_repairs must be nonnegative")
        self.max_repairs = max_repairs

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        ledger = self._ledger_for(budget, provider)
        attempts: list[AttemptLog] = []
        record = self._verified_attempt(
            task,
            provider,
            verifier,
            ledger,
            operator_name="repair_only/draft",
            attempt_number=1,
            run_id=run_id,
        )
        if record is None:
            return self._result(task, budget, attempts)
        attempts.append(record.attempt_log)
        previous = record

        for repair_index in range(1, self.max_repairs + 1):
            if previous.verification.verification_passed:
                break
            repair_prompt = (
                f"{task.prompt}\n\nPrevious candidate:\n```python\n"
                f"{previous.generation.generation_text}\n```\n\n"
                f"Verifier error type: {previous.verification.error_type}\n"
                f"Verifier stderr:\n{previous.verification.stderr}\n\n"
                "Return repaired Python code only."
            )
            repair_task = self._task_with_prompt(task, repair_prompt)
            repair_record = self._verified_attempt(
                repair_task,
                provider,
                verifier,
                ledger,
                operator_name="repair_only/repair",
                attempt_number=repair_index + 1,
                run_id=run_id,
            )
            if repair_record is None:
                break
            previous = repair_record
            attempts.append(previous.attempt_log)
        return self._result(task, budget, attempts)


class PlanThenCodePolicy(BaselinePolicy):
    """Generate a plan, then generate code conditioned on that plan."""

    name = "plan_then_code"

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        ledger = self._ledger_for(budget, provider)
        attempts: list[AttemptLog] = []
        plan_task = self._task_with_prompt(
            task,
            f"{task.prompt}\n\nFirst produce a concise implementation plan.",
        )
        plan_record = self._unverified_attempt(
            plan_task,
            provider,
            ledger,
            operator_name="plan_then_code/plan",
            attempt_number=1,
            run_id=run_id,
            error_type="not_verified_plan",
        )
        if plan_record is None:
            return self._result(task, budget, attempts)
        attempts.append(plan_record.attempt_log)

        code_prompt = (
            f"{task.prompt}\n\nPlan:\n{plan_record.generation.generation_text}\n\n"
            "Return Python code only."
        )
        code_task = self._task_with_prompt(task, code_prompt)
        code_record = self._verified_attempt(
            code_task,
            provider,
            verifier,
            ledger,
            operator_name="plan_then_code/code",
            attempt_number=2,
            run_id=run_id,
        )
        if code_record is not None:
            attempts.append(code_record.attempt_log)
        return self._result(task, budget, attempts)


class LocalRevisionBasicPolicy(BaselinePolicy):
    """Draft code, then revise the local candidate with verifier feedback."""

    name = "local_revision_basic"

    def __init__(self, max_revisions: int = 1):
        if max_revisions < 0:
            raise ValueError("max_revisions must be nonnegative")
        self.max_revisions = max_revisions

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "toy-run",
    ) -> SearchResult:
        ledger = self._ledger_for(budget, provider)
        attempts: list[AttemptLog] = []
        record = self._verified_attempt(
            task,
            provider,
            verifier,
            ledger,
            operator_name="local_revision_basic/draft",
            attempt_number=1,
            run_id=run_id,
        )
        if record is None:
            return self._result(task, budget, attempts)
        attempts.append(record.attempt_log)
        previous = record

        for revision_index in range(1, self.max_revisions + 1):
            if previous.verification.verification_passed:
                break
            revision_prompt = (
                f"{task.prompt}\n\nCurrent local candidate:\n```python\n"
                f"{previous.generation.generation_text}\n```\n\n"
                f"Verifier feedback:\n{previous.verification.stderr}\n\n"
                "Revise the local candidate. Return Python code only."
            )
            revision_task = self._task_with_prompt(task, revision_prompt)
            revision_record = self._verified_attempt(
                revision_task,
                provider,
                verifier,
                ledger,
                operator_name="local_revision_basic/revise",
                attempt_number=revision_index + 1,
                run_id=run_id,
            )
            if revision_record is None:
                break
            previous = revision_record
            attempts.append(previous.attempt_log)
        return self._result(task, budget, attempts)


__all__ = [
    "BaselinePolicy",
    "BestOfNPolicy",
    "GreedyPolicy",
    "LocalRevisionBasicPolicy",
    "ModelProvider",
    "PlanThenCodePolicy",
    "RepairOnlyPolicy",
    "Verifier",
    "best_of_n_success_probability",
]
