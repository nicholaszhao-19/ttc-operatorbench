"""Differential-selection policies for code-generation test-time compute."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ttc_operatorbench.core.schema import (
    AttemptLog,
    Budget,
    Generation,
    SamplingConfig,
    SearchResult,
    Task,
)
from ttc_operatorbench.models.dummy import count_tokens
from ttc_operatorbench.search.baselines import ModelProvider, Verifier
from ttc_operatorbench.tasks.toy_code import ENTRYPOINT_KEY
from ttc_operatorbench.verifiers.python_unit_tests import extract_python_code

BehaviorTrace = tuple[str, ...]
ControllerRegime = Literal["coverage_failure", "selection_failure", "stop_confident"]

_BEHAVIOR_MARKER = "__TTC_BEHAVIOR__="


@dataclass(frozen=True)
class ProbeCall:
    """One candidate-discriminating function call."""

    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()

    def as_literal(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return a Python-literal representation for subprocess execution."""
        return self.args, dict(self.kwargs)


@dataclass(frozen=True)
class DifferentialSelection:
    """Behavior-clustering selection result."""

    selected_index: int
    clusters: tuple[tuple[int, ...], ...]
    distance_matrix: tuple[tuple[float, ...], ...]

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        return tuple(len(cluster) for cluster in self.clusters)

    @property
    def largest_cluster_ratio(self) -> float:
        if not self.clusters:
            return 0.0
        total = sum(len(cluster) for cluster in self.clusters)
        if total == 0:
            return 0.0
        return max(len(cluster) for cluster in self.clusters) / total

    @property
    def top_two_cluster_margin(self) -> float:
        if not self.clusters:
            return 0.0
        sizes = sorted((len(cluster) for cluster in self.clusters), reverse=True)
        top = sizes[0]
        second = sizes[1] if len(sizes) > 1 else 0
        total = sum(sizes)
        if total == 0:
            return 0.0
        return (top - second) / total


@dataclass
class _SelectionLedger:
    budget: Budget
    attempts: int = 0
    tokens: int = 0
    verifier_calls: int = 0
    seconds: float = 0.0

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
        if self.budget.max_tokens is None:
            return True
        return self.tokens + count_tokens(prompt) < self.budget.max_tokens

    def sampling_for(self, prompt: str) -> SamplingConfig:
        if self.budget.max_tokens is None:
            return SamplingConfig(seed_offset=self.attempts)
        remaining_output_tokens = self.budget.max_tokens - self.tokens - count_tokens(prompt)
        return SamplingConfig(
            max_output_tokens=max(1, remaining_output_tokens),
            seed_offset=self.attempts,
        )

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


class DifferentialSelectionPolicy:
    """DiffCodeGen-style behavior-clustering candidate selector.

    This is a lightweight baseline, not a full DiffCodeGen reproduction: it uses
    deterministic probe calls derived from visible task call shapes instead of a
    coverage-guided fuzzer.
    """

    name = "diffcodegen_select"

    def __init__(
        self,
        *,
        n: int = 4,
        probe_limit: int = 16,
        behavior_timeout_seconds: float = 1.0,
        policy_name: str = "diffcodegen_select",
    ):
        if n <= 0:
            raise ValueError("n must be positive")
        if probe_limit <= 0:
            raise ValueError("probe_limit must be positive")
        if behavior_timeout_seconds <= 0:
            raise ValueError("behavior_timeout_seconds must be positive")
        self.n = n
        self.probe_limit = probe_limit
        self.behavior_timeout_seconds = behavior_timeout_seconds
        self.name = policy_name

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "diffcodegen-select-run",
    ) -> SearchResult:
        """Generate candidates, cluster behavior traces, and select one medoid."""
        ledger = _SelectionLedger(budget)
        attempts = _generate_verified_attempts(
            task,
            provider,
            verifier,
            ledger,
            n=self.n,
            run_id=run_id,
            policy_name=self.name,
            operator_name="diffcodegen_select/sample",
        )
        return _select_with_behavior(
            task,
            attempts,
            budget,
            ledger,
            policy_name=self.name,
            probe_limit=self.probe_limit,
            behavior_timeout_seconds=self.behavior_timeout_seconds,
            selection_source="diffcodegen_select",
        )


class BottleneckAwareControllerPolicy:
    """Rule-based controller over sampling, repair, differential selection, and stop."""

    name = "bottleneck_controller"

    def __init__(
        self,
        *,
        min_samples: int = 2,
        max_samples: int = 4,
        probe_limit: int = 16,
        behavior_timeout_seconds: float = 1.0,
        consensus_threshold: float = 0.75,
    ):
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if max_samples < min_samples:
            raise ValueError("max_samples must be at least min_samples")
        if probe_limit <= 0:
            raise ValueError("probe_limit must be positive")
        if behavior_timeout_seconds <= 0:
            raise ValueError("behavior_timeout_seconds must be positive")
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0, 1]")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.probe_limit = probe_limit
        self.behavior_timeout_seconds = behavior_timeout_seconds
        self.consensus_threshold = consensus_threshold

    def run(
        self,
        task: Task,
        provider: ModelProvider,
        verifier: Verifier,
        budget: Budget,
        *,
        run_id: str = "bottleneck-controller-run",
    ) -> SearchResult:
        """Route compute based on a simple coverage-vs-selection bottleneck test."""
        ledger = _SelectionLedger(budget)
        attempts = list(
            _generate_verified_attempts(
                task,
                provider,
                verifier,
                ledger,
                n=self.min_samples,
                run_id=run_id,
                policy_name=self.name,
                operator_name="bottleneck_controller/sample",
            )
        )
        actions: list[str] = [f"sample:{len(attempts)}"]
        analysis = _analyze_attempts(
            task,
            tuple(attempts),
            probe_limit=self.probe_limit,
            behavior_timeout_seconds=self.behavior_timeout_seconds,
        )
        regime = _classify_bottleneck(
            tuple(attempts),
            analysis.selection,
            consensus_threshold=self.consensus_threshold,
        )
        actions.append(f"classify:{regime}")

        if regime == "coverage_failure":
            repair = _generate_repair_attempt(
                task,
                provider,
                verifier,
                ledger,
                attempts,
                run_id=run_id,
                policy_name=self.name,
            )
            if repair is not None:
                attempts.append(repair)
                actions.append("repair")
        elif regime == "selection_failure":
            extra_needed = self.max_samples - len(attempts)
            if extra_needed > 0:
                extra_attempts = _generate_verified_attempts(
                    task,
                    provider,
                    verifier,
                    ledger,
                    n=extra_needed,
                    run_id=run_id,
                    policy_name=self.name,
                    operator_name="bottleneck_controller/sample_more",
                    start_attempt_number=len(attempts) + 1,
                )
                attempts.extend(extra_attempts)
                actions.append(f"sample_more:{len(extra_attempts)}")
        else:
            actions.append("stop_early")

        result = _select_with_behavior(
            task,
            tuple(attempts),
            budget,
            ledger,
            policy_name=self.name,
            probe_limit=self.probe_limit,
            behavior_timeout_seconds=self.behavior_timeout_seconds,
            selection_source="bottleneck_controller",
            public_tie_break=True,
        )
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "controller_actions": tuple(actions),
                    "initial_regime": regime,
                    "min_samples": self.min_samples,
                    "max_samples": self.max_samples,
                    "consensus_threshold": self.consensus_threshold,
                }
            }
        )


@dataclass(frozen=True)
class _AttemptAnalysis:
    probes: tuple[ProbeCall, ...]
    traces: tuple[BehaviorTrace, ...]
    selection: DifferentialSelection | None
    elapsed_seconds: float


def _generate_verified_attempts(
    task: Task,
    provider: ModelProvider,
    verifier: Verifier,
    ledger: _SelectionLedger,
    *,
    n: int,
    run_id: str,
    policy_name: str,
    operator_name: str,
    start_attempt_number: int = 1,
) -> tuple[AttemptLog, ...]:
    attempts: list[AttemptLog] = []
    for offset in range(n):
        attempt = _generate_verified_attempt(
            task,
            provider,
            verifier,
            ledger,
            run_id=run_id,
            policy_name=policy_name,
            operator_name=operator_name,
            attempt_number=start_attempt_number + offset,
            prompt=task.prompt,
        )
        if attempt is None:
            break
        attempts.append(attempt)
    return tuple(attempts)


def _generate_repair_attempt(
    task: Task,
    provider: ModelProvider,
    verifier: Verifier,
    ledger: _SelectionLedger,
    attempts: list[AttemptLog],
    *,
    run_id: str,
    policy_name: str,
) -> AttemptLog | None:
    if not attempts:
        return None
    latest = attempts[-1]
    prompt = (
        f"{task.prompt}\n\nPrevious candidate:\n```python\n"
        f"{latest.generation_text}\n```\n\n"
        f"Verifier error type: {latest.error_type}\n"
        f"Verifier stderr:\n{latest.stderr}\n\n"
        "Return repaired Python code only."
    )
    return _generate_verified_attempt(
        task,
        provider,
        verifier,
        ledger,
        run_id=run_id,
        policy_name=policy_name,
        operator_name="bottleneck_controller/repair",
        attempt_number=len(attempts) + 1,
        prompt=prompt,
    )


def _generate_verified_attempt(
    task: Task,
    provider: ModelProvider,
    verifier: Verifier,
    ledger: _SelectionLedger,
    *,
    run_id: str,
    policy_name: str,
    operator_name: str,
    attempt_number: int,
    prompt: str,
) -> AttemptLog | None:
    if not ledger.can_generate(prompt, requires_verifier=True):
        return None
    task_for_prompt = task.model_copy(update={"prompt": prompt})
    generation = provider.generate(task_for_prompt, ledger.sampling_for(prompt))
    if (
        ledger.budget.max_tokens is not None
        and ledger.tokens + generation.total_tokens > ledger.budget.max_tokens
    ):
        return None

    started_at = time.perf_counter()
    verification = verifier.verify_generation(task, generation)
    verifier_elapsed = time.perf_counter() - started_at
    ledger.record(generation, verifier_elapsed=verifier_elapsed, verifier_called=True)
    return AttemptLog(
        attempt_id=f"{run_id}:{task.task_id}:{policy_name}:{attempt_number}",
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
        selected=False,
        run_id=run_id,
        policy_name=policy_name,
        provider_name=generation.provider_name,
        metadata=generation.metadata,
    )


def _select_with_behavior(
    task: Task,
    attempts: tuple[AttemptLog, ...],
    budget: Budget,
    ledger: _SelectionLedger,
    *,
    policy_name: str,
    probe_limit: int,
    behavior_timeout_seconds: float,
    selection_source: str,
    public_tie_break: bool = False,
) -> SearchResult:
    if not attempts:
        return SearchResult(
            task_id=task.task_id,
            policy_name=policy_name,
            budget=budget,
            attempts=(),
            success=False,
            total_tokens=ledger.tokens,
            total_verifier_calls=ledger.verifier_calls,
            total_seconds=ledger.seconds,
            metadata={"selection_source": selection_source},
        )

    analysis = _analyze_attempts(
        task,
        attempts,
        probe_limit=probe_limit,
        behavior_timeout_seconds=behavior_timeout_seconds,
    )
    selected_index = 0
    selection_metadata: dict[str, Any] = {
        "selection_source": selection_source,
        "candidate_count": len(attempts),
        "probe_count": len(analysis.probes),
        "behavior_elapsed_seconds": analysis.elapsed_seconds,
    }
    if analysis.selection is not None:
        selected_index = analysis.selection.selected_index
        selection_metadata.update(
            {
                "selected_candidate_index": selected_index,
                "cluster_sizes": analysis.selection.cluster_sizes,
                "largest_cluster_ratio": analysis.selection.largest_cluster_ratio,
                "top_two_cluster_margin": analysis.selection.top_two_cluster_margin,
            }
        )

    override_reason: str | None = None
    if public_tie_break and not attempts[selected_index].verification_passed:
        public_passes = [
            index for index, attempt in enumerate(attempts) if attempt.verification_passed
        ]
        if len(public_passes) == 1:
            selected_index = public_passes[0]
            override_reason = "single_public_pass"

    selected_attempts = tuple(
        _attempt_with_selection_and_trace(
            attempt,
            selected=index == selected_index,
            behavior_trace=analysis.traces[index] if index < len(analysis.traces) else (),
        )
        for index, attempt in enumerate(attempts)
    )
    selected_attempt = selected_attempts[selected_index]
    if override_reason is not None:
        selection_metadata["selection_override"] = override_reason
        selection_metadata["selected_candidate_index"] = selected_index

    return SearchResult(
        task_id=task.task_id,
        policy_name=policy_name,
        budget=budget,
        attempts=selected_attempts,
        selected_attempt_id=selected_attempt.attempt_id,
        success=selected_attempt.verification_passed,
        total_tokens=ledger.tokens,
        total_verifier_calls=ledger.verifier_calls,
        total_seconds=ledger.seconds + analysis.elapsed_seconds,
        metadata=selection_metadata,
    )


def _attempt_with_selection_and_trace(
    attempt: AttemptLog,
    *,
    selected: bool,
    behavior_trace: BehaviorTrace,
) -> AttemptLog:
    return attempt.model_copy(
        update={
            "selected": selected,
            "metadata": {
                **attempt.metadata,
                "behavior_trace": behavior_trace,
            },
        }
    )


def _analyze_attempts(
    task: Task,
    attempts: tuple[AttemptLog, ...],
    *,
    probe_limit: int,
    behavior_timeout_seconds: float,
) -> _AttemptAnalysis:
    probes = build_probe_calls(task, limit=probe_limit)
    started_at = time.perf_counter()
    traces = tuple(
        collect_behavior_trace(
            task,
            attempt.generation_text,
            probes,
            timeout_seconds=behavior_timeout_seconds,
        )
        for attempt in attempts
    )
    elapsed_seconds = time.perf_counter() - started_at
    selection = select_consensus_medoid(traces) if traces else None
    return _AttemptAnalysis(
        probes=probes,
        traces=traces,
        selection=selection,
        elapsed_seconds=elapsed_seconds,
    )


def _classify_bottleneck(
    attempts: tuple[AttemptLog, ...],
    selection: DifferentialSelection | None,
    *,
    consensus_threshold: float,
) -> ControllerRegime:
    if not attempts:
        return "coverage_failure"
    public_passes = sum(1 for attempt in attempts if attempt.verification_passed)
    if public_passes == 0:
        return "coverage_failure"
    if selection is None:
        return "selection_failure"
    if selection.largest_cluster_ratio < consensus_threshold:
        return "selection_failure"
    return "stop_confident"


def build_probe_calls(task: Task, *, limit: int = 16) -> tuple[ProbeCall, ...]:
    """Build deterministic behavior probes from policy-visible call shapes."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    entrypoint = _task_entrypoint(task)
    calls: list[ProbeCall] = []
    if entrypoint is not None:
        for test in task.public_tests:
            calls.extend(_calls_from_test(test, entrypoint=entrypoint))
    if not calls:
        calls.append(ProbeCall(args=()))

    probes: list[ProbeCall] = []
    seen: set[str] = set()
    for call in calls:
        for probe in (call, *_mutate_call(call)):
            key = repr(probe.as_literal())
            if key in seen:
                continue
            seen.add(key)
            probes.append(probe)
            if len(probes) >= limit:
                return tuple(probes)
    return tuple(probes)


def collect_behavior_trace(
    task: Task,
    candidate_text: str,
    probes: tuple[ProbeCall, ...],
    *,
    timeout_seconds: float = 1.0,
) -> BehaviorTrace:
    """Execute one candidate over probes and return normalized behavior tokens."""
    if not probes:
        return ()
    entrypoint = _task_entrypoint(task)
    if entrypoint is None:
        return tuple("error:missing_entrypoint" for _ in probes)
    candidate_code = extract_python_code(candidate_text, entrypoint=entrypoint)
    script = _behavior_script(candidate_code, entrypoint=entrypoint, probes=probes)
    with tempfile.TemporaryDirectory(prefix="ttc_diffselect_") as tmp_dir:
        script_path = Path(tmp_dir) / "behavior.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return tuple("error:timeout" for _ in probes)

    marker_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_BEHAVIOR_MARKER)
        ),
        None,
    )
    if marker_line is None:
        stderr = completed.stderr.strip().splitlines()[-1:] or [""]
        return tuple(f"error:process:{completed.returncode}:{stderr[0]}" for _ in probes)
    try:
        raw_results = json.loads(marker_line.removeprefix(_BEHAVIOR_MARKER))
    except json.JSONDecodeError:
        return tuple("error:invalid_behavior_json" for _ in probes)
    if not isinstance(raw_results, list):
        return tuple("error:invalid_behavior_shape" for _ in probes)
    normalized: list[str] = []
    for item in raw_results:
        if isinstance(item, dict):
            normalized.append(_normalize_behavior_item(item))
        else:
            normalized.append(f"error:invalid_behavior_item:{type(item).__name__}")
    if len(normalized) < len(probes):
        normalized.extend("error:missing_behavior" for _ in range(len(probes) - len(normalized)))
    return tuple(normalized[: len(probes)])


def select_consensus_medoid(traces: tuple[BehaviorTrace, ...]) -> DifferentialSelection:
    """Cluster behavior traces into consensus/minority clusters and select a medoid."""
    if not traces:
        raise ValueError("at least one behavior trace is required")
    distance_matrix = _distance_matrix(traces)
    if len(traces) == 1:
        return DifferentialSelection(
            selected_index=0,
            clusters=((0,),),
            distance_matrix=distance_matrix,
        )
    if len(set(traces)) == 1:
        cluster = tuple(range(len(traces)))
        return DifferentialSelection(
            selected_index=0,
            clusters=(cluster,),
            distance_matrix=distance_matrix,
        )

    clusters = _agglomerative_clusters(distance_matrix, target_clusters=2)
    consensus = max(
        clusters,
        key=lambda cluster: (len(cluster), -_cluster_medoid_distance(cluster, distance_matrix)),
    )
    selected_index = _cluster_medoid(consensus, distance_matrix)
    return DifferentialSelection(
        selected_index=selected_index,
        clusters=clusters,
        distance_matrix=distance_matrix,
    )


def _task_entrypoint(task: Task) -> str | None:
    entrypoint = task.allowed_verifier_inputs.get(ENTRYPOINT_KEY)
    if isinstance(entrypoint, str):
        return entrypoint
    metadata_entrypoint = task.metadata.get(ENTRYPOINT_KEY)
    return metadata_entrypoint if isinstance(metadata_entrypoint, str) else None


def _calls_from_test(test: str, *, entrypoint: str) -> tuple[ProbeCall, ...]:
    try:
        module = ast.parse(test)
    except SyntaxError:
        return ()
    calls: list[ProbeCall] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != entrypoint:
            continue
        args = _literal_values(node.args)
        kwargs = _literal_keywords(node.keywords)
        if args is None or kwargs is None:
            continue
        calls.append(ProbeCall(args=args, kwargs=kwargs))
    return tuple(calls)


def _literal_values(nodes: list[ast.expr]) -> tuple[Any, ...] | None:
    values: list[Any] = []
    for node in nodes:
        try:
            values.append(ast.literal_eval(node))
        except (ValueError, SyntaxError):
            return None
    return tuple(values)


def _literal_keywords(nodes: list[ast.keyword]) -> tuple[tuple[str, Any], ...] | None:
    values: list[tuple[str, Any]] = []
    for node in nodes:
        if node.arg is None:
            return None
        try:
            values.append((node.arg, ast.literal_eval(node.value)))
        except (ValueError, SyntaxError):
            return None
    return tuple(values)


def _mutate_call(call: ProbeCall) -> tuple[ProbeCall, ...]:
    mutated: list[ProbeCall] = []
    for index, value in enumerate(call.args):
        for replacement in _mutations_for_value(value):
            args = list(call.args)
            args[index] = replacement
            mutated.append(ProbeCall(args=tuple(args), kwargs=call.kwargs))
    for key, value in call.kwargs:
        for replacement in _mutations_for_value(value):
            kwargs = tuple(
                (existing_key, replacement if existing_key == key else existing_value)
                for existing_key, existing_value in call.kwargs
            )
            mutated.append(ProbeCall(args=call.args, kwargs=kwargs))
    return tuple(mutated)


def _mutations_for_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, bool):
        return (not value,)
    if isinstance(value, int):
        return (-1, 0, 1, value + 1)
    if isinstance(value, str):
        return ("", value.upper(), f"{value}{value}" if value else "x")
    if isinstance(value, list):
        reversed_value = list(reversed(value))
        return ([], reversed_value)
    if isinstance(value, tuple):
        return ((), tuple(reversed(value)))
    if isinstance(value, dict):
        return ({},)
    if value is None:
        return (0, "")
    return ()


def _behavior_script(candidate_code: str, *, entrypoint: str, probes: tuple[ProbeCall, ...]) -> str:
    probe_literals = tuple(probe.as_literal() for probe in probes)
    return (
        f"{candidate_code}\n\n"
        "import json as __ttc_json\n"
        f"__ttc_probes = {probe_literals!r}\n"
        f"__ttc_fn = globals().get({entrypoint!r})\n"
        "__ttc_results = []\n"
        "for __ttc_args, __ttc_kwargs in __ttc_probes:\n"
        "    try:\n"
        "        if not callable(__ttc_fn):\n"
        "            raise NameError('entrypoint is not callable')\n"
        "        __ttc_value = __ttc_fn(*__ttc_args, **__ttc_kwargs)\n"
        "        __ttc_results.append({'ok': True, 'value': repr(__ttc_value)})\n"
        "    except BaseException as __ttc_exc:\n"
        "        __ttc_results.append({\n"
        "            'ok': False,\n"
        "            'error': type(__ttc_exc).__name__,\n"
        "            'message': str(__ttc_exc)[:200],\n"
        "        })\n"
        f"print({(_BEHAVIOR_MARKER)!r} + __ttc_json.dumps(__ttc_results, sort_keys=True))\n"
    )


def _normalize_behavior_item(item: dict[str, Any]) -> str:
    if item.get("ok") is True:
        return f"ok:{item.get('value', '')}"
    error = item.get("error", "unknown")
    message = item.get("message", "")
    return f"err:{error}:{message}"


def _distance_matrix(traces: tuple[BehaviorTrace, ...]) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for left in traces:
        row = tuple(_behavior_distance(left, right) for right in traces)
        rows.append(row)
    return tuple(rows)


def _behavior_distance(left: BehaviorTrace, right: BehaviorTrace) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 1.0
    mismatches = sum(1 for index in range(width) if left[index] != right[index])
    length_penalty = abs(len(left) - len(right))
    return (mismatches + length_penalty) / max(len(left), len(right), 1)


def _agglomerative_clusters(
    distance_matrix: tuple[tuple[float, ...], ...],
    *,
    target_clusters: int,
) -> tuple[tuple[int, ...], ...]:
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(distance_matrix))]
    while len(clusters) > target_clusters:
        best_pair: tuple[int, int] | None = None
        best_distance: float | None = None
        for left_index, left_cluster in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                distance = _cluster_distance(
                    left_cluster,
                    clusters[right_index],
                    distance_matrix,
                )
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (left_index, right_index)
        if best_pair is None:
            break
        left_index, right_index = best_pair
        merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
    return tuple(sorted(clusters, key=lambda cluster: (cluster[0], len(cluster))))


def _cluster_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
    distance_matrix: tuple[tuple[float, ...], ...],
) -> float:
    distances = [
        distance_matrix[left_index][right_index]
        for left_index in left
        for right_index in right
    ]
    if not distances:
        return 1.0
    return sum(distances) / len(distances)


def _cluster_medoid(
    cluster: tuple[int, ...],
    distance_matrix: tuple[tuple[float, ...], ...],
) -> int:
    return min(
        cluster,
        key=lambda index: (
            sum(distance_matrix[index][other] for other in cluster if other != index),
            index,
        ),
    )


def _cluster_medoid_distance(
    cluster: tuple[int, ...],
    distance_matrix: tuple[tuple[float, ...], ...],
) -> float:
    medoid = _cluster_medoid(cluster, distance_matrix)
    return sum(distance_matrix[medoid][other] for other in cluster if other != medoid)


__all__ = [
    "BottleneckAwareControllerPolicy",
    "DifferentialSelection",
    "DifferentialSelectionPolicy",
    "ProbeCall",
    "build_probe_calls",
    "collect_behavior_trace",
    "select_consensus_medoid",
]
