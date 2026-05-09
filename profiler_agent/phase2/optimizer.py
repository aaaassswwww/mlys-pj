from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from profiler_agent.phase2.candidate_store import (
    build_candidate_feedback,
    record_candidate_evaluation,
    write_best_candidate,
    write_phase2_report,
    write_phase2_state,
)
from profiler_agent.phase2.models import CandidateEvaluation, GeneratedCandidate, Phase2OptimizerState
from profiler_agent.runtime_budget import get_runtime_budget_status


def _phase2_stop_buffer_seconds() -> float:
    timeout_raw = str(os.environ.get("OPENAI_TIMEOUT_S", "")).strip()
    try:
        llm_timeout = float(timeout_raw) if timeout_raw else 120.0
    except ValueError:
        llm_timeout = 120.0
    default_buffer = max(90.0, llm_timeout + 15.0)
    raw = str(os.environ.get("PROFILER_AGENT_PHASE2_STOP_BUFFER_SECONDS", "")).strip()
    if not raw:
        return default_buffer
    try:
        value = float(raw)
    except ValueError:
        return default_buffer
    return max(0.0, value)


def _should_stop_before_next_iteration() -> tuple[bool, str]:
    budget = get_runtime_budget_status()
    if not budget.get("enabled", False):
        return False, ""
    if budget.get("expired", False):
        return True, "runtime_budget_expired"
    remaining = float(budget.get("remaining_seconds") or 0.0)
    stop_buffer = _phase2_stop_buffer_seconds()
    if remaining <= stop_buffer:
        return True, f"remaining_runtime_below_stop_buffer:{remaining:.3f}s<={stop_buffer:.3f}s"
    return False, ""


def _fatal_evaluation_stop_reason(evaluation: CandidateEvaluation) -> str:
    for note in evaluation.notes:
        if isinstance(note, str) and note.startswith("fatal_cuda_runtime_error:"):
            return note
    return ""


def _normalize_iteration_limit(max_iterations: int | None) -> int | None:
    if max_iterations is not None and max_iterations > 0:
        return int(max_iterations)
    budget = get_runtime_budget_status()
    if budget.get("enabled", False):
        return None
    return 2


def _phase2_speedup_iteration_target() -> int:
    raw = str(os.environ.get("PROFILER_AGENT_PHASE2_SPEEDUP_ITERATIONS", "")).strip()
    if not raw:
        return 30
    try:
        value = int(raw)
    except ValueError:
        return 30
    return max(1, value)


def _active_iteration_limit(base_limit: int | None, state: Phase2OptimizerState) -> int | None:
    if base_limit is None:
        return None
    if state.current_best_correct_candidate_id is None:
        return base_limit
    return max(base_limit, _phase2_speedup_iteration_target())


@dataclass(frozen=True)
class Phase2OptimizationResult:
    best_candidate_id: str | None
    best_speedup: float
    iterations_run: int
    optimized_lora_path: Path | None
    state: Phase2OptimizerState


def run_phase2_optimization(
    *,
    root_dir: Path,
    max_iterations: int | None,
    candidate_generator: Callable[[Phase2OptimizerState, dict[str, object] | None], GeneratedCandidate],
    candidate_evaluator: Callable[[GeneratedCandidate], CandidateEvaluation],
    bootstrap_candidate: GeneratedCandidate | None = None,
) -> Phase2OptimizationResult:
    state = Phase2OptimizerState()
    last_feedback: dict[str, object] | None = None
    best_path: Path | None = None
    effective_max_iterations = _normalize_iteration_limit(max_iterations)
    def persist() -> None:
        write_phase2_state(root_dir, state=state)
        write_phase2_report(root_dir, state=state, best_candidate_path=best_path)

    try:
        if bootstrap_candidate is not None:
            best_path = write_best_candidate(root_dir, source_code=bootstrap_candidate.source_code, state=state)
            state.llm_revision_history.append(
                {
                    "iteration": 0,
                    "candidate_id": bootstrap_candidate.candidate_id,
                    "rationale": bootstrap_candidate.rationale,
                    "feedback": {"bootstrap": True},
                    "source": bootstrap_candidate.source,
                }
            )
            persist()

        iteration = 1
        while True:
            active_iteration_limit = _active_iteration_limit(effective_max_iterations, state)
            if active_iteration_limit is not None and iteration > active_iteration_limit:
                state.stop_reason = f"max_iterations_reached:{active_iteration_limit}"
                break
            should_stop, stop_reason = _should_stop_before_next_iteration()
            if should_stop:
                state.stop_reason = stop_reason
                break

            state.iteration = iteration
            persist()

            generation_context = {
                "previous_feedback_candidate_id": (
                    (
                        ((last_feedback or {}).get("previous_candidate") or {}).get("candidate_id")
                        if isinstance(last_feedback, dict)
                        else None
                    )
                ),
                "base_candidate_id": state.current_best_candidate_id,
                "reference_like_candidate_id": state.current_best_reference_candidate_id,
                "revision_source_preference": (
                    "previous_candidate_patch_first"
                    if isinstance(last_feedback, dict)
                    and isinstance(last_feedback.get("previous_candidate"), dict)
                    and bool(last_feedback.get("compile_ok"))
                    and not bool(((last_feedback.get("correctness") or {}).get("passed")))
                    else (
                        "reference_like_candidate"
                        if state.current_best_reference_candidate_id is not None
                        and state.current_best_candidate_id != state.current_best_reference_candidate_id
                        else "base_candidate"
                    )
                ),
            }
            candidate = candidate_generator(state, last_feedback)
            evaluation = candidate_evaluator(candidate)

            promoted = record_candidate_evaluation(
                state,
                candidate_id=candidate.candidate_id,
                source_code=candidate.source_code,
                candidate_rationale=candidate.rationale,
                candidate_source=candidate.source,
                evaluation=evaluation,
            )
            if promoted:
                best_path = write_best_candidate(root_dir, source_code=candidate.source_code, state=state)

            if not evaluation.correctness.passed:
                state.correctness_failures.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "max_abs_err": evaluation.correctness.max_abs_err,
                        "rel_l2_err": evaluation.correctness.rel_l2_err,
                    }
                )

            last_feedback = build_candidate_feedback(
                compile_ok=bool(evaluation.compilation.ok) if evaluation.compilation is not None else True,
                correctness=evaluation.correctness.to_dict(),
                per_spec=list(evaluation.per_spec),
                benchmark={
                    "student_median_runtime_ms": evaluation.student_benchmark.median_runtime_ms,
                    "reference_median_runtime_ms": evaluation.reference_benchmark.median_runtime_ms,
                    "speedup": evaluation.speedup,
                },
                profile={
                    "compilation": evaluation.compilation.to_dict() if evaluation.compilation is not None else {},
                    "load": evaluation.load.to_dict() if evaluation.load is not None else {},
                },
                notes=list(evaluation.notes),
                previous_candidate={
                    "candidate_id": candidate.candidate_id,
                    "rationale": candidate.rationale,
                    "source": candidate.source,
                    "source_code": candidate.source_code,
                },
            )
            state.llm_revision_history.append(
                {
                    "iteration": iteration,
                    "candidate_id": candidate.candidate_id,
                    "rationale": candidate.rationale,
                    "feedback": last_feedback,
                    "source": candidate.source,
                    "generation_context": generation_context,
                }
            )
            persist()
            fatal_stop_reason = _fatal_evaluation_stop_reason(evaluation)
            if fatal_stop_reason:
                state.stop_reason = fatal_stop_reason
                break
            iteration += 1

        if not state.stop_reason:
            state.stop_reason = "completed_without_explicit_stop_reason"
        state.done = True
        persist()
    except Exception:
        persist()
        raise

    return Phase2OptimizationResult(
        best_candidate_id=state.current_best_candidate_id,
        best_speedup=state.best_speedup,
        iterations_run=state.iteration,
        optimized_lora_path=best_path,
        state=state,
    )
