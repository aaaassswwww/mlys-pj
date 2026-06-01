from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from profiler_agent.phase3.candidate_store import (
    build_candidate_feedback,
    record_candidate_evaluation,
    write_best_candidate,
    write_phase3_report,
    write_phase3_state,
)
from profiler_agent.phase3.models import GeneratedRuntimeCandidate, Phase3OptimizerState, RuntimeEvaluation
from profiler_agent.runtime_budget import get_runtime_budget_status


def _phase3_stop_buffer_seconds() -> float:
    raw = str(os.environ.get("PROFILER_AGENT_PHASE3_STOP_BUFFER_SECONDS", "")).strip()
    if not raw:
        return 90.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 90.0


def _should_stop_before_next_iteration() -> tuple[bool, str]:
    budget = get_runtime_budget_status()
    if not budget.get("enabled", False):
        return False, ""
    if budget.get("expired", False):
        return True, "runtime_budget_expired"
    remaining = float(budget.get("remaining_seconds") or 0.0)
    stop_buffer = _phase3_stop_buffer_seconds()
    if remaining <= stop_buffer:
        return True, f"remaining_runtime_below_stop_buffer:{remaining:.3f}s<={stop_buffer:.3f}s"
    return False, ""


@dataclass(frozen=True)
class Phase3OptimizationResult:
    best_candidate_id: str | None
    best_speedup: float
    iterations_run: int
    engine_path: Path | None
    state: Phase3OptimizerState


def run_phase3_optimization(
    *,
    root_dir: Path,
    max_iterations: int | None,
    candidate_generator: Callable[[Phase3OptimizerState, dict[str, object] | None], GeneratedRuntimeCandidate],
    candidate_evaluator: Callable[[GeneratedRuntimeCandidate], RuntimeEvaluation],
    bootstrap_candidate: GeneratedRuntimeCandidate | None = None,
) -> Phase3OptimizationResult:
    state = Phase3OptimizerState()
    last_feedback: dict[str, object] | None = None
    best_path: Path | None = None

    def persist() -> None:
        write_phase3_state(root_dir, state=state)
        write_phase3_report(root_dir, state=state, best_candidate_path=best_path)

    try:
        if bootstrap_candidate is not None:
            best_path = write_best_candidate(root_dir, source_code=bootstrap_candidate.source_code, state=state)
            state.llm_revision_history.append(
                {
                    "iteration": 0,
                    "candidate_id": bootstrap_candidate.candidate_id,
                    "rationale": bootstrap_candidate.rationale,
                    "source": bootstrap_candidate.source,
                    "feedback": {"bootstrap": True},
                }
            )
            persist()

        iteration = 1
        while True:
            if max_iterations is not None and max_iterations > 0 and iteration > max_iterations:
                state.stop_reason = f"max_iterations_reached:{max_iterations}"
                break
            should_stop, stop_reason = _should_stop_before_next_iteration()
            if should_stop:
                state.stop_reason = stop_reason
                break

            state.iteration = iteration
            persist()

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

            last_feedback = build_candidate_feedback(
                correctness=evaluation.correctness.to_dict(),
                benchmark=evaluation.benchmark.to_dict(),
                baseline_benchmark=evaluation.baseline_benchmark.to_dict(),
                speedup=evaluation.speedup,
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
                    "source": candidate.source,
                    "feedback": last_feedback,
                }
            )
            state.last_completed_iteration = iteration
            persist()
            iteration += 1

        if not state.stop_reason:
            state.stop_reason = "completed_without_explicit_stop_reason"
        state.done = True
        persist()
    except Exception:
        persist()
        raise

    return Phase3OptimizationResult(
        best_candidate_id=state.current_best_candidate_id,
        best_speedup=state.best_speedup,
        iterations_run=state.iteration,
        engine_path=best_path,
        state=state,
    )
