from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from profiler_agent.phase2.candidate_store import (
    build_candidate_feedback,
    record_candidate_evaluation,
    write_best_candidate,
    write_phase2_report,
)
from profiler_agent.phase2.models import CandidateEvaluation, GeneratedCandidate, Phase2OptimizerState


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
    max_iterations: int,
    candidate_generator: Callable[[Phase2OptimizerState, dict[str, object] | None], GeneratedCandidate],
    candidate_evaluator: Callable[[GeneratedCandidate], CandidateEvaluation],
    bootstrap_candidate: GeneratedCandidate | None = None,
) -> Phase2OptimizationResult:
    state = Phase2OptimizerState()
    last_feedback: dict[str, object] | None = None
    best_path: Path | None = None
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

    for iteration in range(1, max_iterations + 1):
        state.iteration = iteration
        candidate = candidate_generator(state, last_feedback)
        evaluation = candidate_evaluator(candidate)

        promoted = record_candidate_evaluation(
            state,
            candidate_id=candidate.candidate_id,
            source_code=candidate.source_code,
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
            benchmark={
                "student_median_runtime_ms": evaluation.student_benchmark.median_runtime_ms,
                "reference_median_runtime_ms": evaluation.reference_benchmark.median_runtime_ms,
                "speedup": evaluation.speedup,
            },
            profile={
                "compilation": evaluation.compilation.to_dict() if evaluation.compilation is not None else {},
                "load": evaluation.load.to_dict() if evaluation.load is not None else {},
            },
        )
        state.llm_revision_history.append(
            {
                "iteration": iteration,
                "candidate_id": candidate.candidate_id,
                "rationale": candidate.rationale,
                "feedback": last_feedback,
                "source": candidate.source,
            }
        )

    write_phase2_report(root_dir, state=state, best_candidate_path=best_path)

    return Phase2OptimizationResult(
        best_candidate_id=state.current_best_candidate_id,
        best_speedup=state.best_speedup,
        iterations_run=state.iteration,
        optimized_lora_path=best_path,
        state=state,
    )
