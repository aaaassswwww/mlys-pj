from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.phase2.evaluator import (
    build_compile_checked_candidate_evaluator,
    build_ctypes_candidate_runner,
    build_harness_runtime_evaluator,
)
from profiler_agent.phase2.generator import LoraCandidateGenerator
from profiler_agent.phase2.models import CandidateEvaluation, GeneratedCandidate, LoraProblemSpec
from profiler_agent.phase2.optimizer import Phase2OptimizationResult, run_phase2_optimization


def default_problem_specs(
    *,
    hidden_dims: list[int] | None = None,
    low_rank: int = 16,
    num_tokens: int = 32,
    dtype: str = "float32",
    device: str = "cuda",
) -> list[LoraProblemSpec]:
    dims = hidden_dims or [3584, 4096, 4608]
    return [
        LoraProblemSpec(
            hidden_dim=int(dim),
            low_rank=low_rank,
            output_dim=int(dim),
            num_tokens=num_tokens,
            dtype=dtype,
            device=device,
        )
        for dim in dims
    ]


def build_default_phase2_candidate_evaluator(
    *,
    root_dir: Path,
    problem_specs: list[LoraProblemSpec] | None = None,
    candidate_runner: Callable[[GeneratedCandidate, Any, Any, LoraProblemSpec, dict[str, Any], Any], Any] | None = None,
    backend: Any | None = None,
    warmup_runs: int = 3,
    measured_runs: int = 7,
) -> Callable[[GeneratedCandidate], CandidateEvaluation]:
    runtime_runner = candidate_runner or build_ctypes_candidate_runner()
    runtime_evaluator = build_harness_runtime_evaluator(
        problem_specs=problem_specs or default_problem_specs(),
        candidate_runner=runtime_runner,
        backend=backend,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    return build_compile_checked_candidate_evaluator(
        root_dir=root_dir,
        runtime_evaluator=runtime_evaluator,
    )


def run_default_phase2_workflow(
    *,
    root_dir: Path,
    max_iterations: int | None = None,
    llm_client: LLMClient | None = None,
    problem_specs: list[LoraProblemSpec] | None = None,
    candidate_runner: Callable[[GeneratedCandidate, Any, Any, LoraProblemSpec, dict[str, Any], Any], Any] | None = None,
    backend: Any | None = None,
    warmup_runs: int = 3,
    measured_runs: int = 7,
) -> Phase2OptimizationResult:
    artifacts_dir = root_dir / ".agent_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    llm_debug_path = artifacts_dir / "phase2_llm_debug.jsonl"
    if not os.environ.get("PROFILER_AGENT_LLM_DEBUG_PATH", "").strip():
        os.environ["PROFILER_AGENT_LLM_DEBUG_PATH"] = str(llm_debug_path)

    generator = LoraCandidateGenerator(
        llm_client=llm_client,
        debug_dir=artifacts_dir / "phase2_codegen_debug",
    )
    evaluator = build_default_phase2_candidate_evaluator(
        root_dir=root_dir,
        problem_specs=problem_specs,
        candidate_runner=candidate_runner,
        backend=backend,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    return run_phase2_optimization(
        root_dir=root_dir,
        max_iterations=max_iterations,
        candidate_generator=lambda state, feedback: generator.generate_candidate(state=state, feedback=feedback),
        candidate_evaluator=evaluator,
        bootstrap_candidate=generator.bootstrap_candidate(),
    )
