from __future__ import annotations

import os
from pathlib import Path

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.phase3.evaluator import build_phase3_candidate_evaluator
from profiler_agent.phase3.generator import Phase3CandidateGenerator
from profiler_agent.phase3.optimizer import Phase3OptimizationResult, run_phase3_optimization


def run_default_phase3_workflow(
    *,
    root_dir: Path,
    max_iterations: int | None = None,
    llm_client: LLMClient | None = None,
) -> Phase3OptimizationResult:
    artifacts_dir = root_dir / ".agent_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    llm_debug_path = artifacts_dir / "phase3_llm_debug.jsonl"
    if not os.environ.get("PROFILER_AGENT_LLM_DEBUG_PATH", "").strip():
        os.environ["PROFILER_AGENT_LLM_DEBUG_PATH"] = str(llm_debug_path)

    generator = Phase3CandidateGenerator(
        llm_client=llm_client,
        debug_dir=artifacts_dir / "phase3_codegen_debug",
    )
    evaluator = build_phase3_candidate_evaluator(root_dir=root_dir)
    return run_phase3_optimization(
        root_dir=root_dir,
        max_iterations=max_iterations,
        candidate_generator=lambda state, feedback: generator.generate_candidate(state=state, feedback=feedback),
        candidate_evaluator=evaluator,
        bootstrap_candidate=generator.bootstrap_candidate(),
    )
