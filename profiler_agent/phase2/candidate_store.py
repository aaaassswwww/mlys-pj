from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiler_agent.phase2.models import CandidateEvaluation, Phase2OptimizerState


def record_candidate_evaluation(
    state: Phase2OptimizerState,
    *,
    candidate_id: str,
    source_code: str,
    evaluation: CandidateEvaluation,
) -> bool:
    entry = {
        "candidate_id": candidate_id,
        "source_code_preview": source_code[:500],
        "evaluation": evaluation.to_dict(),
    }
    state.candidate_history.append(entry)
    state.benchmark_history.append(
        {
            "candidate_id": candidate_id,
            "student_median_runtime_ms": evaluation.student_benchmark.median_runtime_ms,
            "reference_median_runtime_ms": evaluation.reference_benchmark.median_runtime_ms,
            "speedup": evaluation.speedup,
        }
    )
    if evaluation.compilation is not None and not evaluation.compilation.ok:
        state.compile_errors.append(
            {
                "candidate_id": candidate_id,
                "returncode": evaluation.compilation.returncode,
                "stderr_tail": evaluation.compilation.stderr_tail,
                "command": list(evaluation.compilation.command),
            }
        )

    promoted = False
    if evaluation.correctness.passed and evaluation.speedup >= state.best_speedup:
        state.best_speedup = float(evaluation.speedup)
        state.current_best_candidate_id = candidate_id
        promoted = True
    return promoted


def write_best_candidate(
    root_dir: Path,
    *,
    source_code: str,
    state: Phase2OptimizerState,
) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / "optimized_lora.cu"
    path.write_text(source_code, encoding="utf-8")
    state_path = root_dir / ".agent_artifacts" / "phase2_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_phase2_report(
    root_dir: Path,
    *,
    state: Phase2OptimizerState,
    best_candidate_path: Path | None,
) -> Path:
    report_path = root_dir / ".agent_artifacts" / "phase2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_best_candidate_id": state.current_best_candidate_id,
        "best_speedup": state.best_speedup,
        "iterations_run": state.iteration,
        "candidate_history_count": len(state.candidate_history),
        "correctness_failures_count": len(state.correctness_failures),
        "optimized_lora_path": str(best_candidate_path) if best_candidate_path is not None else "",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def build_candidate_feedback(
    *,
    compile_ok: bool,
    correctness: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "compile_ok": bool(compile_ok),
        "correctness": correctness or {},
        "benchmark": benchmark or {},
        "profile": profile or {},
    }
