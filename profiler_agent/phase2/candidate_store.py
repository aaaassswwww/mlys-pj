from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiler_agent.phase2.models import CandidateEvaluation, Phase2OptimizerState
from profiler_agent.runtime_budget import get_runtime_budget_status

_REFERENCE_DIAG_RE = re.compile(
    r"reference_diagnosis:hidden_dim=(?P<hidden_dim>\d+):"
    r"student_vs_reference_rel_l2=(?P<student_ref>[0-9.eE+-]+):"
    r"student_vs_naive_rel_l2=(?P<student_naive>[0-9.eE+-]+):"
    r"naive_vs_reference_rel_l2=(?P<naive_ref>[0-9.eE+-]+):"
    r"student_closer_to=(?P<closer_to>[a-zA-Z_]+)"
)


def _is_better_incorrect_candidate(state: Phase2OptimizerState, evaluation: CandidateEvaluation) -> bool:
    rel_l2_err = float(evaluation.correctness.rel_l2_err)
    max_abs_err = float(evaluation.correctness.max_abs_err)
    if rel_l2_err < float(state.best_rel_l2_err):
        return True
    if rel_l2_err > float(state.best_rel_l2_err):
        return False
    return max_abs_err < float(state.best_max_abs_err)


def _extract_reference_fit_score(notes: list[str]) -> tuple[float | None, bool]:
    best_score = None
    saw_reference = False
    for note in notes:
        if not isinstance(note, str):
            continue
        match = _REFERENCE_DIAG_RE.fullmatch(note.strip())
        if not match:
            continue
        score = float(match.group("student_ref"))
        closer_to = match.group("closer_to")
        if closer_to == "reference":
            saw_reference = True
        if best_score is None or score < best_score:
            best_score = score
    return best_score, saw_reference


def record_candidate_evaluation(
    state: Phase2OptimizerState,
    *,
    candidate_id: str,
    source_code: str,
    candidate_rationale: str = "",
    candidate_source: str = "",
    evaluation: CandidateEvaluation,
) -> bool:
    entry = {
        "candidate_id": candidate_id,
        "source_code_preview": source_code[:500],
        "source_code": source_code,
        "candidate_rationale": candidate_rationale,
        "candidate_source": candidate_source,
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

    reference_score, reference_like = _extract_reference_fit_score(evaluation.notes)
    if reference_like and reference_score is not None and reference_score < float(state.best_reference_rel_l2_err):
        state.best_reference_rel_l2_err = reference_score
        state.current_best_reference_candidate_id = candidate_id
        state.current_best_reference_source_code = source_code
        state.current_best_reference_rationale = candidate_rationale
        state.current_best_reference_source = candidate_source

    promoted = False
    if evaluation.correctness.passed and evaluation.speedup >= state.best_speedup:
        state.best_speedup = float(evaluation.speedup)
        state.best_rel_l2_err = float(evaluation.correctness.rel_l2_err)
        state.best_max_abs_err = float(evaluation.correctness.max_abs_err)
        state.current_best_candidate_id = candidate_id
        state.current_best_correct_candidate_id = candidate_id
        state.current_best_source_code = source_code
        state.current_best_rationale = candidate_rationale
        state.current_best_source = candidate_source
        promoted = True
    elif state.current_best_correct_candidate_id is None and _is_better_incorrect_candidate(state, evaluation):
        state.best_rel_l2_err = float(evaluation.correctness.rel_l2_err)
        state.best_max_abs_err = float(evaluation.correctness.max_abs_err)
        state.current_best_candidate_id = candidate_id
        state.current_best_source_code = source_code
        state.current_best_rationale = candidate_rationale
        state.current_best_source = candidate_source
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
    write_phase2_state(root_dir, state=state)
    return path


def write_phase2_state(
    root_dir: Path,
    *,
    state: Phase2OptimizerState,
) -> Path:
    state_path = root_dir / ".agent_artifacts" / "phase2_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return state_path


def write_phase2_report(
    root_dir: Path,
    *,
    state: Phase2OptimizerState,
    best_candidate_path: Path | None,
) -> Path:
    report_path = root_dir / ".agent_artifacts" / "phase2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    budget = get_runtime_budget_status()
    payload = {
        "current_best_candidate_id": state.current_best_candidate_id,
        "current_best_correct_candidate_id": state.current_best_correct_candidate_id,
        "current_best_reference_candidate_id": state.current_best_reference_candidate_id,
        "best_speedup": state.best_speedup,
        "best_rel_l2_err": state.best_rel_l2_err,
        "best_max_abs_err": state.best_max_abs_err,
        "best_reference_rel_l2_err": state.best_reference_rel_l2_err,
        "iterations_run": state.iteration,
        "stop_reason": state.stop_reason,
        "candidate_history_count": len(state.candidate_history),
        "correctness_failures_count": len(state.correctness_failures),
        "compile_errors_count": len(state.compile_errors),
        "optimized_lora_path": str(best_candidate_path) if best_candidate_path is not None else "",
        "runtime_budget": budget,
        "recent_candidates": [
            {
                "candidate_id": entry.get("candidate_id"),
                "correctness_passed": (
                    ((entry.get("evaluation") or {}).get("correctness") or {}).get("passed")
                    if isinstance(entry, dict)
                    else None
                ),
                "speedup": ((entry.get("evaluation") or {}).get("speedup") if isinstance(entry, dict) else None),
                "notes": ((entry.get("evaluation") or {}).get("notes") if isinstance(entry, dict) else None),
            }
            for entry in state.candidate_history[-3:]
            if isinstance(entry, dict)
        ],
        "recent_compile_errors": state.compile_errors[-3:],
        "recent_correctness_failures": state.correctness_failures[-3:],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def build_candidate_feedback(
    *,
    compile_ok: bool,
    correctness: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "compile_ok": bool(compile_ok),
        "correctness": correctness or {},
        "benchmark": benchmark or {},
        "profile": profile or {},
        "notes": [str(item) for item in (notes or [])],
    }
