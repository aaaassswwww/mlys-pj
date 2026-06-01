from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_agent.phase3.models import Phase3OptimizerState, RuntimeEvaluation
from profiler_agent.runtime_budget import get_runtime_budget_status


def _is_better_incorrect_candidate(state: Phase3OptimizerState, evaluation: RuntimeEvaluation) -> bool:
    rel_l2_err = float(evaluation.correctness.rel_l2_err)
    max_abs_err = float(evaluation.correctness.max_abs_err)
    if rel_l2_err < float(state.best_rel_l2_err):
        return True
    if rel_l2_err > float(state.best_rel_l2_err):
        return False
    return max_abs_err < float(state.best_max_abs_err)


def record_candidate_evaluation(
    state: Phase3OptimizerState,
    *,
    candidate_id: str,
    source_code: str,
    candidate_rationale: str,
    candidate_source: str,
    evaluation: RuntimeEvaluation,
) -> bool:
    entry = {
        "candidate_id": candidate_id,
        "source_code": source_code,
        "candidate_rationale": candidate_rationale,
        "candidate_source": candidate_source,
        "evaluation": evaluation.to_dict(),
    }
    state.candidate_history.append(entry)
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
    if not evaluation.correctness.passed:
        state.correctness_failures.append(
            {
                "candidate_id": candidate_id,
                "max_abs_err": evaluation.correctness.max_abs_err,
                "rel_l2_err": evaluation.correctness.rel_l2_err,
                "notes": list(evaluation.notes),
            }
        )
    return promoted


def write_best_candidate(root_dir: Path, *, source_code: str, state: Phase3OptimizerState) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / "engine.py"
    path.write_text(source_code, encoding="utf-8")
    write_phase3_state(root_dir, state=state)
    return path


def write_phase3_state(root_dir: Path, *, state: Phase3OptimizerState) -> Path:
    state_path = root_dir / ".agent_artifacts" / "phase3_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return state_path


def write_phase3_report(root_dir: Path, *, state: Phase3OptimizerState, best_candidate_path: Path | None) -> Path:
    report_path = root_dir / ".agent_artifacts" / "phase3_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_best_candidate_id": state.current_best_candidate_id,
        "current_best_correct_candidate_id": state.current_best_correct_candidate_id,
        "best_speedup": state.best_speedup,
        "best_rel_l2_err": state.best_rel_l2_err,
        "best_max_abs_err": state.best_max_abs_err,
        "iterations_run": state.iteration,
        "last_completed_iteration": state.last_completed_iteration,
        "candidate_history_count": len(state.candidate_history),
        "correctness_failures_count": len(state.correctness_failures),
        "stop_reason": state.stop_reason,
        "engine_path": str(best_candidate_path) if best_candidate_path is not None else "",
        "runtime_budget": get_runtime_budget_status(),
        "recent_candidates": [
            {
                "candidate_id": entry.get("candidate_id"),
                "correctness_passed": ((entry.get("evaluation") or {}).get("correctness") or {}).get("passed"),
                "speedup": (entry.get("evaluation") or {}).get("speedup"),
                "aggregate_tokens_per_s": (((entry.get("evaluation") or {}).get("benchmark") or {}).get("aggregate_tokens_per_s")),
                "notes": (entry.get("evaluation") or {}).get("notes"),
            }
            for entry in state.candidate_history[-5:]
        ],
        "recent_correctness_failures": state.correctness_failures[-5:],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if den <= 0.0:
        return 0.0
    return num / den


def _build_phase3_performance_feedback(benchmark: dict[str, Any], baseline_benchmark: dict[str, Any]) -> dict[str, Any]:
    ratios = {
        "prefill_speedup": _safe_ratio(benchmark.get("prefill_tokens_per_s"), baseline_benchmark.get("prefill_tokens_per_s")),
        "decode_speedup": _safe_ratio(benchmark.get("decode_tokens_per_s"), baseline_benchmark.get("decode_tokens_per_s")),
        "mixed_speedup": _safe_ratio(benchmark.get("mixed_tokens_per_s"), baseline_benchmark.get("mixed_tokens_per_s")),
        "aggregate_speedup": _safe_ratio(benchmark.get("aggregate_tokens_per_s"), baseline_benchmark.get("aggregate_tokens_per_s")),
    }
    weakest_metric = min(ratios.items(), key=lambda item: item[1])[0]
    focus_hints = {
        "prefill_speedup": [
            "Improve prompt batching and same-length prefill grouping.",
            "Reduce prompt-side tensor stacking and per-request Python overhead.",
        ],
        "decode_speedup": [
            "Improve incremental decode batching by cache length and reduce per-step Python work.",
            "Reduce cache restacking and repeated tiny tensor allocations in decode.",
        ],
        "mixed_speedup": [
            "Improve request scheduler balance across active arrivals, removals, and decode steps.",
            "Reduce state-management overhead when interleaving prefill, decode, and remove.",
        ],
        "aggregate_speedup": [
            "Seek balanced throughput gains instead of overfitting a single stage.",
        ],
    }
    return {
        "candidate_tokens_per_s": dict(benchmark),
        "baseline_tokens_per_s": dict(baseline_benchmark),
        "speedup_breakdown": ratios,
        "weakest_metric": weakest_metric,
        "focus_hints": focus_hints.get(weakest_metric, []),
    }


def build_candidate_feedback(
    *,
    correctness: dict[str, Any],
    benchmark: dict[str, Any],
    baseline_benchmark: dict[str, Any],
    speedup: float,
    notes: list[str],
    previous_candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "correctness": dict(correctness),
        "benchmark": dict(benchmark),
        "baseline_benchmark": dict(baseline_benchmark),
        "performance_feedback": _build_phase3_performance_feedback(benchmark, baseline_benchmark),
        "speedup": float(speedup),
        "notes": list(notes),
        "previous_candidate": dict(previous_candidate),
    }
