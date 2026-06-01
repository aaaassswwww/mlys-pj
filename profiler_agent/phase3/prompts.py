from __future__ import annotations

import json
from typing import Any

from profiler_agent.phase3.models import Phase3OptimizerState


def build_phase3_generation_system_prompt() -> str:
    return (
        "You are generating a single Python file named engine.py for an LLM inference runtime optimization task. "
        "Return JSON only with keys candidate_id, rationale, and source_code. "
        "Do not use markdown fences. "
        "The generated source must stay self-contained inside engine.py and may import helper modules only from the local runtime package already present in the repository root. "
        "Do not create or reference a workspace/ directory. "
        "The runtime must expose create_engine(model_config, weight_dir, device='cuda') and return an object with methods prefill(request_ids, input_ids), decode(request_ids, token_ids), and remove(request_ids). "
        "Correctness is the hard requirement. Preserve request-local state by request id. "
        "Prefer incremental decode with KV cache, request batching, and low Python overhead. "
        "When revising an existing candidate, make targeted edits instead of rewriting the whole runtime unless the previous design is fundamentally broken. "
        "If a best correct candidate exists, preserve its correctness semantics and focus on throughput improvements. "
        "Do not add CLI entrypoints, main(), print debugging, filesystem writes, network access, or dependency installation."
    )


def _summarize_recent_candidates(state: Phase3OptimizerState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in state.candidate_history[-5:]:
        if not isinstance(item, dict):
            continue
        evaluation = item.get("evaluation") or {}
        correctness = evaluation.get("correctness") or {}
        benchmark = evaluation.get("benchmark") or {}
        baseline_benchmark = evaluation.get("baseline_benchmark") or {}
        prefill_speedup = 0.0
        decode_speedup = 0.0
        mixed_speedup = 0.0
        if baseline_benchmark:
            try:
                prefill_speedup = float(benchmark.get("prefill_tokens_per_s", 0.0)) / max(float(baseline_benchmark.get("prefill_tokens_per_s", 0.0)), 1e-12)
                decode_speedup = float(benchmark.get("decode_tokens_per_s", 0.0)) / max(float(baseline_benchmark.get("decode_tokens_per_s", 0.0)), 1e-12)
                mixed_speedup = float(benchmark.get("mixed_tokens_per_s", 0.0)) / max(float(baseline_benchmark.get("mixed_tokens_per_s", 0.0)), 1e-12)
            except (TypeError, ValueError, ZeroDivisionError):
                prefill_speedup = decode_speedup = mixed_speedup = 0.0
        rows.append(
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_source": item.get("candidate_source"),
                "passed": correctness.get("passed"),
                "rel_l2_err": correctness.get("rel_l2_err"),
                "max_abs_err": correctness.get("max_abs_err"),
                "speedup": evaluation.get("speedup"),
                "aggregate_tokens_per_s": benchmark.get("aggregate_tokens_per_s"),
                "prefill_tokens_per_s": benchmark.get("prefill_tokens_per_s"),
                "decode_tokens_per_s": benchmark.get("decode_tokens_per_s"),
                "mixed_tokens_per_s": benchmark.get("mixed_tokens_per_s"),
                "prefill_speedup": prefill_speedup,
                "decode_speedup": decode_speedup,
                "mixed_speedup": mixed_speedup,
                "notes": evaluation.get("notes"),
            }
        )
    return rows


def _extract_performance_focus(feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    performance_feedback = feedback.get("performance_feedback")
    if not isinstance(performance_feedback, dict):
        return {}
    result = {
        "weakest_metric": performance_feedback.get("weakest_metric"),
        "focus_hints": performance_feedback.get("focus_hints"),
        "speedup_breakdown": performance_feedback.get("speedup_breakdown"),
        "candidate_tokens_per_s": performance_feedback.get("candidate_tokens_per_s"),
        "baseline_tokens_per_s": performance_feedback.get("baseline_tokens_per_s"),
    }
    return result


def _build_strategy_guidance(state: Phase3OptimizerState, feedback: dict[str, Any] | None) -> list[str]:
    guidance = [
        "Keep imports rooted at the repository root runtime package and preserve the public engine API.",
        "Prefer small, local runtime changes that are easy to evaluate and explain.",
        "Avoid introducing completely new helper modules or changing the file contract away from root engine.py.",
    ]
    if state.current_best_correct_candidate_id:
        guidance.extend(
            [
                "A correct candidate already exists; preserve its request-state semantics and correctness behavior.",
                "Focus on throughput: batching policy, cache reuse, avoiding redundant tensor reshapes, and reducing Python overhead.",
                "Do not remove KV cache or revert to full-sequence recomputation.",
            ]
        )
    else:
        guidance.extend(
            [
                "No correct candidate has been locked in yet; prioritize correctness over speed.",
                "Prefer simpler scheduling and obvious cache-safe logic before aggressive optimization.",
            ]
        )
    if isinstance(feedback, dict):
        notes = feedback.get("notes")
        if isinstance(notes, list) and any("correctness_failed" in str(note) for note in notes):
            guidance.append("The previous attempt failed correctness; reduce behavioral drift and keep changes narrowly targeted.")
        if isinstance(notes, list) and any("benchmark_failed" in str(note) for note in notes):
            guidance.append("The previous attempt failed during benchmarking; avoid fragile compile-only or runtime-unsafe ideas.")
        performance_focus = _extract_performance_focus(feedback)
        weakest_metric = performance_focus.get("weakest_metric")
        if isinstance(weakest_metric, str) and weakest_metric:
            guidance.append(
                f"The previous benchmark breakdown shows the weakest area is {weakest_metric}; prioritize changes that help that stage first."
            )
            guidance.append(
                "For this iteration, optimize primarily for that single weakest area instead of trying to improve every throughput metric at once."
            )
    return guidance


def build_phase3_generation_user_prompt(
    *,
    state: Phase3OptimizerState,
    iteration: int,
    feedback: dict[str, Any] | None,
    bootstrap_source_code: str,
) -> str:
    previous_candidate = {}
    if isinstance(feedback, dict):
        candidate = feedback.get("previous_candidate")
        if isinstance(candidate, dict):
            previous_candidate = {
                "candidate_id": candidate.get("candidate_id"),
                "rationale": candidate.get("rationale"),
                "source": candidate.get("source"),
                "source_code": candidate.get("source_code"),
            }

    best_candidate = {}
    if state.current_best_candidate_id and state.current_best_source_code:
        best_candidate = {
            "candidate_id": state.current_best_candidate_id,
            "source": state.current_best_source,
            "rationale": state.current_best_rationale,
            "source_code": state.current_best_source_code,
            "best_speedup": state.best_speedup,
            "best_rel_l2_err": state.best_rel_l2_err,
            "best_max_abs_err": state.best_max_abs_err,
            "is_correct_anchor": state.current_best_correct_candidate_id == state.current_best_candidate_id,
        }
    performance_focus = _extract_performance_focus(feedback)
    optimization_lenses = {
        "prefill_speedup": [
            "same-length prompt grouping",
            "batch assembly overhead",
            "prompt-side cache materialization cost",
        ],
        "decode_speedup": [
            "cache-length grouping",
            "per-step Python overhead",
            "cache stacking and unstacking overhead",
            "tiny tensor allocation frequency",
        ],
        "mixed_speedup": [
            "scheduler balance under arrivals and removals",
            "active-request bookkeeping overhead",
            "interaction between prefill and decode batching policies",
        ],
        "aggregate_speedup": [
            "balanced throughput improvements across prefill, decode, and mixed",
        ],
    }
    weakest_metric = performance_focus.get("weakest_metric")
    suggested_focus_lenses = optimization_lenses.get(weakest_metric, optimization_lenses["aggregate_speedup"])
    single_focus_contract = {
        "enabled": bool(weakest_metric),
        "primary_target_metric": weakest_metric or "aggregate_speedup",
        "instruction": (
            "Make this iteration a single-target optimization pass: choose one weakest metric as the main objective, "
            "keep edits narrow, and avoid broad rewrites intended to improve all stages simultaneously."
        ),
        "guardrails": [
            "Prioritize the primary target metric first.",
            "Preserve correctness.",
            "Try to avoid obvious regressions in the two non-target metrics.",
            "Prefer local changes that can be causally connected to the primary target metric.",
        ],
        "allowed_focus_lenses": suggested_focus_lenses,
    }

    payload = {
        "task": "generate_phase3_engine_candidate",
        "iteration": iteration,
        "goal": "produce a root-level engine.py runtime candidate and improve throughput without breaking correctness",
        "required_contract": {
            "file_name": "engine.py",
            "must_live_at_repo_root": True,
            "forbid_workspace_directory": True,
            "required_factory": "create_engine(model_config, weight_dir, device='cuda')",
            "required_methods": ["prefill", "decode", "remove"],
        },
        "runtime_expectations": {
            "request_state_by_id": True,
            "incremental_decode_with_kv_cache": True,
            "batched_prefill_and_decode_preferred": True,
            "local_runtime_package_only": True,
        },
        "strategy_guidance": _build_strategy_guidance(state, feedback),
        "performance_context": {
            "best_speedup_so_far": state.best_speedup,
            "best_candidate_id": state.current_best_candidate_id,
            "best_correct_candidate_id": state.current_best_correct_candidate_id,
            "current_iteration_focus": weakest_metric or "correctness_or_balanced_throughput",
            "suggested_focus_lenses": suggested_focus_lenses,
            "single_focus_mode": single_focus_contract,
        },
        "previous_feedback": feedback or {},
        "performance_feedback": performance_focus,
        "previous_candidate": previous_candidate,
        "best_candidate": best_candidate,
        "recent_candidates": _summarize_recent_candidates(state),
        "fallback_bootstrap_candidate": {
            "candidate_id": "grouped_compile-v00",
            "source": "bootstrap_template",
            "source_code": bootstrap_source_code,
        },
        "response_schema": {
            "candidate_id": "short_identifier_without_iteration_suffix_if_possible",
            "rationale": "brief explanation of the runtime change",
            "source_code": "complete engine.py source code",
        },
        "iteration_policy": {
            "single_target_runtime_optimization": single_focus_contract,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
