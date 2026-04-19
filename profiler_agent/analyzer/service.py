from __future__ import annotations

from typing import Any

from profiler_agent.analyzer.bound_classifier import analyze_bound
from profiler_agent.analyzer.llm_reasoner import build_llm_analysis
from profiler_agent.report_summary import build_intrinsic_probe_report, build_synthetic_counter_probe_report


def _apply_detector_penalty(analysis: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    detectors = evidence.get("detectors")
    if not isinstance(detectors, dict):
        analysis["confidence_adjusted"] = analysis.get("confidence", 0.0)
        analysis["confidence_penalty"] = 0.0
        analysis["detector_summary"] = {"finding_count": 0, "total_confidence_penalty": 0.0}
        return analysis

    penalty = float(detectors.get("total_confidence_penalty", 0.0) or 0.0)
    base_confidence = float(analysis.get("confidence", 0.0) or 0.0)
    adjusted = max(0.0, min(1.0, base_confidence - penalty))
    analysis["confidence_penalty"] = round(penalty, 3)
    analysis["confidence_adjusted"] = round(adjusted, 3)
    analysis["detector_summary"] = {
        "finding_count": int(detectors.get("finding_count", 0) or 0),
        "total_confidence_penalty": round(penalty, 3),
    }
    analysis["detector_findings"] = detectors.get("findings", [])
    return analysis


def build_analysis(results: dict[str, float], evidence: dict[str, Any]) -> dict[str, Any]:
    # Start from final resolved target values.
    metrics: dict[str, float] = {k: float(v) for k, v in results.items()}

    # Enrich with per-target candidate values for broader signal coverage.
    targets = evidence.get("targets", {})
    if isinstance(targets, dict):
        for _, target_evidence in targets.items():
            if not isinstance(target_evidence, dict):
                continue
            candidates = target_evidence.get("candidates")
            if isinstance(candidates, dict):
                for candidate_name, candidate_value in candidates.items():
                    if isinstance(candidate_value, (int, float)) and not isinstance(candidate_value, bool):
                        metrics[candidate_name] = float(candidate_value)

    baseline = analyze_bound(metrics).to_dict()
    workload_placeholders = evidence.get("workload_placeholders")
    if isinstance(workload_placeholders, dict):
        placeholder_targets = workload_placeholders.get("targets")
        if isinstance(placeholder_targets, list) and placeholder_targets:
            baseline["workload_placeholder_targets"] = [str(item) for item in placeholder_targets]
            baseline["workload_placeholder_count"] = len(placeholder_targets)
            notes = baseline.get("analysis_notes")
            if not isinstance(notes, list):
                notes = []
            notes.append(
                "workload_dependent_targets_without_run_were_left_as_placeholder_zero_values_and_excluded_from_observed_metrics"
            )
            baseline["analysis_notes"] = notes
    intrinsic_probe_report = build_intrinsic_probe_report(evidence)
    if intrinsic_probe_report.get("count", 0):
        baseline["intrinsic_probe_report"] = intrinsic_probe_report
        notes = baseline.get("analysis_notes")
        if not isinstance(notes, list):
            notes = []
        notes.append(
            "intrinsic_probe_report_summarizes_acceptance_reason_ncu_usage_and_semantic_validity_for_synthetic_probe_targets"
        )
        baseline["analysis_notes"] = notes
    synthetic_counter_probe_report = build_synthetic_counter_probe_report(evidence)
    if synthetic_counter_probe_report.get("count", 0):
        baseline["synthetic_counter_probe_report"] = synthetic_counter_probe_report
        accepted_proxy_targets = [
            str(item.get("target"))
            for item in synthetic_counter_probe_report.get("targets", [])
            if isinstance(item, dict) and item.get("accepted")
        ]
        if accepted_proxy_targets:
            baseline["proxy_signal_targets"] = accepted_proxy_targets
            baseline["proxy_signal_count"] = len(accepted_proxy_targets)
        notes = baseline.get("analysis_notes")
        if not isinstance(notes, list):
            notes = []
        notes.append(
            "synthetic_counter_probe_report_summarizes_proxy_counter_measurements_and_marks_them_as_non_workload_observations"
        )
        if accepted_proxy_targets:
            notes.append(
                "accepted_synthetic_counter_probe_targets_were_used_as_proxy_signals_for_bound_inference_not_as_direct_workload_observations"
            )
        baseline["analysis_notes"] = notes
    time_budget = evidence.get("time_budget")
    if isinstance(time_budget, dict) and time_budget.get("timed_out"):
        baseline["time_budget"] = time_budget
        notes = baseline.get("analysis_notes")
        if not isinstance(notes, list):
            notes = []
        notes.append(
            "time_budget_exhausted_before_all_requested_tasks_finished_remaining_targets_were_left_as_partial_timeout_results"
        )
        baseline["analysis_notes"] = notes
    llm_analysis = build_llm_analysis(results=results, evidence=evidence, baseline_analysis=baseline)

    if llm_analysis is None:
        analysis = baseline
        analysis["analysis_source"] = "rules"
    else:
        analysis = dict(baseline)
        analysis["analysis_source"] = "llm"
        analysis["llm_reasoning_summary"] = llm_analysis.get("llm_reasoning_summary", "")
        analysis["bound_type"] = llm_analysis.get("bound_type", analysis.get("bound_type"))
        analysis["confidence"] = float(llm_analysis.get("confidence", analysis.get("confidence", 0.0)))
        llm_bottlenecks = llm_analysis.get("bottlenecks")
        if isinstance(llm_bottlenecks, list) and llm_bottlenecks:
            analysis["bottlenecks"] = llm_bottlenecks

        guardrail_flags: list[str] = []
        if analysis.get("bound_type") == "unknown" and baseline.get("bound_type") != "unknown":
            guardrail_flags.append("llm_unknown_vs_rule_signal_present")
        if float(analysis.get("confidence", 0.0)) > 0.9 and baseline.get("missing_signals"):
            guardrail_flags.append("llm_high_confidence_with_missing_signals")
        analysis["rule_guardrail_flags"] = guardrail_flags

    return _apply_detector_penalty(analysis=analysis, evidence=evidence)
