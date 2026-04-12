from __future__ import annotations

from typing import Any

from profiler_agent.analyzer.bound_classifier import analyze_bound


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

    analysis = analyze_bound(metrics).to_dict()
    return _apply_detector_penalty(analysis=analysis, evidence=evidence)

