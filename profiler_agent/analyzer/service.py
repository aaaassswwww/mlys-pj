from __future__ import annotations

from typing import Any

from profiler_agent.analyzer.bound_classifier import analyze_bound


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

    analysis = analyze_bound(metrics)
    return analysis.to_dict()

