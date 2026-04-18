from __future__ import annotations

from typing import Any


def build_intrinsic_probe_report(evidence: dict[str, Any]) -> dict[str, Any]:
    targets = evidence.get("targets", {})
    if not isinstance(targets, dict):
        return {"count": 0, "accepted_count": 0, "ncu_profiled_count": 0, "targets": []}

    entries: list[dict[str, Any]] = []
    accepted_count = 0
    ncu_profiled_count = 0

    for target, target_evidence in targets.items():
        if not isinstance(target_evidence, dict):
            continue
        if target_evidence.get("measurement_mode") != "synthetic_intrinsic_probe":
            continue

        probe_iteration = target_evidence.get("probe_iteration", {})
        if not isinstance(probe_iteration, dict):
            probe_iteration = {}
        analysis = probe_iteration.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
        probe = target_evidence.get("probe", {})
        if not isinstance(probe, dict):
            probe = {}

        profile_history = probe_iteration.get("state", {})
        if not isinstance(profile_history, dict):
            profile_history = {}
        raw_profile_history = profile_history.get("profile_history", [])
        used_ncu_profile = probe.get("profile_source") == "ncu_csv"
        if isinstance(raw_profile_history, list):
            used_ncu_profile = used_ncu_profile or any(
                isinstance(item, dict) and item.get("source") == "ncu_csv" for item in raw_profile_history
            )

        final_decision = str(probe_iteration.get("final_decision") or "").strip()
        acceptance_reason = str(analysis.get("reason") or final_decision or "not_recorded").strip()
        accepted = final_decision == "accept_measurement"

        if accepted:
            accepted_count += 1
        if used_ncu_profile:
            ncu_profiled_count += 1

        entries.append(
            {
                "target": str(target),
                "accepted": accepted,
                "final_decision": final_decision,
                "acceptance_reason": acceptance_reason,
                "used_ncu_profile": used_ncu_profile,
                "semantic_validity": str(target_evidence.get("semantic_validity") or ""),
                "measurement_mode": str(target_evidence.get("measurement_mode") or ""),
                "confidence": analysis.get("confidence", target_evidence.get("confidence")),
            }
        )

    return {
        "count": len(entries),
        "accepted_count": accepted_count,
        "ncu_profiled_count": ncu_profiled_count,
        "targets": entries,
    }
