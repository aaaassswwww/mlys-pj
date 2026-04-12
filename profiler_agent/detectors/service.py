from __future__ import annotations

import statistics
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _severity_from_spread(relative_spread: float) -> str:
    if relative_spread >= 0.75:
        return "high"
    if relative_spread >= 0.35:
        return "medium"
    return "low"


def _penalty_for_severity(severity: str) -> float:
    if severity == "high":
        return 0.2
    if severity == "medium":
        return 0.1
    return 0.05


def _detect_source_divergence(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    targets = evidence.get("targets", {})
    if not isinstance(targets, dict):
        return findings

    for target, target_evidence in targets.items():
        if not isinstance(target_evidence, dict):
            continue
        candidates = target_evidence.get("candidates")
        if not isinstance(candidates, dict):
            continue
        values = [float(v) for v in candidates.values() if _is_number(v)]
        if len(values) < 2:
            continue
        center = abs(float(statistics.median(values))) + 1e-9
        relative_spread = (max(values) - min(values)) / center
        if relative_spread < 0.35:
            continue
        severity = _severity_from_spread(relative_spread)
        findings.append(
            {
                "id": "source_divergence",
                "target": target,
                "severity": severity,
                "confidence_penalty": _penalty_for_severity(severity),
                "reason": f"Candidate sources diverge (relative_spread={relative_spread:.3f}).",
                "signals": {"candidates": candidates},
            }
        )
    return findings


def _detect_tool_blocking(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    targets = evidence.get("targets", {})
    if not isinstance(targets, dict) or not targets:
        return []

    unavailable = 0
    for _, target_evidence in targets.items():
        if not isinstance(target_evidence, dict):
            continue
        tools = target_evidence.get("tools")
        if not isinstance(tools, dict):
            continue
        ncu_source = ((tools.get("ncu") or {}).get("source")) if isinstance(tools.get("ncu"), dict) else None
        micro_source = (
            ((tools.get("microbench") or {}).get("source")) if isinstance(tools.get("microbench"), dict) else None
        )
        ncu_bad = ncu_source in {"ncu_unavailable", "ncu_failed"}
        micro_bad = micro_source in {"compile_failed", "run_failed", "probe_source_missing", "unsupported_metric"}
        if ncu_bad and micro_bad:
            unavailable += 1

    ratio = unavailable / max(len(targets), 1)
    if ratio < 0.5:
        return []
    severity = "high" if ratio >= 0.8 else "medium"
    return [
        {
            "id": "tool_path_blocking",
            "severity": severity,
            "confidence_penalty": _penalty_for_severity(severity),
            "reason": f"Multiple targets failed across both ncu and microbench paths (ratio={ratio:.2f}).",
            "signals": {"affected_target_ratio": round(ratio, 3)},
        }
    ]


def _detect_clock_lock(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    targets = evidence.get("targets", {})
    if not isinstance(targets, dict):
        return []
    boost = targets.get("actual_boost_clock_mhz")
    if not isinstance(boost, dict):
        return []
    tools = boost.get("tools")
    if not isinstance(tools, dict):
        return []
    clock_probe = tools.get("nvml_clock_probe")
    if not isinstance(clock_probe, dict):
        return []

    std = clock_probe.get("std")
    value_range = clock_probe.get("range")
    sample_count = clock_probe.get("sample_count")
    if not _is_number(std) or not _is_number(value_range) or not _is_number(sample_count):
        return []
    if int(sample_count) < 5:
        return []
    if float(std) > 1.0 or float(value_range) > 2.0:
        return []
    return [
        {
            "id": "clock_lock_or_static_state",
            "target": "actual_boost_clock_mhz",
            "severity": "low",
            "confidence_penalty": 0.05,
            "reason": "Observed SM clock variance is near zero; possible frequency lock or static workload state.",
            "signals": {"std": float(std), "range": float(value_range), "sample_count": int(sample_count)},
        }
    ]


def _detect_resource_mask(results: dict[str, float]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    shmem = results.get("max_shmem_per_block_kb")
    l2 = results.get("l2_cache_capacity_kb")
    if _is_number(shmem) and float(shmem) <= 32.0:
        findings.append(
            {
                "id": "resource_mask_suspected",
                "target": "max_shmem_per_block_kb",
                "severity": "low",
                "confidence_penalty": 0.05,
                "reason": f"Measured shared memory per block is unusually low ({float(shmem):.2f} KB).",
                "signals": {"max_shmem_per_block_kb": float(shmem)},
            }
        )
    if _is_number(l2) and float(l2) <= 256.0:
        findings.append(
            {
                "id": "resource_mask_suspected",
                "target": "l2_cache_capacity_kb",
                "severity": "low",
                "confidence_penalty": 0.05,
                "reason": f"Measured L2 capacity is unusually low ({float(l2):.2f} KB).",
                "signals": {"l2_cache_capacity_kb": float(l2)},
            }
        )
    return findings


def run_detectors(results: dict[str, float], evidence: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    findings.extend(_detect_source_divergence(evidence=evidence))
    findings.extend(_detect_tool_blocking(evidence=evidence))
    findings.extend(_detect_clock_lock(evidence=evidence))
    findings.extend(_detect_resource_mask(results=results))

    total_penalty = round(sum(float(item.get("confidence_penalty", 0.0)) for item in findings), 3)
    return {
        "version": "v1",
        "finding_count": len(findings),
        "total_confidence_penalty": total_penalty,
        "findings": findings,
    }

