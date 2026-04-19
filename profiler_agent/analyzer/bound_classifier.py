from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Bottleneck:
    category: str
    severity: str
    reason: str
    suggestion: str


@dataclass(frozen=True)
class AnalysisResult:
    bound_type: str
    confidence: float
    compute_score: float
    memory_score: float
    observed_metrics: dict[str, float]
    missing_signals: list[str]
    bottlenecks: list[Bottleneck]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bottlenecks"] = [asdict(item) for item in self.bottlenecks]
        return data


_COMPUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "sm_efficiency": (
        "sm_efficiency",
        "sm_utilization",
        "compute_utilization",
        "sm_busy_pct",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "achieved_occupancy": ("achieved_occupancy", "occupancy"),
    "flop_efficiency": ("flop_sp_efficiency", "flop_dp_efficiency", "tensor_core_utilization"),
}

_MEMORY_ALIASES: dict[str, tuple[str, ...]] = {
    "dram_utilization": (
        "dram_utilization",
        "memory_utilization",
        "memory_bw_utilization",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "dram_throughput_gbps": (
        "dram_throughput_gbps",
        "dram_read_throughput",
        "dram_write_throughput",
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
    ),
    "dram_latency_cycles": ("dram_latency_cycles",),
}


def _normalize_percent_like(value: float) -> float:
    if value < 0:
        return 0.0
    if value <= 1.0:
        return value
    if value <= 100.0:
        return value / 100.0
    return 1.0


def _pick_first(metrics: dict[str, float], aliases: tuple[str, ...]) -> Optional[tuple[str, float]]:
    for name in aliases:
        if name in metrics:
            return name, float(metrics[name])
    return None


def _collect_matches(metrics: dict[str, float], aliases: tuple[str, ...]) -> dict[str, float]:
    observed: dict[str, float] = {}
    for name in aliases:
        if name in metrics:
            observed[name] = float(metrics[name])
    return observed


def _memory_latency_penalty(metrics: dict[str, float]) -> float:
    entry = _pick_first(metrics, _MEMORY_ALIASES["dram_latency_cycles"])
    if not entry:
        return 0.0
    _, latency = entry
    # Approximate bins in cycles; intentionally conservative.
    if latency >= 600:
        return 0.35
    if latency >= 400:
        return 0.2
    if latency >= 250:
        return 0.1
    return 0.0


def _compute_score(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    observed: dict[str, float] = {}
    parts: list[float] = []
    for canonical, aliases in _COMPUTE_ALIASES.items():
        match = _pick_first(metrics, aliases)
        if not match:
            continue
        source_name, raw_value = match
        observed.update(_collect_matches(metrics, aliases))
        parts.append(_normalize_percent_like(raw_value))
    if not parts:
        return 0.0, observed
    return sum(parts) / len(parts), observed


def _memory_score(metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    observed: dict[str, float] = {}
    parts: list[float] = []
    util_match = _pick_first(metrics, _MEMORY_ALIASES["dram_utilization"])
    if util_match:
        name, value = util_match
        observed.update(_collect_matches(metrics, _MEMORY_ALIASES["dram_utilization"]))
        parts.append(_normalize_percent_like(value))

    bw_match = _pick_first(metrics, _MEMORY_ALIASES["dram_throughput_gbps"])
    if bw_match:
        name, value = bw_match
        observed.update(_collect_matches(metrics, _MEMORY_ALIASES["dram_throughput_gbps"]))
        # Throughput in GB/s is hard to normalize without hardware peak.
        # Keep this as a weak signal if utilization metric is absent.
        if not util_match:
            parts.append(0.5)

    if not parts:
        base = 0.0
    else:
        base = sum(parts) / len(parts)
    return min(1.0, base + _memory_latency_penalty(metrics)), observed


def analyze_bound(metrics: dict[str, float]) -> AnalysisResult:
    compute_score, compute_observed = _compute_score(metrics)
    memory_score, memory_observed = _memory_score(metrics)

    observed = {**compute_observed, **memory_observed}
    missing_signals: list[str] = []
    if not compute_observed:
        missing_signals.append("compute_signal_missing")
    if not memory_observed:
        missing_signals.append("memory_signal_missing")

    bottlenecks: list[Bottleneck] = []
    bound_type = "unknown"
    confidence = 0.35

    delta = compute_score - memory_score
    if compute_score == 0.0 and memory_score == 0.0:
        bound_type = "unknown"
        confidence = 0.2
        bottlenecks.append(
            Bottleneck(
                category="INSUFFICIENT_SIGNAL",
                severity="MEDIUM",
                reason="No reliable compute or memory utilization metrics were observed.",
                suggestion="Collect SM efficiency/occupancy and DRAM utilization with ncu in the same run.",
            )
        )
    elif delta >= 0.15:
        bound_type = "compute_bound"
        confidence = 0.7 if compute_observed and memory_observed else 0.55
        bottlenecks.append(
            Bottleneck(
                category="COMPUTE_PRESSURE",
                severity="HIGH" if compute_score >= 0.8 else "MEDIUM",
                reason=f"Compute score ({compute_score:.2f}) is higher than memory score ({memory_score:.2f}).",
                suggestion="Improve arithmetic efficiency (kernel fusion, tensor core usage, instruction mix tuning).",
            )
        )
    elif delta <= -0.15:
        bound_type = "memory_bound"
        confidence = 0.7 if compute_observed and memory_observed else 0.55
        bottlenecks.append(
            Bottleneck(
                category="MEMORY_PRESSURE",
                severity="HIGH" if memory_score >= 0.8 else "MEDIUM",
                reason=f"Memory score ({memory_score:.2f}) is higher than compute score ({compute_score:.2f}).",
                suggestion="Reduce memory traffic (coalescing, data reuse, tiling, cache-friendly access).",
            )
        )
    else:
        bound_type = "balanced_or_mixed"
        confidence = 0.55
        bottlenecks.append(
            Bottleneck(
                category="MIXED_LIMIT",
                severity="MEDIUM",
                reason=f"Compute score ({compute_score:.2f}) and memory score ({memory_score:.2f}) are close.",
                suggestion="Profile top kernels and optimize both occupancy and memory access patterns.",
            )
        )

    latency_match = _pick_first(metrics, _MEMORY_ALIASES["dram_latency_cycles"])
    if latency_match and latency_match[1] >= 400:
        bottlenecks.append(
            Bottleneck(
                category="HIGH_DRAM_LATENCY",
                severity="HIGH" if latency_match[1] >= 600 else "MEDIUM",
                reason=f"Observed DRAM latency is high ({latency_match[1]:.1f} cycles).",
                suggestion="Increase parallelism and memory-level locality to hide latency.",
            )
        )

    return AnalysisResult(
        bound_type=bound_type,
        confidence=round(confidence, 3),
        compute_score=round(compute_score, 4),
        memory_score=round(memory_score, 4),
        observed_metrics=observed,
        missing_signals=missing_signals,
        bottlenecks=bottlenecks,
    )

