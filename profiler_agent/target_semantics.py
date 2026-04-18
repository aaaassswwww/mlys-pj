from __future__ import annotations

from dataclasses import asdict, dataclass


_DEVICE_ATTRIBUTE_TARGETS = {
    "device__attribute_max_gpu_frequency_khz",
    "device__attribute_max_mem_frequency_khz",
    "device__attribute_fb_bus_width",
    "launch__sm_count",
}

_INTRINSIC_MICROBENCH_TARGETS = {
    "l1_latency_cycles",
    "l2_latency_cycles",
    "dram_latency_cycles",
    "shared_peak_bandwidth_gbps",
    "global_peak_bandwidth_gbps",
    "l2_cache_capacity_kb",
    "actual_boost_clock_mhz",
    "shmem_bank_conflict_penalty_cycles",
    "max_shmem_per_block_kb",
}


@dataclass(frozen=True)
class TargetSemanticInfo:
    target: str
    semantic_class: str
    workload_dependent: bool
    route_reason: str

    def to_evidence(self) -> dict[str, object]:
        return asdict(self)


def classify_target(target: str) -> TargetSemanticInfo:
    normalized = (target or "").strip()
    lowered = normalized.lower()

    if normalized in _DEVICE_ATTRIBUTE_TARGETS or lowered.startswith("device__attribute_"):
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="device_attribute",
            workload_dependent=False,
            route_reason="device_attribute_name_pattern",
        )

    if normalized in _INTRINSIC_MICROBENCH_TARGETS:
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="intrinsic_microbench",
            workload_dependent=False,
            route_reason="registered_intrinsic_target",
        )

    if "__" in lowered and (
        ".per_second" in lowered
        or "pct_of_peak_sustained_elapsed" in lowered
        or ".avg." in lowered
        or ".sum." in lowered
    ):
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="runtime_throughput_counter",
            workload_dependent=True,
            route_reason="runtime_counter_name_pattern",
        )

    if "__" in lowered:
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="ncu_counter",
            workload_dependent=True,
            route_reason="ncu_counter_name_pattern",
        )

    return TargetSemanticInfo(
        target=normalized,
        semantic_class="unknown",
        workload_dependent=False,
        route_reason="fallback_unknown_semantics",
    )
