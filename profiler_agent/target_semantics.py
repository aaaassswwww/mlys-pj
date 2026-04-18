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
    "shmem_bank_conflict_penalty_cycles",
    "max_shmem_per_block_kb",
}


@dataclass(frozen=True)
class TargetSemanticInfo:
    target: str
    semantic_class: str
    semantic_subclass: str
    measurement_mode_candidate: str
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
            semantic_subclass="device_attribute",
            measurement_mode_candidate="device_attribute_query",
            workload_dependent=False,
            route_reason="device_attribute_name_pattern",
        )

    if normalized in _INTRINSIC_MICROBENCH_TARGETS:
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="intrinsic_probe",
            semantic_subclass="intrinsic_microbench",
            measurement_mode_candidate="synthetic_intrinsic_probe",
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
            semantic_class="workload_counter",
            semantic_subclass="runtime_throughput_counter",
            measurement_mode_candidate="workload_profile_or_synthetic_counter_probe",
            workload_dependent=True,
            route_reason="runtime_counter_name_pattern",
        )

    if "__" in lowered:
        return TargetSemanticInfo(
            target=normalized,
            semantic_class="workload_counter",
            semantic_subclass="ncu_counter",
            measurement_mode_candidate="workload_profile_or_synthetic_counter_probe",
            workload_dependent=True,
            route_reason="ncu_counter_name_pattern",
        )

    return TargetSemanticInfo(
        target=normalized,
        semantic_class="unknown",
        semantic_subclass="unknown",
        measurement_mode_candidate="conservative_fallback",
        workload_dependent=False,
        route_reason="fallback_unknown_semantics",
    )
