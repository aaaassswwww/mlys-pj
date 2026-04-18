from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    unit: str
    min_value: float
    max_value: float
    round_digits: int
    integer_like: bool = False


METRIC_SPECS: dict[str, MetricSpec] = {
    "l1_latency_cycles": MetricSpec(unit="cycles", min_value=0.0, max_value=50000.0, round_digits=3),
    "l2_latency_cycles": MetricSpec(unit="cycles", min_value=0.0, max_value=100000.0, round_digits=3),
    "dram_latency_cycles": MetricSpec(unit="cycles", min_value=0.0, max_value=1000000.0, round_digits=3),
    "shared_peak_bandwidth_gbps": MetricSpec(unit="GB/s", min_value=0.0, max_value=200000.0, round_digits=3),
    "global_peak_bandwidth_gbps": MetricSpec(unit="GB/s", min_value=0.0, max_value=20000.0, round_digits=3),
    "l2_cache_capacity_kb": MetricSpec(unit="KB", min_value=0.0, max_value=1048576.0, round_digits=0, integer_like=True),
    "actual_boost_clock_mhz": MetricSpec(unit="MHz", min_value=0.0, max_value=10000.0, round_digits=3),
    "device__attribute_max_gpu_frequency_khz": MetricSpec(
        unit="kHz", min_value=0.0, max_value=10000000.0, round_digits=0, integer_like=True
    ),
    "device__attribute_max_mem_frequency_khz": MetricSpec(
        unit="kHz", min_value=0.0, max_value=10000000.0, round_digits=0, integer_like=True
    ),
    "device__attribute_fb_bus_width": MetricSpec(
        unit="bits", min_value=0.0, max_value=65536.0, round_digits=0, integer_like=True
    ),
    "launch__sm_count": MetricSpec(unit="count", min_value=0.0, max_value=65536.0, round_digits=0, integer_like=True),
    "max_shmem_per_block_kb": MetricSpec(unit="KB", min_value=0.0, max_value=4096.0, round_digits=3),
    "shmem_bank_conflict_penalty_cycles": MetricSpec(
        unit="cycles", min_value=0.0, max_value=1000000.0, round_digits=3
    ),
}

