from __future__ import annotations

from profiler_agent.target_strategies.actual_boost_clock import ActualBoostClockStrategy
from profiler_agent.target_semantics import TargetSemanticInfo, classify_target
from profiler_agent.target_strategies.base import TargetStrategy
from profiler_agent.target_strategies.device_attributes import DeviceAttributeStrategy
from profiler_agent.target_strategies.dram_latency import DramLatencyCyclesStrategy
from profiler_agent.target_strategies.global_peak_bandwidth import GlobalPeakBandwidthStrategy
from profiler_agent.target_strategies.generic import GenericMetricStrategy
from profiler_agent.target_strategies.l1_latency import L1LatencyCyclesStrategy
from profiler_agent.target_strategies.l2_cache_capacity import L2CacheCapacityStrategy
from profiler_agent.target_strategies.l2_latency import L2LatencyCyclesStrategy
from profiler_agent.target_strategies.max_shmem_per_block import MaxShmemPerBlockStrategy
from profiler_agent.target_strategies.shared_peak_bandwidth import SharedPeakBandwidthStrategy
from profiler_agent.target_strategies.shmem_bank_conflict_penalty import ShmemBankConflictPenaltyStrategy


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, type[TargetStrategy]] = {
            "l1_latency_cycles": L1LatencyCyclesStrategy,
            "l2_latency_cycles": L2LatencyCyclesStrategy,
            "dram_latency_cycles": DramLatencyCyclesStrategy,
            "shared_peak_bandwidth_gbps": SharedPeakBandwidthStrategy,
            "global_peak_bandwidth_gbps": GlobalPeakBandwidthStrategy,
            "l2_cache_capacity_kb": L2CacheCapacityStrategy,
            "max_shmem_per_block_kb": MaxShmemPerBlockStrategy,
            "actual_boost_clock_mhz": ActualBoostClockStrategy,
            "shmem_bank_conflict_penalty_cycles": ShmemBankConflictPenaltyStrategy,
        }
        self._default = GenericMetricStrategy

    def get(self, target: str, semantic: TargetSemanticInfo | None = None) -> TargetStrategy:
        semantic = semantic or classify_target(target)
        cls = self._strategies.get(target, self._default)
        if target not in self._strategies and semantic.semantic_class == "device_attribute":
            cls = DeviceAttributeStrategy
        return cls()
