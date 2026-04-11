from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class L2CacheCapacityStrategy(ProbeFirstMetricStrategy):
    name = "l2_cache_capacity_strategy"
    target_hint = "l2_cache_capacity_kb"

