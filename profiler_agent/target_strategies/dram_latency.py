from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class DramLatencyCyclesStrategy(ProbeFirstMetricStrategy):
    name = "dram_latency_cycles_strategy"
    target_hint = "dram_latency_cycles"
