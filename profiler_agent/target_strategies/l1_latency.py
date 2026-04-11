from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class L1LatencyCyclesStrategy(ProbeFirstMetricStrategy):
    name = "l1_latency_cycles_strategy"
    target_hint = "l1_latency_cycles"

