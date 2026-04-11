from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class L2LatencyCyclesStrategy(ProbeFirstMetricStrategy):
    name = "l2_latency_cycles_strategy"
    target_hint = "l2_latency_cycles"

