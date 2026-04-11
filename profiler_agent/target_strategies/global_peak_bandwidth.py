from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class GlobalPeakBandwidthStrategy(ProbeFirstMetricStrategy):
    name = "global_peak_bandwidth_strategy"
    target_hint = "global_peak_bandwidth_gbps"

