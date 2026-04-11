from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class SharedPeakBandwidthStrategy(ProbeFirstMetricStrategy):
    name = "shared_peak_bandwidth_strategy"
    target_hint = "shared_peak_bandwidth_gbps"

