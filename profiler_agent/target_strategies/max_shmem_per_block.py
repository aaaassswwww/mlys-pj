from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class MaxShmemPerBlockStrategy(ProbeFirstMetricStrategy):
    name = "max_shmem_per_block_strategy"
    target_hint = "max_shmem_per_block_kb"
