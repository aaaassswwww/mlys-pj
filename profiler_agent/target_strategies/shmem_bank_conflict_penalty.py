from __future__ import annotations

from profiler_agent.target_strategies.probe_first_base import ProbeFirstMetricStrategy


class ShmemBankConflictPenaltyStrategy(ProbeFirstMetricStrategy):
    name = "shmem_bank_conflict_penalty_strategy"
    target_hint = "shmem_bank_conflict_penalty_cycles"

