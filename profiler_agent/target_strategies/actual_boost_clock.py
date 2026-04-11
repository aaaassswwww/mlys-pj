from __future__ import annotations

from profiler_agent.target_strategies.base import MeasureContext, MeasureResult
from profiler_agent.target_strategies.generic import GenericMetricStrategy


class ActualBoostClockStrategy(GenericMetricStrategy):
    name = "actual_boost_clock_strategy"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        result = super().measure(ctx)
        evidence = dict(result.evidence)
        evidence["target_hint"] = "actual_boost_clock_mhz"
        return MeasureResult(value=result.value, evidence=evidence)

