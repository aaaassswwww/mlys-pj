from __future__ import annotations

from profiler_agent.target_semantics import classify_target
from profiler_agent.target_strategies.base import MeasureContext, MeasureResult, TargetStrategy
from profiler_agent.tool_adapters.nvml_adapter import query_named_device_attribute


class DeviceAttributeStrategy(TargetStrategy):
    name = "device_attribute_strategy"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        semantic = ctx.target_semantic or classify_target(ctx.target)
        query = query_named_device_attribute(ctx.target)
        evidence = {
            "strategy": self.name,
            "selected_source": "device_attribute_query" if query["value"] is not None else "fallback_default",
            "semantic": semantic.to_evidence(),
            "workload_requirement": {
                "workload_dependent": semantic.workload_dependent,
                "status": "not_required",
            },
            "attribute_query": {
                "backend_chain": query.get("backend_chain"),
                "fallbacks_considered": query.get("fallbacks_considered"),
            },
            "tools": {
                "device_attribute_query": query,
            },
            "run_returncode": ctx.run_returncode,
        }
        return MeasureResult(value=float(query["value"] or 0.0), evidence=evidence)
