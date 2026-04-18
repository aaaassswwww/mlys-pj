from __future__ import annotations

from profiler_agent.probe_iteration import run_probe_iteration
from profiler_agent.target_semantics import classify_target
from profiler_agent.target_strategies.base import MeasureContext, MeasureResult
from profiler_agent.target_strategies.generic import GenericMetricStrategy


class ProbeFirstMetricStrategy(GenericMetricStrategy):
    name = "probe_first_metric_strategy"
    target_hint = "unknown_target"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        semantic = ctx.target_semantic or classify_target(ctx.target)
        probe = run_probe_iteration(target=ctx.target, run_cmd=ctx.run_cmd)
        if probe.value is not None:
            return MeasureResult(
                value=float(probe.value),
                evidence={
                    "strategy": self.name,
                    "target_hint": self.target_hint,
                    "selected_source": "synthetic_intrinsic_probe",
                    "semantic": semantic.to_evidence(),
                    "workload_requirement": {
                        "workload_dependent": semantic.workload_dependent,
                        "status": "not_required",
                    },
                    **probe.evidence,
                },
            )

        result = super().measure(ctx)
        evidence = dict(result.evidence)
        evidence["target_hint"] = self.target_hint
        evidence["probe_fallback"] = probe.evidence
        return MeasureResult(value=result.value, evidence=evidence)
