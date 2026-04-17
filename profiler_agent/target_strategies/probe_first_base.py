from __future__ import annotations

from profiler_agent.target_strategies.base import MeasureContext, MeasureResult
from profiler_agent.target_strategies.generic import GenericMetricStrategy
from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence


class ProbeFirstMetricStrategy(GenericMetricStrategy):
    name = "probe_first_metric_strategy"
    target_hint = "unknown_target"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        probe = measure_metric_with_evidence(ctx.target, ctx.run_cmd)
        if probe.value is not None:
            return MeasureResult(
                value=float(probe.value),
                evidence={
                    "strategy": self.name,
                    "target_hint": self.target_hint,
                    "selected_source": "microbench_probe",
                    "probe": {
                        "source": probe.source,
                        "compile_returncode": probe.compile_returncode,
                        "run_returncode": probe.run_returncode,
                        "parsed_from": probe.parsed_from,
                        "metric_name": probe.metric_name,
                        "sample_count": probe.sample_count,
                        "best_value": probe.best_value,
                        "median_value": probe.median_value,
                        "std_value": probe.std_value,
                        "run_values": probe.run_values,
                        "compile_stderr_tail": probe.compile_stderr_tail,
                        "run_stderr_tail": probe.run_stderr_tail,
                        "source_path": probe.source_path,
                        "generation_source": probe.generation_source,
                        "generation_attempts": probe.generation_attempts,
                        "generation_error": probe.generation_error,
                        "generation_trace": probe.generation_trace,
                    },
                },
            )

        result = super().measure(ctx)
        evidence = dict(result.evidence)
        evidence["target_hint"] = self.target_hint
        evidence["probe_fallback"] = {
            "source": probe.source,
            "compile_returncode": probe.compile_returncode,
            "run_returncode": probe.run_returncode,
            "parsed_from": probe.parsed_from,
            "source_path": probe.source_path,
            "generation_source": probe.generation_source,
            "generation_attempts": probe.generation_attempts,
            "generation_error": probe.generation_error,
            "generation_trace": probe.generation_trace,
        }
        return MeasureResult(value=result.value, evidence=evidence)
