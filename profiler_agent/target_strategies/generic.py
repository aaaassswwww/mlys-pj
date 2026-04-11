from __future__ import annotations

from profiler_agent.fusion.cross_verify import Candidate, fuse_candidates
from profiler_agent.target_strategies.base import MeasureContext, MeasureResult, TargetStrategy
from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence
from profiler_agent.tool_adapters.ncu_adapter import query_metric_with_evidence
from profiler_agent.tool_adapters.nvml_adapter import sample_sm_clock_mhz


class GenericMetricStrategy(TargetStrategy):
    name = "generic_metric"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        candidates: list[Candidate] = []
        candidate_values: dict[str, float] = {}
        tool_evidence: dict[str, object] = {}

        ncu_result = query_metric_with_evidence(ctx.target, ctx.run_cmd)
        tool_evidence["ncu"] = {
            "source": ncu_result.source,
            "returncode": ncu_result.returncode,
            "parse_mode": ncu_result.parse_mode,
            "stderr_tail": ncu_result.stderr_tail,
        }
        if ncu_result.value is not None:
            value = float(ncu_result.value)
            candidates.append(Candidate(source="ncu", value=value))
            candidate_values["ncu"] = value

        probe_result = measure_metric_with_evidence(ctx.target, ctx.run_cmd)
        tool_evidence["microbench"] = {
            "source": probe_result.source,
            "compile_returncode": probe_result.compile_returncode,
            "run_returncode": probe_result.run_returncode,
            "parsed_from": probe_result.parsed_from,
            "metric_name": probe_result.metric_name,
            "sample_count": probe_result.sample_count,
            "best_value": probe_result.best_value,
            "median_value": probe_result.median_value,
            "std_value": probe_result.std_value,
            "compile_stderr_tail": probe_result.compile_stderr_tail,
            "run_stderr_tail": probe_result.run_stderr_tail,
        }
        if probe_result.value is not None:
            value = float(probe_result.value)
            candidates.append(Candidate(source="microbench", value=value))
            candidate_values["microbench"] = value

        if ctx.target == "actual_boost_clock_mhz":
            sm_clock = sample_sm_clock_mhz(sample_count=7, interval_s=0.12)
            if sm_clock is not None:
                value = float(sm_clock)
                candidates.append(Candidate(source="nvml_sm_clock_median", value=value))
                candidate_values["nvml_sm_clock_median"] = value

        fusion_result = fuse_candidates(candidates, default_value=0.0)
        evidence = {
            "strategy": self.name,
            "selected_source": fusion_result.selected_source,
            "num_candidates": len(candidates),
            "candidates": candidate_values,
            "fusion": {
                "method": fusion_result.method,
                "confidence": fusion_result.confidence,
                "retained_sources": fusion_result.retained_sources,
                "dropped_sources": fusion_result.dropped_sources,
            },
            "tools": tool_evidence,
            "run_returncode": ctx.run_returncode,
        }
        return MeasureResult(value=fusion_result.value, evidence=evidence)
