from __future__ import annotations

from profiler_agent.fusion.cross_verify import Candidate, fuse_candidates
from profiler_agent.target_strategies.base import MeasureContext, MeasureResult, TargetStrategy
from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence
from profiler_agent.tool_adapters.ncu_adapter import query_metric_with_evidence
from profiler_agent.tool_adapters.nvml_adapter import sample_sm_clock_stats


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _score_ncu_reliability(source: str, returncode: int, parse_mode: str) -> float:
    if source == "ncu_csv" and returncode == 0:
        score = 0.95
        if parse_mode in {"stdout_tail_numeric", "parse_failed"}:
            score -= 0.15
        return _clamp(score)
    if source in {"ncu_unavailable", "ncu_failed"}:
        return 0.0
    return 0.4


def _score_probe_reliability(sample_count: int | None, std_value: float | None, value: float | None) -> float:
    if value is None:
        return 0.0
    score = 0.75
    if sample_count is not None:
        if sample_count >= 5:
            score += 0.15
        elif sample_count <= 1:
            score -= 0.15
    if std_value is not None and value is not None:
        rel = abs(float(std_value)) / (abs(float(value)) + 1e-9)
        if rel > 0.35:
            score -= 0.2
        elif rel > 0.2:
            score -= 0.1
    return _clamp(score)


def _score_nvml_clock_reliability(stats: dict[str, object]) -> float:
    median = stats.get("median")
    count = stats.get("sample_count")
    std = stats.get("std")
    if not isinstance(median, (int, float)) or not isinstance(count, int):
        return 0.0
    score = 0.8
    if count >= 7:
        score += 0.1
    if isinstance(std, (int, float)):
        rel = abs(float(std)) / (abs(float(median)) + 1e-9)
        if rel > 0.03:
            score -= 0.2
        elif rel > 0.015:
            score -= 0.1
    return _clamp(score)


class GenericMetricStrategy(TargetStrategy):
    name = "generic_metric"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        candidates: list[Candidate] = []
        candidate_values: dict[str, float] = {}
        candidate_reliability: dict[str, float] = {}
        tool_evidence: dict[str, object] = {}

        ncu_result = query_metric_with_evidence(ctx.target, ctx.run_cmd)
        ncu_reliability = _score_ncu_reliability(
            source=ncu_result.source,
            returncode=ncu_result.returncode,
            parse_mode=ncu_result.parse_mode,
        )
        tool_evidence["ncu"] = {
            "source": ncu_result.source,
            "returncode": ncu_result.returncode,
            "parse_mode": ncu_result.parse_mode,
            "stderr_tail": ncu_result.stderr_tail,
            "reliability": ncu_reliability,
        }
        if ncu_result.value is not None:
            value = float(ncu_result.value)
            candidates.append(Candidate(source="ncu", value=value, reliability=ncu_reliability))
            candidate_values["ncu"] = value
            candidate_reliability["ncu"] = ncu_reliability

        probe_result = measure_metric_with_evidence(ctx.target, ctx.run_cmd)
        probe_reliability = _score_probe_reliability(
            sample_count=probe_result.sample_count,
            std_value=probe_result.std_value,
            value=probe_result.value,
        )
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
            "run_values": probe_result.run_values,
            "source_path": probe_result.source_path,
            "generation_source": probe_result.generation_source,
            "generation_attempts": probe_result.generation_attempts,
            "generation_error": probe_result.generation_error,
            "generation_trace": probe_result.generation_trace,
            "compile_stderr_tail": probe_result.compile_stderr_tail,
            "run_stderr_tail": probe_result.run_stderr_tail,
            "reliability": probe_reliability,
        }
        if probe_result.value is not None:
            value = float(probe_result.value)
            candidates.append(Candidate(source="microbench", value=value, reliability=probe_reliability))
            candidate_values["microbench"] = value
            candidate_reliability["microbench"] = probe_reliability

        if ctx.target == "actual_boost_clock_mhz":
            sm_clock_stats = sample_sm_clock_stats(sample_count=7, interval_s=0.12)
            nvml_reliability = _score_nvml_clock_reliability(sm_clock_stats)
            sm_clock_stats = {**sm_clock_stats, "reliability": nvml_reliability}
            tool_evidence["nvml_clock_probe"] = sm_clock_stats
            sm_clock_median = sm_clock_stats.get("median")
            if isinstance(sm_clock_median, (int, float)):
                value = float(sm_clock_median)
                candidates.append(Candidate(source="nvml_sm_clock_median", value=value, reliability=nvml_reliability))
                candidate_values["nvml_sm_clock_median"] = value
                candidate_reliability["nvml_sm_clock_median"] = nvml_reliability

        fusion_result = fuse_candidates(candidates, default_value=0.0)
        evidence = {
            "strategy": self.name,
            "selected_source": fusion_result.selected_source,
            "num_candidates": len(candidates),
            "candidates": candidate_values,
            "candidate_reliability": candidate_reliability,
            "fusion": {
                "method": fusion_result.method,
                "confidence": fusion_result.confidence,
                "retained_sources": fusion_result.retained_sources,
                "dropped_sources": fusion_result.dropped_sources,
                "source_reliability": fusion_result.source_reliability,
            },
            "tools": tool_evidence,
            "run_returncode": ctx.run_returncode,
        }
        return MeasureResult(value=fusion_result.value, evidence=evidence)
