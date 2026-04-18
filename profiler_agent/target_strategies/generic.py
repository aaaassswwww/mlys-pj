from __future__ import annotations

from profiler_agent.probe_iteration import run_probe_iteration
from profiler_agent.target_semantics import classify_target
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
        semantic = ctx.target_semantic or classify_target(ctx.target)
        if semantic.semantic_class == "intrinsic_probe":
            iteration = run_probe_iteration(target=ctx.target, run_cmd=ctx.run_cmd)
            evidence = {
                "strategy": self.name,
                "semantic": semantic.to_evidence(),
                "measurement_mode": "synthetic_intrinsic_probe",
                "semantic_validity": "intrinsic_proxy",
                "workload_requirement": {
                    "workload_dependent": semantic.workload_dependent,
                    "status": "not_required",
                },
                "selected_source": "synthetic_intrinsic_probe" if iteration.value is not None else "fallback_default",
                **iteration.evidence,
                "run_returncode": ctx.run_returncode,
            }
            return MeasureResult(value=float(iteration.value or 0.0), evidence=evidence)
        candidates: list[Candidate] = []
        candidate_values: dict[str, float] = {}
        candidate_reliability: dict[str, float] = {}
        tool_evidence: dict[str, object] = {}
        workload_status = "not_required"
        if semantic.workload_dependent:
            workload_status = "run_command_present" if (ctx.run_cmd or "").strip() else "missing_run_command"

        ncu_source = "skipped_for_semantic_class"
        ncu_returncode = 0
        ncu_parse_mode = "none"
        ncu_stderr_tail = ""
        ncu_stdout_tail = ""
        ncu_reliability = 0.0
        can_use_ncu = semantic.semantic_class == "workload_counter"
        if semantic.workload_dependent and not (ctx.run_cmd or "").strip():
            ncu_source = "workload_run_missing"
            ncu_stderr_tail = "run_skipped_no_command"
        elif can_use_ncu:
            ncu_result = query_metric_with_evidence(ctx.target, ctx.run_cmd)
            ncu_source = ncu_result.source
            ncu_returncode = ncu_result.returncode
            ncu_parse_mode = ncu_result.parse_mode
            ncu_stderr_tail = ncu_result.stderr_tail
            ncu_stdout_tail = ncu_result.stdout_tail
            ncu_reliability = _score_ncu_reliability(
                source=ncu_result.source,
                returncode=ncu_result.returncode,
                parse_mode=ncu_result.parse_mode,
            )
            if ncu_result.value is not None:
                value = float(ncu_result.value)
                candidates.append(Candidate(source="ncu", value=value, reliability=ncu_reliability))
                candidate_values["ncu"] = value
                candidate_reliability["ncu"] = ncu_reliability

        tool_evidence["ncu"] = {
            "source": ncu_source,
            "returncode": ncu_returncode,
            "parse_mode": ncu_parse_mode,
            "stdout_tail": ncu_stdout_tail,
            "stderr_tail": ncu_stderr_tail,
            "reliability": ncu_reliability,
            "semantic_gate": "enabled" if can_use_ncu else "skipped_for_semantic_class",
        }

        probe_source = "skipped_for_semantic_class"
        probe_reliability = 0.0
        probe_compile_returncode = 0
        probe_run_returncode = 0
        probe_parsed_from = "none"
        probe_metric_name = ctx.target
        probe_sample_count = None
        probe_best_value = None
        probe_median_value = None
        probe_std_value = None
        probe_run_values = None
        probe_source_path = None
        probe_generation_source = None
        probe_generation_attempts = None
        probe_generation_error = None
        probe_generation_trace = None
        probe_compile_stderr_tail = ""
        probe_compile_stdout_tail = ""
        probe_run_stderr_tail = ""
        probe_run_stdout_tail = ""
        probe_compile_command = None
        probe_run_command = None
        can_use_probe = semantic.semantic_class == "unknown"
        if semantic.workload_dependent and not (ctx.run_cmd or "").strip():
            probe_source = "workload_run_missing"
        elif can_use_probe:
            probe_result = measure_metric_with_evidence(ctx.target, ctx.run_cmd)
            probe_source = probe_result.source
            probe_compile_returncode = probe_result.compile_returncode
            probe_run_returncode = probe_result.run_returncode
            probe_parsed_from = probe_result.parsed_from
            probe_metric_name = probe_result.metric_name
            probe_sample_count = probe_result.sample_count
            probe_best_value = probe_result.best_value
            probe_median_value = probe_result.median_value
            probe_std_value = probe_result.std_value
            probe_run_values = probe_result.run_values
            probe_source_path = probe_result.source_path
            probe_generation_source = probe_result.generation_source
            probe_generation_attempts = probe_result.generation_attempts
            probe_generation_error = probe_result.generation_error
            probe_generation_trace = probe_result.generation_trace
            probe_compile_stderr_tail = probe_result.compile_stderr_tail
            probe_compile_stdout_tail = probe_result.compile_stdout_tail
            probe_run_stderr_tail = probe_result.run_stderr_tail
            probe_run_stdout_tail = probe_result.run_stdout_tail
            probe_compile_command = probe_result.compile_command
            probe_run_command = probe_result.run_command
            probe_reliability = _score_probe_reliability(
                sample_count=probe_result.sample_count,
                std_value=probe_result.std_value,
                value=probe_result.value,
            )
            if probe_result.value is not None:
                value = float(probe_result.value)
                candidates.append(Candidate(source="microbench", value=value, reliability=probe_reliability))
                candidate_values["microbench"] = value
                candidate_reliability["microbench"] = probe_reliability
        tool_evidence["microbench"] = {
            "source": probe_source,
            "compile_returncode": probe_compile_returncode,
            "run_returncode": probe_run_returncode,
            "parsed_from": probe_parsed_from,
            "metric_name": probe_metric_name,
            "sample_count": probe_sample_count,
            "best_value": probe_best_value,
            "median_value": probe_median_value,
            "std_value": probe_std_value,
            "run_values": probe_run_values,
            "source_path": probe_source_path,
            "generation_source": probe_generation_source,
            "generation_attempts": probe_generation_attempts,
            "generation_error": probe_generation_error,
            "generation_trace": probe_generation_trace,
            "compile_stdout_tail": probe_compile_stdout_tail,
            "compile_stderr_tail": probe_compile_stderr_tail,
            "run_stdout_tail": probe_run_stdout_tail,
            "run_stderr_tail": probe_run_stderr_tail,
            "compile_command": probe_compile_command,
            "run_command": probe_run_command,
            "reliability": probe_reliability,
            "semantic_gate": "enabled" if can_use_probe else "skipped_for_semantic_class",
        }

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
            "semantic": semantic.to_evidence(),
            "measurement_mode": (
                "placeholder_no_run"
                if semantic.workload_dependent and workload_status == "missing_run_command"
                else "workload_profile" if semantic.semantic_class == "workload_counter" else "conservative_fallback"
            ),
            "semantic_validity": (
                "unobserved_placeholder"
                if semantic.workload_dependent and workload_status == "missing_run_command"
                else "direct" if semantic.semantic_class == "workload_counter" else "intrinsic_proxy"
            ),
            "workload_requirement": {
                "workload_dependent": semantic.workload_dependent,
                "status": workload_status,
            },
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
