from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiler_agent.tool_adapters.microbench_adapter import ProbeResult


@dataclass(frozen=True)
class ProbeAnalysisDecision:
    done: bool
    next_action: str
    confidence: float
    needs_ncu_profile: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "next_action": self.next_action,
            "confidence": self.confidence,
            "needs_ncu_profile": self.needs_ncu_profile,
            "reason": self.reason,
        }


def _stability_confidence(result: ProbeResult) -> float:
    if result.value is None:
        return 0.0
    score = 0.55
    if (result.sample_count or 0) >= 3:
        score += 0.15
    if (result.sample_count or 0) >= 5:
        score += 0.1
    if result.std_value is not None and result.value not in {None, 0.0}:
        relative_std = abs(float(result.std_value)) / (abs(float(result.value)) + 1e-9)
        if relative_std > 0.25:
            score -= 0.2
        elif relative_std > 0.12:
            score -= 0.1
    if result.parsed_from in {"stdout_last_numeric", "legacy_metric_equals"}:
        score -= 0.1
    if result.profile_source == "ncu_csv":
        score += 0.05
    return max(0.0, min(1.0, round(score, 3)))


def analyze_probe_round(
    *,
    target: str,
    result: ProbeResult,
    iteration: int,
    history: list[dict[str, Any]],
) -> ProbeAnalysisDecision:
    _ = target, history
    confidence = _stability_confidence(result)

    if result.source == "compile_failed":
        return ProbeAnalysisDecision(False, "repair_compile", 0.0, False, "compile_stage_failed")
    if result.source == "run_failed":
        return ProbeAnalysisDecision(False, "repair_runtime", 0.0, False, "run_stage_failed")
    if result.source in {"llm_generation_failed", "probe_source_missing"}:
        return ProbeAnalysisDecision(False, "change_probe_shape", 0.0, False, "probe_generation_failed")
    if result.source == "unsupported_metric":
        return ProbeAnalysisDecision(True, "unsupported_metric", 0.0, False, "unsupported_metric")

    if result.value is None:
        if result.profile_source in {None, "", "skipped_not_requested"}:
            return ProbeAnalysisDecision(False, "add_ncu_profile", 0.0, True, "no_value_and_profile_not_attempted")
        return ProbeAnalysisDecision(False, "change_probe_shape", 0.0, False, "no_value_after_execution")

    if (result.sample_count or 0) <= 1:
        return ProbeAnalysisDecision(False, "change_probe_shape", confidence, False, "insufficient_sample_count")
    if result.std_value is not None and result.value not in {None, 0.0}:
        relative_std = abs(float(result.std_value)) / (abs(float(result.value)) + 1e-9)
        if relative_std > 0.25:
            return ProbeAnalysisDecision(False, "change_probe_shape", confidence, False, "unstable_measurement")
    if result.parsed_from in {"stdout_last_numeric", "legacy_metric_equals"} and result.profile_source in {None, "", "skipped_not_requested"}:
        return ProbeAnalysisDecision(False, "add_ncu_profile", confidence, True, "weak_parse_signal")
    return ProbeAnalysisDecision(True, "accept_measurement", confidence, False, "measurement_accepted")
