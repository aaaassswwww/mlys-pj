from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from profiler_agent.probe_analysis import analyze_probe_round
from profiler_agent.tool_adapters.microbench_adapter import ProbeResult, measure_metric_with_evidence


def _default_max_iterations() -> int:
    raw = os.environ.get("PROFILER_AGENT_MAX_PROBE_ITERATIONS", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, min(value, 4))


@dataclass
class ProbeIterationState:
    target: str
    iteration: int = 0
    generation_attempts: int = 0
    compile_history: list[dict[str, Any]] = field(default_factory=list)
    run_history: list[dict[str, Any]] = field(default_factory=list)
    profile_history: list[dict[str, Any]] = field(default_factory=list)
    analysis_history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    best_measurement: float | None = None


@dataclass(frozen=True)
class ProbeIterationResult:
    value: float | None
    source: str
    confidence: float
    state: ProbeIterationState
    evidence: dict[str, Any]


def _build_repair_context(result: ProbeResult) -> str:
    parts = [
        f"probe_source={result.source}",
        f"compile_returncode={result.compile_returncode}",
        f"run_returncode={result.run_returncode}",
        f"parsed_from={result.parsed_from}",
    ]
    if result.compile_stderr_tail:
        parts.append(f"compile_stderr_tail={result.compile_stderr_tail[-800:]}")
    if result.run_stderr_tail:
        parts.append(f"run_stderr_tail={result.run_stderr_tail[-800:]}")
    if result.run_stdout_tail:
        parts.append(f"run_stdout_tail={result.run_stdout_tail[-800:]}")
    if result.generation_error:
        parts.append(f"generation_error={result.generation_error[-800:]}")
    return "\n".join(parts)


def _round_confidence(result: ProbeResult, iteration: int) -> float:
    if result.value is None:
        return 0.0
    score = 0.55
    if (result.sample_count or 0) >= 3:
        score += 0.15
    if (result.sample_count or 0) >= 5:
        score += 0.1
    if iteration == 1:
        score += 0.1
    if result.std_value is not None and result.value not in {None, 0.0}:
        relative_std = abs(float(result.std_value)) / (abs(float(result.value)) + 1e-9)
        if relative_std > 0.2:
            score -= 0.15
        elif relative_std > 0.1:
            score -= 0.05
    return max(0.0, min(1.0, round(score, 3)))


def run_probe_iteration(
    *,
    target: str,
    run_cmd: str = "",
    max_probe_iterations: int | None = None,
) -> ProbeIterationResult:
    state = ProbeIterationState(target=target)
    max_rounds = max_probe_iterations or _default_max_iterations()
    prior_error: str | None = None
    force_profile = False
    final_result: ProbeResult | None = None
    final_decision = "iteration_limit_reached"
    final_analysis: dict[str, Any] = {}

    for iteration in range(1, max_rounds + 1):
        state.iteration = iteration
        result = measure_metric_with_evidence(
            metric_name=target,
            run_cmd=run_cmd,
            prior_error=prior_error,
            force_profile=force_profile,
        )
        final_result = result
        state.generation_attempts += int(result.generation_attempts or 0)
        state.compile_history.append(
            {
                "iteration": iteration,
                "returncode": result.compile_returncode,
                "command": result.compile_command,
                "stdout_tail": result.compile_stdout_tail,
                "stderr_tail": result.compile_stderr_tail,
            }
        )
        state.run_history.append(
            {
                "iteration": iteration,
                "returncode": result.run_returncode,
                "command": result.run_command,
                "stdout_tail": result.run_stdout_tail,
                "stderr_tail": result.run_stderr_tail,
                "parsed_from": result.parsed_from,
            }
        )
        state.profile_history.append(
            {
                "iteration": iteration,
                "source": result.profile_source,
                "returncode": result.profile_returncode,
                "parse_mode": result.profile_parse_mode,
                "command": result.profile_command,
                "stdout_tail": result.profile_stdout_tail,
                "stderr_tail": result.profile_stderr_tail,
            }
        )
        analysis = analyze_probe_round(
            target=target,
            result=result,
            iteration=iteration,
            history=state.analysis_history,
        )
        final_analysis = analysis.to_dict()
        state.analysis_history.append(
            {
                "iteration": iteration,
                "source": result.source,
                "next_action": analysis.next_action,
                "value": result.value,
                "sample_count": result.sample_count,
                "confidence": analysis.confidence,
                "reason": analysis.reason,
                "needs_ncu_profile": analysis.needs_ncu_profile,
            }
        )
        if result.value is not None:
            state.best_measurement = float(result.value)
        if analysis.done:
            state.done = True
            final_decision = analysis.next_action
            break
        prior_error = _build_repair_context(result)
        if analysis.next_action == "add_ncu_profile":
            force_profile = True
            prior_error = f"{prior_error}\nnext_action=add_ncu_profile_for_probe_binary"
        final_decision = analysis.next_action

    assert final_result is not None
    confidence = _round_confidence(final_result, state.iteration)
    evidence = {
        "measurement_mode": "synthetic_intrinsic_probe",
        "semantic_validity": "intrinsic_proxy",
        "probe_iteration": {
            "iteration_count": state.iteration,
            "final_decision": final_decision,
            "accepted_round": state.iteration if state.done and final_result.value is not None else None,
            "analysis": final_analysis,
            "state": asdict(state),
        },
        "probe": {
            "source": final_result.source,
            "compile_returncode": final_result.compile_returncode,
            "run_returncode": final_result.run_returncode,
            "parsed_from": final_result.parsed_from,
            "metric_name": final_result.metric_name,
            "sample_count": final_result.sample_count,
            "best_value": final_result.best_value,
            "median_value": final_result.median_value,
            "std_value": final_result.std_value,
            "run_values": final_result.run_values,
            "compile_stderr_tail": final_result.compile_stderr_tail,
            "run_stderr_tail": final_result.run_stderr_tail,
            "compile_stdout_tail": final_result.compile_stdout_tail,
            "run_stdout_tail": final_result.run_stdout_tail,
            "compile_command": final_result.compile_command,
            "run_command": final_result.run_command,
            "source_path": final_result.source_path,
            "generation_source": final_result.generation_source,
            "generation_attempts": final_result.generation_attempts,
            "generation_error": final_result.generation_error,
            "generation_trace": final_result.generation_trace,
            "profile_source": final_result.profile_source,
            "profile_returncode": final_result.profile_returncode,
            "profile_parse_mode": final_result.profile_parse_mode,
            "profile_command": final_result.profile_command,
            "profile_stdout_tail": final_result.profile_stdout_tail,
            "profile_stderr_tail": final_result.profile_stderr_tail,
        },
        "confidence": confidence,
    }
    return ProbeIterationResult(
        value=final_result.value,
        source=final_result.source,
        confidence=confidence,
        state=state,
        evidence=evidence,
    )
