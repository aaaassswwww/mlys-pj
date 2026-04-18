from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiler_agent.analyzer.service import build_analysis
from profiler_agent.detectors.service import run_detectors
from profiler_agent.io.logger import build_logger
from profiler_agent.io.write_results import write_analysis, write_evidence, write_results
from profiler_agent.orchestrator.task_planner import build_task_plan
from profiler_agent.report_summary import build_intrinsic_probe_report, build_synthetic_counter_probe_report
from profiler_agent.runtime_budget import build_timeout_metadata, get_runtime_budget_status, runtime_budget_expired
from profiler_agent.schema.result_schema import normalize_results_with_specs
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.target_semantics import classify_target
from profiler_agent.target_strategies.base import MeasureContext
from profiler_agent.target_strategies.registry import StrategyRegistry
from profiler_agent.tool_adapters.binary_runner import RunResult, run_executable


@dataclass
class PipelineOutput:
    results_path: Path
    evidence_path: Path
    analysis_path: Path
    run_result: RunResult


def _build_workload_placeholder_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    targets = evidence.get("targets", {})
    if not isinstance(targets, dict):
        return {"count": 0, "targets": [], "reason": ""}
    placeholder_targets: list[str] = []
    for target, target_evidence in targets.items():
        if not isinstance(target_evidence, dict):
            continue
        workload_req = target_evidence.get("workload_requirement")
        if not isinstance(workload_req, dict):
            continue
        if workload_req.get("status") != "missing_run_command":
            continue
        placeholder_targets.append(str(target))
    return {
        "count": len(placeholder_targets),
        "targets": placeholder_targets,
        "reason": "workload_dependent_targets_without_run_use_placeholder_zero_values",
    }


def _mark_timeout_targets(
    *,
    pending_targets: list[str],
    results: dict[str, float],
    evidence: dict[str, Any],
) -> None:
    if not pending_targets:
        return
    timeout_meta = build_timeout_metadata(
        reason="time_budget_exhausted_before_all_targets_completed",
        skipped_targets=pending_targets,
    )
    evidence["time_budget"] = timeout_meta
    for target in pending_targets:
        if target not in results:
            results[target] = 0.0
        semantic = classify_target(target)
        evidence["targets"][target] = {
            "strategy": "time_budget_short_circuit",
            "semantic": semantic.to_evidence(),
            "measurement_mode": "timeout_partial_result",
            "semantic_validity": "not_observed_due_to_timeout",
            "selected_source": "time_budget_short_circuit",
            "workload_requirement": {
                "workload_dependent": semantic.workload_dependent,
                "status": "not_attempted_due_to_time_budget",
            },
            "timeout": {
                "reason": timeout_meta["reason"],
                "remaining_seconds": timeout_meta["remaining_seconds"],
            },
            "run_returncode": evidence.get("run", {}).get("returncode", 0),
        }


def execute(spec: TargetSpec, out_dir: Path) -> PipelineOutput:
    logger = build_logger()
    if spec.run:
        logger.info("Running target executable once: %s", spec.run)
    else:
        logger.info("No run command provided; skipping workload execution")
    run_result = run_executable(spec.run)

    registry = StrategyRegistry()
    results: dict[str, float] = {}
    evidence: dict[str, Any] = {
        "run": {
            "command": run_result.command,
            "returncode": run_result.returncode,
            "stdout_tail": (run_result.stdout or "")[-1000:],
            "stderr_tail": (run_result.stderr or "")[-1000:],
        },
        "targets": {},
        "time_budget": get_runtime_budget_status(),
    }

    planned_targets = build_task_plan(spec.targets)
    measured_targets: list[str] = []
    for target in planned_targets:
        if runtime_budget_expired():
            remaining_targets = [item for item in planned_targets if item not in measured_targets]
            _mark_timeout_targets(pending_targets=remaining_targets, results=results, evidence=evidence)
            break
        semantic = classify_target(target)
        strategy = registry.get(target, semantic=semantic)
        logger.info("Measuring target '%s' using strategy '%s'", target, strategy.name)
        ctx = MeasureContext(
            target=target,
            run_cmd=spec.run,
            run_returncode=run_result.returncode,
            run_stdout=run_result.stdout,
            run_stderr=run_result.stderr,
            target_semantic=semantic,
        )
        measurement = strategy.measure(ctx)
        results[target] = float(measurement.value)
        target_evidence = dict(measurement.evidence)
        target_evidence.setdefault("semantic", semantic.to_evidence())
        evidence["targets"][target] = target_evidence
        measured_targets.append(target)

    if not runtime_budget_expired():
        pending_targets = [target for target in spec.targets if target not in results]
        for target in pending_targets:
            results[target] = 0.0

    if not (isinstance(evidence.get("time_budget"), dict) and evidence["time_budget"].get("timed_out")):
        evidence["time_budget"] = get_runtime_budget_status()
    normalized_results, result_quality = normalize_results_with_specs(results=results, expected_targets=spec.targets)
    evidence["result_quality"] = result_quality
    evidence["workload_placeholders"] = _build_workload_placeholder_summary(evidence)
    evidence["intrinsic_probe_report"] = build_intrinsic_probe_report(evidence)
    evidence["synthetic_counter_probe_report"] = build_synthetic_counter_probe_report(evidence)
    evidence["detectors"] = run_detectors(results=normalized_results, evidence=evidence)
    analysis = build_analysis(results=normalized_results, evidence=evidence)
    results_path = write_results(out_dir, normalized_results, spec.targets)
    evidence_path = write_evidence(out_dir, evidence)
    analysis_path = write_analysis(out_dir, analysis)
    return PipelineOutput(
        results_path=results_path,
        evidence_path=evidence_path,
        analysis_path=analysis_path,
        run_result=run_result,
    )
