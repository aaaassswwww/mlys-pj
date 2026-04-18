from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiler_agent.analyzer.service import build_analysis
from profiler_agent.detectors.service import run_detectors
from profiler_agent.io.logger import build_logger
from profiler_agent.io.write_results import write_analysis, write_evidence, write_results
from profiler_agent.orchestrator.task_planner import build_task_plan
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
    }

    for target in build_task_plan(spec.targets):
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

    normalized_results, result_quality = normalize_results_with_specs(results=results, expected_targets=spec.targets)
    evidence["result_quality"] = result_quality
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
