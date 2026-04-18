from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.orchestrator.pipeline import execute
from profiler_agent.probe_iteration import ProbeIterationResult, ProbeIterationState
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.tool_adapters.binary_runner import RunResult
from profiler_agent.tool_adapters.microbench_adapter import ProbeResult
from profiler_agent.tool_adapters.ncu_adapter import NcuQueryResult


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures/golden") / name


def _mk_probe(
    value: float | None = None,
    source: str = "microbench_probe",
    sample_count: int | None = None,
    best: float | None = None,
    median: float | None = None,
    std: float | None = None,
) -> ProbeResult:
    ok = source == "microbench_probe"
    return ProbeResult(
        value=value,
        source=source,
        compile_returncode=0 if ok else 127,
        run_returncode=0 if ok else 127,
        compile_stdout_tail="",
        compile_stderr_tail="" if ok else "nvcc_not_found",
        run_stdout_tail="",
        run_stderr_tail="",
        compile_command=None,
        run_command=None,
        parsed_from="multi_run_median[structured_metric_value]" if value is not None else "none",
        metric_name=None,
        sample_count=sample_count,
        best_value=best,
        median_value=median,
        std_value=std,
        run_values=None,
    )


def _mk_probe_iteration_result(
    value: float | None = None,
    source: str = "microbench_probe",
    sample_count: int | None = None,
    best: float | None = None,
    median: float | None = None,
    std: float | None = None,
) -> ProbeIterationResult:
    state = ProbeIterationState(
        target="dram_latency_cycles",
        iteration=1,
        generation_attempts=1,
        compile_history=[],
        run_history=[],
        profile_history=[],
        analysis_history=[],
        done=value is not None,
        best_measurement=value,
    )
    return ProbeIterationResult(
        value=value,
        source=source,
        confidence=0.9 if value is not None else 0.0,
        state=state,
        evidence={
            "measurement_mode": "synthetic_intrinsic_probe",
            "semantic_validity": "intrinsic_proxy",
            "probe_iteration": {
                "iteration_count": 1,
                "final_decision": "accept_measurement" if value is not None else "change_probe_shape",
                "accepted_round": 1 if value is not None else None,
                "state": {
                    "target": state.target,
                    "iteration": state.iteration,
                    "generation_attempts": state.generation_attempts,
                    "compile_history": [],
                    "run_history": [],
                    "profile_history": [],
                    "analysis_history": [],
                    "done": state.done,
                    "best_measurement": state.best_measurement,
                },
            },
            "probe": {
                "source": source,
                "compile_returncode": 0 if value is not None else 127,
                "run_returncode": 0 if value is not None else 127,
                "parsed_from": "structured_metric_value" if value is not None else "none",
                "metric_name": "dram_latency_cycles",
                "sample_count": sample_count,
                "best_value": best,
                "median_value": median,
                "std_value": std,
                "run_values": None,
                "compile_stderr_tail": "" if value is not None else "nvcc_not_found",
                "run_stderr_tail": "",
                "compile_stdout_tail": "",
                "run_stdout_tail": "",
                "compile_command": None,
                "run_command": None,
                "source_path": None,
                "generation_source": "llm_generated",
                "generation_attempts": 1,
                "generation_error": "" if value is not None else "nvcc_not_found",
                "generation_trace": [],
            },
            "confidence": 0.9 if value is not None else 0.0,
        },
    )


def _project_outputs(out_dir: Path) -> dict[str, object]:
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    evidence = json.loads((out_dir / "evidence.json").read_text(encoding="utf-8"))
    analysis = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))

    return {
        "results": results,
        "evidence": {
            "run_returncode": evidence["run"]["returncode"],
            "detectors": {
                "finding_count": evidence["detectors"]["finding_count"],
                "total_confidence_penalty": evidence["detectors"]["total_confidence_penalty"],
                "ids": [item["id"] for item in evidence["detectors"]["findings"]],
            },
            "targets": {
                target: {
                    "selected_source": target_ev.get("selected_source"),
                    "fusion_method": (target_ev.get("fusion") or {}).get("method"),
                    "fusion_confidence": (target_ev.get("fusion") or {}).get("confidence"),
                }
                for target, target_ev in evidence["targets"].items()
            },
        },
        "analysis": {
            "bound_type": analysis["bound_type"],
            "confidence": analysis["confidence"],
            "confidence_penalty": analysis["confidence_penalty"],
            "confidence_adjusted": analysis["confidence_adjusted"],
            "missing_signals": analysis["missing_signals"],
            "bottleneck_categories": [item["category"] for item in analysis["bottlenecks"]],
        },
    }


class GoldenFixtureTests(unittest.TestCase):
    def _run_case(self, case_name: str) -> dict[str, object]:
        if case_name == "success":
            spec = TargetSpec(targets=["dram_latency_cycles", "actual_boost_clock_mhz"], run="cmd /c echo x")

            def probe_first(metric_name: str, run_cmd: str) -> ProbeResult:
                _ = run_cmd
                if metric_name == "dram_latency_cycles":
                    return _mk_probe(420.0, sample_count=5, best=410.0, median=420.0, std=6.0)
                return _mk_probe(None, source="unsupported_metric")

            def generic_probe(metric_name: str, run_cmd: str) -> ProbeResult:
                _ = metric_name, run_cmd
                return _mk_probe(None, source="unsupported_metric")

            clock_stats = {
                "sample_count": 7,
                "median": 2500.0,
                "std": 10.0,
                "min": 2490.0,
                "max": 2520.0,
                "range": 30.0,
                "values": [2490.0, 2500.0, 2505.0, 2510.0, 2498.0, 2520.0, 2495.0],
            }
        else:
            spec = TargetSpec(
                targets=[
                    "dram_latency_cycles",
                    "actual_boost_clock_mhz",
                    "max_shmem_per_block_kb",
                    "l2_cache_capacity_kb",
                ],
                run="cmd /c echo x",
            )

            def probe_first(metric_name: str, run_cmd: str) -> ProbeResult:
                _ = metric_name, run_cmd
                return _mk_probe(None, source="compile_failed")

            def generic_probe(metric_name: str, run_cmd: str) -> ProbeResult:
                _ = metric_name, run_cmd
                return _mk_probe(None, source="compile_failed")

            clock_stats = {
                "sample_count": 7,
                "median": 210.0,
                "std": 0.0,
                "min": 210.0,
                "max": 210.0,
                "range": 0.0,
                "values": [210.0] * 7,
            }

        def ncu_query(metric_name: str, run_cmd: str) -> NcuQueryResult:
            _ = metric_name, run_cmd
            return NcuQueryResult(
                value=None,
                source="ncu_unavailable",
                returncode=127,
                parse_mode="none",
                command=["ncu", "--metrics", metric_name],
                stdout_tail="",
                stderr_tail="ncu_not_found_or_timeout",
            )

        out_dir = Path("tests/.tmp") / f"golden_{case_name}_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (
                patch(
                    "profiler_agent.orchestrator.pipeline.run_executable",
                    return_value=RunResult(command=spec.run, returncode=0, stdout="ok", stderr=""),
                ),
                patch(
                    "profiler_agent.target_strategies.probe_first_base.run_probe_iteration",
                    side_effect=lambda target, run_cmd: _mk_probe_iteration_result(
                        value=probe_first(target, run_cmd).value,
                        source=probe_first(target, run_cmd).source,
                        sample_count=probe_first(target, run_cmd).sample_count,
                        best=probe_first(target, run_cmd).best_value,
                        median=probe_first(target, run_cmd).median_value,
                        std=probe_first(target, run_cmd).std_value,
                    ),
                ),
                patch(
                    "profiler_agent.target_strategies.generic.measure_metric_with_evidence",
                    side_effect=generic_probe,
                ),
                patch(
                    "profiler_agent.target_strategies.generic.query_metric_with_evidence",
                    side_effect=ncu_query,
                ),
                patch(
                    "profiler_agent.target_strategies.generic.sample_sm_clock_stats",
                    return_value=clock_stats,
                ),
            ):
                execute(spec=spec, out_dir=out_dir)
                return _project_outputs(out_dir=out_dir)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def _assert_case_matches_fixture(self, case_name: str, fixture_name: str) -> None:
        actual = self._run_case(case_name=case_name)
        expected = json.loads(_fixture_path(fixture_name).read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)

    def test_success_case_matches_golden_projection(self) -> None:
        self._assert_case_matches_fixture("success", "success_projection.json")

    def test_degraded_case_matches_golden_projection(self) -> None:
        self._assert_case_matches_fixture("degraded", "degraded_projection.json")


if __name__ == "__main__":
    unittest.main()
