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


class PipelineSmokeTests(unittest.TestCase):
    @staticmethod
    def _synthetic_counter_probe_result(target: str, value: float = 987.0) -> ProbeIterationResult:
        state = ProbeIterationState(
            target=target,
            iteration=1,
            generation_attempts=1,
            compile_history=[],
            run_history=[],
            profile_history=[{"iteration": 1, "source": "ncu_csv"}],
            analysis_history=[],
            done=True,
            best_measurement=value,
        )
        return ProbeIterationResult(
            value=value,
            source="microbench_probe",
            confidence=0.82,
            state=state,
            evidence={
                "measurement_mode": "synthetic_counter_probe",
                "semantic_validity": "synthetic_counter_proxy",
                "probe_iteration": {
                    "iteration_count": 1,
                    "final_decision": "accept_measurement",
                    "accepted_round": 1,
                    "analysis": {
                        "next_action": "accept_measurement",
                        "reason": "synthetic_counter_probe_accepted",
                        "confidence": 0.82,
                    },
                    "state": {
                        "target": target,
                        "iteration": 1,
                        "generation_attempts": 1,
                        "compile_history": [],
                        "run_history": [],
                        "profile_history": [{"iteration": 1, "source": "ncu_csv"}],
                        "analysis_history": [],
                        "done": True,
                        "best_measurement": value,
                    },
                },
                "probe": {
                    "source": "microbench_probe",
                    "compile_returncode": 0,
                    "run_returncode": 0,
                    "parsed_from": "multi_run_median[structured_metric_value]",
                    "metric_name": target,
                    "sample_count": 5,
                    "best_value": value - 3.0,
                    "median_value": value,
                    "std_value": 4.0,
                    "run_values": [value - 3.0, value, value + 2.0],
                    "compile_stderr_tail": "",
                    "run_stderr_tail": "",
                    "compile_stdout_tail": "",
                    "run_stdout_tail": "",
                    "compile_command": ["nvcc"],
                    "run_command": ["probe.exe"],
                    "source_path": None,
                    "generation_source": "llm_generated",
                    "generation_attempts": 1,
                    "generation_error": "",
                    "generation_trace": [],
                    "profile_source": "ncu_csv",
                    "profile_returncode": 0,
                    "profile_parse_mode": "csv_metric_value",
                    "profile_command": ["ncu"],
                    "profile_stdout_tail": "",
                    "profile_stderr_tail": "",
                },
                "confidence": 0.82,
            },
        )

    def test_execute_writes_results_and_evidence(self) -> None:
        spec = TargetSpec(
            targets=["dram_latency_cycles", "actual_boost_clock_mhz"],
            run="cmd /c echo smoke_test_binary",
        )
        out_dir = Path("tests/.tmp") / f"smoke_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = execute(spec, out_dir)

            self.assertTrue(output.results_path.exists())
            self.assertTrue(output.evidence_path.exists())
            self.assertTrue(output.analysis_path.exists())

            results = json.loads(output.results_path.read_text(encoding="utf-8"))
            evidence = json.loads(output.evidence_path.read_text(encoding="utf-8"))
            self.assertIn("dram_latency_cycles", results)
            self.assertIn("actual_boost_clock_mhz", results)
            self.assertIsInstance(results["dram_latency_cycles"], (int, float))
            self.assertIn("result_quality", evidence)
            self.assertIn("detectors", evidence)
            self.assertEqual(
                evidence["targets"]["dram_latency_cycles"]["semantic"]["semantic_class"],
                "intrinsic_probe",
            )
            self.assertEqual(
                evidence["targets"]["dram_latency_cycles"]["measurement_mode"],
                "synthetic_intrinsic_probe",
            )
            self.assertEqual(evidence["intrinsic_probe_report"]["count"], 1)
            self.assertEqual(evidence["intrinsic_probe_report"]["targets"][0]["target"], "dram_latency_cycles")
            self.assertEqual(
                evidence["intrinsic_probe_report"]["targets"][0]["semantic_validity"],
                "intrinsic_proxy",
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_execute_without_run_still_writes_outputs(self) -> None:
        spec = TargetSpec(
            targets=["dram_latency_cycles"],
            run="",
        )
        out_dir = Path("tests/.tmp") / f"smoke_norun_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = execute(spec, out_dir)

            self.assertTrue(output.results_path.exists())
            evidence = json.loads(output.evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["run"]["command"], "")
            self.assertEqual(evidence["run"]["returncode"], 0)
            self.assertEqual(evidence["run"]["stderr_tail"], "run_skipped_no_command")
            self.assertEqual(
                evidence["targets"]["dram_latency_cycles"]["workload_requirement"]["status"],
                "not_required",
            )
            self.assertEqual(
                evidence["targets"]["dram_latency_cycles"]["semantic_validity"],
                "intrinsic_proxy",
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.target_strategies.generic.run_probe_iteration")
    def test_execute_without_run_uses_synthetic_counter_probe(
        self,
        mock_probe_iteration: unittest.mock.Mock,
    ) -> None:
        target = "dram__bytes_read.sum.per_second"
        mock_probe_iteration.return_value = self._synthetic_counter_probe_result(target)
        spec = TargetSpec(
            targets=[target],
            run="",
        )
        out_dir = Path("tests/.tmp") / f"smoke_placeholder_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = execute(spec, out_dir)
            evidence = json.loads(output.evidence_path.read_text(encoding="utf-8"))
            analysis = json.loads(output.analysis_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["targets"][target]["measurement_mode"], "synthetic_counter_probe")
            self.assertEqual(evidence["targets"][target]["semantic_validity"], "synthetic_counter_proxy")
            self.assertEqual(
                evidence["targets"][target]["workload_requirement"]["status"],
                "missing_run_command_replaced_with_synthetic_probe",
            )
            self.assertEqual(evidence["workload_placeholders"]["count"], 0)
            self.assertEqual(evidence["synthetic_counter_probe_report"]["count"], 1)
            self.assertEqual(evidence["synthetic_counter_probe_report"]["ncu_profiled_count"], 1)
            self.assertEqual(analysis["synthetic_counter_probe_report"]["count"], 1)
            self.assertIn(
                "synthetic_counter_probe_report_summarizes_proxy_counter_measurements_and_marks_them_as_non_workload_observations",
                analysis["analysis_notes"],
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.target_strategies.device_attributes.query_named_device_attribute")
    def test_execute_device_attribute_target_uses_dedicated_attribute_evidence(
        self,
        mock_query: unittest.mock.Mock,
    ) -> None:
        mock_query.return_value = {
            "target": "launch__sm_count",
            "source": "cuda_runtime_attribute",
            "value": 114.0,
            "field": "cudaDeviceGetAttribute(16)",
            "unit": "count",
            "command": ["cudart64_130.dll"],
            "returncode": 0,
            "stdout_tail": "device=0 raw_value=114",
            "stderr_tail": "",
            "backend_chain": ["cuda_runtime_attribute"],
            "fallbacks_considered": ["nvidia_smi_query"],
        }
        spec = TargetSpec(targets=["launch__sm_count"], run="")
        out_dir = Path("tests/.tmp") / f"smoke_attr_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = execute(spec, out_dir)
            evidence = json.loads(output.evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["targets"]["launch__sm_count"]["strategy"], "device_attribute_strategy")
            self.assertEqual(evidence["targets"]["launch__sm_count"]["semantic"]["semantic_class"], "device_attribute")
            self.assertEqual(evidence["targets"]["launch__sm_count"]["measurement_mode"], "device_attribute_query")
            self.assertEqual(
                evidence["targets"]["launch__sm_count"]["tools"]["device_attribute_query"]["source"],
                "cuda_runtime_attribute",
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
