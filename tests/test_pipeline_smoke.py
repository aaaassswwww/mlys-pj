from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.orchestrator.pipeline import execute
from profiler_agent.schema.target_spec_schema import TargetSpec


class PipelineSmokeTests(unittest.TestCase):
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
                "intrinsic_microbench",
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
            self.assertEqual(
                evidence["targets"]["launch__sm_count"]["tools"]["device_attribute_query"]["source"],
                "cuda_runtime_attribute",
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
