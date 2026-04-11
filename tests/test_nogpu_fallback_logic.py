from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.orchestrator.pipeline import execute
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.tool_adapters.binary_runner import RunResult


class NoGpuFallbackLogicTests(unittest.TestCase):
    @patch("profiler_agent.orchestrator.pipeline.run_executable")
    @patch("profiler_agent.tool_adapters.ncu_adapter.subprocess.run")
    @patch("profiler_agent.tool_adapters.microbench_adapter.shutil.which")
    def test_pipeline_survives_without_gpu_tools(
        self,
        mock_which: unittest.mock.Mock,
        mock_ncu_run: unittest.mock.Mock,
        mock_run_exec: unittest.mock.Mock,
    ) -> None:
        mock_which.return_value = None  # nvcc unavailable
        mock_ncu_run.side_effect = FileNotFoundError("ncu missing")
        mock_run_exec.return_value = RunResult(
            command="dummy",
            returncode=0,
            stdout="ok",
            stderr="",
        )

        spec = TargetSpec(
            targets=["dram_latency_cycles", "actual_boost_clock_mhz"],
            run="cmd /c echo x",
        )
        out_dir = Path("tests/.tmp") / f"nogpu_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = execute(spec=spec, out_dir=out_dir)
            self.assertTrue(output.results_path.exists())
            self.assertTrue(output.evidence_path.exists())
            self.assertTrue(output.analysis_path.exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
