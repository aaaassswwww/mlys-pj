from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.multi_agent import MultiAgentCoordinator, MultiAgentRequest
from profiler_agent.orchestrator.pipeline import PipelineOutput
from profiler_agent.tool_adapters.binary_runner import RunResult


def _fake_execute(spec, out_dir: Path) -> PipelineOutput:
    _ = spec
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    evidence_path = out_dir / "evidence.json"
    analysis_path = out_dir / "analysis.json"
    results_path.write_text(json.dumps({"dram_latency_cycles": 420.0}), encoding="utf-8")
    evidence_path.write_text(json.dumps({"run": {"returncode": 0}}), encoding="utf-8")
    analysis_path.write_text(
        json.dumps(
            {
                "bound_type": "memory_bound",
                "confidence": 0.7,
                "confidence_adjusted": 0.6,
                "bottlenecks": [{"category": "MEMORY_PRESSURE"}],
            }
        ),
        encoding="utf-8",
    )
    return PipelineOutput(
        results_path=results_path,
        evidence_path=evidence_path,
        analysis_path=analysis_path,
        run_result=RunResult(command="cmd /c echo x", returncode=0, stdout="ok", stderr=""),
    )


class MultiAgentFrameworkTests(unittest.TestCase):
    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_coordinator_runs_full_framework(self, _mock_execute: unittest.mock.Mock) -> None:
        out_dir = Path("tests/.tmp") / f"ma_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = MultiAgentRequest(
                targets=["dram_latency_cycles", "actual_boost_clock_mhz"],
                run="cmd /c echo x",
                objective="analyze gpu profiling",
                out_dir=out_dir,
            )
            coordinator = MultiAgentCoordinator()
            result = coordinator.run(request)

            self.assertTrue((result.out_dir / "results.json").exists())
            self.assertIn("tool_calls", result.outputs)
            self.assertIn("pipeline", result.outputs)
            self.assertIn("interpretation", result.outputs)
            self.assertIn("next_actions", result.outputs)
            self.assertIn("nvml", result.plan.selected_tools)
            self.assertGreaterEqual(len(result.trace), 4)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
