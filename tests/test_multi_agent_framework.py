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
            self.assertTrue((result.out_dir / "agent_state.json").exists())
            self.assertEqual(result.outputs["iteration_control"]["executed_rounds"], 2)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_selects_device_attribute_tool_for_phase2_targets(
        self,
        _mock_execute: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"ma_attr_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = MultiAgentRequest(
                targets=["launch__sm_count"],
                run="",
                objective="collect device attributes",
                out_dir=out_dir,
            )
            coordinator = MultiAgentCoordinator()
            result = coordinator.run(request)
            self.assertIn("device_attribute", result.plan.selected_tools)
            self.assertNotIn("microbench", result.plan.selected_tools)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_persists_iteration_history_across_runs(self, _mock_execute: unittest.mock.Mock) -> None:
        out_dir = Path("tests/.tmp") / f"ma_state_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            coordinator = MultiAgentCoordinator()
            request = MultiAgentRequest(
                targets=["dram_latency_cycles"],
                run="cmd /c echo x",
                objective="track state",
                out_dir=out_dir,
            )
            coordinator.run(request)
            coordinator.run(request)

            agent_state = json.loads((out_dir / "agent_state.json").read_text(encoding="utf-8"))
            self.assertEqual(agent_state["iteration"], 4)
            self.assertEqual(len(agent_state["selected_tools_history"]), 4)
            self.assertGreaterEqual(len(agent_state["analysis_history"]), 4)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_respects_max_iterations_metadata(self, _mock_execute: unittest.mock.Mock) -> None:
        out_dir = Path("tests/.tmp") / f"ma_max_iter_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            coordinator = MultiAgentCoordinator()
            request = MultiAgentRequest(
                targets=["dram_latency_cycles"],
                run="cmd /c echo x",
                objective="track state",
                out_dir=out_dir,
                metadata={"max_iterations": 1},
            )
            result = coordinator.run(request)
            self.assertEqual(result.outputs["iteration_control"]["executed_rounds"], 1)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_second_round_applies_refinement_directive(self, _mock_execute: unittest.mock.Mock) -> None:
        out_dir = Path("tests/.tmp") / f"ma_refine_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            coordinator = MultiAgentCoordinator()
            request = MultiAgentRequest(
                targets=["dram_latency_cycles"],
                run="cmd /c echo x",
                objective="track state",
                out_dir=out_dir,
            )
            result = coordinator.run(request)
            self.assertEqual(result.outputs["iteration_control"]["executed_rounds"], 2)
            second_round = result.outputs["iterations"][1]
            self.assertIn("ncu", second_round["selected_tools"])
            self.assertIn("ncu", second_round["round_directive"]["forced_tools"])
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.multi_agent.executor.execute", side_effect=_fake_execute)
    def test_multi_agent_does_not_rerun_without_run_command(self, _mock_execute: unittest.mock.Mock) -> None:
        out_dir = Path("tests/.tmp") / f"ma_no_run_iter_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            coordinator = MultiAgentCoordinator()
            request = MultiAgentRequest(
                targets=["dram__bytes_read.sum.per_second"],
                run="",
                objective="track state",
                out_dir=out_dir,
            )
            result = coordinator.run(request)
            self.assertEqual(result.outputs["iteration_control"]["executed_rounds"], 1)
            completed = [msg for msg in result.trace if msg.action == "iteration_completed"]
            self.assertEqual(completed[-1].content["decision_reason"], "rerun_requested_but_run_command_missing")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
