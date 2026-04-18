from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from profiler_agent.multi_agent.executor import ExecutorAgent
from profiler_agent.multi_agent.models import ExecutionStep, MultiAgentRequest, MultiAgentState


class ExecutorPhase3Tests(unittest.TestCase):
    def test_executor_reports_missing_run_command_stage_when_empty(self) -> None:
        agent = ExecutorAgent()
        result = agent._run_tool_executor("")
        self.assertTrue(result["run_stage"]["skipped"])
        self.assertEqual(result["run_stage"]["reason"], "run_command_missing")

    @patch("profiler_agent.multi_agent.executor.probe_command")
    def test_ncu_missing_binary_stops_profile_stage(self, mock_probe: unittest.mock.Mock) -> None:
        from profiler_agent.runtime_tools import CommandProbe

        mock_probe.return_value = CommandProbe(
            command=["ncu", "--version"],
            available=False,
            resolved_path=None,
            returncode=127,
            stdout_tail="",
            stderr_tail="required_command_not_found:ncu",
        )
        agent = ExecutorAgent()
        result = agent._run_tool_ncu("dram__bytes_read.sum.per_second", "python -c \"print(1)\"")
        self.assertEqual(result["query"]["source"], "command_missing")
        self.assertEqual(result["profile_stage"]["error_type"], "command_missing")

    @patch("profiler_agent.multi_agent.executor.shutil.which")
    def test_microbench_missing_nvcc_stops_compile_stage(self, mock_which: unittest.mock.Mock) -> None:
        mock_which.return_value = None
        agent = ExecutorAgent()
        result = agent._run_tool_microbench("dram_latency_cycles", "")
        self.assertEqual(result["source"], "command_missing")
        self.assertEqual(result["compile_stage"]["error_type"], "command_missing")
        self.assertTrue(result["run_stage"]["skipped"])

    @patch("profiler_agent.multi_agent.executor.shutil.which", return_value=None)
    def test_run_tools_records_missing_nsys(self, _mock_which: unittest.mock.Mock) -> None:
        state = MultiAgentState(request=MultiAgentRequest(targets=["dram_latency_cycles"], run="", out_dir=Path("tests/.tmp")))
        step = ExecutionStep(id="tool_execution", owner="executor_agent", action="run_tools", payload={"tools": ["nsys"]})
        agent = ExecutorAgent()
        agent.run_tools(state=state, step=step)
        self.assertFalse(state.outputs["tool_calls"]["nsys"]["available"])
        self.assertEqual(state.outputs["tool_calls"]["nsys"]["profile_stage"]["error_type"], "command_missing")


if __name__ == "__main__":
    unittest.main()
