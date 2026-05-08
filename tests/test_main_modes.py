from __future__ import annotations

import argparse
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.main import main
from profiler_agent.multi_agent.models import AgentMessage, ExecutionPlan, ExecutionStep, MultiAgentResult
from profiler_agent.orchestrator.pipeline import PipelineOutput
from profiler_agent.phase2.optimizer import Phase2OptimizationResult
from profiler_agent.phase2.models import Phase2OptimizerState
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.tool_adapters.binary_runner import RunResult


class MainModeTests(unittest.TestCase):
    @patch("profiler_agent.main.execute")
    @patch("profiler_agent.main.initialize_runtime_budget")
    @patch("profiler_agent.main.load_target_spec")
    @patch("profiler_agent.main.parse_args")
    def test_single_mode_keeps_existing_pipeline_path(
        self,
        mock_parse_args: unittest.mock.Mock,
        mock_load_spec: unittest.mock.Mock,
        mock_initialize_budget: unittest.mock.Mock,
        mock_execute: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"main_single_{uuid4().hex}"
        mock_parse_args.return_value = argparse.Namespace(
            spec=Path("inputs/target_spec.json"),
            out=out_dir,
            mode="single",
            objective="",
            phase2_iterations=15,
            llm_secret_file=None,
            llm_base_url="",
            llm_model="",
        )
        mock_load_spec.return_value = TargetSpec(targets=["dram_latency_cycles"], run="cmd /c echo x")
        mock_execute.return_value = PipelineOutput(
            results_path=out_dir / "results.json",
            evidence_path=out_dir / "evidence.json",
            analysis_path=out_dir / "analysis.json",
            run_result=RunResult(command="cmd /c echo x", returncode=0, stdout="", stderr=""),
        )

        rc = main()
        self.assertEqual(rc, 0)
        mock_initialize_budget.assert_called_once()
        mock_execute.assert_called_once()

    @patch("profiler_agent.main.execute")
    @patch("profiler_agent.main.load_target_spec")
    @patch("profiler_agent.main.parse_args")
    def test_single_mode_allows_missing_run(
        self,
        mock_parse_args: unittest.mock.Mock,
        mock_load_spec: unittest.mock.Mock,
        mock_execute: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"main_single_norun_{uuid4().hex}"
        mock_parse_args.return_value = argparse.Namespace(
            spec=Path("inputs/target_spec.json"),
            out=out_dir,
            mode="single",
            objective="",
            phase2_iterations=15,
            llm_secret_file=None,
            llm_base_url="",
            llm_model="",
        )
        mock_load_spec.return_value = TargetSpec(targets=["dram_latency_cycles"], run="")
        mock_execute.return_value = PipelineOutput(
            results_path=out_dir / "results.json",
            evidence_path=out_dir / "evidence.json",
            analysis_path=out_dir / "analysis.json",
            run_result=RunResult(command="", returncode=0, stdout="", stderr="run_skipped_no_command"),
        )

        rc = main()
        self.assertEqual(rc, 0)
        mock_execute.assert_called_once()

    @patch("profiler_agent.main.MultiAgentCoordinator")
    @patch("profiler_agent.main.load_target_spec")
    @patch("profiler_agent.main.parse_args")
    def test_multi_mode_writes_plan_and_trace(
        self,
        mock_parse_args: unittest.mock.Mock,
        mock_load_spec: unittest.mock.Mock,
        mock_coordinator_cls: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"main_multi_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            mock_parse_args.return_value = argparse.Namespace(
                spec=Path("inputs/target_spec.json"),
                out=out_dir,
                mode="multi",
                objective="analyze this run",
                phase2_iterations=15,
                llm_secret_file=None,
                llm_base_url="",
                llm_model="",
            )
            mock_load_spec.return_value = TargetSpec(targets=["dram_latency_cycles"], run="cmd /c echo x")

            coordinator = mock_coordinator_cls.return_value
            plan = ExecutionPlan(
                intent="gpu_profiling",
                selected_tools=["executor", "ncu"],
                steps=[ExecutionStep(id="profiling_execution", owner="executor_agent", action="run_pipeline", payload={})],
            )
            trace = [
                AgentMessage(sender="router_agent", recipient="planner_agent", action="route_intent", content={}),
            ]
            coordinator.run.return_value = MultiAgentResult(
                out_dir=out_dir,
                outputs={
                    "pipeline": {
                        "results_path": str(out_dir / "results.json"),
                        "evidence_path": str(out_dir / "evidence.json"),
                        "analysis_path": str(out_dir / "analysis.json"),
                    }
                },
                trace=trace,
                plan=plan,
            )

            rc = main()
            self.assertEqual(rc, 0)

            plan_path = out_dir / "multi_agent_plan.json"
            trace_path = out_dir / "multi_agent_trace.json"
            self.assertTrue(plan_path.exists())
            self.assertTrue(trace_path.exists())

            plan_json = json.loads(plan_path.read_text(encoding="utf-8"))
            trace_json = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(plan_json["intent"], "gpu_profiling")
            self.assertEqual(trace_json[0]["sender"], "router_agent")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.main.run_default_phase2_workflow")
    @patch("profiler_agent.main.load_target_spec")
    @patch("profiler_agent.main.parse_args")
    def test_phase2_mode_uses_phase2_workflow_without_loading_spec(
        self,
        mock_parse_args: unittest.mock.Mock,
        mock_load_spec: unittest.mock.Mock,
        mock_phase2_workflow: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"main_phase2_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            optimized_path = out_dir / "optimized_lora.cu"
            mock_parse_args.return_value = argparse.Namespace(
                spec=Path("inputs/target_spec.json"),
                out=out_dir,
                mode="phase2",
                objective="",
                phase2_iterations=3,
                llm_secret_file=None,
                llm_base_url="",
                llm_model="",
            )
            mock_phase2_workflow.return_value = Phase2OptimizationResult(
                best_candidate_id="cand-1",
                best_speedup=1.25,
                iterations_run=3,
                optimized_lora_path=optimized_path,
                state=Phase2OptimizerState(iteration=3, current_best_candidate_id="cand-1", best_speedup=1.25),
            )

            rc = main()
            self.assertEqual(rc, 0)
            mock_phase2_workflow.assert_called_once_with(
                root_dir=out_dir,
                max_iterations=3,
                llm_client=None,
            )
            mock_load_spec.assert_not_called()
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    @patch("profiler_agent.main.OpenAICompatibleLLMClient")
    @patch("profiler_agent.main.run_default_phase2_workflow")
    @patch("profiler_agent.main.load_target_spec")
    @patch("profiler_agent.main.parse_args")
    def test_phase2_mode_builds_llm_client_from_secret_file(
        self,
        mock_parse_args: unittest.mock.Mock,
        mock_load_spec: unittest.mock.Mock,
        mock_phase2_workflow: unittest.mock.Mock,
        mock_llm_client_cls: unittest.mock.Mock,
    ) -> None:
        out_dir = Path("tests/.tmp") / f"main_phase2_secret_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            secret_path = out_dir / "llm_secret.txt"
            secret_path.write_text("secret\n", encoding="utf-8")
            optimized_path = out_dir / "optimized_lora.cu"
            llm_client = object()
            mock_llm_client_cls.from_secret_file.return_value = llm_client
            mock_parse_args.return_value = argparse.Namespace(
                spec=Path("inputs/target_spec.json"),
                out=out_dir,
                mode="phase2",
                objective="",
                phase2_iterations=2,
                llm_secret_file=secret_path,
                llm_base_url="https://example.invalid/v1",
                llm_model="gpt-5.4",
            )
            mock_phase2_workflow.return_value = Phase2OptimizationResult(
                best_candidate_id="cand-1",
                best_speedup=1.25,
                iterations_run=2,
                optimized_lora_path=optimized_path,
                state=Phase2OptimizerState(iteration=2, current_best_candidate_id="cand-1", best_speedup=1.25),
            )

            rc = main()
            self.assertEqual(rc, 0)
            mock_llm_client_cls.from_secret_file.assert_called_once_with(
                secret_path,
                base_url="https://example.invalid/v1",
                model="gpt-5.4",
            )
            mock_phase2_workflow.assert_called_once_with(
                root_dir=out_dir,
                max_iterations=2,
                llm_client=llm_client,
            )
            mock_load_spec.assert_not_called()
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
