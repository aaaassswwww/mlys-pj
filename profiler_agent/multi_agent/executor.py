from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.models import AgentMessage, ExecutionStep, MultiAgentState
from profiler_agent.orchestrator.pipeline import execute
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.tool_adapters.binary_runner import run_executable
from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence
from profiler_agent.tool_adapters.ncu_adapter import query_metric_with_evidence
from profiler_agent.tool_adapters.nvml_adapter import sample_sm_clock_stats


def _tail(text: str, n: int = 500) -> str:
    return (text or "")[-n:]


def _run_cmd(argv: list[str], timeout_s: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout_s)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


class ExecutorAgent:
    name = "executor_agent"

    def _run_tool_executor(self, run_cmd: str) -> dict[str, Any]:
        if not (run_cmd or "").strip():
            return {
                "ok": True,
                "skipped": True,
                "reason": "run_command_missing",
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "run_skipped_no_command",
            }
        run = run_executable(run_cmd, timeout_s=30)
        return {
            "ok": run.returncode == 0,
            "returncode": run.returncode,
            "stdout_tail": _tail(run.stdout),
            "stderr_tail": _tail(run.stderr),
        }

    def _run_tool_ncu(self, target: str, run_cmd: str) -> dict[str, Any]:
        version = _run_cmd(["ncu", "--version"], timeout_s=15)
        query = query_metric_with_evidence(metric_name=target, run_cmd=run_cmd)
        return {
            "available": shutil.which("ncu") is not None,
            "version": version,
            "query": {
                "source": query.source,
                "returncode": query.returncode,
                "parse_mode": query.parse_mode,
                "value": query.value,
                "stderr_tail": query.stderr_tail,
            },
        }

    def _run_tool_microbench(self, target: str, run_cmd: str) -> dict[str, Any]:
        probe = measure_metric_with_evidence(metric_name=target, run_cmd=run_cmd)
        return {
            "value": probe.value,
            "source": probe.source,
            "compile_returncode": probe.compile_returncode,
            "run_returncode": probe.run_returncode,
            "parsed_from": probe.parsed_from,
            "sample_count": probe.sample_count,
            "median_value": probe.median_value,
            "std_value": probe.std_value,
        }

    def _run_tool_nvml(self) -> dict[str, Any]:
        stats = sample_sm_clock_stats(sample_count=5, interval_s=0.1)
        return {"stats": stats}

    def _run_tool_nsys(self) -> dict[str, Any]:
        return {
            "available": shutil.which("nsys") is not None,
            "version": _run_cmd(["nsys", "--version"], timeout_s=15),
        }

    def _run_tool_torch_profiler(self) -> dict[str, Any]:
        has_torch = importlib.util.find_spec("torch") is not None
        if not has_torch:
            return {"available": False, "error": "torch_not_installed"}
        version = _run_cmd(["python", "-c", "import torch; print(torch.__version__)"], timeout_s=15)
        return {"available": True, "version": version}

    def run_tools(self, state: MultiAgentState, step: ExecutionStep) -> AgentMessage:
        payload = step.payload or {}
        tools = payload.get("tools")
        if not isinstance(tools, list):
            tools = ["executor", "ncu", "microbench"]
        targets = payload.get("targets")
        if not isinstance(targets, list) or not targets:
            targets = list(state.request.targets)
        primary_target = str(targets[0]) if targets else "dram_latency_cycles"
        run_cmd = str(payload.get("run") or state.request.run)

        outputs: dict[str, Any] = {}
        for tool in tools:
            if tool == "executor":
                outputs[tool] = self._run_tool_executor(run_cmd=run_cmd)
            elif tool == "ncu":
                outputs[tool] = self._run_tool_ncu(target=primary_target, run_cmd=run_cmd)
            elif tool == "microbench":
                outputs[tool] = self._run_tool_microbench(target=primary_target, run_cmd=run_cmd)
            elif tool == "nvml":
                outputs[tool] = self._run_tool_nvml()
            elif tool == "nsys":
                outputs[tool] = self._run_tool_nsys()
            elif tool == "torch_profiler":
                outputs[tool] = self._run_tool_torch_profiler()
            else:
                outputs[str(tool)] = {"ok": False, "error": "unknown_tool"}

        state.outputs["tool_calls"] = outputs
        return AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="tools_done",
            content={"tools": list(outputs.keys())},
        )

    def execute_pipeline(self, state: MultiAgentState, out_dir: Path) -> AgentMessage:
        spec = TargetSpec(targets=list(state.request.targets), run=state.request.run)
        output = execute(spec=spec, out_dir=out_dir)
        state.outputs["pipeline"] = {
            "results_path": str(output.results_path),
            "evidence_path": str(output.evidence_path),
            "analysis_path": str(output.analysis_path),
            "run_returncode": output.run_result.returncode,
        }
        return AgentMessage(
            sender=self.name,
            recipient="interpreter_agent",
            action="pipeline_done",
            content=dict(state.outputs["pipeline"]),
        )

    def execute_step(self, step: ExecutionStep, state: MultiAgentState, out_dir: Path) -> AgentMessage:
        if step.action == "run_tools":
            return self.run_tools(state=state, step=step)
        if step.action != "run_pipeline":
            return AgentMessage(
                sender=self.name,
                recipient="coordinator",
                action="noop",
                content={"step_id": step.id, "action": step.action},
            )
        return self.execute_pipeline(state=state, out_dir=out_dir)
