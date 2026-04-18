from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.models import AgentMessage, ExecutionStageResult, ExecutionStep, MultiAgentState
from profiler_agent.orchestrator.pipeline import execute
from profiler_agent.runtime_tools import parse_command_argv, probe_command, probe_python_module
from profiler_agent.schema.target_spec_schema import TargetSpec
from profiler_agent.target_semantics import classify_target
from profiler_agent.tool_adapters.binary_runner import run_executable
from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence
from profiler_agent.tool_adapters.ncu_adapter import query_metric_with_evidence
from profiler_agent.tool_adapters.nvml_adapter import query_named_device_attribute, sample_sm_clock_stats


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

    @staticmethod
    def _stage_result(
        *,
        stage: str,
        ok: bool,
        command: list[str] | None = None,
        returncode: int = 0,
        skipped: bool = False,
        reason: str = "",
        error_type: str = "",
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> dict[str, Any]:
        return asdict(
            ExecutionStageResult(
                stage=stage,
                ok=ok,
                command=list(command or []),
                returncode=returncode,
                skipped=skipped,
                reason=reason,
                error_type=error_type,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
        )

    def _missing_command_stage(self, *, stage: str, command: list[str]) -> dict[str, Any]:
        executable = command[0] if command else "<empty>"
        return self._stage_result(
            stage=stage,
            ok=False,
            command=command,
            returncode=127,
            reason="required_command_missing",
            error_type="command_missing",
            stderr_tail=f"required_command_not_found:{executable}",
        )

    def _run_tool_executor(self, run_cmd: str) -> dict[str, Any]:
        if not (run_cmd or "").strip():
            return {
                "ok": True,
                "skipped": True,
                "reason": "run_command_missing",
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "run_skipped_no_command",
                "run_stage": self._stage_result(
                    stage="run",
                    ok=True,
                    skipped=True,
                    reason="run_command_missing",
                    stderr_tail="run_skipped_no_command",
                ),
            }
        run_argv = parse_command_argv(run_cmd)
        command_probe = probe_command([run_argv[0]]) if run_argv else probe_command([])
        if not command_probe.available:
            return {
                "ok": False,
                "returncode": 127,
                "stdout_tail": "",
                "stderr_tail": command_probe.stderr_tail,
                "run_stage": self._missing_command_stage(stage="run", command=run_argv[:1]),
            }
        run = run_executable(run_cmd, timeout_s=30)
        return {
            "ok": run.returncode == 0,
            "returncode": run.returncode,
            "stdout_tail": _tail(run.stdout),
            "stderr_tail": _tail(run.stderr),
            "run_stage": self._stage_result(
                stage="run",
                ok=run.returncode == 0,
                command=run_argv,
                returncode=run.returncode,
                stderr_tail=_tail(run.stderr),
                stdout_tail=_tail(run.stdout),
            ),
        }

    def _run_tool_ncu(self, target: str, run_cmd: str) -> dict[str, Any]:
        semantic = classify_target(target)
        version_probe = probe_command(["ncu", "--version"], timeout_s=15)
        if not version_probe.available:
            missing_stage = self._missing_command_stage(stage="profile", command=["ncu"])
            return {
                "semantic": semantic.to_evidence(),
                "available": False,
                "version": version_probe.to_dict(),
                "profile_stage": missing_stage,
                "query": {
                    "source": "command_missing",
                    "returncode": 127,
                    "parse_mode": "none",
                    "value": None,
                    "command": ["ncu"],
                    "stdout_tail": "",
                    "stderr_tail": version_probe.stderr_tail,
                },
            }
        version = _run_cmd(["ncu", "--version"], timeout_s=15)
        query = query_metric_with_evidence(metric_name=target, run_cmd=run_cmd)
        return {
            "semantic": semantic.to_evidence(),
            "available": True,
            "version": version,
            "profile_stage": self._stage_result(
                stage="profile",
                ok=query.returncode == 0 and query.source not in {"ncu_failed", "ncu_unavailable", "workload_run_missing"},
                command=query.command or ["ncu"],
                returncode=query.returncode,
                skipped=query.source == "workload_run_missing",
                reason="workload_run_missing" if query.source == "workload_run_missing" else "",
                error_type="tool_failed" if query.source in {"ncu_failed", "ncu_unavailable"} else "",
                stdout_tail=query.stdout_tail,
                stderr_tail=query.stderr_tail,
            ),
            "query": {
                "source": query.source,
                "returncode": query.returncode,
                "parse_mode": query.parse_mode,
                "value": query.value,
                "command": query.command,
                "stdout_tail": query.stdout_tail,
                "stderr_tail": query.stderr_tail,
            },
        }

    def _run_tool_microbench(self, target: str, run_cmd: str) -> dict[str, Any]:
        semantic = classify_target(target)
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None:
            compile_stage = self._missing_command_stage(stage="compile", command=["nvcc"])
            return {
                "semantic": semantic.to_evidence(),
                "value": None,
                "source": "command_missing",
                "compile_returncode": 127,
                "run_returncode": 0,
                "parsed_from": "none",
                "sample_count": None,
                "median_value": None,
                "std_value": None,
                "compile_command": ["nvcc"],
                "run_command": None,
                "compile_stderr_tail": compile_stage["stderr_tail"],
                "run_stderr_tail": "",
                "compile_stage": compile_stage,
                "run_stage": self._stage_result(
                    stage="run",
                    ok=False,
                    skipped=True,
                    reason="compile_stage_failed",
                ),
            }
        probe = measure_metric_with_evidence(metric_name=target, run_cmd=run_cmd)
        compile_ok = probe.compile_returncode == 0
        run_ok = compile_ok and probe.run_returncode == 0 and probe.source != "run_failed"
        return {
            "semantic": semantic.to_evidence(),
            "value": probe.value,
            "source": probe.source,
            "compile_returncode": probe.compile_returncode,
            "run_returncode": probe.run_returncode,
            "parsed_from": probe.parsed_from,
            "sample_count": probe.sample_count,
            "median_value": probe.median_value,
            "std_value": probe.std_value,
            "compile_command": probe.compile_command,
            "run_command": probe.run_command,
            "compile_stderr_tail": probe.compile_stderr_tail,
            "run_stderr_tail": probe.run_stderr_tail,
            "compile_stage": self._stage_result(
                stage="compile",
                ok=compile_ok,
                command=list(probe.compile_command or ["nvcc"]),
                returncode=probe.compile_returncode,
                error_type="tool_failed" if not compile_ok else "",
                stderr_tail=probe.compile_stderr_tail,
                stdout_tail=probe.compile_stdout_tail,
            ),
            "run_stage": self._stage_result(
                stage="run",
                ok=run_ok,
                command=list(probe.run_command or []),
                returncode=probe.run_returncode,
                skipped=not compile_ok,
                reason="compile_stage_failed" if not compile_ok else "",
                error_type="tool_failed" if compile_ok and not run_ok else "",
                stderr_tail=probe.run_stderr_tail,
                stdout_tail=probe.run_stdout_tail,
            ),
        }

    def _run_tool_device_attribute(self, target: str) -> dict[str, Any]:
        semantic = classify_target(target)
        query = query_named_device_attribute(target)
        return {
            "semantic": semantic.to_evidence(),
            "value": query.get("value"),
            "source": query.get("source"),
            "field": query.get("field"),
            "unit": query.get("unit"),
            "command": query.get("command"),
            "returncode": query.get("returncode"),
            "stdout_tail": query.get("stdout_tail"),
            "stderr_tail": query.get("stderr_tail"),
            "backend_chain": query.get("backend_chain"),
        }

    def _run_tool_nvml(self) -> dict[str, Any]:
        stats = sample_sm_clock_stats(sample_count=5, interval_s=0.1)
        return {"stats": stats}

    def _run_tool_nsys(self) -> dict[str, Any]:
        probe = probe_command(["nsys", "--version"], timeout_s=15)
        if not probe.available:
            return {
                "available": False,
                "version": probe.to_dict(),
                "profile_stage": self._missing_command_stage(stage="profile", command=["nsys"]),
            }
        return {
            "available": True,
            "version": _run_cmd(["nsys", "--version"], timeout_s=15),
            "profile_stage": self._stage_result(stage="profile", ok=True, command=["nsys", "--version"]),
        }

    def _run_tool_torch_profiler(self) -> dict[str, Any]:
        module_probe = probe_python_module("torch")
        if not module_probe.available:
            return {
                "available": False,
                "module_probe": module_probe.to_dict(),
                "profile_stage": self._stage_result(
                    stage="profile",
                    ok=False,
                    reason="python_module_missing",
                    error_type="module_missing",
                    stderr_tail="python_module_not_found:torch",
                ),
            }
        version = _run_cmd(["python", "-c", "import torch; print(torch.__version__)"], timeout_s=15)
        return {
            "available": True,
            "module_probe": module_probe.to_dict(),
            "version": version,
            "profile_stage": self._stage_result(stage="profile", ok=True, command=["python", "-c", "import torch"]),
        }

    def run_tools(self, state: MultiAgentState, step: ExecutionStep) -> AgentMessage:
        payload = step.payload or {}
        tools = payload.get("tools")
        if not isinstance(tools, list):
            tools = ["executor", "ncu", "microbench"]
        targets = payload.get("targets")
        if not isinstance(targets, list) or not targets:
            targets = list(state.request.targets)
        tool_targets = payload.get("tool_targets")
        if not isinstance(tool_targets, dict):
            tool_targets = {}
        primary_target = str(targets[0]) if targets else "dram_latency_cycles"
        run_cmd = str(payload.get("run") or state.request.run)

        outputs: dict[str, Any] = {}
        for tool in tools:
            selected_target = str(tool_targets.get(tool) or primary_target)
            if tool == "executor":
                outputs[tool] = self._run_tool_executor(run_cmd=run_cmd)
            elif tool == "ncu":
                outputs[tool] = self._run_tool_ncu(target=selected_target, run_cmd=run_cmd)
            elif tool == "microbench":
                outputs[tool] = self._run_tool_microbench(target=selected_target, run_cmd=run_cmd)
            elif tool == "device_attribute":
                outputs[tool] = self._run_tool_device_attribute(target=selected_target)
            elif tool == "nvml":
                outputs[tool] = self._run_tool_nvml()
            elif tool == "nsys":
                outputs[tool] = self._run_tool_nsys()
            elif tool == "torch_profiler":
                outputs[tool] = self._run_tool_torch_profiler()
            else:
                outputs[str(tool)] = {"ok": False, "error": "unknown_tool"}
            if isinstance(outputs.get(tool), dict):
                outputs[tool]["selected_target"] = selected_target if tool in {"ncu", "microbench", "device_attribute"} else None

        state.outputs["tool_calls"] = outputs
        return AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="tools_done",
            content={"tools": list(outputs.keys()), "tool_targets": tool_targets},
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
