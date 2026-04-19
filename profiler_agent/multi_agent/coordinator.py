from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from profiler_agent.agent_state import AgentStateRecord, load_agent_state, save_agent_state
from profiler_agent.multi_agent.executor import ExecutorAgent
from profiler_agent.multi_agent.interpreter import InterpreterAgent
from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient
from profiler_agent.multi_agent.models import AgentMessage, ExecutionPlan, MultiAgentRequest, MultiAgentResult, MultiAgentState
from profiler_agent.multi_agent.planner import PlannerAgent
from profiler_agent.multi_agent.router import RouterAgent
from profiler_agent.runtime_budget import build_timeout_metadata, runtime_budget_expired
from profiler_agent.target_semantics import classify_target


class MultiAgentCoordinator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
        self.router = RouterAgent(llm_client=self.llm_client)
        self.planner = PlannerAgent(llm_client=self.llm_client)
        self.executor = ExecutorAgent()
        self.interpreter = InterpreterAgent(llm_client=self.llm_client)

    @staticmethod
    def _agent_state_path(out_dir: Path) -> Path:
        return out_dir / "agent_state.json"

    @staticmethod
    def _tag_message(message: AgentMessage, execution_round: int) -> AgentMessage:
        return AgentMessage(
            sender=message.sender,
            recipient=message.recipient,
            action=message.action,
            content={**message.content, "execution_round": execution_round},
        )

    @staticmethod
    def _max_iterations(request: MultiAgentRequest) -> int:
        raw = request.metadata.get("max_iterations", 2)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 2
        return max(1, min(value, 4))

    @staticmethod
    def _extract_tool_errors(tool_calls: dict[str, object]) -> list[dict[str, object]]:
        errors: list[dict[str, object]] = []
        for tool_name, payload in tool_calls.items():
            if not isinstance(payload, dict):
                continue
            for stage_key in ("compile_stage", "run_stage", "profile_stage"):
                stage = payload.get(stage_key)
                if isinstance(stage, dict) and not bool(stage.get("ok", False)) and not bool(stage.get("skipped", False)):
                    errors.append(
                        {
                            "tool": tool_name,
                            "stage": stage.get("stage", stage_key),
                            "reason": stage.get("reason", ""),
                            "error_type": stage.get("error_type", ""),
                            "returncode": stage.get("returncode", 0),
                            "stderr_tail": stage.get("stderr_tail", ""),
                        }
                    )
            if payload.get("source") == "command_missing":
                errors.append(
                    {
                        "tool": tool_name,
                        "stage": "tool_entry",
                        "reason": "required_command_missing",
                        "error_type": "command_missing",
                        "returncode": payload.get("compile_returncode", payload.get("returncode", 127)),
                        "stderr_tail": payload.get("compile_stderr_tail", payload.get("stderr_tail", "")),
                    }
                )
        return errors

    @staticmethod
    def _load_json_if_exists(path_str: object) -> dict[str, object]:
        if not isinstance(path_str, str):
            return {}
        path = Path(path_str)
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalized_actions(next_actions: list[object]) -> list[str]:
        return [str(item).strip().lower() for item in next_actions if str(item).strip()]

    @staticmethod
    def _normalize_targets(raw_targets: object, request_targets: list[str]) -> list[str]:
        if not isinstance(raw_targets, list):
            return []
        allowed = set(request_targets)
        normalized: list[str] = []
        for item in raw_targets:
            target = str(item).strip()
            if not target or target not in allowed or target in normalized:
                continue
            normalized.append(target)
        return normalized

    @staticmethod
    def _matches_any(texts: list[str], patterns: tuple[str, ...]) -> bool:
        return any(any(pattern in text for pattern in patterns) for text in texts)

    @staticmethod
    def _build_round_directive(state: MultiAgentState) -> dict[str, object]:
        next_actions = state.outputs.get("next_actions", [])
        if not isinstance(next_actions, list):
            next_actions = []
        focus_targets = MultiAgentCoordinator._normalize_targets(state.outputs.get("next_targets", []), state.request.targets)
        normalized_actions = MultiAgentCoordinator._normalized_actions(next_actions)
        tool_calls = state.outputs.get("tool_calls", {})
        errors = MultiAgentCoordinator._extract_tool_errors(tool_calls) if isinstance(tool_calls, dict) else []

        forced_tools: list[str] = []
        reasons: list[str] = []
        if focus_targets and focus_targets != list(state.request.targets):
            reasons.append("refinement_requested_target_focus")
        if MultiAgentCoordinator._matches_any(
            normalized_actions,
            (
                "probe_repair_compile:",
                "probe_repair_runtime:",
                "probe_change_probe_shape:",
                "probe_add_ncu_profile:",
            ),
        ):
            forced_tools.append("microbench")
            reasons.append("probe_refinement_requested_microbench_rerun")
        if MultiAgentCoordinator._matches_any(
            normalized_actions,
            ("probe_add_ncu_profile:",),
        ):
            forced_tools.append("ncu")
            reasons.append("probe_refinement_requested_ncu_profile")
        if MultiAgentCoordinator._matches_any(
            normalized_actions,
            (
                "collect_ncu_",
                "collect_compute_",
                "profile sm efficiency",
                "measure dram utilization",
                "assess compute and memory throughput",
                "sm efficiency metrics",
                "compute and memory throughput",
                "dram utilization rates",
            ),
        ):
            forced_tools.append("ncu")
            reasons.append("next_actions_requested_ncu_focus")
        if MultiAgentCoordinator._matches_any(
            normalized_actions,
            ("collect_nsys_", "nsys timeline", "timeline"),
        ):
            forced_tools.append("nsys")
            reasons.append("next_actions_requested_nsys_focus")
        if MultiAgentCoordinator._matches_any(
            normalized_actions,
            ("frequency data", "gpu and memory frequency", "clock data"),
        ):
            forced_tools.append("nvml")
            reasons.append("next_actions_requested_clock_focus")
        if any(error.get("tool") == "microbench" and error.get("error_type") == "tool_failed" for error in errors):
            forced_tools.append("microbench")
            reasons.append("retry_microbench_after_recoverable_failure")
        if any(error.get("tool") == "ncu" and error.get("error_type") == "tool_failed" for error in errors):
            forced_tools.append("ncu")
            reasons.append("retry_ncu_after_recoverable_failure")

        return {
            "forced_tools": sorted(dict.fromkeys(forced_tools)),
            "focus_targets": focus_targets or list(state.request.targets),
            "reasons": reasons,
            "source_next_actions": [str(item) for item in next_actions],
        }

    def _update_persistent_state(self, state: MultiAgentState, out_dir: Path, execution_round: int) -> Path:
        record = state.persistent_state
        record.iteration += 1
        record.request_targets = list(state.request.targets)
        record.request_run = state.request.run
        record.objective = state.request.objective
        record.routed_intent = state.routed_intent
        record.selected_tools_history.append(list(state.selected_tools))
        record.target_categories = {
            target: classify_target(target).to_evidence()
            for target in state.request.targets
        }

        tool_calls = state.outputs.get("tool_calls", {})
        if isinstance(tool_calls, dict):
            record.error_history.extend(self._extract_tool_errors(tool_calls))

        pipeline = state.outputs.get("pipeline", {})
        if isinstance(pipeline, dict):
            results_obj = self._load_json_if_exists(pipeline.get("results_path"))
            if results_obj:
                record.metrics_history.append(
                    {"iteration": record.iteration, "execution_round": execution_round, "results": results_obj}
                )
            analysis_obj = self._load_json_if_exists(pipeline.get("analysis_path"))
            if analysis_obj:
                record.analysis_history.append(
                    {"iteration": record.iteration, "execution_round": execution_round, "analysis": analysis_obj}
                )
                bound_type = analysis_obj.get("bound_type")
                if isinstance(bound_type, str) and bound_type.strip():
                    record.current_bottleneck = bound_type

        next_actions = state.outputs.get("next_actions")
        if isinstance(next_actions, list):
            record.recommended_next_actions.append([str(item) for item in next_actions])
        next_targets = self._normalize_targets(state.outputs.get("next_targets", []), state.request.targets)
        if next_targets:
            record.recommended_next_targets.append(list(next_targets))

        record.done = bool(state.outputs.get("pipeline"))
        record.last_out_dir = str(out_dir)
        return save_agent_state(state.agent_state_path or self._agent_state_path(out_dir), record)

    @staticmethod
    def _should_iterate(state: MultiAgentState, execution_round: int, max_iterations: int) -> tuple[bool, str]:
        if execution_round >= max_iterations:
            return False, "max_iterations_reached"

        tool_calls = state.outputs.get("tool_calls", {})
        recoverable_errors = []
        if isinstance(tool_calls, dict):
            for error in MultiAgentCoordinator._extract_tool_errors(tool_calls):
                if error.get("error_type") == "tool_failed":
                    recoverable_errors.append(error)

        next_actions = state.outputs.get("next_actions", [])
        if not isinstance(next_actions, list):
            next_actions = []
        normalized_actions = MultiAgentCoordinator._normalized_actions(next_actions)
        pipeline = state.outputs.get("pipeline", {})
        evidence_obj = MultiAgentCoordinator._load_json_if_exists(pipeline.get("evidence_path")) if isinstance(pipeline, dict) else {}
        synthetic_counter_probe_report = evidence_obj.get("synthetic_counter_probe_report", {})
        accepted_synthetic_counter_probes = 0
        if isinstance(synthetic_counter_probe_report, dict):
            accepted_synthetic_counter_probes = int(synthetic_counter_probe_report.get("accepted_count", 0) or 0)
        if (
            not (state.request.run or "").strip()
            and (
                any("improve_signal_coverage_by_using_realistic_workload_run_command" in str(action) for action in next_actions)
                or MultiAgentCoordinator._matches_any(
                    normalized_actions,
                    (
                        "profile sm efficiency",
                        "measure dram utilization",
                        "assess compute and memory throughput",
                        "workload command becomes available",
                    ),
                )
            )
        ):
            if accepted_synthetic_counter_probes > 0:
                return False, "no_run_input_finalized_with_synthetic_counter_probes"
            return False, "no_run_input_finalized_with_placeholders"
        probe_rerun_requested = MultiAgentCoordinator._matches_any(
            normalized_actions,
            (
                "probe_repair_compile:",
                "probe_repair_runtime:",
                "probe_change_probe_shape:",
                "probe_add_ncu_profile:",
            ),
        )
        rerun_requested = any(
            any(token in str(action) for token in ("re-run_pipeline", "collect_ncu_", "collect_compute_", "collect_nsys_"))
            for action in next_actions
        )
        rerun_requested = rerun_requested or probe_rerun_requested or MultiAgentCoordinator._matches_any(
            normalized_actions,
            (
                "profile sm efficiency",
                "measure dram utilization",
                "assess compute and memory throughput",
                "frequency data",
                "gpu and memory frequency",
            ),
        )

        if recoverable_errors:
            return True, "recoverable_tool_failure"
        if probe_rerun_requested:
            return True, "synthetic_probe_refinement_requested_rerun"
        if rerun_requested and (state.request.run or "").strip():
            return True, "refinement_requested_rerun"
        if rerun_requested and not (state.request.run or "").strip():
            if accepted_synthetic_counter_probes > 0:
                return False, "no_run_input_finalized_with_synthetic_counter_probes"
            return False, "no_run_input_finalized_with_placeholders"
        if not (state.request.run or "").strip() and accepted_synthetic_counter_probes > 0:
            return False, "no_run_input_finalized_with_synthetic_counter_probes"
        return False, "no_followup_condition_met"

    def _run_single_round(
        self,
        *,
        state: MultiAgentState,
        out_dir: Path,
        execution_round: int,
    ) -> ExecutionPlan:
        state.trace.append(
            AgentMessage(
                sender="coordinator",
                recipient="all_agents",
                action="iteration_started",
                content={"execution_round": execution_round},
            )
        )
        plan, plan_message = self.planner.build_plan(state)
        state.selected_tools = list(plan.selected_tools)
        state.trace.append(self._tag_message(plan_message, execution_round))

        for step in plan.steps:
            if step.owner == "executor_agent":
                state.trace.append(self._tag_message(self.executor.execute_step(step=step, state=state, out_dir=out_dir), execution_round))
                continue
            if step.id == "result_interpretation":
                state.trace.append(self._tag_message(self.interpreter.summarize_outputs(state), execution_round))
                continue
            if step.id == "iterative_refinement":
                state.trace.append(self._tag_message(self.interpreter.propose_next_actions(state), execution_round))
                continue
            state.trace.append(
                AgentMessage(
                    sender="coordinator",
                    recipient=step.owner,
                    action="step_skipped",
                    content={"step_id": step.id, "execution_round": execution_round},
                )
            )

        persisted_path = self._update_persistent_state(state=state, out_dir=out_dir, execution_round=execution_round)
        state.outputs["agent_state_path"] = str(persisted_path)
        state.outputs.setdefault("iterations", []).append(
            {
                "execution_round": execution_round,
                "selected_tools": list(state.selected_tools),
                "round_directive": dict(state.round_directive),
                "planned_targets": list(plan_message.content.get("planned_targets", state.request.targets)),
                "pipeline": dict(state.outputs.get("pipeline", {})),
                "interpretation": dict(state.outputs.get("interpretation", {})),
                "next_actions": list(state.outputs.get("next_actions", [])),
                "next_targets": list(state.outputs.get("next_targets", [])),
            }
        )
        return plan

    def run(self, request: MultiAgentRequest) -> MultiAgentResult:
        out_dir = request.out_dir or Path("outputs") / f"multi_agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        agent_state_path = self._agent_state_path(out_dir)
        persistent_state = load_agent_state(agent_state_path)
        state = MultiAgentState(request=request, persistent_state=persistent_state, agent_state_path=agent_state_path)
        intent, route_message = self.router.route(request)
        state.routed_intent = intent
        state.trace.append(route_message)
        last_plan: ExecutionPlan | None = None
        max_iterations = self._max_iterations(request)
        for execution_round in range(1, max_iterations + 1):
            if runtime_budget_expired():
                state.outputs["time_budget"] = build_timeout_metadata(
                    reason="time_budget_exhausted_before_next_iteration",
                    skipped_targets=list(state.request.targets),
                )
                state.trace.append(
                    AgentMessage(
                        sender="coordinator",
                        recipient="all_agents",
                        action="iteration_completed",
                        content={
                            "execution_round": execution_round,
                            "should_continue": False,
                            "decision_reason": "time_budget_exhausted",
                        },
                    )
                )
                break
            if execution_round > 1:
                state.round_directive = self._build_round_directive(state)
            else:
                state.round_directive = {}
            last_plan = self._run_single_round(state=state, out_dir=out_dir, execution_round=execution_round)
            should_continue, reason = self._should_iterate(
                state=state,
                execution_round=execution_round,
                max_iterations=max_iterations,
            )
            state.trace.append(
                AgentMessage(
                    sender="coordinator",
                    recipient="all_agents",
                    action="iteration_completed",
                    content={
                        "execution_round": execution_round,
                        "should_continue": should_continue,
                        "decision_reason": reason,
                    },
                )
            )
            if not should_continue:
                break

        state.outputs["iteration_control"] = {
            "max_iterations": max_iterations,
            "executed_rounds": len(state.outputs.get("iterations", [])),
        }
        if last_plan is None:
            last_plan = ExecutionPlan(intent=state.routed_intent, selected_tools=[], steps=[])
        return MultiAgentResult(out_dir=out_dir, outputs=dict(state.outputs), trace=list(state.trace), plan=last_plan)
