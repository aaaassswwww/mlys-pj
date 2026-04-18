from __future__ import annotations

import json
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.multi_agent.models import AgentMessage, ExecutionPlan, ExecutionStep, MultiAgentState
from profiler_agent.target_semantics import classify_target


def _effective_targets(state: MultiAgentState) -> list[str]:
    directive = state.round_directive if isinstance(state.round_directive, dict) else {}
    raw = directive.get("focus_targets")
    if not isinstance(raw, list):
        return list(state.request.targets)
    selected = [str(item).strip() for item in raw if str(item).strip()]
    return selected or list(state.request.targets)


def _select_tools(targets: list[str]) -> list[str]:
    tools = {"executor"}
    semantics = [classify_target(target) for target in targets]
    if any(item.semantic_class == "device_attribute" for item in semantics):
        tools.add("device_attribute")
    if any(item.semantic_class == "workload_counter" for item in semantics):
        tools.add("ncu")
    if any(item.semantic_class in {"intrinsic_probe", "unknown"} for item in semantics):
        tools.add("microbench")
    if "actual_boost_clock_mhz" in targets:
        tools.add("nvml")
    return sorted(tools)


def _pick_focus_target(targets: list[str], semantic_classes: set[str]) -> str | None:
    for target in targets:
        semantic = classify_target(target)
        if semantic.semantic_class in semantic_classes:
            return target
    return targets[0] if targets else None


class PlannerAgent:
    name = "planner_agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _parse_selected_tools(llm_json: dict[str, Any]) -> list[str]:
        raw = llm_json.get("selected_tools")
        if raw is None:
            raw = llm_json.get("tools")
        if raw is None:
            raw = llm_json.get("tool_selection")

        if isinstance(raw, str):
            raw = [token.strip() for token in raw.replace(";", ",").split(",")]
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def build_plan(self, state: MultiAgentState) -> tuple[ExecutionPlan, AgentMessage]:
        effective_targets = _effective_targets(state)
        tools = _select_tools(effective_targets)
        directive = state.round_directive if isinstance(state.round_directive, dict) else {}
        forced_tools = [str(item) for item in directive.get("forced_tools", []) if str(item).strip()]
        for item in forced_tools:
            tools.append(item)
        tools = sorted(dict.fromkeys(tools))
        source = "rules"
        fallback_reason = "llm_not_enabled"
        llm_attempted = False
        if self.llm_client is not None and self.llm_client.is_enabled():
            llm_attempted = True
            system_prompt = (
                "You are a planning assistant. Return JSON only with key 'selected_tools' "
                "as a list from: executor,ncu,microbench,nvml,nsys,torch_profiler."
            )
            user_prompt = json.dumps(
                {
                    "intent": state.routed_intent,
                    "targets": effective_targets,
                    "request_targets": state.request.targets,
                    "objective": state.request.objective,
                    "previous_errors": state.persistent_state.error_history[-3:],
                    "previous_selected_tools": state.persistent_state.selected_tools_history[-3:],
                    "round_directive": directive,
                },
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if not isinstance(llm_json, dict):
                fallback_reason = "llm_no_response_or_invalid_json"
            else:
                parsed = set(self._parse_selected_tools(llm_json))
                allow = {"executor", "device_attribute", "ncu", "microbench", "nvml", "nsys", "torch_profiler"}
                selected = sorted(parsed.intersection(allow))
                if parsed and "executor" not in selected:
                    selected.insert(0, "executor")
                if parsed and selected:
                    tools = selected
                    source = "llm"
                    fallback_reason = ""
                elif parsed:
                    fallback_reason = "llm_tools_not_allowed"
                else:
                    fallback_reason = "llm_missing_selected_tools"
        for item in forced_tools:
            tools.append(item)
        tools = sorted(dict.fromkeys(tools))

        tool_targets: dict[str, str] = {}
        if "ncu" in tools:
            focus = _pick_focus_target(effective_targets, {"workload_counter"})
            if focus is not None:
                tool_targets["ncu"] = focus
        if "microbench" in tools:
            focus = _pick_focus_target(effective_targets, {"intrinsic_probe", "unknown"})
            if focus is not None:
                tool_targets["microbench"] = focus
        if "device_attribute" in tools:
            focus = _pick_focus_target(effective_targets, {"device_attribute"})
            if focus is not None:
                tool_targets["device_attribute"] = focus

        steps = [
            ExecutionStep(
                id="task_understanding",
                owner="planner_agent",
                action="summarize_task",
                payload={"objective": state.request.objective or "hardware intrinsic profiling"},
            ),
            ExecutionStep(
                id="profiling_plan_generation",
                owner="planner_agent",
                action="select_tools",
                payload={"tools": tools},
            ),
            ExecutionStep(
                id="tool_execution",
                owner="executor_agent",
                action="run_tools",
                payload={
                    "tools": tools,
                    "targets": effective_targets,
                    "run": state.request.run,
                    "tool_targets": tool_targets,
                    "round_directive": directive,
                },
            ),
            ExecutionStep(
                id="profiling_execution",
                owner="executor_agent",
                action="run_pipeline",
                payload={"targets": effective_targets, "run": state.request.run},
            ),
            ExecutionStep(
                id="result_interpretation",
                owner="interpreter_agent",
                action="summarize_outputs",
                payload={},
            ),
            ExecutionStep(
                id="iterative_refinement",
                owner="interpreter_agent",
                action="propose_next_actions",
                payload={},
            ),
        ]
        plan = ExecutionPlan(intent=state.routed_intent, selected_tools=tools, steps=steps)
        message = AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="plan_ready",
            content={
                "intent": plan.intent,
                "selected_tools": plan.selected_tools,
                "planning_source": source,
                "step_ids": [step.id for step in plan.steps],
                "llm_attempted": llm_attempted,
                "planning_fallback_reason": fallback_reason if source == "rules" else "",
                "loaded_iteration": state.persistent_state.iteration,
                "round_directive": directive,
                "planned_targets": effective_targets,
                "tool_targets": tool_targets,
            },
        )
        return plan, message
