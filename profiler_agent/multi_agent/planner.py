from __future__ import annotations

import json
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.multi_agent.models import AgentMessage, ExecutionPlan, ExecutionStep, MultiAgentState


def _select_tools(targets: list[str]) -> list[str]:
    tools = {"executor", "ncu", "microbench"}
    if "actual_boost_clock_mhz" in targets:
        tools.add("nvml")
    return sorted(tools)


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
        tools = _select_tools(state.request.targets)
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
                    "targets": state.request.targets,
                    "objective": state.request.objective,
                },
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if not isinstance(llm_json, dict):
                fallback_reason = "llm_no_response_or_invalid_json"
            else:
                parsed = set(self._parse_selected_tools(llm_json))
                allow = {"executor", "ncu", "microbench", "nvml", "nsys", "torch_profiler"}
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
                payload={"tools": tools, "targets": state.request.targets, "run": state.request.run},
            ),
            ExecutionStep(
                id="profiling_execution",
                owner="executor_agent",
                action="run_pipeline",
                payload={"targets": state.request.targets, "run": state.request.run},
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
            },
        )
        return plan, message
