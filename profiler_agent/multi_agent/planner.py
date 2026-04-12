from __future__ import annotations

import json

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

    def build_plan(self, state: MultiAgentState) -> tuple[ExecutionPlan, AgentMessage]:
        tools = _select_tools(state.request.targets)
        source = "rules"
        if self.llm_client is not None and self.llm_client.is_enabled():
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
            llm_tools = llm_json.get("selected_tools") if isinstance(llm_json, dict) else None
            if isinstance(llm_tools, list):
                parsed = {str(item).strip() for item in llm_tools if str(item).strip()}
                allow = {"executor", "ncu", "microbench", "nvml", "nsys", "torch_profiler"}
                selected = sorted(parsed.intersection(allow))
                if "executor" not in selected:
                    selected.insert(0, "executor")
                if selected:
                    tools = selected
                    source = "llm"

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
            },
        )
        return plan, message
