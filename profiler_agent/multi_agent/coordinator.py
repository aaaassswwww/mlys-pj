from __future__ import annotations

from datetime import datetime
from pathlib import Path

from profiler_agent.multi_agent.executor import ExecutorAgent
from profiler_agent.multi_agent.interpreter import InterpreterAgent
from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient
from profiler_agent.multi_agent.models import AgentMessage, MultiAgentRequest, MultiAgentResult, MultiAgentState
from profiler_agent.multi_agent.planner import PlannerAgent
from profiler_agent.multi_agent.router import RouterAgent


class MultiAgentCoordinator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
        self.router = RouterAgent(llm_client=self.llm_client)
        self.planner = PlannerAgent(llm_client=self.llm_client)
        self.executor = ExecutorAgent()
        self.interpreter = InterpreterAgent(llm_client=self.llm_client)

    def run(self, request: MultiAgentRequest) -> MultiAgentResult:
        out_dir = request.out_dir or Path("outputs") / f"multi_agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        state = MultiAgentState(request=request)
        intent, route_message = self.router.route(request)
        state.routed_intent = intent
        state.trace.append(route_message)

        plan, plan_message = self.planner.build_plan(state)
        state.selected_tools = list(plan.selected_tools)
        state.trace.append(plan_message)

        for step in plan.steps:
            if step.owner == "executor_agent":
                state.trace.append(self.executor.execute_step(step=step, state=state, out_dir=out_dir))
                continue
            if step.id == "result_interpretation":
                state.trace.append(self.interpreter.summarize_outputs(state))
                continue
            if step.id == "iterative_refinement":
                state.trace.append(self.interpreter.propose_next_actions(state))
                continue
            state.trace.append(
                AgentMessage(
                    sender="coordinator",
                    recipient=step.owner,
                    action="step_skipped",
                    content={"step_id": step.id},
                )
            )

        return MultiAgentResult(out_dir=out_dir, outputs=dict(state.outputs), trace=list(state.trace), plan=plan)
