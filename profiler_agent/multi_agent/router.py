from __future__ import annotations

import json

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.multi_agent.models import AgentMessage, MultiAgentRequest


class RouterAgent:
    name = "router_agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _rule_based_intent(request: MultiAgentRequest) -> str:
        intent = "gpu_profiling"
        if request.objective.strip():
            lowered = request.objective.lower()
            if "autotune" in lowered or "tuning" in lowered:
                intent = "gpu_profiling_and_tuning"
            elif "explain" in lowered or "analysis" in lowered:
                intent = "gpu_profiling_explain"
        return intent

    def route(self, request: MultiAgentRequest) -> tuple[str, AgentMessage]:
        intent = self._rule_based_intent(request)
        source = "rules"
        if self.llm_client is not None and self.llm_client.is_enabled():
            system_prompt = (
                "You are a strict intent router for a GPU profiling multi-agent system. "
                "Return JSON only with key 'intent'. Allowed values: "
                "gpu_profiling, gpu_profiling_explain, gpu_profiling_and_tuning."
            )
            user_prompt = json.dumps(
                {"objective": request.objective, "targets": request.targets, "run": request.run},
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            llm_intent = llm_json.get("intent") if isinstance(llm_json, dict) else None
            if isinstance(llm_intent, str) and llm_intent in {
                "gpu_profiling",
                "gpu_profiling_explain",
                "gpu_profiling_and_tuning",
            }:
                intent = llm_intent
                source = "llm"

        message = AgentMessage(
            sender=self.name,
            recipient="planner_agent",
            action="route_intent",
            content={
                "intent": intent,
                "decision_source": source,
                "target_count": len(request.targets),
                "has_objective": bool(request.objective.strip()),
            },
        )
        return intent, message
