from __future__ import annotations

from profiler_agent.multi_agent.models import MultiAgentRequest, MultiAgentResult

__all__ = ["MultiAgentCoordinator", "MultiAgentRequest", "MultiAgentResult"]


def __getattr__(name: str):
    if name == "MultiAgentCoordinator":
        from profiler_agent.multi_agent.coordinator import MultiAgentCoordinator

        return MultiAgentCoordinator
    raise AttributeError(name)
