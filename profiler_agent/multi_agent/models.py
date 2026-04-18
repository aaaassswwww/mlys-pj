from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultiAgentRequest:
    targets: list[str]
    run: str = ""
    objective: str = ""
    out_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    recipient: str
    action: str
    content: dict[str, Any]


@dataclass(frozen=True)
class ExecutionStep:
    id: str
    owner: str
    action: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    intent: str
    selected_tools: list[str]
    steps: list[ExecutionStep]


@dataclass
class MultiAgentState:
    request: MultiAgentRequest
    routed_intent: str = "gpu_profiling"
    selected_tools: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    trace: list[AgentMessage] = field(default_factory=list)


@dataclass(frozen=True)
class MultiAgentResult:
    out_dir: Path
    outputs: dict[str, Any]
    trace: list[AgentMessage]
    plan: ExecutionPlan
