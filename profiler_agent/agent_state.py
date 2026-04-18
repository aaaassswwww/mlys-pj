from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentStateRecord:
    iteration: int = 0
    request_targets: list[str] = field(default_factory=list)
    request_run: str = ""
    objective: str = ""
    routed_intent: str = "gpu_profiling"
    target_categories: dict[str, dict[str, object]] = field(default_factory=dict)
    selected_tools_history: list[list[str]] = field(default_factory=list)
    metrics_history: list[dict[str, object]] = field(default_factory=list)
    analysis_history: list[dict[str, object]] = field(default_factory=list)
    error_history: list[dict[str, object]] = field(default_factory=list)
    recommended_next_actions: list[list[str]] = field(default_factory=list)
    recommended_next_targets: list[list[str]] = field(default_factory=list)
    current_bottleneck: str = "unknown"
    done: bool = False
    last_out_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_agent_state(path: Path) -> AgentStateRecord:
    if not path.exists():
        return AgentStateRecord()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return AgentStateRecord()
    known_fields = {field_name for field_name in AgentStateRecord.__dataclass_fields__}
    filtered = {key: value for key, value in raw.items() if key in known_fields}
    return AgentStateRecord(**filtered)


def save_agent_state(path: Path, state: AgentStateRecord) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path
