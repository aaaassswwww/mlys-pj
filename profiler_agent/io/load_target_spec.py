from __future__ import annotations

import json
from pathlib import Path

from profiler_agent.schema.target_spec_schema import TargetSpec, validate_target_spec


def load_target_spec(path: Path) -> TargetSpec:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return validate_target_spec(raw)

