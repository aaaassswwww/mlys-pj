from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MeasureContext:
    target: str
    run_cmd: str
    run_returncode: int | None = None
    run_stdout: str | None = None
    run_stderr: str | None = None


@dataclass(frozen=True)
class MeasureResult:
    value: float
    evidence: dict[str, Any]


class TargetStrategy:
    name = "base"

    def measure(self, ctx: MeasureContext) -> MeasureResult:
        raise NotImplementedError

