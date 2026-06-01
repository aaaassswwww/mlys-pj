from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeneratedRuntimeCandidate:
    candidate_id: str
    source_code: str
    rationale: str = ""
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCorrectnessResult:
    passed: bool
    max_abs_err: float
    rel_l2_err: float
    checked_cases: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeBenchmarkResult:
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    mixed_tokens_per_s: float
    aggregate_tokens_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeEvaluation:
    candidate_id: str
    correctness: RuntimeCorrectnessResult
    benchmark: RuntimeBenchmarkResult
    baseline_benchmark: RuntimeBenchmarkResult
    speedup: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Phase3OptimizerState:
    iteration: int = 0
    last_completed_iteration: int = 0
    current_best_candidate_id: str | None = None
    current_best_correct_candidate_id: str | None = None
    current_best_source_code: str = ""
    current_best_rationale: str = ""
    current_best_source: str = ""
    best_speedup: float = 0.0
    best_rel_l2_err: float = float("inf")
    best_max_abs_err: float = float("inf")
    candidate_history: list[dict[str, Any]] = field(default_factory=list)
    correctness_failures: list[dict[str, Any]] = field(default_factory=list)
    llm_revision_history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

