from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_id: str
    source_code: str
    rationale: str = ""
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoraProblemSpec:
    hidden_dim: int
    low_rank: int = 16
    output_dim: int | None = None
    num_tokens: int = 32
    dtype: str = "float32"
    device: str = "cuda"

    def resolved_output_dim(self) -> int:
        return self.output_dim if self.output_dim is not None else self.hidden_dim


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs_err: float
    rel_l2_err: float
    rtol: float
    atol: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    warmup_runs: int
    measured_runs: int
    median_runtime_ms: float
    min_runtime_ms: float
    max_runtime_ms: float
    all_runtime_ms: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompilationResult:
    ok: bool
    command: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadResult:
    ok: bool
    library_path: str
    symbol_name: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    correctness: CorrectnessResult
    student_benchmark: BenchmarkResult
    reference_benchmark: BenchmarkResult
    speedup: float
    notes: list[str] = field(default_factory=list)
    compilation: CompilationResult | None = None
    load: LoadResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Phase2OptimizerState:
    iteration: int = 0
    current_best_candidate_id: str | None = None
    current_best_correct_candidate_id: str | None = None
    current_best_source_code: str = ""
    current_best_rationale: str = ""
    current_best_source: str = ""
    current_best_reference_candidate_id: str | None = None
    current_best_reference_source_code: str = ""
    current_best_reference_rationale: str = ""
    current_best_reference_source: str = ""
    best_reference_rel_l2_err: float = float("inf")
    best_speedup: float = 0.0
    best_rel_l2_err: float = float("inf")
    best_max_abs_err: float = float("inf")
    candidate_history: list[dict[str, Any]] = field(default_factory=list)
    compile_errors: list[dict[str, Any]] = field(default_factory=list)
    correctness_failures: list[dict[str, Any]] = field(default_factory=list)
    benchmark_history: list[dict[str, Any]] = field(default_factory=list)
    llm_revision_history: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
