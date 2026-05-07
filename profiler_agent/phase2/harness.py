from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Protocol

from profiler_agent.phase2.models import BenchmarkResult, CorrectnessResult, LoraProblemSpec


class TensorBackend(Protocol):
    def randn(self, *shape: int, device: str, dtype: str): ...

    def matmul(self, lhs, rhs): ...

    def transpose(self, value): ...

    def allclose(self, lhs, rhs, *, rtol: float, atol: float) -> bool: ...

    def max_abs_err(self, lhs, rhs) -> float: ...

    def rel_l2_err(self, lhs, rhs) -> float: ...

    def synchronize(self) -> None: ...


class PythonListBackend:
    """Tiny fallback backend used by tests and no-torch environments."""

    def randn(self, *shape: int, device: str, dtype: str):
        _ = device, dtype
        rows, cols = shape
        return [[float((r + 1) * (c + 2)) / 100.0 for c in range(cols)] for r in range(rows)]

    def matmul(self, lhs, rhs):
        rows = len(lhs)
        shared = len(lhs[0]) if lhs else 0
        cols = len(rhs[0]) if rhs else 0
        out = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                acc = 0.0
                for k in range(shared):
                    acc += float(lhs[r][k]) * float(rhs[k][c])
                out[r][c] = acc
        return out

    def transpose(self, value):
        return [list(row) for row in zip(*value)]

    def allclose(self, lhs, rhs, *, rtol: float, atol: float) -> bool:
        for row_l, row_r in zip(lhs, rhs):
            for a, b in zip(row_l, row_r):
                if abs(a - b) > atol + rtol * abs(b):
                    return False
        return True

    def max_abs_err(self, lhs, rhs) -> float:
        best = 0.0
        for row_l, row_r in zip(lhs, rhs):
            for a, b in zip(row_l, row_r):
                best = max(best, abs(float(a) - float(b)))
        return best

    def rel_l2_err(self, lhs, rhs) -> float:
        num = 0.0
        denom = 0.0
        for row_l, row_r in zip(lhs, rhs):
            for a, b in zip(row_l, row_r):
                diff = float(a) - float(b)
                num += diff * diff
                denom += float(b) * float(b)
        if denom <= 0.0:
            return 0.0 if num <= 0.0 else math.inf
        return math.sqrt(num / denom)

    def synchronize(self) -> None:
        return None


def resolve_backend() -> TensorBackend:
    try:
        import torch  # type: ignore
    except Exception:
        return PythonListBackend()

    class TorchBackend:
        def randn(self, *shape: int, device: str, dtype: str):
            dtype_obj = getattr(torch, dtype, torch.float32)
            if device.startswith("cuda") and not torch.cuda.is_available():
                device_name = "cpu"
            else:
                device_name = device
            return torch.randn(*shape, device=device_name, dtype=dtype_obj)

        def matmul(self, lhs, rhs):
            return lhs @ rhs

        def transpose(self, value):
            return value.transpose(0, 1)

        def allclose(self, lhs, rhs, *, rtol: float, atol: float) -> bool:
            return bool(torch.allclose(lhs, rhs, rtol=rtol, atol=atol))

        def max_abs_err(self, lhs, rhs) -> float:
            return float((lhs - rhs).abs().max().item())

        def rel_l2_err(self, lhs, rhs) -> float:
            diff = (lhs - rhs).reshape(-1)
            ref = rhs.reshape(-1)
            denom = float(torch.linalg.norm(ref).item())
            if denom <= 0.0:
                return 0.0
            return float(torch.linalg.norm(diff).item() / denom)

        def synchronize(self) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return TorchBackend()


def generate_lora_inputs(spec: LoraProblemSpec, backend: TensorBackend | None = None) -> dict[str, Any]:
    b = backend or resolve_backend()
    out_dim = spec.resolved_output_dim()
    return {
        "W": b.randn(out_dim, spec.hidden_dim, device=spec.device, dtype=spec.dtype),
        "X": b.randn(spec.hidden_dim, spec.num_tokens, device=spec.device, dtype=spec.dtype),
        "A": b.randn(out_dim, spec.low_rank, device=spec.device, dtype=spec.dtype),
        "B": b.randn(spec.hidden_dim, spec.low_rank, device=spec.device, dtype=spec.dtype),
    }


def lora_reference_forward(inputs: dict[str, Any], backend: TensorBackend | None = None):
    b = backend or resolve_backend()
    wx = b.matmul(inputs["W"], inputs["X"])
    btx = b.matmul(b.transpose(inputs["B"]), inputs["X"])
    abtx = b.matmul(inputs["A"], btx)
    if isinstance(wx, list):
        rows = len(wx)
        cols = len(wx[0]) if wx else 0
        return [[float(wx[r][c]) + float(abtx[r][c]) for c in range(cols)] for r in range(rows)]
    return wx + abtx


def check_correctness(
    student_output,
    reference_output,
    *,
    backend: TensorBackend | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> CorrectnessResult:
    b = backend or resolve_backend()
    passed = b.allclose(student_output, reference_output, rtol=rtol, atol=atol)
    return CorrectnessResult(
        passed=passed,
        max_abs_err=float(b.max_abs_err(student_output, reference_output)),
        rel_l2_err=float(b.rel_l2_err(student_output, reference_output)),
        rtol=rtol,
        atol=atol,
    )


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    backend: TensorBackend | None = None,
    warmup_runs: int = 5,
    measured_runs: int = 20,
    timer_fn: Callable[[], float] | None = None,
) -> BenchmarkResult:
    b = backend or resolve_backend()
    clock = timer_fn or time.perf_counter

    for _ in range(max(0, warmup_runs)):
        fn()
        b.synchronize()

    samples: list[float] = []
    for _ in range(max(1, measured_runs)):
        start = clock()
        fn()
        b.synchronize()
        end = clock()
        samples.append((end - start) * 1000.0)

    return BenchmarkResult(
        warmup_runs=max(0, warmup_runs),
        measured_runs=max(1, measured_runs),
        median_runtime_ms=float(statistics.median(samples)),
        min_runtime_ms=float(min(samples)),
        max_runtime_ms=float(max(samples)),
        all_runtime_ms=[float(x) for x in samples],
    )


def compute_speedup(reference: BenchmarkResult, student: BenchmarkResult) -> float:
    if student.median_runtime_ms <= 0.0:
        return 0.0
    return float(reference.median_runtime_ms / student.median_runtime_ms)


def empty_benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
        warmup_runs=0,
        measured_runs=0,
        median_runtime_ms=0.0,
        min_runtime_ms=0.0,
        max_runtime_ms=0.0,
        all_runtime_ms=[],
    )
