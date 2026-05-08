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


def get_torch_precision_environment() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
        }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "matmul_allow_tf32": (
            getattr(torch.backends.cuda.matmul, "allow_tf32", None)
            if cuda_available and hasattr(torch.backends, "cuda")
            else None
        ),
        "cudnn_allow_tf32": (
            getattr(torch.backends.cudnn, "allow_tf32", None)
            if cuda_available and hasattr(torch.backends, "cudnn")
            else None
        ),
        "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: "unsupported")(),
    }


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


def _to_python_matrix(value, *, max_rows: int | None = None, max_cols: int | None = None):
    if isinstance(value, list):
        rows = value if max_rows is None else value[:max_rows]
        if max_cols is None:
            return [[float(item) for item in row] for row in rows]
        return [[float(item) for item in row[:max_cols]] for row in rows]
    try:
        sliced = value
        if max_rows is not None or max_cols is not None:
            row_slice = slice(None if max_rows is None else max_rows)
            col_slice = slice(None if max_cols is None else max_cols)
            sliced = value[row_slice, col_slice]
        detached = sliced.detach().cpu()
        return [[float(item) for item in row] for row in detached.tolist()]
    except Exception:
        try:
            materialized = value.tolist()
        except Exception as exc:
            raise RuntimeError("unable_to_materialize_tensor_for_reference_diagnosis") from exc
        rows = materialized if max_rows is None else materialized[:max_rows]
        if max_cols is None:
            return [[float(item) for item in row] for row in rows]
        return [[float(item) for item in row[:max_cols]] for row in rows]


def _naive_lora_reference_subset(
    inputs: dict[str, Any],
    *,
    output_rows: int,
    output_cols: int,
) -> list[list[float]]:
    W = _to_python_matrix(inputs["W"], max_rows=output_rows, max_cols=None)
    X = _to_python_matrix(inputs["X"], max_rows=None, max_cols=output_cols)
    A = _to_python_matrix(inputs["A"], max_rows=output_rows, max_cols=None)
    B = _to_python_matrix(inputs["B"], max_rows=None, max_cols=None)
    d = len(X)
    rank = len(A[0]) if A else 0
    temp = [[0.0 for _ in range(output_cols)] for _ in range(rank)]
    for k in range(rank):
        for j in range(output_cols):
            acc = 0.0
            for i in range(d):
                acc += float(B[i][k]) * float(X[i][j])
            temp[k][j] = acc
    Y = [[0.0 for _ in range(output_cols)] for _ in range(output_rows)]
    for i in range(output_rows):
        for j in range(output_cols):
            acc = 0.0
            for t in range(d):
                acc += float(W[i][t]) * float(X[t][j])
            for k in range(rank):
                acc += float(A[i][k]) * float(temp[k][j])
            Y[i][j] = acc
    return Y


def build_reference_diagnosis(
    student_output,
    reference_output,
    inputs: dict[str, Any],
    *,
    max_rows: int = 64,
    max_cols: int = 8,
) -> dict[str, float | str]:
    output_rows = min(max_rows, len(inputs["W"]) if isinstance(inputs["W"], list) else int(inputs["W"].shape[0]))
    output_cols = min(max_cols, len(inputs["X"][0]) if isinstance(inputs["X"], list) else int(inputs["X"].shape[1]))
    metrics = PythonListBackend()
    student_subset = _to_python_matrix(student_output, max_rows=output_rows, max_cols=output_cols)
    reference_subset = _to_python_matrix(reference_output, max_rows=output_rows, max_cols=output_cols)
    naive_subset = _naive_lora_reference_subset(inputs, output_rows=output_rows, output_cols=output_cols)
    student_vs_reference_rel = float(metrics.rel_l2_err(student_subset, reference_subset))
    student_vs_naive_rel = float(metrics.rel_l2_err(student_subset, naive_subset))
    naive_vs_reference_rel = float(metrics.rel_l2_err(naive_subset, reference_subset))
    return {
        "subset_rows": float(output_rows),
        "subset_cols": float(output_cols),
        "student_vs_reference_rel_l2_err": student_vs_reference_rel,
        "student_vs_reference_max_abs_err": float(metrics.max_abs_err(student_subset, reference_subset)),
        "student_vs_naive_rel_l2_err": student_vs_naive_rel,
        "student_vs_naive_max_abs_err": float(metrics.max_abs_err(student_subset, naive_subset)),
        "naive_vs_reference_rel_l2_err": naive_vs_reference_rel,
        "naive_vs_reference_max_abs_err": float(metrics.max_abs_err(naive_subset, reference_subset)),
        "student_closer_to": ("naive" if student_vs_naive_rel < student_vs_reference_rel else "reference"),
    }


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
