from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from profiler_agent.phase2.harness import (
    benchmark_callable,
    check_correctness,
    compute_speedup,
    empty_benchmark_result,
    generate_lora_inputs,
    lora_reference_forward,
    resolve_backend,
)
from profiler_agent.phase2.models import (
    BenchmarkResult,
    CandidateEvaluation,
    CompilationResult,
    CorrectnessResult,
    GeneratedCandidate,
    LoadResult,
    LoraProblemSpec,
)
from profiler_agent.runtime_tools import resolve_command_path, tail_text

_ENTRYPOINT_SYMBOL = "launch_optimized_lora"


@dataclass(frozen=True)
class CandidateArtifactPaths:
    source_path: Path
    library_path: Path


def _looks_like_fatal_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    fatal_markers = (
        "illegal memory access",
        "device-side assert",
        "device side assert",
        "unspecified launch failure",
        "misaligned address",
        "warp illegal address",
        "cuda error",
        "cublas_status_execution_failed",
        "context is destroyed",
    )
    return any(marker in message for marker in fatal_markers)


def _runtime_failure_evaluation(
    *,
    candidate_id: str,
    notes: list[str],
    rtol: float,
    atol: float,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        correctness=CorrectnessResult(
            passed=False,
            max_abs_err=float("inf"),
            rel_l2_err=float("inf"),
            rtol=rtol,
            atol=atol,
        ),
        student_benchmark=empty_benchmark_result(),
        reference_benchmark=empty_benchmark_result(),
        speedup=0.0,
        notes=notes,
    )


def _subprocess_timeout_seconds() -> int:
    raw = str(os.environ.get("PROFILER_AGENT_PHASE2_SUBPROCESS_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return 600
    try:
        value = int(raw)
    except ValueError:
        return 600
    return max(30, value)


def _runtime_eval_worker_path(root_dir: Path) -> Path:
    return root_dir / "profiler_agent" / "phase2" / "runtime_eval_worker.py"


def _build_subprocess_env(root_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "").strip()
    root_text = str(root_dir)
    if existing:
        if root_text not in existing.split(os.pathsep):
            env["PYTHONPATH"] = root_text + os.pathsep + existing
    else:
        env["PYTHONPATH"] = root_text
    return env


def _candidate_evaluation_from_dict(payload: dict[str, Any]) -> CandidateEvaluation:
    correctness_data = payload.get("correctness") or {}
    student_data = payload.get("student_benchmark") or {}
    reference_data = payload.get("reference_benchmark") or {}
    compilation_data = payload.get("compilation")
    load_data = payload.get("load")
    return CandidateEvaluation(
        candidate_id=str(payload.get("candidate_id", "")),
        correctness=CorrectnessResult(
            passed=bool(correctness_data.get("passed", False)),
            max_abs_err=float(correctness_data.get("max_abs_err", float("inf"))),
            rel_l2_err=float(correctness_data.get("rel_l2_err", float("inf"))),
            rtol=float(correctness_data.get("rtol", 1e-4)),
            atol=float(correctness_data.get("atol", 1e-4)),
        ),
        student_benchmark=BenchmarkResult(
            warmup_runs=int(student_data.get("warmup_runs", 0)),
            measured_runs=int(student_data.get("measured_runs", 0)),
            median_runtime_ms=float(student_data.get("median_runtime_ms", 0.0)),
            min_runtime_ms=float(student_data.get("min_runtime_ms", 0.0)),
            max_runtime_ms=float(student_data.get("max_runtime_ms", 0.0)),
            all_runtime_ms=[float(x) for x in student_data.get("all_runtime_ms", [])],
        ),
        reference_benchmark=BenchmarkResult(
            warmup_runs=int(reference_data.get("warmup_runs", 0)),
            measured_runs=int(reference_data.get("measured_runs", 0)),
            median_runtime_ms=float(reference_data.get("median_runtime_ms", 0.0)),
            min_runtime_ms=float(reference_data.get("min_runtime_ms", 0.0)),
            max_runtime_ms=float(reference_data.get("max_runtime_ms", 0.0)),
            all_runtime_ms=[float(x) for x in reference_data.get("all_runtime_ms", [])],
        ),
        speedup=float(payload.get("speedup", 0.0)),
        notes=[str(x) for x in payload.get("notes", [])],
        compilation=(
            CompilationResult(
                ok=bool(compilation_data.get("ok", False)),
                command=[str(x) for x in compilation_data.get("command", [])],
                returncode=int(compilation_data.get("returncode", 0)),
                stdout_tail=str(compilation_data.get("stdout_tail", "")),
                stderr_tail=str(compilation_data.get("stderr_tail", "")),
                output_path=str(compilation_data.get("output_path", "")),
            )
            if isinstance(compilation_data, dict)
            else None
        ),
        load=(
            LoadResult(
                ok=bool(load_data.get("ok", False)),
                library_path=str(load_data.get("library_path", "")),
                symbol_name=str(load_data.get("symbol_name", _ENTRYPOINT_SYMBOL)),
                error=str(load_data.get("error", "")),
            )
            if isinstance(load_data, dict)
            else None
        ),
    )


def _aggregate_benchmarks(items: list[BenchmarkResult]) -> BenchmarkResult:
    if not items:
        return empty_benchmark_result()
    all_samples: list[float] = []
    for item in items:
        all_samples.extend(float(x) for x in item.all_runtime_ms)
    if not all_samples:
        return empty_benchmark_result()
    all_samples.sort()
    warmup_runs = max(item.warmup_runs for item in items)
    measured_runs = len(all_samples)
    mid = measured_runs // 2
    if measured_runs % 2 == 0:
        median = (all_samples[mid - 1] + all_samples[mid]) / 2.0
    else:
        median = all_samples[mid]
    return BenchmarkResult(
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        median_runtime_ms=float(median),
        min_runtime_ms=float(all_samples[0]),
        max_runtime_ms=float(all_samples[-1]),
        all_runtime_ms=all_samples,
    )


def _default_output_allocator(spec: LoraProblemSpec, inputs: dict[str, Any]):
    x = inputs.get("X")
    shape = (spec.resolved_output_dim(), spec.num_tokens)
    if x is not None and hasattr(x, "new_empty"):
        return x.new_empty(shape)
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch_output_allocator_unavailable") from exc
    device = getattr(x, "device", spec.device)
    dtype = getattr(x, "dtype", torch.float32)
    return torch.empty(shape, device=device, dtype=dtype)


def _default_stream_resolver(inputs: dict[str, Any]) -> int:
    x = inputs.get("X")
    device = getattr(x, "device", None)
    device_type = getattr(device, "type", "")
    if device_type != "cuda":
        return 0
    try:
        import torch  # type: ignore
    except Exception:
        return 0
    try:
        stream = torch.cuda.current_stream(device=device)
        return int(getattr(stream, "cuda_stream", 0) or 0)
    except Exception:
        return 0


def _tensor_ptr(value: Any, *, name: str) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise RuntimeError(f"tensor_pointer_unavailable:{name}")
    ptr = int(data_ptr())
    if ptr == 0:
        raise RuntimeError(f"tensor_pointer_is_null:{name}")
    return ptr


def invoke_launch_optimized_lora(
    symbol: Any,
    *,
    inputs: dict[str, Any],
    output: Any,
    spec: LoraProblemSpec,
    stream_ptr: int = 0,
) -> Any:
    symbol.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    symbol.restype = None
    symbol(
        ctypes.c_void_p(_tensor_ptr(inputs["W"], name="W")),
        ctypes.c_void_p(_tensor_ptr(inputs["X"], name="X")),
        ctypes.c_void_p(_tensor_ptr(inputs["A"], name="A")),
        ctypes.c_void_p(_tensor_ptr(inputs["B"], name="B")),
        ctypes.c_void_p(_tensor_ptr(output, name="Y")),
        ctypes.c_int(spec.hidden_dim),
        ctypes.c_int(spec.num_tokens),
        ctypes.c_void_p(stream_ptr),
    )
    return output


def _shared_library_suffix() -> str:
    if os.name == "nt":
        return ".dll"
    return ".so"


def build_candidate_artifact_paths(root_dir: Path, candidate_id: str) -> CandidateArtifactPaths:
    candidate_dir = root_dir / ".agent_artifacts" / "phase2_candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return CandidateArtifactPaths(
        source_path=candidate_dir / "optimized_lora.cu",
        library_path=candidate_dir / f"optimized_lora{_shared_library_suffix()}",
    )


def write_candidate_source(paths: CandidateArtifactPaths, candidate: GeneratedCandidate) -> Path:
    paths.source_path.write_text(candidate.source_code, encoding="utf-8")
    return paths.source_path


def build_nvcc_shared_library_command(source_path: Path, library_path: Path) -> list[str]:
    nvcc = resolve_command_path("nvcc")
    executable = nvcc if nvcc is not None else "nvcc"
    argv = [
        executable,
        str(source_path),
        "-O3",
        "-std=c++14",
        "-shared",
        "-o",
        str(library_path),
    ]
    if os.name == "nt":
        argv[4:4] = ["-Xcompiler", "/wd4819"]
    else:
        argv[4:4] = ["-Xcompiler", "-fPIC"]
    return argv


def compile_candidate_source(source_path: Path, library_path: Path) -> CompilationResult:
    nvcc = resolve_command_path("nvcc")
    command = build_nvcc_shared_library_command(source_path, library_path)
    if nvcc is None:
        return CompilationResult(
            ok=False,
            command=command,
            returncode=127,
            stdout_tail="",
            stderr_tail="required_command_not_found:nvcc",
            output_path=str(library_path),
        )
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0 and not stderr.strip() and stdout.strip():
        stderr = stdout
    return CompilationResult(
        ok=completed.returncode == 0,
        command=command,
        returncode=completed.returncode,
        stdout_tail=tail_text(stdout, 1000),
        stderr_tail=tail_text(stderr, 1000),
        output_path=str(library_path),
    )


def load_compiled_candidate(library_path: Path, *, symbol_name: str = _ENTRYPOINT_SYMBOL) -> LoadResult:
    if not library_path.exists():
        return LoadResult(
            ok=False,
            library_path=str(library_path),
            symbol_name=symbol_name,
            error="compiled_library_missing",
        )
    try:
        handle = ctypes.CDLL(str(library_path))
    except OSError as exc:
        return LoadResult(
            ok=False,
            library_path=str(library_path),
            symbol_name=symbol_name,
            error=f"library_load_failed:{exc}",
        )
    if not hasattr(handle, symbol_name):
        return LoadResult(
            ok=False,
            library_path=str(library_path),
            symbol_name=symbol_name,
            error=f"missing_required_symbol:{symbol_name}",
        )
    return LoadResult(
        ok=True,
        library_path=str(library_path),
        symbol_name=symbol_name,
        error="",
    )


def build_ctypes_candidate_runner(
    *,
    output_allocator: Callable[[LoraProblemSpec, dict[str, Any]], Any] | None = None,
    stream_resolver: Callable[[dict[str, Any]], int] | None = None,
    symbol_name: str = _ENTRYPOINT_SYMBOL,
) -> Callable[[GeneratedCandidate, CandidateArtifactPaths, LoadResult, LoraProblemSpec, dict[str, Any], Any], Any]:
    allocator = output_allocator or _default_output_allocator
    resolve_stream = stream_resolver or _default_stream_resolver

    def runner(
        candidate: GeneratedCandidate,
        paths: CandidateArtifactPaths,
        load_result: LoadResult,
        spec: LoraProblemSpec,
        inputs: dict[str, Any],
        backend: Any,
    ) -> Any:
        _ = candidate, paths
        if not load_result.ok:
            raise RuntimeError(f"load_result_not_ok:{load_result.error}")
        handle = ctypes.CDLL(load_result.library_path)
        if not hasattr(handle, symbol_name):
            raise RuntimeError(f"missing_required_symbol:{symbol_name}")
        symbol = getattr(handle, symbol_name)
        output = allocator(spec, inputs)
        stream_ptr = int(resolve_stream(inputs))
        invoke_launch_optimized_lora(
            symbol,
            inputs=inputs,
            output=output,
            spec=spec,
            stream_ptr=stream_ptr,
        )
        synchronize = getattr(backend, "synchronize", None)
        if callable(synchronize):
            synchronize()
        return output

    return runner


def build_harness_runtime_evaluator(
    *,
    problem_specs: list[LoraProblemSpec],
    candidate_runner: Callable[[GeneratedCandidate, CandidateArtifactPaths, LoadResult, LoraProblemSpec, dict[str, Any], Any], Any],
    reference_runner: Callable[[dict[str, Any], Any], Any] | None = None,
    backend: Any | None = None,
    warmup_runs: int = 3,
    measured_runs: int = 7,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Callable[[GeneratedCandidate, CandidateArtifactPaths, LoadResult], CandidateEvaluation]:
    if not problem_specs:
        raise ValueError("problem_specs must be non-empty")

    def runtime_evaluator(candidate: GeneratedCandidate, paths: CandidateArtifactPaths, load_result: LoadResult) -> CandidateEvaluation:
        runtime_backend = backend or resolve_backend()
        ref_runner = reference_runner or lora_reference_forward
        correctness_results: list[CorrectnessResult] = []
        student_benchmarks: list[BenchmarkResult] = []
        reference_benchmarks: list[BenchmarkResult] = []
        notes: list[str] = []

        for spec in problem_specs:
            try:
                inputs = generate_lora_inputs(spec, backend=runtime_backend)
                reference_output = ref_runner(inputs, runtime_backend)
            except Exception as exc:
                if _looks_like_fatal_cuda_runtime_error(exc):
                    notes.append(f"fatal_cuda_runtime_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                else:
                    notes.append(f"runtime_environment_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                return _runtime_failure_evaluation(
                    candidate_id=candidate.candidate_id,
                    notes=notes,
                    rtol=rtol,
                    atol=atol,
                )
            try:
                student_output = candidate_runner(candidate, paths, load_result, spec, inputs, runtime_backend)
            except Exception as exc:
                if _looks_like_fatal_cuda_runtime_error(exc):
                    notes.append(f"fatal_cuda_runtime_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                else:
                    notes.append(f"candidate_runner_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                return _runtime_failure_evaluation(
                    candidate_id=candidate.candidate_id,
                    notes=notes,
                    rtol=rtol,
                    atol=atol,
                )
            try:
                correctness = check_correctness(
                    student_output,
                    reference_output,
                    backend=runtime_backend,
                    rtol=rtol,
                    atol=atol,
                )
            except Exception as exc:
                if _looks_like_fatal_cuda_runtime_error(exc):
                    notes.append(f"fatal_cuda_runtime_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                else:
                    notes.append(f"correctness_check_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                return _runtime_failure_evaluation(
                    candidate_id=candidate.candidate_id,
                    notes=notes,
                    rtol=rtol,
                    atol=atol,
                )
            correctness_results.append(correctness)
            if not correctness.passed:
                notes.append(f"correctness_failed:hidden_dim={spec.hidden_dim}")
                continue

            try:
                reference_benchmarks.append(
                    benchmark_callable(
                        lambda i=inputs: ref_runner(i, runtime_backend),
                        backend=runtime_backend,
                        warmup_runs=warmup_runs,
                        measured_runs=measured_runs,
                    )
                )
                student_benchmarks.append(
                    benchmark_callable(
                        lambda s=spec, i=inputs: candidate_runner(candidate, paths, load_result, s, i, runtime_backend),
                        backend=runtime_backend,
                        warmup_runs=warmup_runs,
                        measured_runs=measured_runs,
                    )
                )
            except Exception as exc:
                if _looks_like_fatal_cuda_runtime_error(exc):
                    notes.append(f"fatal_cuda_runtime_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                else:
                    notes.append(f"benchmark_error:hidden_dim={spec.hidden_dim}:{type(exc).__name__}")
                return _runtime_failure_evaluation(
                    candidate_id=candidate.candidate_id,
                    notes=notes,
                    rtol=rtol,
                    atol=atol,
                )

        overall_passed = all(item.passed for item in correctness_results)
        max_abs_err = max((item.max_abs_err for item in correctness_results), default=0.0)
        rel_l2_err = max((item.rel_l2_err for item in correctness_results), default=0.0)
        correctness_summary = CorrectnessResult(
            passed=overall_passed,
            max_abs_err=float(max_abs_err),
            rel_l2_err=float(rel_l2_err),
            rtol=rtol,
            atol=atol,
        )
        reference_summary = _aggregate_benchmarks(reference_benchmarks)
        student_summary = _aggregate_benchmarks(student_benchmarks)
        speedup = compute_speedup(reference_summary, student_summary) if overall_passed else 0.0
        if overall_passed:
            notes.append("correctness_passed_all_specs")
        else:
            notes.append("correctness_not_yet_passing")
        if not student_benchmarks or not reference_benchmarks:
            notes.append("benchmark_partial_or_skipped_due_to_correctness")

        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            correctness=correctness_summary,
            student_benchmark=student_summary,
            reference_benchmark=reference_summary,
            speedup=speedup,
            notes=notes,
        )

    return runtime_evaluator


def build_subprocess_runtime_evaluator(
    *,
    root_dir: Path,
    problem_specs: list[LoraProblemSpec],
    warmup_runs: int = 3,
    measured_runs: int = 7,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Callable[[GeneratedCandidate, CandidateArtifactPaths, LoadResult], CandidateEvaluation]:
    def runtime_evaluator(candidate: GeneratedCandidate, paths: CandidateArtifactPaths, load_result: LoadResult) -> CandidateEvaluation:
        candidate_dir = paths.source_path.parent
        request_path = candidate_dir / "runtime_eval_request.json"
        response_path = candidate_dir / "runtime_eval_response.json"
        request_payload = {
            "candidate_id": candidate.candidate_id,
            "library_path": str(paths.library_path),
            "load_result": load_result.to_dict(),
            "problem_specs": [
                {
                    "hidden_dim": spec.hidden_dim,
                    "low_rank": spec.low_rank,
                    "output_dim": spec.output_dim,
                    "num_tokens": spec.num_tokens,
                    "dtype": spec.dtype,
                    "device": spec.device,
                }
                for spec in problem_specs
            ],
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "rtol": rtol,
            "atol": atol,
        }
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
        if response_path.exists():
            response_path.unlink()

        command = [
            sys.executable,
            str(_runtime_eval_worker_path(root_dir)),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_subprocess_timeout_seconds(),
                cwd=str(root_dir),
                env=_build_subprocess_env(root_dir),
            )
        except subprocess.TimeoutExpired as exc:
            return _runtime_failure_evaluation(
                candidate_id=candidate.candidate_id,
                notes=[f"runtime_subprocess_timeout:returncode=124", f"runtime_subprocess_stderr:{tail_text((exc.stderr or ''), 500)}"],
                rtol=rtol,
                atol=atol,
            )

        if completed.returncode != 0:
            return _runtime_failure_evaluation(
                candidate_id=candidate.candidate_id,
                notes=[
                    f"runtime_subprocess_failed:returncode={completed.returncode}",
                    f"runtime_subprocess_stderr:{tail_text(completed.stderr or '', 500)}",
                ],
                rtol=rtol,
                atol=atol,
            )
        if not response_path.exists():
            return _runtime_failure_evaluation(
                candidate_id=candidate.candidate_id,
                notes=["runtime_subprocess_missing_response"],
                rtol=rtol,
                atol=atol,
            )
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return _runtime_failure_evaluation(
                candidate_id=candidate.candidate_id,
                notes=["runtime_subprocess_invalid_response_json"],
                rtol=rtol,
                atol=atol,
            )
        if not isinstance(payload, dict):
            return _runtime_failure_evaluation(
                candidate_id=candidate.candidate_id,
                notes=["runtime_subprocess_invalid_response_shape"],
                rtol=rtol,
                atol=atol,
            )
        return _candidate_evaluation_from_dict(payload)

    return runtime_evaluator


def build_compile_checked_candidate_evaluator(
    *,
    root_dir: Path,
    runtime_evaluator: Callable[[GeneratedCandidate, CandidateArtifactPaths, LoadResult], CandidateEvaluation] | None = None,
) -> Callable[[GeneratedCandidate], CandidateEvaluation]:
    def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
        paths = build_candidate_artifact_paths(root_dir, candidate.candidate_id)
        write_candidate_source(paths, candidate)
        compilation = compile_candidate_source(paths.source_path, paths.library_path)
        if not compilation.ok:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                correctness=CorrectnessResult(
                    passed=False,
                    max_abs_err=float("inf"),
                    rel_l2_err=float("inf"),
                    rtol=1e-4,
                    atol=1e-4,
                ),
                student_benchmark=empty_benchmark_result(),
                reference_benchmark=empty_benchmark_result(),
                speedup=0.0,
                notes=["compile_failed"],
                compilation=compilation,
                load=LoadResult(
                    ok=False,
                    library_path=str(paths.library_path),
                    symbol_name=_ENTRYPOINT_SYMBOL,
                    error="compile_failed",
                ),
            )

        load_result = load_compiled_candidate(paths.library_path)
        if not load_result.ok:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                correctness=CorrectnessResult(
                    passed=False,
                    max_abs_err=float("inf"),
                    rel_l2_err=float("inf"),
                    rtol=1e-4,
                    atol=1e-4,
                ),
                student_benchmark=empty_benchmark_result(),
                reference_benchmark=empty_benchmark_result(),
                speedup=0.0,
                notes=["load_failed"],
                compilation=compilation,
                load=load_result,
            )

        if runtime_evaluator is None:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                correctness=CorrectnessResult(
                    passed=False,
                    max_abs_err=0.0,
                    rel_l2_err=0.0,
                    rtol=1e-4,
                    atol=1e-4,
                ),
                student_benchmark=empty_benchmark_result(),
                reference_benchmark=empty_benchmark_result(),
                speedup=0.0,
                notes=["runtime_evaluator_not_connected_yet"],
                compilation=compilation,
                load=load_result,
            )

        evaluation = runtime_evaluator(candidate, paths, load_result)
        return CandidateEvaluation(
            candidate_id=evaluation.candidate_id,
            correctness=evaluation.correctness,
            student_benchmark=evaluation.student_benchmark,
            reference_benchmark=evaluation.reference_benchmark,
            speedup=evaluation.speedup,
            notes=list(evaluation.notes),
            compilation=compilation,
            load=load_result,
        )

    return evaluator
