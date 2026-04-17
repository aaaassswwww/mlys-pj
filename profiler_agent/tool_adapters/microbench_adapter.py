from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from profiler_agent.codegen.generator import ProbeCodeGenerator


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_DEFAULT_PROBE_REPEAT = 5
_MAX_PROBE_REPEAT = 20
_DEFAULT_GENERATION_RETRY = 2
_PROBE_MAP: dict[str, str] = {
    "l1_latency_cycles": "l1_latency_cycles",
    "l2_latency_cycles": "l2_latency_cycles",
    "max_shmem_per_block_kb": "max_shmem_per_block",
    "dram_latency_cycles": "dram_latency_cycles",
    "shared_peak_bandwidth_gbps": "shared_peak_bandwidth_gbps",
    "global_peak_bandwidth_gbps": "global_peak_bandwidth_gbps",
    "l2_cache_capacity_kb": "l2_cache_capacity_kb",
    "shmem_bank_conflict_penalty_cycles": "shmem_bank_conflict_penalty_cycles",
}


@dataclass(frozen=True)
class ProbeResult:
    value: Optional[float]
    source: str
    compile_returncode: int
    run_returncode: int
    compile_stdout_tail: str
    compile_stderr_tail: str
    run_stdout_tail: str
    run_stderr_tail: str
    parsed_from: str
    metric_name: Optional[str] = None
    sample_count: Optional[int] = None
    best_value: Optional[float] = None
    median_value: Optional[float] = None
    std_value: Optional[float] = None
    run_values: Optional[list[float]] = None
    source_path: Optional[str] = None
    generation_source: Optional[str] = None
    generation_attempts: Optional[int] = None
    generation_error: Optional[str] = None
    generation_trace: Optional[list[dict[str, object]]] = None


def _repo_root() -> Path:
    # profiler_agent/tool_adapters/microbench_adapter.py -> repo root
    return Path(__file__).resolve().parents[2]


def _probe_source_path(probe_name: str) -> Path:
    return _repo_root() / "probes" / probe_name / "probe.cu"


def _generated_probe_source_path(metric_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", metric_name.strip()).strip("_") or "unknown_metric"
    return _repo_root() / "outputs" / "generated_probes" / safe / "probe.cu"


def _probe_binary_path(probe_name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _repo_root() / "outputs" / "probes" / f"{probe_name}{suffix}"


def _generated_probe_binary_path(metric_name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", metric_name.strip()).strip("_") or "unknown_metric"
    return _repo_root() / "outputs" / "generated_probes" / f"{safe}{suffix}"


def _extract_last_numeric(text: str) -> Optional[float]:
    matches = _NUMERIC_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_key_value_tokens(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip(",")
        if key:
            parsed[key] = value
    return parsed


def _parse_probe_output(metric_name: str, stdout: str) -> tuple[Optional[float], str, dict[str, float | int | str]]:
    for line in reversed((stdout or "").splitlines()):
        kv = _parse_key_value_tokens(line)
        if "metric" in kv and "value" in kv:
            metric = kv.get("metric", "")
            if metric and metric != metric_name:
                continue
            value = _extract_last_numeric(kv.get("value", ""))
            if value is None:
                continue
            meta: dict[str, float | int | str] = {"metric_name": metric}
            if "samples" in kv:
                samples = _extract_last_numeric(kv["samples"])
                if samples is not None:
                    meta["sample_count"] = int(samples)
            if "best" in kv:
                best = _extract_last_numeric(kv["best"])
                if best is not None:
                    meta["best_value"] = best
            if "median" in kv:
                median = _extract_last_numeric(kv["median"])
                if median is not None:
                    meta["median_value"] = median
            if "std" in kv:
                std = _extract_last_numeric(kv["std"])
                if std is not None:
                    meta["std_value"] = std
            return value, "structured_metric_value", meta

        # Backward-compatible parse: "<metric>=<value>"
        if "=" in line:
            key, raw = line.split("=", 1)
            if key.strip() == metric_name:
                value = _extract_last_numeric(raw)
                if value is not None:
                    return value, "legacy_metric_equals", {"metric_name": key.strip()}

    for line in reversed((stdout or "").splitlines()):
        value = _extract_last_numeric(line)
        if value is not None:
            return value, "stdout_last_numeric", {}
    return None, "none", {}


def _compile_probe(source: Path, binary: Path) -> tuple[int, str, str]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return 127, "", "nvcc_not_found"

    binary.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        nvcc,
        str(source),
        "-O3",
        "-std=c++14",
        "-o",
        str(binary),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=240)
    return completed.returncode, completed.stdout, completed.stderr


def _run_probe(binary: Path) -> tuple[int, str, str]:
    if not binary.exists():
        return 127, "", "probe_binary_not_found"
    completed = subprocess.run([str(binary)], capture_output=True, text=True, check=False, timeout=180)
    return completed.returncode, completed.stdout, completed.stderr


def _probe_repeat_count() -> int:
    raw = os.environ.get("PROFILER_AGENT_PROBE_REPEAT")
    if raw is None:
        return _DEFAULT_PROBE_REPEAT
    value = _extract_last_numeric(raw)
    if value is None:
        return _DEFAULT_PROBE_REPEAT
    return max(1, min(int(value), _MAX_PROBE_REPEAT))


def _prefer_lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return (
        "latency" in lowered
        or "penalty" in lowered
        or lowered.endswith("_cycles")
    )


def _aggregate_best_value(metric_name: str, values: list[float]) -> float:
    if _prefer_lower_is_better(metric_name):
        return min(values)
    return max(values)


def _probe_source_mode() -> str:
    mode = os.environ.get("PROFILER_AGENT_PROBE_SOURCE_MODE", "llm_generated").strip().lower()
    if mode not in {"llm_generated", "static_fallback"}:
        return "llm_generated"
    return mode


def _generation_retry_count() -> int:
    raw = os.environ.get("PROFILER_AGENT_GENERATION_RETRY")
    value = _extract_last_numeric(raw or "")
    if value is None:
        return _DEFAULT_GENERATION_RETRY
    return max(0, min(int(value), 5))


def _disable_static_fallback() -> bool:
    raw = os.environ.get("PROFILER_AGENT_DISABLE_STATIC_FALLBACK", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _select_probe_source(
    *,
    metric_name: str,
    run_cmd: str,
) -> tuple[Optional[Path], str, int, str, list[dict[str, object]]]:
    _ = run_cmd
    mode = _probe_source_mode()
    generator = ProbeCodeGenerator()
    retries = _generation_retry_count()
    attempts = 0
    last_error = ""
    generation_trace: list[dict[str, object]] = []

    if mode == "llm_generated":
        if not generator.is_enabled():
            last_error = "llm_disabled"
            generation_trace.append(
                {
                    "attempt": 0,
                    "type": "llm_generation",
                    "ok": False,
                    "error": "llm_disabled",
                }
            )
        else:
            out_dir = _repo_root() / "outputs" / "generated_probes"
            prior_error: str | None = None
            for _ in range(retries + 1):
                attempts += 1
                gen = generator.generate_probe(metric=metric_name, out_dir=out_dir, prior_error=prior_error)
                generation_trace.append(
                    {
                        "attempt": attempts,
                        "type": "llm_generation",
                        "ok": bool(gen.ok),
                        "error": gen.error,
                        "source_path": str(gen.source_path),
                        "has_prior_error": bool(prior_error),
                    }
                )
                if gen.ok:
                    return gen.source_path, gen.source_type, attempts, "", generation_trace
                last_error = gen.error
                prior_error = gen.error
        if _disable_static_fallback():
            generation_trace.append(
                {
                    "attempt": attempts + 1,
                    "type": "fallback",
                    "ok": False,
                    "error": "static_fallback_disabled",
                }
            )
            return None, "llm_generated_only", attempts, last_error or "llm_generation_failed", generation_trace

    probe_name = _PROBE_MAP.get(metric_name)
    if probe_name is None:
        return None, "unsupported_metric", attempts, last_error or "unsupported_metric", generation_trace
    source = _probe_source_path(probe_name)
    generation_trace.append(
        {
            "attempt": attempts + 1,
            "type": "fallback",
            "ok": source.exists(),
            "error": "",
            "source_path": str(source),
            "fallback_kind": "static_probe",
        }
    )
    return source, "static_probe", attempts, last_error, generation_trace


def measure_metric_with_evidence(metric_name: str, run_cmd: str) -> ProbeResult:
    source, source_type, generation_attempts, generation_error, generation_trace = _select_probe_source(
        metric_name=metric_name,
        run_cmd=run_cmd,
    )
    if source is None:
        result_source = "unsupported_metric"
        if source_type == "llm_generated_only":
            result_source = "llm_generation_failed"
        return ProbeResult(
            value=None,
            source=result_source,
            compile_returncode=0,
            run_returncode=0,
            compile_stdout_tail="",
            compile_stderr_tail="",
            run_stdout_tail="",
            run_stderr_tail="",
            parsed_from="none",
            metric_name=metric_name,
            generation_source=source_type,
            generation_attempts=generation_attempts,
            generation_error=generation_error or "probe_source_missing",
            generation_trace=generation_trace,
        )

    binary = (
        _generated_probe_binary_path(metric_name)
        if source_type == "llm_generated"
        else _probe_binary_path(_PROBE_MAP.get(metric_name, metric_name))
    )
    if not source.exists():
        return ProbeResult(
            value=None,
            source="probe_source_missing",
            compile_returncode=127,
            run_returncode=127,
            compile_stdout_tail="",
            compile_stderr_tail=f"missing_source:{source}",
            run_stdout_tail="",
            run_stderr_tail="",
            parsed_from="none",
            metric_name=metric_name,
            source_path=str(source),
            generation_source=source_type,
            generation_attempts=generation_attempts,
            generation_error=generation_error or "probe_source_path_not_found",
            generation_trace=generation_trace,
        )

    compile_rc, compile_out, compile_err = _compile_probe(source, binary)
    # If LLM-generated source fails to compile, attempt one or more repairs.
    if compile_rc != 0 and source_type == "llm_generated":
        generator = ProbeCodeGenerator()
        retry = _generation_retry_count()
        prior_error = (compile_err or "")[-1500:]
        for _ in range(retry):
            fix = generator.generate_probe(
                metric=metric_name,
                out_dir=_repo_root() / "outputs" / "generated_probes",
                prior_error=prior_error,
            )
            generation_attempts += 1
            if not fix.ok:
                prior_error = fix.error or prior_error
                generation_error = fix.error
                continue
            source = fix.source_path
            compile_rc, compile_out, compile_err = _compile_probe(source, binary)
            if compile_rc == 0:
                generation_error = ""
                break
            prior_error = (compile_err or "")[-1500:]

    if compile_rc != 0:
        return ProbeResult(
            value=None,
            source="compile_failed",
            compile_returncode=compile_rc,
            run_returncode=0,
            compile_stdout_tail=(compile_out or "")[-1000:],
            compile_stderr_tail=(compile_err or "")[-1000:],
            run_stdout_tail="",
            run_stderr_tail="",
            parsed_from="none",
            metric_name=metric_name,
            source_path=str(source),
            generation_source=source_type,
            generation_attempts=generation_attempts,
            generation_error=generation_error or (compile_err or "")[-500:],
            generation_trace=generation_trace,
        )

    repeat = _probe_repeat_count()
    parsed_modes: list[str] = []
    run_values: list[float] = []
    run_stds: list[float] = []
    metric_name_seen: Optional[str] = None
    run_out = ""
    run_err = ""
    run_rc = 0

    for _ in range(repeat):
        run_rc, run_out, run_err = _run_probe(binary)
        if run_rc != 0:
            return ProbeResult(
                value=None,
                source="run_failed",
                compile_returncode=compile_rc,
                run_returncode=run_rc,
                compile_stdout_tail=(compile_out or "")[-1000:],
                compile_stderr_tail=(compile_err or "")[-1000:],
                run_stdout_tail=(run_out or "")[-1000:],
                run_stderr_tail=(run_err or "")[-1000:],
                parsed_from="none",
                metric_name=metric_name,
                source_path=str(source),
                generation_source=source_type,
                generation_attempts=generation_attempts,
                generation_error=generation_error,
                generation_trace=generation_trace,
            )

        value, parsed_from, meta = _parse_probe_output(metric_name=metric_name, stdout=run_out or "")
        parsed_modes.append(parsed_from)
        if value is not None:
            run_values.append(float(value))
        if meta.get("std_value") is not None:
            run_stds.append(float(meta["std_value"]))
        if metric_name_seen is None and meta.get("metric_name") is not None:
            metric_name_seen = str(meta.get("metric_name"))

    if run_values:
        aggregate_value = float(statistics.median(run_values))
        aggregate_best = _aggregate_best_value(metric_name=metric_name, values=run_values)
        aggregate_std = float(statistics.pstdev(run_values)) if len(run_values) > 1 else 0.0
        parsed_from = f"multi_run_median[{parsed_modes[-1]}]"
        return ProbeResult(
            value=aggregate_value,
            source="microbench_probe",
            compile_returncode=compile_rc,
            run_returncode=run_rc,
            compile_stdout_tail=(compile_out or "")[-1000:],
            compile_stderr_tail=(compile_err or "")[-1000:],
            run_stdout_tail=(run_out or "")[-1000:],
            run_stderr_tail=(run_err or "")[-1000:],
            parsed_from=parsed_from,
            metric_name=metric_name_seen or metric_name,
            sample_count=len(run_values),
            best_value=float(aggregate_best),
            median_value=aggregate_value,
            std_value=aggregate_std,
            run_values=[float(v) for v in run_values],
            source_path=str(source),
            generation_source=source_type,
            generation_attempts=generation_attempts,
            generation_error=generation_error,
            generation_trace=generation_trace,
        )

    # Parse failed across all runs; preserve parser-level hints when available.
    parsed_from = parsed_modes[-1] if parsed_modes else "none"
    return ProbeResult(
        value=None,
        source="microbench_probe",
        compile_returncode=compile_rc,
        run_returncode=run_rc,
        compile_stdout_tail=(compile_out or "")[-1000:],
        compile_stderr_tail=(compile_err or "")[-1000:],
        run_stdout_tail=(run_out or "")[-1000:],
        run_stderr_tail=(run_err or "")[-1000:],
        parsed_from=parsed_from,
        metric_name=metric_name_seen or metric_name,
        sample_count=0,
        best_value=None,
        median_value=None,
        std_value=float(statistics.mean(run_stds)) if run_stds else None,
        run_values=[],
        source_path=str(source),
        generation_source=source_type,
        generation_attempts=generation_attempts,
        generation_error=generation_error,
        generation_trace=generation_trace,
    )


def measure_metric(metric_name: str, run_cmd: str) -> Optional[float]:
    result = measure_metric_with_evidence(metric_name=metric_name, run_cmd=run_cmd)
    return result.value
