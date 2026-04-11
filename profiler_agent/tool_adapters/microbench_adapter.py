from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
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


def _repo_root() -> Path:
    # profiler_agent/tool_adapters/microbench_adapter.py -> repo root
    return Path(__file__).resolve().parents[2]


def _probe_source_path(probe_name: str) -> Path:
    return _repo_root() / "probes" / probe_name / "probe.cu"


def _probe_binary_path(probe_name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _repo_root() / "outputs" / "probes" / f"{probe_name}{suffix}"


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


def measure_metric_with_evidence(metric_name: str, run_cmd: str) -> ProbeResult:
    _ = run_cmd
    probe_name = _PROBE_MAP.get(metric_name)
    if probe_name is None:
        return ProbeResult(
            value=None,
            source="unsupported_metric",
            compile_returncode=0,
            run_returncode=0,
            compile_stdout_tail="",
            compile_stderr_tail="",
            run_stdout_tail="",
            run_stderr_tail="",
            parsed_from="none",
        )

    source = _probe_source_path(probe_name)
    binary = _probe_binary_path(probe_name)
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
        )

    compile_rc, compile_out, compile_err = _compile_probe(source, binary)
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
        )

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
        )

    value, parsed_from, meta = _parse_probe_output(metric_name=metric_name, stdout=run_out or "")
    return ProbeResult(
        value=value,
        source="microbench_probe",
        compile_returncode=compile_rc,
        run_returncode=run_rc,
        compile_stdout_tail=(compile_out or "")[-1000:],
        compile_stderr_tail=(compile_err or "")[-1000:],
        run_stdout_tail=(run_out or "")[-1000:],
        run_stderr_tail=(run_err or "")[-1000:],
        parsed_from=parsed_from,
        metric_name=str(meta.get("metric_name")) if meta.get("metric_name") is not None else metric_name,
        sample_count=int(meta["sample_count"]) if "sample_count" in meta else None,
        best_value=float(meta["best_value"]) if "best_value" in meta else None,
        median_value=float(meta["median_value"]) if "median_value" in meta else None,
        std_value=float(meta["std_value"]) if "std_value" in meta else None,
    )


def measure_metric(metric_name: str, run_cmd: str) -> Optional[float]:
    result = measure_metric_with_evidence(metric_name=metric_name, run_cmd=run_cmd)
    return result.value
