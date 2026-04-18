from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Optional


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CUDA_DEVICE_ATTRIBUTES: dict[str, tuple[int, float, str]] = {
    "device__attribute_max_gpu_frequency_khz": (13, 1.0, "kHz"),
    "launch__sm_count": (16, 1.0, "count"),
    "device__attribute_max_mem_frequency_khz": (36, 1.0, "kHz"),
    "device__attribute_fb_bus_width": (37, 1.0, "bits"),
}
_NVIDIA_SMI_FIELD_MAP: dict[str, tuple[str, float, str]] = {
    "device__attribute_max_gpu_frequency_khz": ("clocks.max.sm", 1000.0, "kHz"),
    "device__attribute_max_mem_frequency_khz": ("clocks.max.memory", 1000.0, "kHz"),
    "device__attribute_fb_bus_width": ("memory.bus_width", 1.0, "bits"),
    "launch__sm_count": ("multiprocessor_count", 1.0, "count"),
}


def _extract_last_numeric(text: str) -> Optional[float]:
    matches = _NUMERIC_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _candidate_cudart_paths() -> list[str]:
    candidates: list[str] = []
    if os.name == "nt":
        env_roots = [
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_ROOT"),
        ]
        for root in env_roots:
            if not root:
                continue
            for path in Path(root).glob("bin/cudart64_*.dll"):
                candidates.append(str(path))
        program_files = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if program_files.exists():
            for path in sorted(program_files.glob("v*/bin/cudart64_*.dll"), reverse=True):
                candidates.append(str(path))
        for name in ("cudart64_130.dll", "cudart64_120.dll", "cudart64_110.dll"):
            candidates.append(name)
    else:
        env_roots = [
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_ROOT"),
            "/usr/local/cuda",
        ]
        for root in env_roots:
            if not root:
                continue
            for pattern in ("lib64/libcudart.so*", "targets/*/lib/libcudart.so*"):
                for path in Path(root).glob(pattern):
                    candidates.append(str(path))
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _load_cudart() -> tuple[object | None, str | None, str | None]:
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    errors: list[str] = []
    for candidate in _candidate_cudart_paths():
        try:
            return loader(candidate), candidate, None
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    return None, None, "; ".join(errors[-3:]) if errors else "cudart_not_found"


def _query_cuda_device_attribute_once(target: str) -> dict[str, object]:
    mapped = _CUDA_DEVICE_ATTRIBUTES.get(target)
    if mapped is None:
        return {
            "target": target,
            "source": "unsupported_device_attribute",
            "value": None,
            "field": None,
            "unit": None,
            "command": [],
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "unsupported_device_attribute_target",
        }

    cudart, library_path, load_error = _load_cudart()
    if cudart is None:
        return {
            "target": target,
            "source": "cuda_runtime_unavailable",
            "value": None,
            "field": f"cudaDeviceGetAttribute({mapped[0]})",
            "unit": mapped[2],
            "command": [library_path] if library_path else [],
            "returncode": 127,
            "stdout_tail": "",
            "stderr_tail": load_error or "cudart_not_found",
        }

    device = ctypes.c_int()
    get_device_rc = int(cudart.cudaGetDevice(ctypes.byref(device)))
    if get_device_rc != 0:
        count = ctypes.c_int()
        count_rc = int(cudart.cudaGetDeviceCount(ctypes.byref(count)))
        if count_rc != 0 or int(count.value) <= 0:
            return {
                "target": target,
                "source": "cuda_runtime_query_failed",
                "value": None,
                "field": f"cudaDeviceGetAttribute({mapped[0]})",
                "unit": mapped[2],
                "command": [str(library_path or "cudart")],
                "returncode": get_device_rc if get_device_rc != 0 else count_rc,
                "stdout_tail": "",
                "stderr_tail": f"cudaGetDevice={get_device_rc}; cudaGetDeviceCount={count_rc}; device_count={int(count.value)}",
            }
        device = ctypes.c_int(0)

    raw_value = ctypes.c_int()
    attribute_id, scale, unit = mapped
    attr_rc = int(cudart.cudaDeviceGetAttribute(ctypes.byref(raw_value), attribute_id, int(device.value)))
    if attr_rc != 0:
        return {
            "target": target,
            "source": "cuda_runtime_query_failed",
            "value": None,
            "field": f"cudaDeviceGetAttribute({attribute_id})",
            "unit": unit,
            "command": [str(library_path or "cudart")],
            "returncode": attr_rc,
            "stdout_tail": "",
            "stderr_tail": f"cudaDeviceGetAttribute_failed device={int(device.value)} attribute={attribute_id}",
        }

    return {
        "target": target,
        "source": "cuda_runtime_attribute",
        "value": float(raw_value.value) * scale,
        "field": f"cudaDeviceGetAttribute({attribute_id})",
        "unit": unit,
        "command": [str(library_path or "cudart")],
        "returncode": 0,
        "stdout_tail": f"device={int(device.value)} raw_value={int(raw_value.value)}",
        "stderr_tail": "",
    }


def _query_gpu_field_once(field: str) -> tuple[Optional[float], dict[str, object]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, {
            "field": field,
            "command": cmd,
            "source": "nvidia_smi_unavailable",
            "returncode": 127,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    value = _extract_last_numeric(stdout)
    return value, {
        "field": field,
        "command": cmd,
        "source": "nvidia_smi_query",
        "returncode": completed.returncode,
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def _query_current_sm_clock_once() -> Optional[float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=clocks.current.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].strip())
    except ValueError:
        return None


def read_current_sm_clock_mhz() -> Optional[float]:
    return _query_current_sm_clock_once()


def sample_sm_clock_mhz(sample_count: int = 5, interval_s: float = 0.15) -> Optional[float]:
    if sample_count <= 0:
        return None

    values: list[float] = []
    for idx in range(sample_count):
        value = _query_current_sm_clock_once()
        if value is not None:
            values.append(value)
        if idx != sample_count - 1:
            time.sleep(interval_s)

    if not values:
        return None
    return float(statistics.median(values))


def sample_sm_clock_stats(sample_count: int = 7, interval_s: float = 0.12) -> dict[str, float | int | list[float] | None]:
    if sample_count <= 0:
        return {
            "sample_count": 0,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "values": [],
        }

    values: list[float] = []
    for idx in range(sample_count):
        value = _query_current_sm_clock_once()
        if value is not None and math.isfinite(value):
            values.append(float(value))
        if idx != sample_count - 1:
            time.sleep(interval_s)

    if not values:
        return {
            "sample_count": 0,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "values": [],
        }

    min_value = min(values)
    max_value = max(values)
    return {
        "sample_count": len(values),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min_value),
        "max": float(max_value),
        "range": float(max_value - min_value),
        "values": [float(v) for v in values],
    }


def query_named_device_attribute(target: str) -> dict[str, object]:
    cuda_query = _query_cuda_device_attribute_once(target)
    if cuda_query["value"] is not None:
        return {
            **cuda_query,
            "backend_chain": ["cuda_runtime_attribute"],
            "fallbacks_considered": ["nvidia_smi_query"],
        }

    mapped = _NVIDIA_SMI_FIELD_MAP.get(target)
    if mapped is None:
        return cuda_query

    field, scale, unit = mapped
    raw_value, details = _query_gpu_field_once(field)
    value = None if raw_value is None else float(raw_value) * scale
    if value is not None:
        return {
            "target": target,
            "source": details["source"],
            "value": value,
            "field": field,
            "unit": unit,
            "command": details["command"],
            "returncode": details["returncode"],
            "stdout_tail": details["stdout_tail"],
            "stderr_tail": details["stderr_tail"],
            "backend_chain": [str(cuda_query["source"]), str(details["source"])],
            "fallbacks_considered": ["cuda_runtime_attribute", "nvidia_smi_query"],
            "fallback_reason": cuda_query["stderr_tail"],
        }

    return {
        "target": target,
        "source": str(cuda_query["source"]),
        "value": None,
        "field": field,
        "unit": unit,
        "command": {
            "cuda_runtime": cuda_query["command"],
            "nvidia_smi": details["command"],
        },
        "returncode": details["returncode"] if int(details["returncode"]) != 0 else cuda_query["returncode"],
        "stdout_tail": details["stdout_tail"],
        "stderr_tail": f"cuda_runtime={cuda_query['stderr_tail']}; nvidia_smi={details['stderr_tail']}",
        "backend_chain": [str(cuda_query["source"]), str(details["source"])],
        "fallbacks_considered": ["cuda_runtime_attribute", "nvidia_smi_query"],
    }
