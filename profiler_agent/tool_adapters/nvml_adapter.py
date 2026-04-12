from __future__ import annotations

import math
import statistics
import subprocess
import time
from typing import Optional


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
