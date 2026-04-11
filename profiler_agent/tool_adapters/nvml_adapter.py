from __future__ import annotations

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
