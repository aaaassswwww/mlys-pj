from __future__ import annotations

import os
import time
from typing import Any


_ENABLE_KEYS = {"1", "true", "yes", "on"}
_DEFAULT_MAX_RUNTIME_SECONDS = 29 * 60
_START_TIME_ENV = "PROFILER_AGENT_BUDGET_START_EPOCH"


def _enabled() -> bool:
    raw = os.environ.get("PROFILER_AGENT_ENABLE_TIME_BUDGET", "").strip().lower()
    return raw in _ENABLE_KEYS


def _max_runtime_seconds() -> int:
    raw = os.environ.get("PROFILER_AGENT_MAX_RUNTIME_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_MAX_RUNTIME_SECONDS
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_MAX_RUNTIME_SECONDS
    return max(1, value)


def initialize_runtime_budget() -> dict[str, Any]:
    enabled = _enabled()
    if not enabled:
        return {
            "enabled": False,
            "max_runtime_seconds": None,
            "start_epoch": None,
            "elapsed_seconds": 0.0,
            "remaining_seconds": None,
            "expired": False,
        }
    start_raw = os.environ.get(_START_TIME_ENV, "").strip()
    if not start_raw:
        start_epoch = time.time()
        os.environ[_START_TIME_ENV] = f"{start_epoch:.6f}"
    else:
        try:
            start_epoch = float(start_raw)
        except ValueError:
            start_epoch = time.time()
            os.environ[_START_TIME_ENV] = f"{start_epoch:.6f}"
    return get_runtime_budget_status()


def get_runtime_budget_status() -> dict[str, Any]:
    enabled = _enabled()
    if not enabled:
        return {
            "enabled": False,
            "max_runtime_seconds": None,
            "start_epoch": None,
            "elapsed_seconds": 0.0,
            "remaining_seconds": None,
            "expired": False,
        }
    start_raw = os.environ.get(_START_TIME_ENV, "").strip()
    try:
        start_epoch = float(start_raw)
    except ValueError:
        start_epoch = time.time()
        os.environ[_START_TIME_ENV] = f"{start_epoch:.6f}"
    max_runtime_seconds = _max_runtime_seconds()
    elapsed_seconds = max(0.0, time.time() - start_epoch)
    remaining_seconds = max(0.0, float(max_runtime_seconds) - elapsed_seconds)
    return {
        "enabled": True,
        "max_runtime_seconds": max_runtime_seconds,
        "start_epoch": start_epoch,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "remaining_seconds": round(remaining_seconds, 3),
        "expired": elapsed_seconds >= float(max_runtime_seconds),
    }


def runtime_budget_expired() -> bool:
    return bool(get_runtime_budget_status().get("expired", False))


def build_timeout_metadata(*, reason: str, skipped_targets: list[str] | None = None) -> dict[str, Any]:
    status = get_runtime_budget_status()
    return {
        **status,
        "timed_out": bool(status.get("expired", False)),
        "reason": reason,
        "skipped_targets": list(skipped_targets or []),
    }
