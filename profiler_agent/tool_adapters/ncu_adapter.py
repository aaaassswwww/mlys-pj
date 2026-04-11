from __future__ import annotations

import csv
import io
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class NcuQueryResult:
    value: Optional[float]
    source: str
    returncode: int
    parse_mode: str
    stdout_tail: str
    stderr_tail: str


def _extract_last_numeric(text: str) -> Optional[float]:
    matches = _NUMERIC_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _normalize_metric_name(text: str) -> str:
    return (text or "").strip().strip('"').lower()


def _parse_ncu_csv(metric_name: str, stdout: str) -> tuple[Optional[float], str]:
    normalized_target = _normalize_metric_name(metric_name)
    reader = csv.reader(io.StringIO(stdout or ""))
    metric_values: list[float] = []

    for row in reader:
        if not row:
            continue
        normalized_row = [_normalize_metric_name(col) for col in row]
        if normalized_target in normalized_row:
            # ncu row often contains metric name and value in adjacent columns.
            for col in reversed(row):
                value = _extract_last_numeric(col)
                if value is not None:
                    metric_values.append(value)
                    break

    if metric_values:
        return metric_values[-1], "csv_metric_row"

    # fallback: line scan with explicit metric name
    for line in reversed((stdout or "").splitlines()):
        if _normalize_metric_name(metric_name) in _normalize_metric_name(line):
            value = _extract_last_numeric(line)
            if value is not None:
                return value, "line_metric_match"

    # last-chance generic numeric tail
    value = _extract_last_numeric(stdout or "")
    if value is not None:
        return value, "stdout_tail_numeric"
    return None, "parse_failed"


def query_metric_with_evidence(metric_name: str, run_cmd: str) -> NcuQueryResult:
    argv = [
        "ncu",
        "--metrics",
        metric_name,
        "--csv",
        "--target-processes",
        "all",
    ]
    argv.extend(shlex.split(run_cmd, posix=False))

    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=180)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return NcuQueryResult(
            value=None,
            source="ncu_unavailable",
            returncode=127,
            parse_mode="none",
            stdout_tail="",
            stderr_tail="ncu_not_found_or_timeout",
        )

    if completed.returncode != 0:
        return NcuQueryResult(
            value=None,
            source="ncu_failed",
            returncode=completed.returncode,
            parse_mode="none",
            stdout_tail=(completed.stdout or "")[-1000:],
            stderr_tail=(completed.stderr or "")[-1000:],
        )

    value, parse_mode = _parse_ncu_csv(metric_name=metric_name, stdout=completed.stdout or "")
    return NcuQueryResult(
        value=value,
        source="ncu_csv",
        returncode=completed.returncode,
        parse_mode=parse_mode,
        stdout_tail=(completed.stdout or "")[-1000:],
        stderr_tail=(completed.stderr or "")[-1000:],
    )


def query_metric(metric_name: str, run_cmd: str) -> Optional[float]:
    result = query_metric_with_evidence(metric_name=metric_name, run_cmd=run_cmd)
    return result.value
