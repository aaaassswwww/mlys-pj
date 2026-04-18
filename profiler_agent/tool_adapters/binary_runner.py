from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


def run_executable(run_cmd: str, timeout_s: int = 120) -> RunResult:
    try:
        # Prefer POSIX-style splitting so quoted payloads (e.g. python -c "...") are unwrapped.
        argv = shlex.split(run_cmd, posix=True)
    except ValueError:
        # Fallback for malformed input that still may run with legacy behavior.
        argv = shlex.split(run_cmd, posix=False)
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return RunResult(
        command=run_cmd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

