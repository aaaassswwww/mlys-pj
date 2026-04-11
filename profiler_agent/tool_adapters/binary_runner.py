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

