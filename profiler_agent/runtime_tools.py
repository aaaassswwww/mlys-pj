from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass


def tail_text(text: str, n: int = 500) -> str:
    return (text or "")[-n:]


def parse_command_argv(command: str) -> list[str]:
    if not (command or "").strip():
        return []
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return shlex.split(command, posix=False)


def resolve_command_path(executable: str) -> str | None:
    return shutil.which(executable)


@dataclass(frozen=True)
class CommandProbe:
    command: list[str]
    available: bool
    resolved_path: str | None
    returncode: int
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_command(command: list[str], timeout_s: int = 10) -> CommandProbe:
    if not command:
        return CommandProbe(
            command=[],
            available=False,
            resolved_path=None,
            returncode=127,
            stdout_tail="",
            stderr_tail="required_command_not_found:<empty>",
        )
    resolved = resolve_command_path(command[0])
    if resolved is None:
        return CommandProbe(
            command=list(command),
            available=False,
            resolved_path=None,
            returncode=127,
            stdout_tail="",
            stderr_tail=f"required_command_not_found:{command[0]}",
        )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_s)
        return CommandProbe(
            command=list(command),
            available=True,
            resolved_path=resolved,
            returncode=completed.returncode,
            stdout_tail=tail_text(completed.stdout),
            stderr_tail=tail_text(completed.stderr),
        )
    except FileNotFoundError:
        return CommandProbe(
            command=list(command),
            available=False,
            resolved_path=resolved,
            returncode=127,
            stdout_tail="",
            stderr_tail=f"required_command_not_found:{command[0]}",
        )
    except subprocess.TimeoutExpired as exc:
        return CommandProbe(
            command=list(command),
            available=True,
            resolved_path=resolved,
            returncode=124,
            stdout_tail=tail_text(getattr(exc, "stdout", "") or ""),
            stderr_tail=f"command_timeout:{command[0]}",
        )


@dataclass(frozen=True)
class PythonModuleProbe:
    module_name: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_python_module(module_name: str) -> PythonModuleProbe:
    spec = importlib.util.find_spec(module_name)
    return PythonModuleProbe(
        module_name=module_name,
        available=spec is not None,
        detail="module_importable" if spec is not None else "python_module_not_found",
    )
