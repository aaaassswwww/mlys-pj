from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_run_sh_exists(workspace: Path) -> CheckResult:
    path = workspace / "run.sh"
    if not path.exists():
        return CheckResult("run.sh_exists", False, f"missing: {path}")
    return CheckResult("run.sh_exists", True, f"found: {path}")


def _check_run_sh_contract(workspace: Path) -> CheckResult:
    path = workspace / "run.sh"
    if not path.exists():
        return CheckResult("run.sh_contract", False, "run.sh missing")
    text = _read_text(path)
    if "/target/target_spec.json" not in text:
        return CheckResult("run.sh_contract", False, "missing /target/target_spec.json usage")
    # allow either direct '/workspace/output.json' or variable-based composition.
    workspace_mentioned = "/workspace" in text
    output_json_mentioned = "output.json" in text
    if not (workspace_mentioned and output_json_mentioned):
        return CheckResult("run.sh_contract", False, "missing single output.json contract under /workspace")
    return CheckResult("run.sh_contract", True, "contains target spec path and single output path contract")


def _check_main_modes(workspace: Path) -> CheckResult:
    main_path = workspace / "profiler_agent" / "main.py"
    if not main_path.exists():
        return CheckResult("main_modes", False, f"missing: {main_path}")
    text = _read_text(main_path)
    tokens = ['choices=["single", "multi", "phase2"]', "--mode", "--spec", "--out", "--phase2-iterations"]
    missing = [token for token in tokens if token not in text]
    if missing:
        return CheckResult("main_modes", False, f"missing token(s): {missing}")
    return CheckResult("main_modes", True, "single/multi/phase2 mode entry is present")


def _check_llm_env_contract(workspace: Path) -> CheckResult:
    path = workspace / "profiler_agent" / "multi_agent" / "llm_client.py"
    if not path.exists():
        return CheckResult("llm_env_contract", False, f"missing: {path}")
    text = _read_text(path)
    required = ['os.environ.get("API_KEY"', 'os.environ.get("BASE_URL"', 'os.environ.get("BASE_MODEL"']
    missing = [token for token in required if token not in text]
    if missing:
        return CheckResult("llm_env_contract", False, f"missing env var support: {missing}")
    return CheckResult("llm_env_contract", True, "supports API_KEY / BASE_URL / BASE_MODEL")


def _check_single_output_artifacts(workspace: Path) -> CheckResult:
    matches = list(workspace.glob("output.*"))
    if len(matches) <= 1:
        return CheckResult("single_output_rule", True, f"current output.* count={len(matches)}")
    return CheckResult(
        "single_output_rule",
        False,
        f"found multiple output.* files ({len(matches)}): {[p.name for p in matches]}",
    )


def _check_run_sh_executable(workspace: Path) -> CheckResult:
    path = workspace / "run.sh"
    if not path.exists():
        return CheckResult("run.sh_executable_hint", False, "run.sh missing")
    if os.name == "nt":
        return CheckResult("run.sh_executable_hint", True, "windows host: executable bit check skipped")
    st_mode = path.stat().st_mode
    if st_mode & 0o111:
        return CheckResult("run.sh_executable_hint", True, "run.sh executable bit set")
    return CheckResult("run.sh_executable_hint", False, "run.sh is not executable (chmod +x run.sh)")


def _check_container_run(workspace: Path) -> CheckResult:
    run_sh = workspace / "run.sh"
    if not run_sh.exists():
        return CheckResult("container_run", False, "run.sh missing")
    try:
        completed = subprocess.run(
            ["bash", str(run_sh)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return CheckResult("container_run", False, f"unable to execute run.sh: {exc}")

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-500:]
        return CheckResult("container_run", False, f"run.sh failed rc={completed.returncode}, tail={tail}")

    outputs = list(workspace.glob("output.*"))
    if len(outputs) != 1:
        return CheckResult("container_run", False, f"expected exactly 1 output.*, got {len(outputs)}")
    return CheckResult("container_run", True, f"run.sh succeeded, output={outputs[0].name}")


def _run_checks(checks: list[Callable[[Path], CheckResult]], workspace: Path) -> list[CheckResult]:
    return [check(workspace) for check in checks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-submit checks for evaluator compatibility")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(".").resolve(),
        help="project root (defaults to current directory)",
    )
    parser.add_argument(
        "--run-container-check",
        action="store_true",
        help="execute run.sh (recommended only inside evaluation-like container where /workspace and /target exist)",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    checks: list[Callable[[Path], CheckResult]] = [
        _check_run_sh_exists,
        _check_run_sh_contract,
        _check_run_sh_executable,
        _check_main_modes,
        _check_llm_env_contract,
        _check_single_output_artifacts,
    ]
    if args.run_container_check:
        checks.append(_check_container_run)

    results = _run_checks(checks=checks, workspace=workspace)
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")

    failed = [item for item in results if not item.ok]
    summary = {"workspace": str(workspace), "total": len(results), "failed": len(failed)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
