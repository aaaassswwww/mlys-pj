from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from profiler_agent.phase2.evaluator import _build_subprocess_env, _runtime_eval_worker_bootstrap
from profiler_agent.runtime_tools import resolve_command_path


def _tool_check_result(tool: str, optional: bool = False) -> dict[str, object]:
    resolved = resolve_command_path(tool)
    ok = resolved is not None or optional
    detail = resolved or f"required_command_not_found:{tool}"
    return {
        "name": tool,
        "ok": ok,
        "optional": optional,
        "detail": detail,
    }


def _collect_tool_checks(required_tools: list[str], optional_tools: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for tool in required_tools:
        results.append(_tool_check_result(tool, optional=False))
    for tool in optional_tools:
        results.append(_tool_check_result(tool, optional=True))
    return results


def _run_unittest_suite(workspace: Path) -> dict[str, object]:
    suite = [
        "tests.test_agent_state",
        "tests.test_runtime_tools",
        "tests.test_executor_phase3",
        "tests.test_device_attribute_adapter",
        "tests.test_device_attribute_strategy",
        "tests.test_metric_specs_phase2",
        "tests.test_schema",
        "tests.test_pipeline_smoke",
        "tests.test_main_modes",
        "tests.test_microbench_adapter",
        "tests.test_ncu_adapter",
        "tests.test_registry",
        "tests.test_target_semantics",
        "tests.test_golden_fixtures",
        "tests.test_multi_agent_framework",
        "tests.test_multi_agent_llm_usage",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *suite, "-v"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "suite": suite,
    }


def _phase2_runtime_worker_import_check(workspace: Path) -> dict[str, object]:
    worker_path = workspace / "profiler_agent" / "phase2" / "runtime_eval_worker.py"
    package_path = workspace / "profiler_agent" / "__init__.py"
    phase2_init_path = workspace / "profiler_agent" / "phase2" / "__init__.py"
    temp_parent = workspace / "tests" / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="phase2_worker_probe_", dir=str(temp_parent)))
    try:
        response_path = temp_root / "probe_response.json"
        command = [
            sys.executable,
            "-c",
            _runtime_eval_worker_bootstrap(),
            "--probe-import-only",
        ]
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=_build_subprocess_env(workspace),
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": command,
            "cwd": str(workspace),
            "worker_exists": worker_path.exists(),
            "package_init_exists": package_path.exists(),
            "phase2_init_exists": phase2_init_path.exists(),
            "response_path": str(response_path),
            "spec_found_in_parent_process": importlib.util.find_spec("profiler_agent.phase2.runtime_eval_worker") is not None,
            "stdout_tail": str(completed.stdout or "")[-2000:],
            "stderr_tail": str(completed.stderr or "")[-2000:],
        }
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Server-side self check for profiler agent")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--skip-unittest", action="store_true")
    parser.add_argument(
        "--required-tools",
        default="python,nvidia-smi",
        help="comma-separated required commands",
    )
    parser.add_argument(
        "--optional-tools",
        default="nvcc,ncu,nsys",
        help="comma-separated optional commands to report",
    )
    parser.add_argument("--skip-phase2-runtime-check", action="store_true")
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    required_tools = [item.strip() for item in args.required_tools.split(",") if item.strip()]
    optional_tools = [item.strip() for item in args.optional_tools.split(",") if item.strip()]

    tool_results = _collect_tool_checks(required_tools=required_tools, optional_tools=optional_tools)
    unittest_result = None if args.skip_unittest else _run_unittest_suite(workspace)
    phase2_runtime_check = None if args.skip_phase2_runtime_check else _phase2_runtime_worker_import_check(workspace)

    failed_required = [item for item in tool_results if not item["ok"] and not item["optional"]]
    summary = {
        "workspace": str(workspace),
        "tool_checks": tool_results,
        "unittest": unittest_result,
        "phase2_runtime_subprocess_check": phase2_runtime_check,
        "failed_required_tool_count": len(failed_required),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if failed_required:
        return 1
    if phase2_runtime_check is not None and not phase2_runtime_check["ok"]:
        return 1
    if unittest_result is not None and not unittest_result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
