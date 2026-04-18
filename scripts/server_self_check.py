from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    required_tools = [item.strip() for item in args.required_tools.split(",") if item.strip()]
    optional_tools = [item.strip() for item in args.optional_tools.split(",") if item.strip()]

    tool_results = _collect_tool_checks(required_tools=required_tools, optional_tools=optional_tools)
    unittest_result = None if args.skip_unittest else _run_unittest_suite(workspace)

    failed_required = [item for item in tool_results if not item["ok"] and not item["optional"]]
    summary = {
        "workspace": str(workspace),
        "tool_checks": tool_results,
        "unittest": unittest_result,
        "failed_required_tool_count": len(failed_required),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if failed_required:
        return 1
    if unittest_result is not None and not unittest_result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
