from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from scripts.server_self_check import (
    _collect_tool_checks,
    _phase2_runtime_worker_import_check,
    _tool_check_result,
    main,
)


class ServerSelfCheckTests(unittest.TestCase):
    @patch("scripts.server_self_check.resolve_command_path")
    def test_tool_check_result_reports_missing_tool(self, mock_resolve: unittest.mock.Mock) -> None:
        mock_resolve.return_value = None
        result = _tool_check_result("ncu")
        self.assertFalse(result["ok"])
        self.assertIn("required_command_not_found:ncu", result["detail"])

    @patch("scripts.server_self_check.resolve_command_path")
    def test_collect_tool_checks_marks_optional_tool(self, mock_resolve: unittest.mock.Mock) -> None:
        mock_resolve.side_effect = lambda tool: None if tool == "nsys" else "/usr/bin/fake"
        results = _collect_tool_checks(required_tools=["python"], optional_tools=["nsys"])
        self.assertTrue(results[0]["ok"])
        self.assertTrue(results[1]["optional"])

    @patch("scripts.server_self_check.resolve_command_path")
    @patch("scripts.server_self_check.subprocess.run")
    def test_main_succeeds_when_mocked_checks_pass(
        self,
        mock_run: unittest.mock.Mock,
        mock_resolve: unittest.mock.Mock,
    ) -> None:
        mock_resolve.return_value = "/usr/bin/fake"
        mock_run.return_value.returncode = 0
        workspace = Path("tests/.tmp") / f"server_self_check_{uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            rc = main(["--workspace", str(workspace), "--skip-unittest"])
            self.assertEqual(rc, 0)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @patch("scripts.server_self_check.subprocess.run")
    def test_phase2_runtime_worker_import_check_reports_success(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        result = _phase2_runtime_worker_import_check(Path(".").resolve())
        self.assertTrue(result["ok"])
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["worker_exists"])
        self.assertIn("--probe-import-only", result["command"])

    @patch("scripts.server_self_check.resolve_command_path")
    @patch("scripts.server_self_check.subprocess.run")
    def test_main_fails_when_phase2_runtime_check_fails(
        self,
        mock_run: unittest.mock.Mock,
        mock_resolve: unittest.mock.Mock,
    ) -> None:
        mock_resolve.return_value = "/usr/bin/fake"

        def side_effect(*args, **kwargs):
            _ = args, kwargs
            completed = unittest.mock.Mock()
            completed.returncode = 1
            completed.stdout = ""
            completed.stderr = "ModuleNotFoundError: No module named 'profiler_agent.phase2.runtime_eval_worker'"
            return completed

        mock_run.side_effect = side_effect
        rc = main(["--workspace", str(Path(".").resolve()), "--skip-unittest"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
