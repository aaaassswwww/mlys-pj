from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from profiler_agent.runtime_tools import parse_command_argv, probe_command, probe_python_module


class RuntimeToolsTests(unittest.TestCase):
    def test_parse_command_argv_handles_quoted_payload(self) -> None:
        argv = parse_command_argv('python -c "print(123)"')
        self.assertEqual(argv[0], "python")
        self.assertIn("print(123)", argv[-1])

    @patch("profiler_agent.runtime_tools.resolve_command_path")
    def test_probe_command_reports_missing_binary(self, mock_resolve: unittest.mock.Mock) -> None:
        mock_resolve.return_value = None
        result = probe_command(["ncu", "--version"])
        self.assertFalse(result.available)
        self.assertEqual(result.returncode, 127)
        self.assertIn("required_command_not_found:ncu", result.stderr_tail)

    @patch("profiler_agent.runtime_tools.subprocess.run")
    @patch("profiler_agent.runtime_tools.resolve_command_path")
    def test_probe_command_runs_available_binary(
        self,
        mock_resolve: unittest.mock.Mock,
        mock_run: unittest.mock.Mock,
    ) -> None:
        mock_resolve.return_value = "/usr/bin/python3"
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python3", "--version"], returncode=0, stdout="Python 3.11.0", stderr=""
        )
        result = probe_command(["python3", "--version"])
        self.assertTrue(result.available)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Python", result.stdout_tail)

    @patch("profiler_agent.runtime_tools.importlib.util.find_spec")
    def test_probe_python_module_reports_availability(self, mock_find_spec: unittest.mock.Mock) -> None:
        mock_find_spec.return_value = object()
        result = probe_python_module("torch")
        self.assertTrue(result.available)
        self.assertEqual(result.detail, "module_importable")


if __name__ == "__main__":
    unittest.main()
