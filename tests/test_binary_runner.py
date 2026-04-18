from __future__ import annotations

import sys
import unittest

from profiler_agent.tool_adapters.binary_runner import run_executable


class BinaryRunnerTests(unittest.TestCase):
    def test_run_executable_handles_python_c_with_multiline_payload(self) -> None:
        cmd = (
            f'"{sys.executable}" -c "s=0\n'
            "for i in range(10): s+=i*i\n"
            'print(s)"'
        )
        result = run_executable(cmd)
        self.assertEqual(result.returncode, 0)
        self.assertIn("285", result.stdout)

    def test_run_executable_reports_missing_command(self) -> None:
        result = run_executable("definitely_missing_command_12345 --version")
        self.assertEqual(result.returncode, 127)
        self.assertIn("required_command_not_found:definitely_missing_command_12345", result.stderr)


if __name__ == "__main__":
    unittest.main()
