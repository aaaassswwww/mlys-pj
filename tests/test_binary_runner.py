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


if __name__ == "__main__":
    unittest.main()

