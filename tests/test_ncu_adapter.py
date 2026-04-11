from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from profiler_agent.tool_adapters.ncu_adapter import query_metric_with_evidence


class NcuAdapterTests(unittest.TestCase):
    @patch("profiler_agent.tool_adapters.ncu_adapter.subprocess.run")
    def test_parse_metric_from_csv_row(self, mock_run: unittest.mock.Mock) -> None:
        stdout = (
            '"ID","Kernel Name","Metric Name","Metric Value"\n'
            '"1","k","dram_latency_cycles","441.5"\n'
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ncu"], returncode=0, stdout=stdout, stderr=""
        )
        result = query_metric_with_evidence("dram_latency_cycles", "cmd /c echo x")
        self.assertEqual(result.parse_mode, "csv_metric_row")
        self.assertEqual(result.value, 441.5)

    @patch("profiler_agent.tool_adapters.ncu_adapter.subprocess.run")
    def test_failed_ncu_returns_none_value(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ncu"], returncode=2, stdout="", stderr="error"
        )
        result = query_metric_with_evidence("dram_latency_cycles", "cmd /c echo x")
        self.assertIsNone(result.value)
        self.assertEqual(result.source, "ncu_failed")


if __name__ == "__main__":
    unittest.main()

