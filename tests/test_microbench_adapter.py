from __future__ import annotations

import unittest
from unittest.mock import patch

from profiler_agent.tool_adapters.microbench_adapter import measure_metric_with_evidence


class MicrobenchAdapterTests(unittest.TestCase):
    def test_unsupported_metric(self) -> None:
        result = measure_metric_with_evidence("unknown_metric", "dummy")
        self.assertIsNone(result.value)
        self.assertEqual(result.source, "unsupported_metric")

    @patch("profiler_agent.tool_adapters.microbench_adapter._probe_source_path")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    def test_successful_probe_parse(
        self,
        mock_run_probe: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_source_path: unittest.mock.Mock,
    ) -> None:
        mock_source_path.return_value.exists.return_value = True
        mock_compile_probe.return_value = (0, "", "")
        mock_run_probe.return_value = (
            0,
            "metric=dram_latency_cycles value=412 samples=5 median=415 best=412 std=2.5\n",
            "",
        )

        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.source, "microbench_probe")
        self.assertEqual(result.value, 412.0)
        self.assertEqual(result.parsed_from, "structured_metric_value")
        self.assertEqual(result.sample_count, 5)
        self.assertEqual(result.best_value, 412.0)
        self.assertEqual(result.median_value, 415.0)


if __name__ == "__main__":
    unittest.main()
