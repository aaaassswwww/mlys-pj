from __future__ import annotations

import os
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
        self.assertTrue(result.parsed_from.startswith("multi_run_median["))
        self.assertEqual(result.sample_count, 5)
        self.assertEqual(result.best_value, 412.0)
        self.assertEqual(result.median_value, 412.0)
        self.assertEqual(result.std_value, 0.0)
        self.assertEqual(result.run_values, [412.0] * 5)

    @patch.dict(os.environ, {"PROFILER_AGENT_PROBE_REPEAT": "3"}, clear=False)
    @patch("profiler_agent.tool_adapters.microbench_adapter._probe_source_path")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    def test_probe_repeat_env_applies(
        self,
        mock_run_probe: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_source_path: unittest.mock.Mock,
    ) -> None:
        mock_source_path.return_value.exists.return_value = True
        mock_compile_probe.return_value = (0, "", "")
        mock_run_probe.return_value = (
            0,
            "metric=dram_latency_cycles value=500\n",
            "",
        )

        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.sample_count, 3)
        self.assertEqual(mock_run_probe.call_count, 3)

    @patch.dict(
        os.environ,
        {
            "PROFILER_AGENT_PROBE_SOURCE_MODE": "llm_generated",
            "PROFILER_AGENT_DISABLE_STATIC_FALLBACK": "1",
        },
        clear=False,
    )
    @patch("profiler_agent.tool_adapters.microbench_adapter.ProbeCodeGenerator.is_enabled")
    def test_disable_static_fallback_returns_llm_generation_failed(
        self,
        mock_is_enabled: unittest.mock.Mock,
    ) -> None:
        mock_is_enabled.return_value = False
        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.source, "llm_generation_failed")
        self.assertEqual(result.generation_source, "llm_generated_only")
        self.assertIsInstance(result.generation_trace, list)
        assert result.generation_trace is not None
        self.assertTrue(any(item.get("error") == "static_fallback_disabled" for item in result.generation_trace))


if __name__ == "__main__":
    unittest.main()
