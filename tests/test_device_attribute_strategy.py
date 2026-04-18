from __future__ import annotations

import unittest
from unittest.mock import patch

from profiler_agent.target_semantics import classify_target
from profiler_agent.target_strategies.base import MeasureContext
from profiler_agent.target_strategies.device_attributes import DeviceAttributeStrategy


class DeviceAttributeStrategyTests(unittest.TestCase):
    @patch("profiler_agent.target_strategies.device_attributes.query_named_device_attribute")
    def test_device_attribute_strategy_records_semantics_in_evidence(self, mock_query: unittest.mock.Mock) -> None:
        mock_query.return_value = {
            "target": "device__attribute_max_gpu_frequency_khz",
            "source": "nvidia_smi_query",
            "value": 2100000.0,
            "field": "clocks.max.sm",
            "unit": "kHz",
            "command": ["nvidia-smi"],
            "returncode": 0,
            "stdout_tail": "2100",
            "stderr_tail": "",
        }

        strategy = DeviceAttributeStrategy()
        ctx = MeasureContext(
            target="device__attribute_max_gpu_frequency_khz",
            run_cmd="",
            target_semantic=classify_target("device__attribute_max_gpu_frequency_khz"),
        )

        result = strategy.measure(ctx)
        self.assertEqual(result.value, 2100000.0)
        self.assertEqual(result.evidence["semantic"]["semantic_class"], "device_attribute")
        self.assertEqual(result.evidence["selected_source"], "device_attribute_query")
        self.assertIn("backend_chain", result.evidence["attribute_query"])


if __name__ == "__main__":
    unittest.main()
