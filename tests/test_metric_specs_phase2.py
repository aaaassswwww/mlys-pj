from __future__ import annotations

import unittest

from profiler_agent.schema.metric_specs import METRIC_SPECS


class MetricSpecsPhase2Tests(unittest.TestCase):
    def test_phase2_device_attribute_targets_have_metric_specs(self) -> None:
        for target in (
            "device__attribute_max_gpu_frequency_khz",
            "device__attribute_max_mem_frequency_khz",
            "device__attribute_fb_bus_width",
            "launch__sm_count",
        ):
            self.assertIn(target, METRIC_SPECS)


if __name__ == "__main__":
    unittest.main()
