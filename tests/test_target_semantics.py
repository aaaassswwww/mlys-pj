from __future__ import annotations

import unittest

from profiler_agent.target_semantics import classify_target


class TargetSemanticsTests(unittest.TestCase):
    def test_device_attribute_classification(self) -> None:
        semantic = classify_target("device__attribute_max_gpu_frequency_khz")
        self.assertEqual(semantic.semantic_class, "device_attribute")
        self.assertFalse(semantic.workload_dependent)

    def test_runtime_counter_classification(self) -> None:
        semantic = classify_target("dram__bytes_read.sum.per_second")
        self.assertEqual(semantic.semantic_class, "workload_counter")
        self.assertEqual(semantic.semantic_subclass, "runtime_throughput_counter")
        self.assertTrue(semantic.workload_dependent)

    def test_intrinsic_classification(self) -> None:
        semantic = classify_target("dram_latency_cycles")
        self.assertEqual(semantic.semantic_class, "intrinsic_probe")
        self.assertEqual(semantic.semantic_subclass, "intrinsic_microbench")
        self.assertEqual(semantic.measurement_mode_candidate, "synthetic_intrinsic_probe")
        self.assertFalse(semantic.workload_dependent)


if __name__ == "__main__":
    unittest.main()
