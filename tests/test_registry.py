from __future__ import annotations

import unittest

from profiler_agent.target_strategies.registry import StrategyRegistry


class RegistryTests(unittest.TestCase):
    def test_known_target_uses_specific_strategy(self) -> None:
        registry = StrategyRegistry()
        strategy = registry.get("actual_boost_clock_mhz")
        self.assertEqual(strategy.name, "actual_boost_clock_strategy")

    def test_unknown_target_uses_generic_strategy(self) -> None:
        registry = StrategyRegistry()
        strategy = registry.get("unknown_metric")
        self.assertEqual(strategy.name, "generic_metric")

    def test_device_attribute_target_uses_dedicated_strategy(self) -> None:
        registry = StrategyRegistry()
        strategy = registry.get("device__attribute_max_gpu_frequency_khz")
        self.assertEqual(strategy.name, "device_attribute_strategy")

    def test_new_hardware_intrinsic_targets_are_registered(self) -> None:
        registry = StrategyRegistry()
        self.assertEqual(registry.get("l1_latency_cycles").name, "l1_latency_cycles_strategy")
        self.assertEqual(registry.get("l2_latency_cycles").name, "l2_latency_cycles_strategy")
        self.assertEqual(registry.get("shared_peak_bandwidth_gbps").name, "shared_peak_bandwidth_strategy")
        self.assertEqual(registry.get("global_peak_bandwidth_gbps").name, "global_peak_bandwidth_strategy")
        self.assertEqual(registry.get("l2_cache_capacity_kb").name, "l2_cache_capacity_strategy")
        self.assertEqual(
            registry.get("shmem_bank_conflict_penalty_cycles").name,
            "shmem_bank_conflict_penalty_strategy",
        )


if __name__ == "__main__":
    unittest.main()
