from __future__ import annotations

import unittest

from profiler_agent.schema.target_spec_schema import validate_target_spec


class TargetSpecSchemaTests(unittest.TestCase):
    def test_validate_target_spec_success_and_dedup(self) -> None:
        spec = validate_target_spec(
            {
                "targets": ["dram_latency_cycles", "dram_latency_cycles", "actual_boost_clock_mhz"],
                "run": "demo_binary",
            }
        )
        self.assertEqual(spec.targets, ["dram_latency_cycles", "actual_boost_clock_mhz"])
        self.assertEqual(spec.run, "demo_binary")

    def test_validate_target_spec_missing_run(self) -> None:
        with self.assertRaises(ValueError):
            validate_target_spec({"targets": ["dram_latency_cycles"]})


if __name__ == "__main__":
    unittest.main()

