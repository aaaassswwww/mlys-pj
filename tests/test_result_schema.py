from __future__ import annotations

import unittest

from profiler_agent.schema.result_schema import normalize_results_with_specs


class ResultSchemaTests(unittest.TestCase):
    def test_normalize_results_clamps_and_rounds(self) -> None:
        normalized, quality = normalize_results_with_specs(
            {
                "actual_boost_clock_mhz": 22000.12345,
                "l2_cache_capacity_kb": 6144.7,
            },
            expected_targets=["actual_boost_clock_mhz", "l2_cache_capacity_kb"],
        )
        self.assertEqual(normalized["actual_boost_clock_mhz"], 10000.0)
        self.assertEqual(normalized["l2_cache_capacity_kb"], 6145)
        self.assertGreaterEqual(quality["issue_count"], 1)

    def test_unknown_target_keeps_numeric(self) -> None:
        normalized, quality = normalize_results_with_specs(
            {"unknown_metric": 12.3456},
            expected_targets=["unknown_metric"],
        )
        self.assertEqual(normalized["unknown_metric"], 12.3456)
        self.assertEqual(quality["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()

