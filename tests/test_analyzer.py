from __future__ import annotations

import unittest

from profiler_agent.analyzer.service import build_analysis


class AnalyzerTests(unittest.TestCase):
    def test_memory_bound_classification(self) -> None:
        results = {
            "dram_utilization": 88.0,
            "sm_efficiency": 41.0,
            "dram_latency_cycles": 430.0,
        }
        evidence = {"targets": {}}
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["bound_type"], "memory_bound")
        self.assertGreaterEqual(analysis["memory_score"], analysis["compute_score"])

    def test_compute_bound_classification(self) -> None:
        results = {
            "sm_efficiency": 91.0,
            "achieved_occupancy": 0.82,
            "dram_utilization": 30.0,
        }
        evidence = {"targets": {}}
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["bound_type"], "compute_bound")


if __name__ == "__main__":
    unittest.main()

