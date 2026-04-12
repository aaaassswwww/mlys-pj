from __future__ import annotations

import unittest

from profiler_agent.detectors.service import run_detectors


class DetectorTests(unittest.TestCase):
    def test_detects_source_divergence_and_resource_mask(self) -> None:
        results = {
            "max_shmem_per_block_kb": 16.0,
            "l2_cache_capacity_kb": 192.0,
        }
        evidence = {
            "targets": {
                "dram_latency_cycles": {
                    "candidates": {"ncu": 120.0, "microbench": 260.0},
                    "tools": {
                        "ncu": {"source": "ncu_csv"},
                        "microbench": {"source": "microbench_probe"},
                    },
                }
            }
        }
        report = run_detectors(results=results, evidence=evidence)
        self.assertGreaterEqual(report["finding_count"], 2)
        finding_ids = {item["id"] for item in report["findings"]}
        self.assertIn("source_divergence", finding_ids)
        self.assertIn("resource_mask_suspected", finding_ids)
        self.assertGreater(report["total_confidence_penalty"], 0.0)

    def test_detects_tool_path_blocking(self) -> None:
        results = {"dram_latency_cycles": 0.0}
        evidence = {
            "targets": {
                "t1": {
                    "tools": {
                        "ncu": {"source": "ncu_unavailable"},
                        "microbench": {"source": "compile_failed"},
                    }
                },
                "t2": {
                    "tools": {
                        "ncu": {"source": "ncu_failed"},
                        "microbench": {"source": "run_failed"},
                    }
                },
            }
        }
        report = run_detectors(results=results, evidence=evidence)
        finding_ids = {item["id"] for item in report["findings"]}
        self.assertIn("tool_path_blocking", finding_ids)


if __name__ == "__main__":
    unittest.main()

