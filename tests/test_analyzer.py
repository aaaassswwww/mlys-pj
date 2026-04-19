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

    def test_detector_penalty_applies_to_adjusted_confidence(self) -> None:
        results = {
            "sm_efficiency": 60.0,
            "dram_utilization": 55.0,
        }
        evidence = {
            "targets": {},
            "detectors": {
                "finding_count": 1,
                "total_confidence_penalty": 0.2,
                "findings": [{"id": "source_divergence"}],
            },
        }
        analysis = build_analysis(results, evidence)
        self.assertIn("confidence_adjusted", analysis)
        self.assertLessEqual(analysis["confidence_adjusted"], analysis["confidence"])
        self.assertEqual(analysis["confidence_penalty"], 0.2)

    def test_analysis_marks_workload_placeholder_targets(self) -> None:
        results = {"dram__bytes_read.sum.per_second": 0.0}
        evidence = {
            "targets": {},
            "workload_placeholders": {
                "count": 1,
                "targets": ["dram__bytes_read.sum.per_second"],
                "reason": "workload_dependent_targets_without_run_use_placeholder_zero_values",
            },
        }
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["workload_placeholder_count"], 1)
        self.assertIn("dram__bytes_read.sum.per_second", analysis["workload_placeholder_targets"])

    def test_analysis_includes_intrinsic_probe_report_summary(self) -> None:
        results = {"dram_latency_cycles": 420.0}
        evidence = {
            "targets": {
                "dram_latency_cycles": {
                    "measurement_mode": "synthetic_intrinsic_probe",
                    "semantic_validity": "intrinsic_proxy",
                    "probe_iteration": {
                        "final_decision": "accept_measurement",
                        "analysis": {
                            "next_action": "accept_measurement",
                            "reason": "measurement_accepted",
                            "confidence": 0.9,
                        },
                        "state": {
                            "profile_history": [
                                {"source": "skipped_not_requested"},
                                {"source": "ncu_csv"},
                            ]
                        },
                    },
                    "probe": {
                        "profile_source": "ncu_csv",
                    },
                }
            }
        }
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["intrinsic_probe_report"]["count"], 1)
        self.assertEqual(analysis["intrinsic_probe_report"]["accepted_count"], 1)
        self.assertEqual(analysis["intrinsic_probe_report"]["ncu_profiled_count"], 1)
        self.assertEqual(
            analysis["intrinsic_probe_report"]["targets"][0]["acceptance_reason"],
            "measurement_accepted",
        )
        self.assertIn(
            "intrinsic_probe_report_summarizes_acceptance_reason_ncu_usage_and_semantic_validity_for_synthetic_probe_targets",
            analysis["analysis_notes"],
        )

    def test_analysis_includes_synthetic_counter_probe_report_summary(self) -> None:
        results = {"dram__bytes_read.sum.per_second": 981.0}
        evidence = {
            "targets": {
                "dram__bytes_read.sum.per_second": {
                    "measurement_mode": "synthetic_counter_probe",
                    "semantic_validity": "synthetic_counter_proxy",
                    "probe_iteration": {
                        "final_decision": "accept_measurement",
                        "analysis": {
                            "next_action": "accept_measurement",
                            "reason": "synthetic_counter_probe_accepted",
                            "confidence": 0.84,
                        },
                        "state": {
                            "profile_history": [
                                {"source": "ncu_csv"},
                            ]
                        },
                    },
                    "probe": {
                        "profile_source": "ncu_csv",
                    },
                }
            }
        }
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["synthetic_counter_probe_report"]["count"], 1)
        self.assertEqual(analysis["synthetic_counter_probe_report"]["accepted_count"], 1)
        self.assertEqual(analysis["synthetic_counter_probe_report"]["ncu_profiled_count"], 1)
        self.assertEqual(
            analysis["synthetic_counter_probe_report"]["targets"][0]["semantic_validity"],
            "synthetic_counter_proxy",
        )
        self.assertIn(
            "synthetic_counter_probe_report_summarizes_proxy_counter_measurements_and_marks_them_as_non_workload_observations",
            analysis["analysis_notes"],
        )

    def test_analysis_uses_accepted_synthetic_counter_probe_as_proxy_signal(self) -> None:
        results = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 76.05,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 62.91,
            "dram__bytes_read.sum.per_second": 233393513034.7,
            "dram__bytes_write.sum.per_second": 845692047377.33,
        }
        evidence = {
            "targets": {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
                    "measurement_mode": "synthetic_counter_probe",
                    "semantic_validity": "synthetic_counter_proxy",
                    "probe_iteration": {
                        "final_decision": "accept_measurement",
                        "analysis": {
                            "next_action": "accept_measurement",
                            "reason": "measurement_accepted",
                            "confidence": 0.85,
                        },
                        "state": {"profile_history": [{"source": "ncu_csv"}]},
                    },
                    "probe": {"profile_source": "ncu_csv"},
                },
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": {
                    "measurement_mode": "synthetic_counter_probe",
                    "semantic_validity": "synthetic_counter_proxy",
                    "probe_iteration": {
                        "final_decision": "accept_measurement",
                        "analysis": {
                            "next_action": "accept_measurement",
                            "reason": "measurement_accepted",
                            "confidence": 0.85,
                        },
                        "state": {"profile_history": [{"source": "ncu_csv"}]},
                    },
                    "probe": {"profile_source": "ncu_csv"},
                },
            }
        }
        analysis = build_analysis(results, evidence)
        self.assertEqual(analysis["bound_type"], "balanced_or_mixed")
        self.assertGreater(analysis["compute_score"], 0.0)
        self.assertGreater(analysis["memory_score"], 0.0)
        self.assertIn("sm__throughput.avg.pct_of_peak_sustained_elapsed", analysis["observed_metrics"])
        self.assertIn("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", analysis["observed_metrics"])
        self.assertIn("dram__bytes_read.sum.per_second", analysis["observed_metrics"])
        self.assertIn("dram__bytes_write.sum.per_second", analysis["observed_metrics"])
        self.assertEqual(analysis["proxy_signal_count"], 2)
        self.assertIn(
            "accepted_synthetic_counter_probe_targets_were_used_as_proxy_signals_for_bound_inference_not_as_direct_workload_observations",
            analysis["analysis_notes"],
        )


if __name__ == "__main__":
    unittest.main()

