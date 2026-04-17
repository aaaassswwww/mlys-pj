from __future__ import annotations

import unittest
from unittest.mock import patch

from profiler_agent.analyzer.service import build_analysis


class AnalyzerLlmReasoningTests(unittest.TestCase):
    @patch("profiler_agent.analyzer.service.build_llm_analysis")
    def test_analysis_prefers_llm_when_available(self, mock_llm_analysis: unittest.mock.Mock) -> None:
        mock_llm_analysis.return_value = {
            "bound_type": "memory_bound",
            "confidence": 0.88,
            "bottlenecks": [
                {
                    "category": "MEMORY_PRESSURE",
                    "severity": "HIGH",
                    "reason": "LLM inferred high memory pressure.",
                    "suggestion": "Reduce memory traffic.",
                }
            ],
            "llm_reasoning_summary": "Memory throughput dominates runtime.",
        }
        out = build_analysis(
            results={"dram_latency_cycles": 500.0},
            evidence={"targets": {}, "detectors": {"finding_count": 0, "total_confidence_penalty": 0.0, "findings": []}},
        )
        self.assertEqual(out["analysis_source"], "llm")
        self.assertEqual(out["bound_type"], "memory_bound")
        self.assertIn("llm_reasoning_summary", out)


if __name__ == "__main__":
    unittest.main()

