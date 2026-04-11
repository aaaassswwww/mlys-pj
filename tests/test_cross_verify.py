from __future__ import annotations

import unittest

from profiler_agent.fusion.cross_verify import Candidate, fuse_candidates


class CrossVerifyTests(unittest.TestCase):
    def test_fuse_candidates_empty(self) -> None:
        result = fuse_candidates([])
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.selected_source, "fallback_default")
        self.assertEqual(result.confidence, 0.0)

    def test_fuse_candidates_multi_source_robust_median(self) -> None:
        result = fuse_candidates(
            [
                Candidate(source="ncu", value=100.0),
                Candidate(source="microbench", value=102.0),
                Candidate(source="nvml", value=600.0),
            ]
        )
        self.assertEqual(result.method, "robust_median")
        self.assertAlmostEqual(result.value, 101.0)
        self.assertIn("ncu", result.retained_sources)
        self.assertIn("microbench", result.retained_sources)


if __name__ == "__main__":
    unittest.main()

