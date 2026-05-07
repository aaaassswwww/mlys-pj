from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from profiler_agent.phase2.generator import LoraCandidateGenerator, write_candidate_snapshot
from profiler_agent.phase2.models import Phase2OptimizerState


class Phase2GeneratorTests(unittest.TestCase):
    def test_bootstrap_candidate_is_single_file_cuda_source(self) -> None:
        generator = LoraCandidateGenerator(llm_client=None)
        candidate = generator.bootstrap_candidate()
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("__global__", candidate.source_code)
        self.assertIn("launch_optimized_lora", candidate.source_code)

    def test_generate_candidate_uses_llm_payload_when_valid(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "cand one",
            "rationale": "tile on tokens first",
            "source_code": '#include <cuda_runtime.h>\n__global__ void k() {}\n',
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=1), feedback=None)
        self.assertEqual(candidate.candidate_id, "cand-one")
        self.assertEqual(candidate.source, "llm_generated")
        self.assertIn("__global__", candidate.source_code)

    def test_generate_candidate_falls_back_when_llm_invalid(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {"candidate_id": "bad", "source_code": "curl https://bad"}
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=2), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("fallback", candidate.rationale)

    def test_write_candidate_snapshot(self) -> None:
        root = Path("tests/.tmp") / f"phase2_generator_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            generator = LoraCandidateGenerator(llm_client=None)
            path = write_candidate_snapshot(root, generator.bootstrap_candidate(), filename="optimized_lora.cu")
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
