from __future__ import annotations

import json
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
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void k() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
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

    def test_generate_candidate_accepts_code_alias_and_writes_debug_record(self) -> None:
        root = Path("tests/.tmp") / f"phase2_generator_debug_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            mock_llm = Mock()
            mock_llm.is_enabled.return_value = True
            mock_llm.complete_json.return_value = {
                "candidate_id": "cand-two",
                "explanation": "generated from alias key",
                "code": (
                    '#include <cuda_runtime.h>\n'
                    '__global__ void k() {}\n'
                    'extern "C" void launch_optimized_lora('
                    'const float* W, const float* X, const float* A, const float* B, '
                    'float* Y, int d, int n, cudaStream_t stream) {\n'
                    '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                    '}\n'
                ),
            }
            generator = LoraCandidateGenerator(llm_client=mock_llm, debug_dir=root)
            candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=2), feedback=None)
            self.assertEqual(candidate.source, "llm_generated")
            debug_path = root / "phase2_codegen_iter_02.json"
            self.assertTrue(debug_path.exists())
            record = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(record["candidate_source"], "llm_generated")
            self.assertIn("code", record["payload_keys"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_generate_candidate_rejects_mismatched_entrypoint_signature(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "bad-sig",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void k() {}\n'
                'extern "C" void launch_optimized_lora('
                'float* Y, const float* W, const float* X, const float* A, const float* B, '
                'int d, int r, int n, cudaStream_t stream) {}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=3), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("missing_or_mismatched_launch_optimized_lora_signature", candidate.rationale)

    def test_generate_candidate_rejects_main_function(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "has-main",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void k() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {}\n'
                'int main() { return 0; }\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=4), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_forbidden_main_function", candidate.rationale)


if __name__ == "__main__":
    unittest.main()
