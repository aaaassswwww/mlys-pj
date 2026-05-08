from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from profiler_agent.phase2.generator import LoraCandidateGenerator, write_candidate_snapshot
from profiler_agent.phase2.models import Phase2OptimizerState
from profiler_agent.phase2.prompts import build_lora_generation_user_prompt


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

    def test_generate_candidate_rejects_runtime_rank_sized_shared_arrays(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "shared-rank",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void k(int r) {\n'
                '  __attribute__((shared)) float a_tile[64 * r];\n'
                '}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=5), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_runtime_rank_sized_shared_array", candidate.rationale)

    def test_generate_candidate_rejects_host_side_test_harness(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "host-harness",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void k() {}\n'
                'void run_lora_forward() { cudaMemcpy(0, 0, 0, cudaMemcpyHostToDevice); }\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=6), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_forbidden_host_side_test_harness", candidate.rationale)

    def test_generate_candidate_rejects_malformed_global_qualifier(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "bad-global",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                'global__ void temp_kernel() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=7), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_malformed_global_qualifier", candidate.rationale)

    def test_generate_candidate_rejects_half_intrinsics_without_cuda_fp16_include(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "bad-half",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void temp_kernel(float a, float b, float* out) {\n'
                '  out[0] = __half2float(__float2half_rn(a * b));\n'
                '}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=8), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("uses_half_intrinsics_without_cuda_fp16_include", candidate.rationale)

    def test_generate_candidate_rejects_cublas_dependency(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "bad-cublas",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '#include <cublas_v2.h>\n'
                '__global__ void k() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  cublasHandle_t handle;\n'
                '  cublasCreate(&handle);\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=9), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_cublas_dependency_not_supported_by_current_build", candidate.rationale)

    def test_generate_candidate_rejects_unsupported_tf32_intrinsics(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "bad-tf32",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '__global__ void temp_kernel(float a, float* out) {\n'
                '  out[0] = __float2tf32(a);\n'
                '}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=10), feedback=None)
        self.assertEqual(candidate.source, "bootstrap_template")
        self.assertIn("contains_unsupported_tf32_intrinsic_for_current_build", candidate.rationale)

    def test_generate_candidate_allows_half_intrinsics_with_cuda_fp16_include(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "half-allowed",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '#include <cuda_fp16.h>\n'
                '__global__ void temp_kernel(float a, float b, float* out) {\n'
                '  out[0] = __half2float(__float2half_rn(a * b));\n'
                '}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=10), feedback=None)
        self.assertEqual(candidate.source, "llm_generated")

    def test_user_prompt_switches_to_correctness_first_when_error_is_small(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(),
            iteration=11,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 4.2e-4,
                    "max_abs_err": 0.65,
                }
            },
        )
        self.assertIn('"optimization_priority": "correctness_first"', prompt)
        self.assertIn('"candidate_strategy": "match_reference_float32_semantics"', prompt)
        self.assertIn("match the reference float32 matmul behavior as closely as possible", prompt)
        self.assertIn("prefer a simple two-kernel design over fused or tiled kernels", prompt)
        self.assertIn("avoid shared-memory tiling, vectorized casts, and warp-level tricks", prompt)

    def test_user_prompt_switches_to_fit_torch_reference_when_diagnosis_says_naive(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(),
            iteration=12,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 4.0e-4,
                    "max_abs_err": 0.59,
                },
                "notes": [
                    "reference_diagnosis:hidden_dim=4096:student_vs_reference_rel_l2=4.000000e-04:student_vs_naive_rel_l2=1.000000e-06:naive_vs_reference_rel_l2=4.010000e-04:student_closer_to=naive"
                ],
            },
        )
        self.assertIn('"candidate_strategy": "fit_torch_reference_over_naive"', prompt)
        self.assertIn('"dominant_student_closer_to": "naive"', prompt)
        self.assertIn("if diagnosis shows student_closer_to=naive", prompt)
        self.assertIn("Do not simply regenerate the same naive two-kernel global-memory implementation", prompt)

    def test_user_prompt_switches_to_fit_torch_tf32_reference_when_tf32_enabled(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(),
            iteration=13,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 4.0e-4,
                    "max_abs_err": 0.59,
                },
                "notes": [
                    "reference_diagnosis:hidden_dim=4096:student_vs_reference_rel_l2=4.000000e-04:student_vs_naive_rel_l2=1.000000e-06:naive_vs_reference_rel_l2=4.010000e-04:student_closer_to=naive",
                    "torch_precision_env:torch_available=True:cuda_available=True:matmul_allow_tf32=True:cudnn_allow_tf32=True:float32_matmul_precision=high",
                ],
            },
        )
        self.assertIn('"candidate_strategy": "fit_torch_tf32_reference"', prompt)
        self.assertIn('"matmul_allow_tf32": "True"', prompt)
        self.assertIn("better match the TF32-backed torch reference", prompt)
        self.assertIn("Do not treat half precision as equivalent to TF32.", prompt)
        self.assertIn("Do not use unsupported pseudo-TF32 intrinsics such as __float2tf32", prompt)

    def test_user_prompt_includes_best_candidate_as_revision_base(self) -> None:
        state = Phase2OptimizerState(
            current_best_candidate_id="best-so-far",
            current_best_source_code=(
                '#include <cuda_runtime.h>\n'
                '__global__ void temp_kernel() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {}\n'
            ),
            current_best_rationale="closest candidate so far",
            current_best_source="llm_generated",
            best_rel_l2_err=4.0e-4,
            best_max_abs_err=0.6,
        )
        prompt = build_lora_generation_user_prompt(
            state=state,
            iteration=14,
            best_speedup=0.0,
            feedback=None,
        )
        self.assertIn('"revision_mode": "modify_best_candidate_instead_of_regenerating_from_scratch"', prompt)
        self.assertIn('"candidate_id": "best-so-far"', prompt)
        self.assertIn('"selection_reason": "best_incorrect_candidate_closest_to_passing"', prompt)
        self.assertIn("treat the preferred revision source candidate as the starting point and revise it instead of discarding it", prompt)

    def test_user_prompt_prefers_reference_like_candidate_for_tf32_strategy(self) -> None:
        state = Phase2OptimizerState(
            current_best_candidate_id="best-naive-like",
            current_best_source_code="naive source",
            current_best_rationale="closest overall",
            current_best_source="llm_generated",
            best_rel_l2_err=4.0e-4,
            best_max_abs_err=0.6,
            current_best_reference_candidate_id="best-reference-like",
            current_best_reference_source_code="reference-like source",
            current_best_reference_rationale="closest to torch reference",
            current_best_reference_source="llm_generated",
            best_reference_rel_l2_err=3.2e-4,
        )
        prompt = build_lora_generation_user_prompt(
            state=state,
            iteration=15,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 4.0e-4,
                    "max_abs_err": 0.59,
                },
                "notes": [
                    "reference_diagnosis:hidden_dim=4096:student_vs_reference_rel_l2=4.000000e-04:student_vs_naive_rel_l2=1.000000e-06:naive_vs_reference_rel_l2=4.010000e-04:student_closer_to=naive",
                    "torch_precision_env:torch_available=True:cuda_available=True:matmul_allow_tf32=True:cudnn_allow_tf32=True:float32_matmul_precision=high",
                ],
            },
        )
        self.assertIn('"candidate_strategy": "fit_torch_tf32_reference"', prompt)
        self.assertIn('"revision_source_preference": "reference_like_candidate"', prompt)
        self.assertIn('"candidate_id": "best-reference-like"', prompt)
        self.assertIn("prefer revising that candidate over the naive-like base_candidate", prompt)


if __name__ == "__main__":
    unittest.main()
