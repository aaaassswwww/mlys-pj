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
        self.assertIn("#include <torch/extension.h>", candidate.source_code)
        self.assertIn("torch::matmul", candidate.source_code)
        self.assertIn("forward(", candidate.source_code)
        self.assertIn("PYBIND11_MODULE(TORCH_EXTENSION_NAME", candidate.source_code)

    def test_generate_candidate_uses_deterministic_reference_safe_seed_on_first_iteration(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=1), feedback=None)
        self.assertEqual(candidate.source, "deterministic_reference_safe")
        self.assertIn("#include <torch/extension.h>", candidate.source_code)
        self.assertIn("torch::matmul", candidate.source_code)
        mock_llm.complete_json.assert_not_called()

    def test_generate_candidate_switches_to_deterministic_speedup_family_after_correctness(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        state = Phase2OptimizerState(
            iteration=16,
            current_best_candidate_id="aten_inplace_addmm_bt_contiguous-v13",
            current_best_correct_candidate_id="aten_inplace_addmm_bt_contiguous-v13",
            current_best_source_code=(
                "#include <torch/extension.h>\n"
                "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) { return W; }\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
            ),
            current_best_source="deterministic_reference_safe",
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={"correctness": {"passed": True, "rel_l2_err": 0.0, "max_abs_err": 0.0}},
        )
        self.assertEqual(candidate.source, "deterministic_speedup_middle_route")
        self.assertIn("aten_", candidate.candidate_id)
        self.assertIn("torch::matmul", candidate.source_code)
        self.assertIn("PYBIND11_MODULE(TORCH_EXTENSION_NAME", candidate.source_code)
        mock_llm.complete_json.assert_not_called()

    def test_generate_candidate_focuses_aten_speedup_search_on_same_bt_shape_as_best_correct(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        state = Phase2OptimizerState(
            iteration=9,
            current_best_candidate_id="aten_out_addmm_bt_view-v05",
            current_best_correct_candidate_id="aten_out_addmm_bt_view-v05",
            current_best_source_code=(
                "#include <torch/extension.h>\n"
                "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {\n"
                "  auto temp = torch::matmul(B.transpose(0, 1), X);\n"
                "  auto wx = torch::matmul(W, X);\n"
                "  auto y = torch::addmm(wx, A, temp, 1.0, 1.0);\n"
                "  return y;\n"
                "}\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
            ),
            current_best_source="deterministic_speedup_middle_route",
            llm_revision_history=[
                {
                    "iteration": 5,
                    "candidate_id": "aten_out_addmm_bt_view-v05",
                    "generation_context": {"revision_source_preference": "current_best_correct_candidate"},
                },
            ],
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={"correctness": {"passed": True, "rel_l2_err": 0.0, "max_abs_err": 0.0}},
        )
        self.assertEqual(candidate.source, "deterministic_speedup_middle_route")
        self.assertTrue(candidate.candidate_id.startswith("aten_inplace_addmm_bt_view-v"))
        self.assertNotIn("bt_contiguous", candidate.candidate_id)
        self.assertNotIn("functional_addmm", candidate.candidate_id)
        mock_llm.complete_json.assert_not_called()

    def test_generate_candidate_locks_bt_view_speedup_search_to_inplace_only(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        state = Phase2OptimizerState(
            iteration=10,
            current_best_candidate_id="aten_inplace_addmm_bt_view-v04",
            current_best_correct_candidate_id="aten_inplace_addmm_bt_view-v04",
            current_best_source_code=(
                "#include <torch/extension.h>\n"
                "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {\n"
                "  auto temp = torch::matmul(B.transpose(0, 1), X);\n"
                "  auto out = torch::matmul(W, X);\n"
                "  out.addmm_(A, temp, 1.0, 1.0);\n"
                "  return out;\n"
                "}\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
            ),
            current_best_source="deterministic_speedup_middle_route",
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={"correctness": {"passed": True, "rel_l2_err": 0.0, "max_abs_err": 0.0}},
        )
        self.assertTrue(candidate.candidate_id.startswith("aten_inplace_addmm_bt_view-v"))
        mock_llm.complete_json.assert_not_called()

    def test_bootstrap_aten_template_avoids_extra_copy_for_inplace_path(self) -> None:
        generator = LoraCandidateGenerator(llm_client=None)
        candidate = generator.bootstrap_candidate()
        self.assertNotIn("copy_(", candidate.source_code)
        self.assertNotIn("auto Y_t = torch::empty", candidate.source_code)
        self.assertIn("return out;", candidate.source_code)

    def test_bootstrap_aten_template_avoids_unnecessary_contiguous_and_type_checks(self) -> None:
        generator = LoraCandidateGenerator(llm_client=None)
        candidate = generator.bootstrap_candidate()
        self.assertNotIn("W.contiguous()", candidate.source_code)
        self.assertNotIn("X.contiguous()", candidate.source_code)
        self.assertNotIn("A.contiguous()", candidate.source_code)
        self.assertNotIn("B.contiguous()", candidate.source_code)
        self.assertNotIn("scalar_type()", candidate.source_code)
        self.assertNotIn("is_cuda()", candidate.source_code)
        self.assertIn("torch::matmul(B.transpose(0, 1).contiguous(), X)", candidate.source_code)

    def test_generate_candidate_stops_forcing_rank16_speedup_family_after_repeated_failures(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "speedup-fallback",
            "rationale": "fallback to correct family engineering tweak",
            "source_code": (
                '#include <cuda_runtime.h>\n'
                '#include <cublas_v2.h>\n'
                '__global__ void k() {}\n'
                'extern "C" void launch_optimized_lora('
                'const float* W, const float* X, const float* A, const float* B, '
                'float* Y, int d, int n, cudaStream_t stream) {\n'
                '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                '}\n'
            ),
        }
        state = Phase2OptimizerState(
            iteration=20,
            current_best_candidate_id="all-gemm-cublas-safe-tf32-v13",
            current_best_correct_candidate_id="all-gemm-cublas-safe-tf32-v13",
            current_best_source_code=(
                "#include <cuda_runtime.h>\n"
                "#include <cublas_v2.h>\n"
                'extern "C" void launch_optimized_lora(const float* W, const float* X, const float* A, const float* B, float* Y, int d, int n, cudaStream_t stream) {}\n'
            ),
            current_best_source="deterministic_reference_safe",
            correctness_failures=[
                {"candidate_id": "cublas-rank16-update-rank16-scalar-b128-v17"},
                {"candidate_id": "cublas-rank16-update-rank16-vec4-b256-v18"},
                {"candidate_id": "cublas-rank16-update-rank16-shape-vec4-v19"},
            ],
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={"correctness": {"passed": True, "rel_l2_err": 0.0, "max_abs_err": 0.0}},
        )
        self.assertEqual(candidate.source, "llm_generated")
        self.assertEqual(candidate.candidate_id, "speedup-fallback")
        mock_llm.complete_json.assert_called_once()

    def test_generate_candidate_uses_llm_payload_when_valid_after_seed_iteration(self) -> None:
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
        candidate = generator.generate_candidate(state=Phase2OptimizerState(iteration=2), feedback=None)
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

    def test_generate_candidate_prefers_programmatic_local_enumeration_on_stable_patch_chain(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        state = Phase2OptimizerState(
            iteration=14,
            llm_revision_history=[
                {
                    "iteration": 11,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v11",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 12,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v12",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 13,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
            ],
        )
        source = (
            '#include <cuda_runtime.h>\n'
            'float tf32_round_float(float x) { return x; }\n'
            'extern "C" void launch_optimized_lora('
            'const float* W, const float* X, const float* A, const float* B, '
            'float* Y, int d, int n, cudaStream_t stream) {\n'
            '  float temp[16];\n'
            '  float acc = 0.0f;\n'
            '  temp[0] = tf32_round_float(acc);\n'
            '}\n'
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={
                "compile_ok": True,
                "correctness": {"passed": False, "rel_l2_err": 3.0e-4, "max_abs_err": 0.5},
                "previous_candidate": {
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "rationale": "latest patch",
                    "source": "llm_generated",
                    "source_code": source,
                },
            },
        )
        self.assertEqual(candidate.source, "programmatic_mutation")
        self.assertIn("programmatic_local_enumeration", candidate.rationale)
        mock_llm.complete_json.assert_not_called()

    def test_generate_candidate_falls_back_to_llm_when_programmatic_mutation_not_applicable(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "cand-fallback",
            "rationale": "llm fallback",
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
        state = Phase2OptimizerState(
            iteration=14,
            llm_revision_history=[
                {
                    "iteration": 11,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v11",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 12,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v12",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 13,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
            ],
        )
        generator = LoraCandidateGenerator(llm_client=mock_llm)
        candidate = generator.generate_candidate(
            state=state,
            feedback={
                "compile_ok": True,
                "correctness": {"passed": False, "rel_l2_err": 3.0e-4, "max_abs_err": 0.5},
                "previous_candidate": {
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "rationale": "latest patch",
                    "source": "llm_generated",
                    "source_code": (
                        '#include <cuda_runtime.h>\n'
                        'extern "C" void launch_optimized_lora('
                        'const float* W, const float* X, const float* A, const float* B, '
                        'float* Y, int d, int n, cudaStream_t stream) {\n'
                        '  (void)W; (void)X; (void)A; (void)B; (void)Y; (void)d; (void)n; (void)stream;\n'
                        '}\n'
                    ),
                },
            },
        )
        self.assertEqual(candidate.source, "llm_generated")
        mock_llm.complete_json.assert_called_once()

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

    def test_generate_candidate_allows_cublas_dependency(self) -> None:
        mock_llm = Mock()
        mock_llm.is_enabled.return_value = True
        mock_llm.complete_json.return_value = {
            "candidate_id": "ok-cublas",
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
        self.assertEqual(candidate.source, "llm_generated")

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
        self.assertIn("Do not revert to a fully plain-float32 two-kernel baseline", prompt)
        self.assertIn("Do not treat half precision as equivalent to TF32.", prompt)
        self.assertIn("Do not use unsupported pseudo-TF32 intrinsics such as __float2tf32", prompt)
        self.assertIn("Explicitly consider mixed numeric paths across stages", prompt)
        self.assertIn("Treat the temp = B^T X stage, the W*X stage, and the A*temp stage as independently tunable numeric paths", prompt)
        self.assertIn("prefer transplanting one local numeric-path idea from the reference_like_candidate into the base_candidate", prompt)
        self.assertIn("Avoid whole-program rewrites that switch the complete candidate from one family to another", prompt)

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

    def test_user_prompt_focuses_worst_hidden_dim_when_close_but_failing(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(),
            iteration=16,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 4.0e-4,
                    "max_abs_err": 0.59,
                },
                "notes": [
                    "reference_diagnosis:hidden_dim=3584:student_vs_reference_rel_l2=3.800000e-04:student_vs_naive_rel_l2=1.000000e-06:naive_vs_reference_rel_l2=3.810000e-04:student_closer_to=naive",
                    "reference_diagnosis:hidden_dim=4096:student_vs_reference_rel_l2=4.400000e-05:student_vs_naive_rel_l2=3.700000e-04:naive_vs_reference_rel_l2=3.680000e-04:student_closer_to=reference",
                    "reference_diagnosis:hidden_dim=4608:student_vs_reference_rel_l2=2.300000e-05:student_vs_naive_rel_l2=4.000000e-04:naive_vs_reference_rel_l2=4.030000e-04:student_closer_to=reference",
                ],
            },
        )
        self.assertIn('"focus_hidden_dim": 3584', prompt)
        self.assertIn('"balance_priority": "high"', prompt)
        self.assertIn("if one hidden_dim is clearly worse than the others, use it as a diagnostic hint, but do not overfit to it", prompt)
        self.assertIn("prefer balanced improvements across 3584, 4096, and 4608", prompt)
        self.assertIn("treat regressions on already-strong hidden_dims as important failures", prompt)

    def test_user_prompt_switches_to_speedup_after_correctness(self) -> None:
        state = Phase2OptimizerState(
            current_best_candidate_id="correct-best",
            current_best_correct_candidate_id="correct-best",
            current_best_source_code="correct source",
            current_best_rationale="correct and fast baseline",
            current_best_source="llm_generated",
            best_speedup=1.2,
        )
        prompt = build_lora_generation_user_prompt(
            state=state,
            iteration=17,
            best_speedup=1.2,
            feedback={
                "correctness": {
                    "passed": True,
                    "rel_l2_err": 0.0,
                    "max_abs_err": 0.0,
                }
            },
        )
        self.assertIn('"optimization_priority": "speedup_after_correctness"', prompt)
        self.assertIn('"candidate_strategy": "speedup_preserve_correctness"', prompt)
        self.assertIn('"revision_source_preference": "current_best_correct_candidate"', prompt)
        self.assertIn('"patch_discipline": "strict_speedup_lock"', prompt)
        self.assertIn('"correct_family_candidate": {', prompt)
        self.assertIn("preserve its numerical behavior and optimize speed cautiously", prompt)
        self.assertIn("Do not leave the current correct candidate family", prompt)

    def test_user_prompt_tf32_strategy_mentions_balanced_multi_dim_behavior(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(),
            iteration=18,
            best_speedup=0.0,
            feedback={
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 3.0e-4,
                    "max_abs_err": 0.55,
                },
                "notes": [
                    "reference_diagnosis:hidden_dim=3584:student_vs_reference_rel_l2=2.700000e-04:student_vs_naive_rel_l2=2.000000e-04:naive_vs_reference_rel_l2=6.600000e-05:student_closer_to=naive",
                    "reference_diagnosis:hidden_dim=4096:student_vs_reference_rel_l2=4.100000e-04:student_vs_naive_rel_l2=7.000000e-05:naive_vs_reference_rel_l2=4.700000e-04:student_closer_to=naive",
                    "reference_diagnosis:hidden_dim=4608:student_vs_reference_rel_l2=2.700000e-04:student_vs_naive_rel_l2=8.000000e-05:naive_vs_reference_rel_l2=3.700000e-04:student_closer_to=naive",
                    "torch_precision_env:torch_available=True:cuda_available=True:matmul_allow_tf32=True:cudnn_allow_tf32=True:float32_matmul_precision=high",
                ],
            },
        )
        self.assertIn('"candidate_strategy": "fit_torch_tf32_reference"', prompt)
        self.assertIn("Optimize for balanced behavior across all tested hidden dimensions", prompt)

    def test_user_prompt_prefers_structured_per_spec_feedback_and_previous_candidate(self) -> None:
        prompt = build_lora_generation_user_prompt(
            state=Phase2OptimizerState(
                current_best_candidate_id="best-overall",
                current_best_source_code="best overall source",
                current_best_rationale="best overall rationale",
                current_best_source="llm_generated",
                current_best_reference_candidate_id="best-reference",
                current_best_reference_source_code="best reference source",
                current_best_reference_rationale="best reference rationale",
                current_best_reference_source="llm_generated",
            ),
            iteration=19,
            best_speedup=0.0,
            feedback={
                "compile_ok": True,
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 3.0e-4,
                    "max_abs_err": 0.55,
                },
                "per_spec": [
                    {
                        "hidden_dim": 3584,
                        "num_tokens": 32,
                        "passed": False,
                        "rel_l2_err": 3.2e-4,
                        "max_abs_err": 0.6,
                        "reference_diagnosis": {
                            "student_vs_reference_rel_l2_err": 3.2e-4,
                            "student_vs_naive_rel_l2_err": 3.3e-4,
                            "naive_vs_reference_rel_l2_err": 7.0e-5,
                            "student_closer_to": "reference",
                        },
                    },
                    {
                        "hidden_dim": 4096,
                        "num_tokens": 32,
                        "passed": False,
                        "rel_l2_err": 2.0e-4,
                        "max_abs_err": 0.4,
                        "reference_diagnosis": {
                            "student_vs_reference_rel_l2_err": 2.0e-4,
                            "student_vs_naive_rel_l2_err": 3.8e-4,
                            "naive_vs_reference_rel_l2_err": 4.0e-4,
                            "student_closer_to": "reference",
                        },
                    },
                ],
                "notes": [
                    "torch_precision_env:torch_available=True:cuda_available=True:matmul_allow_tf32=True:cudnn_allow_tf32=True:float32_matmul_precision=high",
                ],
                "previous_candidate": {
                    "candidate_id": "prev-cand",
                    "rationale": "latest patch",
                    "source": "llm_generated",
                    "source_code": "prev source",
                },
            },
        )
        self.assertIn('"revision_source_preference": "previous_candidate_patch_first"', prompt)
        self.assertIn('"previous_candidate": {', prompt)
        self.assertIn('"candidate_id": "prev-cand"', prompt)
        self.assertIn('"per_spec_feedback": [', prompt)
        self.assertIn("treat the structured per_spec feedback as the main source of truth", prompt)
        self.assertIn("revise the immediately previous candidate in place", prompt)

    def test_user_prompt_enforces_strict_local_patch_for_stable_family_chain(self) -> None:
        state = Phase2OptimizerState(
            llm_revision_history=[
                {
                    "iteration": 11,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v11",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 12,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v12",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
                {
                    "iteration": 13,
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "generation_context": {"revision_source_preference": "previous_candidate_patch_first"},
                },
            ]
        )
        prompt = build_lora_generation_user_prompt(
            state=state,
            iteration=14,
            best_speedup=0.0,
            feedback={
                "compile_ok": True,
                "correctness": {
                    "passed": False,
                    "rel_l2_err": 3.0e-4,
                    "max_abs_err": 0.55,
                },
                "previous_candidate": {
                    "candidate_id": "simple_two_kernel_tf32_stage_selective_v13",
                    "rationale": "latest patch",
                    "source": "llm_generated",
                    "source_code": "prev source",
                },
            },
        )
        self.assertIn('"patch_discipline": "strict_local_patch"', prompt)
        self.assertIn('"mutation_execution_mode": "guided_local_enumeration"', prompt)
        self.assertIn('"selected_mutation_plan": {', prompt)
        self.assertIn('"guided_mutation_plans": [', prompt)
        self.assertIn('"family_name": "simple_two_kernel_tf32_stage_selective"', prompt)
        self.assertIn('"local_mutation_axes": [', prompt)
        self.assertIn('"axis": "wx_numeric_path"', prompt)
        self.assertIn('"max_changes_from_previous_candidate": 1', prompt)
        self.assertIn("do not introduce a new candidate family", prompt)
        self.assertIn("apply only a local numeric-path patch", prompt)
        self.assertIn("choose one or at most two axes", prompt)
        self.assertIn("execute the selected_mutation_plan faithfully", prompt)


if __name__ == "__main__":
    unittest.main()
