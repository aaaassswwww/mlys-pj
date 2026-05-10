from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient
from profiler_agent.phase2.models import GeneratedCandidate, Phase2OptimizerState
from profiler_agent.phase2.prompts import build_lora_generation_system_prompt, build_lora_generation_user_prompt

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_ENTRYPOINT_SIGNATURE_RE = re.compile(
    r'extern\s+"C"\s+void\s+launch_optimized_lora\s*\(\s*'
    r'const\s+float\s*\*\s*W\s*,\s*'
    r'const\s+float\s*\*\s*X\s*,\s*'
    r'const\s+float\s*\*\s*A\s*,\s*'
    r'const\s+float\s*\*\s*B\s*,\s*'
    r'float\s*\*\s*Y\s*,\s*'
    r'int\s+d\s*,\s*'
    r'int\s+n\s*,\s*'
    r'cudaStream_t\s+stream\s*'
    r'\)',
    flags=re.DOTALL,
)
_FORWARD_SIGNATURE_RE = re.compile(
    r"torch::Tensor\s+forward\s*\(\s*"
    r"torch::Tensor\s+W\s*,\s*"
    r"torch::Tensor\s+X\s*,\s*"
    r"torch::Tensor\s+A\s*,\s*"
    r"torch::Tensor\s+B\s*"
    r"\)",
    flags=re.DOTALL,
)
_PYBIND11_MODULE_RE = re.compile(
    r"PYBIND11_MODULE\s*\(\s*TORCH_EXTENSION_NAME\s*,\s*m\s*\)",
    flags=re.DOTALL,
)
_SHARED_RUNTIME_RANK_RE = re.compile(
    r'(?:__shared__|__attribute__\s*\(\(\s*shared\s*\)\))\s+[^;\n\[]+\[[^\]\n]*\br\b[^\]\n]*\]',
    flags=re.IGNORECASE,
)
_FORBIDDEN_HOST_TEST_RE = re.compile(
    r"\b(?:int\s+main\s*\(|run_lora_forward\s*\(|cudaMemcpy(?:Async)?\s*\([^;]*cudaMemcpyHostToDevice)",
    flags=re.IGNORECASE,
)
_HALF_INTRINSIC_RE = re.compile(r"__(?:float2half(?:_rn)?|half2float)\s*\(")
_UNSUPPORTED_TF32_INTRINSIC_RE = re.compile(
    r"__(?:float2tf32|nv_float2tf32|nv_tf32_to_float)\s*\(",
    flags=re.IGNORECASE,
)
_CANDIDATE_FAMILY_SUFFIX_RE = re.compile(r"(?:[_-]?v\d+|[_-]rev\d+|[_-]fix\d+)$", flags=re.IGNORECASE)
_FMA_SELF_ACCUM_RE = re.compile(
    r"(?P<indent>\s*)(?P<acc>[A-Za-z_]\w*)\s*=\s*fmaf\(\s*(?P<a>[^,;\n]+?)\s*,\s*(?P<b>[^,;\n]+?)\s*,\s*(?P=acc)\s*\)\s*;",
    flags=re.MULTILINE,
)
_PLAIN_SELF_ACCUM_RE = re.compile(
    r"(?P<indent>\s*)(?P<acc>[A-Za-z_]\w*)\s*\+=\s*(?P<a>[^;=\n]+?)\s*\*\s*(?P<b>[^;=\n]+?)\s*;",
    flags=re.MULTILINE,
)
_TEMP_ASSIGN_ROUNDED_RE = re.compile(
    r"(?P<lhs>\b(?:temp|T)\s*\[[^\]]+\]\s*=\s*)tf32_round_float\(\s*(?P<rhs>[A-Za-z_]\w*)\s*\)\s*;",
    flags=re.MULTILINE,
)
_TEMP_ASSIGN_PLAIN_RE = re.compile(
    r"(?P<lhs>\b(?:temp|T)\s*\[[^\]]+\]\s*=\s*)(?P<rhs>[A-Za-z_]\w*)\s*;",
    flags=re.MULTILINE,
)
_TF32_HELPER_RET_RE = re.compile(
    r"return\s+tf32_round_float\(\s*a\s*\)\s*\*\s*tf32_round_float\(\s*b\s*\)\s*;",
    flags=re.MULTILINE,
)
_PLAIN_HELPER_RET_RE = re.compile(
    r"return\s+a\s*\*\s*b\s*;",
    flags=re.MULTILINE,
)
_HALFMUL_EXPR_RE = re.compile(
    r"__half2float\(\s*__hmul\(\s*__float2half(?:_rn)?\(\s*(?P<a>[^()]+?)\s*\)\s*,\s*__float2half(?:_rn)?\(\s*(?P<b>[^()]+?)\s*\)\s*\)\s*\)",
    flags=re.MULTILINE,
)


def _strip_fenced_code(text: str) -> str:
    code = text.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", code, count=1)
        if code.endswith("```"):
            code = code[:-3]
    return code.strip()


def _sanitize_candidate_id(candidate_id: str, *, default: str) -> str:
    cleaned = _SAFE_ID_RE.sub("-", (candidate_id or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or default


def _normalize_source_code(source_code: str) -> str:
    normalized = _strip_fenced_code(source_code).replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized + "\n"


def _validate_source_code(source_code: str) -> tuple[bool, str]:
    lowered = source_code.lower()
    banned = ("wget ", "curl ", "git clone", "http://", "https://")
    for item in banned:
        if item in lowered:
            return False, f"contains_banned_pattern:{item.strip()}"
    if "#include \"" in source_code:
        return False, "contains_local_include_not_allowed_for_single_file_submission"
    if "__global__" not in source_code and "cuda" not in lowered:
        return False, "missing_cuda_shape"
    if re.search(r"\bint\s+main\s*\(", source_code):
        return False, "contains_forbidden_main_function"
    if _FORBIDDEN_HOST_TEST_RE.search(source_code):
        return False, "contains_forbidden_host_side_test_harness"
    if re.search(r"\bglobal__\b", source_code):
        return False, "contains_malformed_global_qualifier"
    has_launch = "launch_optimized_lora" in source_code
    has_direct_forward = _FORWARD_SIGNATURE_RE.search(source_code) and _PYBIND11_MODULE_RE.search(source_code)
    if has_launch:
        if not _ENTRYPOINT_SIGNATURE_RE.search(source_code):
            return False, "missing_or_mismatched_launch_optimized_lora_signature"
        if re.search(r"launch_optimized_lora\s*\([^)]*\bint\s+r\b", source_code):
            return False, "launch_optimized_lora_must_not_accept_runtime_rank_parameter"
    elif not has_direct_forward:
        return False, "missing_supported_extension_entrypoint"
    if _SHARED_RUNTIME_RANK_RE.search(source_code):
        return False, "contains_runtime_rank_sized_shared_array"
    if _UNSUPPORTED_TF32_INTRINSIC_RE.search(source_code):
        return False, "contains_unsupported_tf32_intrinsic_for_current_build"
    if _HALF_INTRINSIC_RE.search(source_code) and "#include <cuda_fp16.h>" not in source_code:
        return False, "uses_half_intrinsics_without_cuda_fp16_include"
    return True, ""


def _candidate_family(candidate_id: str) -> str:
    value = candidate_id.strip()
    if not value:
        return ""
    previous = None
    while previous != value:
        previous = value
        value = _CANDIDATE_FAMILY_SUFFIX_RE.sub("", value).rstrip("_-")
    return value


def _extract_stable_patch_family(state: Phase2OptimizerState) -> dict[str, Any] | None:
    history = [item for item in state.llm_revision_history if isinstance(item, dict) and int(item.get("iteration", 0) or 0) > 0]
    if len(history) < 3:
        return None
    recent = history[-3:]
    candidate_ids: list[str] = []
    for item in recent:
        candidate_id = item.get("candidate_id")
        generation_context = item.get("generation_context")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            return None
        if not isinstance(generation_context, dict):
            return None
        if str(generation_context.get("revision_source_preference", "")) != "previous_candidate_patch_first":
            return None
        candidate_ids.append(candidate_id)
    families = [_candidate_family(candidate_id) for candidate_id in candidate_ids]
    if not families or any(not family for family in families):
        return None
    if len(set(families)) != 1:
        return None
    return {
        "family_name": families[0],
        "recent_candidate_ids": candidate_ids,
    }


def _programmatic_mutation_plans() -> list[dict[str, str]]:
    return [
        {"plan_id": "temp_round_disable", "intent": "remove explicit tf32_round_float only at the temp stage boundary"},
        {"plan_id": "temp_round_enable", "intent": "add explicit tf32_round_float only at the temp stage boundary"},
        {"plan_id": "lowrank_fmaf_to_plain_acc", "intent": "replace only low-rank self-referential fmaf accumulators with plain multiply-add"},
        {"plan_id": "lowrank_plain_acc_to_fmaf", "intent": "replace only low-rank plain multiply-add accumulators with self-referential fmaf"},
        {"plan_id": "temp_halfmul_to_plainmul", "intent": "replace only temp-stage halfmul expressions with plain multiplication"},
        {"plan_id": "lowrank_halfmul_to_plainmul", "intent": "replace only low-rank halfmul expressions with plain multiplication"},
    ]


def _apply_programmatic_mutation(source_code: str, *, plan_id: str) -> tuple[str, bool]:
    mutated = source_code
    changed = False
    if plan_id == "temp_round_disable":
        mutated, count = _TEMP_ASSIGN_ROUNDED_RE.subn(lambda m: f"{m.group('lhs')}{m.group('rhs')};", mutated)
        changed = count > 0
    elif plan_id == "temp_round_enable":
        def enable_temp_round(match: re.Match[str]) -> str:
            lhs = match.group("lhs")
            rhs = match.group("rhs")
            return f"{lhs}tf32_round_float({rhs});"
        mutated, count = _TEMP_ASSIGN_PLAIN_RE.subn(enable_temp_round, mutated)
        changed = count > 0
    elif plan_id == "lowrank_fmaf_to_plain_acc":
        lines = mutated.splitlines()
        out_lines: list[str] = []
        in_lowrank = False
        local_changed = False
        for line in lines:
            if "for (int k = 0;" in line or "for(int k = 0;" in line:
                in_lowrank = True
            if in_lowrank:
                new_line, count = _FMA_SELF_ACCUM_RE.subn(
                    lambda m: f"{m.group('indent')}{m.group('acc')} += ({m.group('a').strip()}) * ({m.group('b').strip()});",
                    line,
                )
                if count > 0:
                    local_changed = True
                line = new_line
            out_lines.append(line)
            if in_lowrank and "}" in line:
                in_lowrank = False
        mutated = "\n".join(out_lines) + ("\n" if source_code.endswith("\n") else "")
        changed = local_changed
    elif plan_id == "lowrank_plain_acc_to_fmaf":
        lines = mutated.splitlines()
        out_lines = []
        in_lowrank = False
        local_changed = False
        for line in lines:
            if "for (int k = 0;" in line or "for(int k = 0;" in line:
                in_lowrank = True
            if in_lowrank:
                new_line, count = _PLAIN_SELF_ACCUM_RE.subn(
                    lambda m: f"{m.group('indent')}{m.group('acc')} = fmaf({m.group('a').strip()}, {m.group('b').strip()}, {m.group('acc')});",
                    line,
                )
                if count > 0:
                    local_changed = True
                line = new_line
            out_lines.append(line)
            if in_lowrank and "}" in line:
                in_lowrank = False
        mutated = "\n".join(out_lines) + ("\n" if source_code.endswith("\n") else "")
        changed = local_changed
    elif plan_id == "temp_halfmul_to_plainmul":
        lines = mutated.splitlines()
        out_lines = []
        in_temp = False
        local_changed = False
        for line in lines:
            if "compute_temp" in line and "(" in line:
                in_temp = True
            if in_temp:
                new_line, count = _HALFMUL_EXPR_RE.subn(
                    lambda m: f"(({m.group('a').strip()}) * ({m.group('b').strip()}))",
                    line,
                )
                if count > 0:
                    local_changed = True
                line = new_line
            out_lines.append(line)
            if in_temp and line.strip() == "}":
                in_temp = False
        mutated = "\n".join(out_lines) + ("\n" if source_code.endswith("\n") else "")
        changed = local_changed
    elif plan_id == "lowrank_halfmul_to_plainmul":
        lines = mutated.splitlines()
        out_lines = []
        in_lowrank = False
        local_changed = False
        for line in lines:
            if "for (int k = 0;" in line or "for(int k = 0;" in line:
                in_lowrank = True
            if in_lowrank:
                new_line, count = _HALFMUL_EXPR_RE.subn(
                    lambda m: f"(({m.group('a').strip()}) * ({m.group('b').strip()}))",
                    line,
                )
                if count > 0:
                    local_changed = True
                line = new_line
            out_lines.append(line)
            if in_lowrank and "}" in line:
                in_lowrank = False
        mutated = "\n".join(out_lines) + ("\n" if source_code.endswith("\n") else "")
        changed = local_changed
    return mutated, changed


def build_bootstrap_lora_source() -> str:
    return build_reference_safe_aten_source(bt_contiguous=True, addmm_mode="inplace")


def build_reference_safe_aten_source(
    *,
    bt_contiguous: bool,
    addmm_mode: str,
) -> str:
    bt_expr = "B.transpose(0, 1).contiguous()" if bt_contiguous else "B.transpose(0, 1)"
    if addmm_mode == "out":
        addmm_block = (
            "    auto Y_t = torch::empty({W.size(0), X.size(1)}, W.options());\n"
            "    auto wx = torch::matmul(W, X);\n"
            "    at::addmm_out(Y_t, wx, A, temp, 1.0, 1.0);\n"
            "    return Y_t;\n"
        )
    elif addmm_mode == "functional":
        addmm_block = (
            "    auto wx = torch::matmul(W, X);\n"
            "    return torch::addmm(wx, A, temp, 1.0, 1.0);\n"
        )
    else:
        addmm_block = (
            "    auto out = torch::matmul(W, X);\n"
            "    out.addmm_(A, temp, 1.0, 1.0);\n"
            "    return out;\n"
        )
    return (
        "#include <torch/extension.h>\n"
        "#include <ATen/ATen.h>\n"
        "\n"
        "namespace {\n"
        "constexpr int RANK = 16;\n"
        "\n"
        "}  // namespace\n"
        "\n"
        "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {\n"
        "    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2, \"all inputs must be rank-2\");\n"
        "    TORCH_CHECK(W.size(0) == W.size(1), \"W must be square\");\n"
        "    TORCH_CHECK(X.size(0) == W.size(1), \"X rows must match W columns\");\n"
        "    TORCH_CHECK(A.size(0) == W.size(0) && A.size(1) == RANK, \"A must be [d, 16]\");\n"
        "    TORCH_CHECK(B.size(0) == W.size(0) && B.size(1) == RANK, \"B must be [d, 16]\");\n"
        f"    auto temp = torch::matmul({bt_expr}, X);\n"
        f"{addmm_block}"
        "}\n"
        "\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
        "    m.def(\"forward\", &forward, \"LoRA forward\");\n"
        "}\n"
    )


def build_reference_safe_cublas_source(*, use_tf32_math: bool) -> str:
    tf32_math_block = (
        "#if defined(CUBLAS_TF32_TENSOR_OP_MATH)\n"
        "    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);\n"
        "#endif\n"
        if use_tf32_math
        else
        "#if defined(CUBLAS_DEFAULT_MATH)\n"
        "    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);\n"
        "#endif\n"
    )
    compute_type = "CUBLAS_COMPUTE_32F_FAST_TF32" if use_tf32_math else "CUBLAS_COMPUTE_32F"
    algo = "CUBLAS_GEMM_DEFAULT_TENSOR_OP" if use_tf32_math else "CUBLAS_GEMM_DEFAULT"
    return (
        "#include <cuda_runtime.h>\n"
        "#include <cublas_v2.h>\n"
        "\n"
        "// Correctness-safe reference family: keep the large GEMMs on cuBLAS and\n"
        "// only orchestrate the rank-16 LoRA path here. This mirrors the safer\n"
        "// structure used by projects that do not struggle with correctness.\n"
        "namespace {\n"
        "constexpr int RANK = 16;\n"
        "\n"
        "inline cublasStatus_t gemm_row_major(\n"
        "    cublasHandle_t handle,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* C) {\n"
        "    const float alpha = 1.0f;\n"
        "    const float beta = 0.0f;\n"
        "    return cublasGemmEx(\n"
        "        handle,\n"
        "        CUBLAS_OP_N,\n"
        "        CUBLAS_OP_N,\n"
        "        n,\n"
        "        m,\n"
        "        k,\n"
        "        &alpha,\n"
        "        B,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        "        A,\n"
        "        CUDA_R_32F,\n"
        "        k,\n"
        "        &beta,\n"
        "        C,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        f"        {compute_type},\n"
        f"        {algo});\n"
        "}\n"
        "\n"
        "inline cublasStatus_t gemm_row_major_a_transpose_b(\n"
        "    cublasHandle_t handle,\n"
        "    int a_rows,\n"
        "    int a_cols,\n"
        "    int n,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* C) {\n"
        "    const float alpha = 1.0f;\n"
        "    const float beta = 0.0f;\n"
        "    return cublasGemmEx(\n"
        "        handle,\n"
        "        CUBLAS_OP_N,\n"
        "        CUBLAS_OP_T,\n"
        "        n,\n"
        "        a_cols,\n"
        "        a_rows,\n"
        "        &alpha,\n"
        "        B,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        "        A,\n"
        "        CUDA_R_32F,\n"
        "        a_cols,\n"
        "        &beta,\n"
        "        C,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        f"        {compute_type},\n"
        f"        {algo});\n"
        "}\n"
        "}  // namespace\n"
        "\n"
        "extern \"C\" void launch_optimized_lora(\n"
        "    const float* W,\n"
        "    const float* X,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* Y,\n"
        "    int d,\n"
        "    int n,\n"
        "    cudaStream_t stream) {\n"
        "    if (W == nullptr || X == nullptr || A == nullptr || B == nullptr || Y == nullptr) {\n"
        "        return;\n"
        "    }\n"
        "    if (d <= 0 || n <= 0) {\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    cublasHandle_t handle = nullptr;\n"
        "    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {\n"
        "        return;\n"
        "    }\n"
        "    cublasSetStream(handle, stream);\n"
        f"{tf32_math_block}"
        "\n"
        "    float* temp = nullptr;\n"
        "    float* delta = nullptr;\n"
        "    const size_t temp_bytes = static_cast<size_t>(RANK) * static_cast<size_t>(n) * sizeof(float);\n"
        "    const size_t delta_bytes = static_cast<size_t>(d) * static_cast<size_t>(n) * sizeof(float);\n"
        "    cudaError_t alloc_temp = cudaMalloc(reinterpret_cast<void**>(&temp), temp_bytes);\n"
        "    cudaError_t alloc_delta = cudaMalloc(reinterpret_cast<void**>(&delta), delta_bytes);\n"
        "    if (alloc_temp != cudaSuccess || alloc_delta != cudaSuccess || temp == nullptr || delta == nullptr) {\n"
        "        if (temp != nullptr) cudaFree(temp);\n"
        "        if (delta != nullptr) cudaFree(delta);\n"
        "        cublasDestroy(handle);\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    cublasStatus_t st = gemm_row_major(handle, d, n, d, W, X, Y);\n"
        "    if (st == CUBLAS_STATUS_SUCCESS) {\n"
        "        st = gemm_row_major_a_transpose_b(handle, d, RANK, n, B, X, temp);\n"
        "    }\n"
        "    if (st == CUBLAS_STATUS_SUCCESS) {\n"
        "        st = gemm_row_major(handle, d, n, RANK, A, temp, delta);\n"
        "    }\n"
        "    if (st == CUBLAS_STATUS_SUCCESS) {\n"
        "        const float alpha = 1.0f;\n"
        "        st = cublasSaxpy(handle, d * n, &alpha, delta, 1, Y, 1);\n"
        "    }\n"
        "\n"
        "    cudaFree(temp);\n"
        "    cudaFree(delta);\n"
        "    cublasDestroy(handle);\n"
        "}\n"
    )


def build_cublas_rank16_update_source(
    *,
    block_size: int,
    vector_width: int,
    shape_dispatch: bool,
    use_tf32_math: bool,
) -> str:
    tf32_math_block = (
        "#if defined(CUBLAS_TF32_TENSOR_OP_MATH)\n"
        "    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);\n"
        "#endif\n"
        if use_tf32_math
        else
        "#if defined(CUBLAS_DEFAULT_MATH)\n"
        "    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);\n"
        "#endif\n"
    )
    compute_type = "CUBLAS_COMPUTE_32F_FAST_TF32" if use_tf32_math else "CUBLAS_COMPUTE_32F"
    algo = "CUBLAS_GEMM_DEFAULT_TENSOR_OP" if use_tf32_math else "CUBLAS_GEMM_DEFAULT"
    shape_dispatch_flag = "1" if shape_dispatch else "0"
    return (
        "#include <cuda_runtime.h>\n"
        "#include <cublas_v2.h>\n"
        "#include <stdint.h>\n"
        "\n"
        "namespace {\n"
        "constexpr int RANK = 16;\n"
        f"constexpr int LORA_BLOCK_SIZE = {block_size};\n"
        f"constexpr int LORA_VECTOR_WIDTH = {vector_width};\n"
        f"constexpr int LORA_SHAPE_DISPATCH = {shape_dispatch_flag};\n"
        "thread_local cublasHandle_t tls_handle = nullptr;\n"
        "thread_local cudaStream_t tls_stream = nullptr;\n"
        "thread_local float* tls_temp = nullptr;\n"
        "thread_local size_t tls_temp_capacity = 0;\n"
        "\n"
        "inline cublasStatus_t gemm_row_major(\n"
        "    cublasHandle_t handle,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* C) {\n"
        "    const float alpha = 1.0f;\n"
        "    const float beta = 0.0f;\n"
        "    return cublasGemmEx(\n"
        "        handle,\n"
        "        CUBLAS_OP_N,\n"
        "        CUBLAS_OP_N,\n"
        "        n,\n"
        "        m,\n"
        "        k,\n"
        "        &alpha,\n"
        "        B,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        "        A,\n"
        "        CUDA_R_32F,\n"
        "        k,\n"
        "        &beta,\n"
        "        C,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        f"        {compute_type},\n"
        f"        {algo});\n"
        "}\n"
        "\n"
        "inline cublasStatus_t gemm_row_major_a_transpose_b(\n"
        "    cublasHandle_t handle,\n"
        "    int a_rows,\n"
        "    int a_cols,\n"
        "    int n,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* C) {\n"
        "    const float alpha = 1.0f;\n"
        "    const float beta = 0.0f;\n"
        "    return cublasGemmEx(\n"
        "        handle,\n"
        "        CUBLAS_OP_N,\n"
        "        CUBLAS_OP_T,\n"
        "        n,\n"
        "        a_cols,\n"
        "        a_rows,\n"
        "        &alpha,\n"
        "        B,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        "        A,\n"
        "        CUDA_R_32F,\n"
        "        a_cols,\n"
        "        &beta,\n"
        "        C,\n"
        "        CUDA_R_32F,\n"
        "        n,\n"
        f"        {compute_type},\n"
        f"        {algo});\n"
        "}\n"
        "\n"
        "__global__ void rank16_add_scalar_kernel(\n"
        "    float* __restrict__ y,\n"
        "    const float* __restrict__ A,\n"
        "    const float* __restrict__ T,\n"
        "    int d,\n"
        "    int n) {\n"
        "    const int64_t total = static_cast<int64_t>(d) * static_cast<int64_t>(n);\n"
        "    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);\n"
        "    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {\n"
        "        const int row = static_cast<int>(idx / n);\n"
        "        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);\n"
        "        const float* a_row = A + static_cast<int64_t>(row) * RANK;\n"
        "        float acc = 0.0f;\n"
        "#pragma unroll\n"
        "        for (int k = 0; k < RANK; ++k) {\n"
        "            acc = fmaf(a_row[k], T[static_cast<int64_t>(k) * n + col], acc);\n"
        "        }\n"
        "        y[idx] += acc;\n"
        "    }\n"
        "}\n"
        "\n"
        "__global__ void rank16_add_vec4_kernel(\n"
        "    float* __restrict__ y,\n"
        "    const float* __restrict__ A,\n"
        "    const float* __restrict__ T,\n"
        "    int d,\n"
        "    int n) {\n"
        "    const int vec_cols = n / 4;\n"
        "    const int64_t total = static_cast<int64_t>(d) * static_cast<int64_t>(vec_cols);\n"
        "    const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);\n"
        "    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {\n"
        "        const int row = static_cast<int>(idx / vec_cols);\n"
        "        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * vec_cols) * 4;\n"
        "        const float* a_row = A + static_cast<int64_t>(row) * RANK;\n"
        "        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
        "#pragma unroll\n"
        "        for (int k = 0; k < RANK; ++k) {\n"
        "            const float aval = a_row[k];\n"
        "            const float4 tv = *reinterpret_cast<const float4*>(T + static_cast<int64_t>(k) * n + col);\n"
        "            acc.x = fmaf(aval, tv.x, acc.x);\n"
        "            acc.y = fmaf(aval, tv.y, acc.y);\n"
        "            acc.z = fmaf(aval, tv.z, acc.z);\n"
        "            acc.w = fmaf(aval, tv.w, acc.w);\n"
        "        }\n"
        "        float4 yv = *reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * n + col);\n"
        "        yv.x += acc.x;\n"
        "        yv.y += acc.y;\n"
        "        yv.z += acc.z;\n"
        "        yv.w += acc.w;\n"
        "        *reinterpret_cast<float4*>(y + static_cast<int64_t>(row) * n + col) = yv;\n"
        "    }\n"
        "}\n"
        "\n"
        "inline bool ensure_handle(cudaStream_t stream) {\n"
        "    if (tls_handle == nullptr) {\n"
        "        if (cublasCreate(&tls_handle) != CUBLAS_STATUS_SUCCESS) {\n"
        "            tls_handle = nullptr;\n"
        "            return false;\n"
        "        }\n"
        f"{tf32_math_block}"
        "    }\n"
        "    if (tls_stream != stream) {\n"
        "        if (cublasSetStream(tls_handle, stream) != CUBLAS_STATUS_SUCCESS) {\n"
        "            return false;\n"
        "        }\n"
        "        tls_stream = stream;\n"
        "    }\n"
        "    return true;\n"
        "}\n"
        "\n"
        "inline bool ensure_temp(size_t required_elems) {\n"
        "    if (required_elems <= tls_temp_capacity && tls_temp != nullptr) {\n"
        "        return true;\n"
        "    }\n"
        "    if (tls_temp != nullptr) {\n"
        "        cudaFree(tls_temp);\n"
        "        tls_temp = nullptr;\n"
        "        tls_temp_capacity = 0;\n"
        "    }\n"
        "    if (cudaMalloc(reinterpret_cast<void**>(&tls_temp), required_elems * sizeof(float)) != cudaSuccess) {\n"
        "        tls_temp = nullptr;\n"
        "        tls_temp_capacity = 0;\n"
        "        return false;\n"
        "    }\n"
        "    tls_temp_capacity = required_elems;\n"
        "    return true;\n"
        "}\n"
        "\n"
        "inline void launch_rank16_add(float* y, const float* A, const float* T, int d, int n, cudaStream_t stream) {\n"
        "    int threads = LORA_BLOCK_SIZE;\n"
        "    if (LORA_SHAPE_DISPATCH) {\n"
        "        if (d <= 3840) {\n"
        "            threads = 128;\n"
        "        } else if (d <= 4352) {\n"
        "            threads = 256;\n"
        "        } else {\n"
        "            threads = 512;\n"
        "        }\n"
        "    }\n"
        "    if (LORA_VECTOR_WIDTH == 4 && (n % 4 == 0)) {\n"
        "        const int64_t total = static_cast<int64_t>(d) * static_cast<int64_t>(n / 4);\n"
        "        const int blocks = static_cast<int>((total + threads - 1) / threads);\n"
        "        rank16_add_vec4_kernel<<<blocks, threads, 0, stream>>>(y, A, T, d, n);\n"
        "    } else {\n"
        "        const int64_t total = static_cast<int64_t>(d) * static_cast<int64_t>(n);\n"
        "        const int blocks = static_cast<int>((total + threads - 1) / threads);\n"
        "        rank16_add_scalar_kernel<<<blocks, threads, 0, stream>>>(y, A, T, d, n);\n"
        "    }\n"
        "}\n"
        "}  // namespace\n"
        "\n"
        "extern \"C\" void launch_optimized_lora(\n"
        "    const float* W,\n"
        "    const float* X,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* Y,\n"
        "    int d,\n"
        "    int n,\n"
        "    cudaStream_t stream) {\n"
        "    if (W == nullptr || X == nullptr || A == nullptr || B == nullptr || Y == nullptr) {\n"
        "        return;\n"
        "    }\n"
        "    if (d <= 0 || n <= 0) {\n"
        "        return;\n"
        "    }\n"
        "    if (!ensure_handle(stream)) {\n"
        "        return;\n"
        "    }\n"
        "    const size_t temp_elems = static_cast<size_t>(RANK) * static_cast<size_t>(n);\n"
        "    if (!ensure_temp(temp_elems)) {\n"
        "        return;\n"
        "    }\n"
        "    cublasStatus_t st = gemm_row_major(tls_handle, d, n, d, W, X, Y);\n"
        "    if (st != CUBLAS_STATUS_SUCCESS) {\n"
        "        return;\n"
        "    }\n"
        "    st = gemm_row_major_a_transpose_b(tls_handle, d, RANK, n, B, X, tls_temp);\n"
        "    if (st != CUBLAS_STATUS_SUCCESS) {\n"
        "        return;\n"
        "    }\n"
        "    launch_rank16_add(Y, A, tls_temp, d, n, stream);\n"
        "}\n"
    )


def _is_cublas_candidate_signature(*, candidate_id: str | None = None, source_code: str | None = None) -> bool:
    if isinstance(candidate_id, str) and "cublas" in candidate_id.lower():
        return True
    if isinstance(source_code, str):
        lowered = source_code.lower()
        if "#include <cublas_v2.h>" in lowered or "cublas" in lowered:
            return True
    return False


def _is_aten_middle_route_candidate_signature(*, candidate_id: str | None = None, source_code: str | None = None) -> bool:
    if isinstance(candidate_id, str):
        lowered_id = candidate_id.lower()
        if lowered_id.startswith("aten-") or "aten_" in lowered_id or "cublas-all-gemm-tf32" in lowered_id:
            return True
    if isinstance(source_code, str):
        lowered = source_code.lower()
        if "#include <torch/extension.h>" in lowered or "torch::matmul" in lowered or "addmm_" in lowered:
            return True
    return False


def _speedup_aten_search_space() -> list[dict[str, Any]]:
    return [
        {"name": "aten_inplace_addmm_bt_contiguous", "bt_contiguous": True, "addmm_mode": "inplace"},
        {"name": "aten_out_addmm_bt_contiguous", "bt_contiguous": True, "addmm_mode": "out"},
        {"name": "aten_functional_addmm_bt_contiguous", "bt_contiguous": True, "addmm_mode": "functional"},
        {"name": "aten_inplace_addmm_bt_view", "bt_contiguous": False, "addmm_mode": "inplace"},
        {"name": "aten_out_addmm_bt_view", "bt_contiguous": False, "addmm_mode": "out"},
    ]


def _infer_aten_bt_contiguous(*, candidate_id: str | None = None, source_code: str | None = None) -> bool | None:
    if isinstance(candidate_id, str):
        lowered = candidate_id.lower()
        if "bt_contiguous" in lowered:
            return True
        if "bt_view" in lowered:
            return False
    if isinstance(source_code, str):
        if "transpose(0, 1).contiguous()" in source_code:
            return True
        if "transpose(0, 1)" in source_code:
            return False
    return None


def _infer_aten_addmm_mode(*, candidate_id: str | None = None, source_code: str | None = None) -> str | None:
    if isinstance(candidate_id, str):
        lowered = candidate_id.lower()
        if "functional_addmm" in lowered:
            return "functional"
        if "out_addmm" in lowered:
            return "out"
        if "inplace_addmm" in lowered:
            return "inplace"
    if isinstance(source_code, str):
        if "torch::addmm(" in source_code:
            return "functional"
        if "addmm_out" in source_code:
            return "out"
        if "addmm_(" in source_code:
            return "inplace"
    return None


def _focused_speedup_aten_search_space(state: Phase2OptimizerState) -> list[dict[str, Any]]:
    bt_contiguous = _infer_aten_bt_contiguous(
        candidate_id=state.current_best_correct_candidate_id,
        source_code=state.current_best_source_code,
    )
    addmm_mode = _infer_aten_addmm_mode(
        candidate_id=state.current_best_correct_candidate_id,
        source_code=state.current_best_source_code,
    )
    if bt_contiguous is None or addmm_mode is None:
        return _speedup_aten_search_space()

    if not bt_contiguous:
        modes = ["inplace"] if addmm_mode != "inplace" else ["inplace"]
    else:
        if addmm_mode == "out":
            modes = ["inplace"]
        elif addmm_mode == "inplace":
            modes = ["out"]
        else:
            modes = ["out", "inplace"]

    return [
        {
            "name": f"aten_{mode}_addmm_{'bt_contiguous' if bt_contiguous else 'bt_view'}",
            "bt_contiguous": bt_contiguous,
            "addmm_mode": mode,
        }
        for mode in modes
    ]


def _build_speedup_aten_candidate(*, state: Phase2OptimizerState, iteration: int) -> GeneratedCandidate:
    prior_speedup_candidates = [
        item
        for item in state.llm_revision_history
        if isinstance(item, dict)
        and isinstance(item.get("candidate_id"), str)
        and (_is_aten_middle_route_candidate_signature(candidate_id=str(item.get("candidate_id"))))
    ]
    configs = _focused_speedup_aten_search_space(state)
    tried_names = {
        _candidate_family(str(item.get("candidate_id")))
        for item in prior_speedup_candidates
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    config = next((item for item in configs if item["name"] not in tried_names), None)
    if config is None:
        config = configs[len(prior_speedup_candidates) % len(configs)]
    candidate_id = f"{config['name']}-v{iteration:02d}"
    return GeneratedCandidate(
        candidate_id=candidate_id,
        source_code=build_reference_safe_aten_source(
            bt_contiguous=bool(config["bt_contiguous"]),
            addmm_mode=str(config["addmm_mode"]),
        ),
        rationale=(
            "deterministic middle-route speedup candidate: keep the implementation inside the "
            "ATen/PyTorch extension contract, vary B^T contiguity and addmm style, and avoid raw cuBLAS symbols"
        ),
        source="deterministic_speedup_middle_route",
    )


def _should_force_speedup_aten_family(
    *,
    state: Phase2OptimizerState,
) -> bool:
    if state.current_best_correct_candidate_id is None:
        return False
    return _is_aten_middle_route_candidate_signature(
        candidate_id=state.current_best_correct_candidate_id,
        source_code=state.current_best_source_code,
    )


def _build_reference_safe_aten_candidate(*, iteration: int) -> GeneratedCandidate:
    configs = _speedup_aten_search_space()
    config = configs[(max(0, iteration - 1)) % len(configs)]
    return GeneratedCandidate(
        candidate_id=f"{config['name']}-v{iteration:02d}",
        source_code=build_reference_safe_aten_source(
            bt_contiguous=bool(config["bt_contiguous"]),
            addmm_mode=str(config["addmm_mode"]),
        ),
        rationale=(
            "deterministic correctness-safe middle-route candidate that keeps the large matmuls in the "
            "ATen/PyTorch extension path instead of using raw cuBLAS symbols"
        ),
        source="deterministic_reference_safe",
    )


def _should_force_reference_safe_aten_family(
    *,
    state: Phase2OptimizerState,
    feedback: dict[str, Any] | None,
) -> bool:
    if state.current_best_correct_candidate_id is not None:
        return False
    if state.iteration == 1:
        return True
    if not isinstance(feedback, dict):
        return False
    previous_candidate = feedback.get("previous_candidate")
    if isinstance(previous_candidate, dict):
        if _is_aten_middle_route_candidate_signature(
            candidate_id=previous_candidate.get("candidate_id"),
            source_code=previous_candidate.get("source_code"),
        ):
            return True
    if _is_aten_middle_route_candidate_signature(
        candidate_id=state.current_best_candidate_id,
        source_code=state.current_best_source_code,
    ):
        return True
    return False


def _speedup_rank16_search_space() -> list[dict[str, Any]]:
    return [
        {"name": "rank16-scalar-b128", "block_size": 128, "vector_width": 1, "shape_dispatch": False},
        {"name": "rank16-scalar-b256", "block_size": 256, "vector_width": 1, "shape_dispatch": False},
        {"name": "rank16-scalar-b512", "block_size": 512, "vector_width": 1, "shape_dispatch": False},
        {"name": "rank16-vec4-b128", "block_size": 128, "vector_width": 4, "shape_dispatch": False},
        {"name": "rank16-vec4-b256", "block_size": 256, "vector_width": 4, "shape_dispatch": False},
        {"name": "rank16-vec4-b512", "block_size": 512, "vector_width": 4, "shape_dispatch": False},
        {"name": "rank16-shape-scalar", "block_size": 256, "vector_width": 1, "shape_dispatch": True},
        {"name": "rank16-shape-vec4", "block_size": 256, "vector_width": 4, "shape_dispatch": True},
    ]


def _build_speedup_rank16_candidate(*, state: Phase2OptimizerState, iteration: int) -> GeneratedCandidate:
    prior_speedup_candidates = [
        item
        for item in state.llm_revision_history
        if isinstance(item, dict)
        and isinstance(item.get("candidate_id"), str)
        and "cublas-rank16-update" in str(item.get("candidate_id"))
    ]
    configs = _speedup_rank16_search_space()
    config = configs[len(prior_speedup_candidates) % len(configs)]
    candidate_id = f"cublas-rank16-update-{config['name']}-v{iteration:02d}"
    return GeneratedCandidate(
        candidate_id=candidate_id,
        source_code=build_cublas_rank16_update_source(
            block_size=int(config["block_size"]),
            vector_width=int(config["vector_width"]),
            shape_dispatch=bool(config["shape_dispatch"]),
            use_tf32_math=True,
        ),
        rationale=(
            "deterministic speedup family candidate: keep W@X and B^T@X on cuBLAS TF32 GEMMs, "
            "replace A@temp plus Y+=delta with a custom rank-16 update kernel, and enumerate only launch/vectorization settings"
        ),
        source="deterministic_speedup_family",
    )


def _should_force_speedup_rank16_family(
    *,
    state: Phase2OptimizerState,
) -> bool:
    if state.current_best_correct_candidate_id is None:
        return False
    rank16_failures = 0
    for item in state.correctness_failures:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.startswith("cublas-rank16-update-"):
            rank16_failures += 1
    if rank16_failures >= 3:
        return False
    return _is_cublas_candidate_signature(
        candidate_id=state.current_best_correct_candidate_id,
        source_code=state.current_best_source_code,
    )


def _build_reference_safe_cublas_candidate(*, iteration: int) -> GeneratedCandidate:
    use_tf32_math = (iteration % 2) == 1
    mode = "tf32" if use_tf32_math else "defaultmath"
    return GeneratedCandidate(
        candidate_id=f"all-gemm-cublas-safe-{mode}-v{iteration:02d}",
        source_code=build_reference_safe_cublas_source(use_tf32_math=use_tf32_math),
        rationale=(
            "deterministic correctness-safe candidate that keeps W@X, B^T@X, and A@temp on cuBLAS-backed GEMMs "
            f"with {mode} math mode, avoiding freeform numeric-path mutations until correctness passes"
        ),
        source="deterministic_reference_safe",
    )


def _should_force_reference_safe_cublas_family(
    *,
    state: Phase2OptimizerState,
    feedback: dict[str, Any] | None,
) -> bool:
    if state.current_best_correct_candidate_id is not None:
        return False
    if state.iteration == 1:
        return True
    if not isinstance(feedback, dict):
        return False
    previous_candidate = feedback.get("previous_candidate")
    if isinstance(previous_candidate, dict):
        if _is_cublas_candidate_signature(
            candidate_id=previous_candidate.get("candidate_id"),
            source_code=previous_candidate.get("source_code"),
        ):
            return True
    if _is_cublas_candidate_signature(
        candidate_id=state.current_best_candidate_id,
        source_code=state.current_best_source_code,
    ):
        return True
    if _is_cublas_candidate_signature(
        candidate_id=state.current_best_reference_candidate_id,
        source_code=state.current_best_reference_source_code,
    ):
        return True
    return False


class LoraCandidateGenerator:
    def __init__(self, llm_client: LLMClient | None = None, *, debug_dir: Path | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
        self.debug_dir = debug_dir

    def is_enabled(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_enabled()

    def bootstrap_candidate(self) -> GeneratedCandidate:
        return GeneratedCandidate(
            candidate_id="bootstrap-aten-safe",
            source_code=build_bootstrap_lora_source(),
            rationale="correctness-safe bootstrap using the ATen/PyTorch extension path for the large matmuls",
            source="bootstrap_template",
        )

    def _write_generation_debug(
        self,
        *,
        iteration: int,
        payload: dict[str, Any] | None,
        fallback_reason: str,
        candidate: GeneratedCandidate,
    ) -> None:
        if self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "iteration": iteration,
                "fallback_reason": fallback_reason,
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "candidate_rationale": candidate.rationale,
                "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                "payload_preview": payload if isinstance(payload, dict) else None,
            }
            path = self.debug_dir / f"phase2_codegen_iter_{iteration:02d}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _extract_source_code(payload: dict[str, Any]) -> str:
        for key in ("source_code", "code", "source", "kernel_code", "cuda_code"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _extract_rationale(payload: dict[str, Any]) -> str:
        for key in ("rationale", "reasoning_summary", "explanation", "notes"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _maybe_generate_programmatic_candidate(
        self,
        *,
        state: Phase2OptimizerState,
        feedback: dict[str, Any] | None,
        default_id: str,
    ) -> tuple[GeneratedCandidate | None, str]:
        if not isinstance(feedback, dict):
            return None, ""
        correctness = feedback.get("correctness")
        if not isinstance(correctness, dict):
            return None, ""
        if bool(correctness.get("passed")):
            return None, ""
        rel_l2_err = correctness.get("rel_l2_err")
        if not isinstance(rel_l2_err, (int, float)) or float(rel_l2_err) > 1e-3:
            return None, ""
        stable_family = _extract_stable_patch_family(state)
        if not isinstance(stable_family, dict):
            return None, ""
        previous_candidate = feedback.get("previous_candidate")
        if not isinstance(previous_candidate, dict):
            return None, ""
        base_source = previous_candidate.get("source_code")
        base_id = previous_candidate.get("candidate_id")
        if not isinstance(base_source, str) or not base_source.strip():
            return None, ""
        if not isinstance(base_id, str) or not base_id.strip():
            return None, ""
        family = stable_family.get("family_name")
        if not isinstance(family, str) or not family:
            return None, ""
        if _candidate_family(base_id) != family:
            return None, ""
        plans = _programmatic_mutation_plans()
        plan = plans[(max(0, state.iteration - 1)) % len(plans)]
        mutated_source, changed = _apply_programmatic_mutation(base_source, plan_id=plan["plan_id"])
        if not changed:
            for fallback_plan in plans:
                mutated_source, changed = _apply_programmatic_mutation(base_source, plan_id=fallback_plan["plan_id"])
                if changed:
                    plan = fallback_plan
                    break
        if not changed:
            return None, ""
        normalized = _normalize_source_code(mutated_source)
        ok, error = _validate_source_code(normalized)
        if not ok:
            return None, f"programmatic_mutation_invalid:{plan['plan_id']}:{error}"
        candidate = GeneratedCandidate(
            candidate_id=_sanitize_candidate_id(f"{family}-{plan['plan_id']}-{state.iteration:02d}", default=default_id),
            source_code=normalized,
            rationale=(
                f"programmatic_local_enumeration:{plan['plan_id']} "
                f"from previous candidate {base_id} by applying a single local numeric-path mutation"
            ),
            source="programmatic_mutation",
        )
        return candidate, f"programmatic_mutation:{plan['plan_id']}"

    def generate_candidate(
        self,
        *,
        state: Phase2OptimizerState,
        feedback: dict[str, Any] | None,
    ) -> GeneratedCandidate:
        default_id = f"iter-{state.iteration:02d}"
        if not self.is_enabled():
            bootstrap = self.bootstrap_candidate()
            candidate = GeneratedCandidate(
                candidate_id=default_id,
                source_code=bootstrap.source_code,
                rationale="llm_disabled_using_bootstrap_candidate",
                source="bootstrap_template",
            )
            self._write_generation_debug(
                iteration=state.iteration,
                payload=None,
                fallback_reason="llm_disabled",
                candidate=candidate,
            )
            return candidate

        if _should_force_speedup_aten_family(state=state):
            candidate = _build_speedup_aten_candidate(state=state, iteration=state.iteration)
            self._write_generation_debug(
                iteration=state.iteration,
                payload={"mode": "deterministic_speedup_middle_route_family"},
                fallback_reason="deterministic_speedup_middle_route_family_locked",
                candidate=candidate,
            )
            return candidate

        if _should_force_reference_safe_aten_family(state=state, feedback=feedback):
            candidate = _build_reference_safe_aten_candidate(iteration=state.iteration)
            self._write_generation_debug(
                iteration=state.iteration,
                payload={"mode": "deterministic_reference_safe_middle_route_family_locked"},
                fallback_reason="deterministic_reference_safe_middle_route_family_locked",
                candidate=candidate,
            )
            return candidate

        programmatic_candidate, programmatic_reason = self._maybe_generate_programmatic_candidate(
            state=state,
            feedback=feedback,
            default_id=default_id,
        )
        if programmatic_candidate is not None:
            self._write_generation_debug(
                iteration=state.iteration,
                payload={
                    "mode": "programmatic_local_enumeration",
                    "feedback_preview": {
                        "previous_candidate_id": ((feedback.get("previous_candidate") or {}).get("candidate_id") if isinstance(feedback, dict) else None),
                        "rel_l2_err": ((feedback.get("correctness") or {}).get("rel_l2_err") if isinstance(feedback, dict) else None),
                    },
                },
                fallback_reason=programmatic_reason,
                candidate=programmatic_candidate,
            )
            return programmatic_candidate

        payload = self.llm_client.complete_json(
            system_prompt=build_lora_generation_system_prompt(),
            user_prompt=build_lora_generation_user_prompt(
                state=state,
                iteration=state.iteration,
                best_speedup=state.best_speedup,
                feedback=feedback,
            ),
        )
        if not isinstance(payload, dict):
            fallback = self.bootstrap_candidate()
            candidate = GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale="llm_returned_no_json_fallback_to_bootstrap",
                source="bootstrap_template",
            )
            self._write_generation_debug(
                iteration=state.iteration,
                payload=None,
                fallback_reason="llm_returned_no_json",
                candidate=candidate,
            )
            return candidate

        source_code = self._extract_source_code(payload)
        rationale = self._extract_rationale(payload)
        candidate_id = payload.get("candidate_id", default_id)
        if not isinstance(source_code, str) or not source_code.strip():
            fallback = self.bootstrap_candidate()
            candidate = GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale="llm_missing_source_code_fallback_to_bootstrap",
                source="bootstrap_template",
            )
            self._write_generation_debug(
                iteration=state.iteration,
                payload=payload,
                fallback_reason="llm_missing_source_code",
                candidate=candidate,
            )
            return candidate

        normalized = _normalize_source_code(source_code)
        ok, error = _validate_source_code(normalized)
        if not ok:
            fallback = self.bootstrap_candidate()
            candidate = GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale=f"llm_generated_invalid_source_fallback:{error}",
                source="bootstrap_template",
            )
            self._write_generation_debug(
                iteration=state.iteration,
                payload=payload,
                fallback_reason=f"llm_generated_invalid_source:{error}",
                candidate=candidate,
            )
            return candidate

        candidate = GeneratedCandidate(
            candidate_id=_sanitize_candidate_id(str(candidate_id), default=default_id),
            source_code=normalized,
            rationale=str(rationale) if isinstance(rationale, str) else "",
            source="llm_generated",
        )
        self._write_generation_debug(
            iteration=state.iteration,
            payload=payload,
            fallback_reason="",
            candidate=candidate,
        )
        return candidate


def write_candidate_snapshot(root_dir: Path, candidate: GeneratedCandidate, *, filename: str) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / filename
    path.write_text(candidate.source_code, encoding="utf-8")
    return path
