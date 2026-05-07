from __future__ import annotations

import json
from typing import Any


def build_lora_generation_system_prompt() -> str:
    return (
        "You are generating a single-file CUDA C++ candidate for LoRA operator optimization. "
        "Return JSON only. Do not download any external benchmark, code, or dependency. "
        "Generate source code from scratch for one self-contained file named optimized_lora.cu. "
        "The code must be a single-file CUDA C++ candidate, suitable for iterative local compilation, "
        "correctness checking, benchmarking, profiling, and later revision. "
        "Do not wrap code in markdown fences. "
        "The source code must export exactly one evaluator entrypoint with this exact ABI: "
        "extern \"C\" void launch_optimized_lora("
        "const float* W, const float* X, const float* A, const float* B, "
        "float* Y, int d, int n, cudaStream_t stream). "
        "Do not define main(). Do not add host-side testing scaffolds, example drivers, "
        "or alternate exported entrypoints. "
        "All tensors use row-major layout. Treat W as [d, d], X as [d, n], A as [d, 16], "
        "B as [d, 16], and Y as [d, n]. The required math is: "
        "temp[k, j] = sum_i B[i, k] * X[i, j], and "
        "Y[i, j] = sum_t W[i, t] * X[t, j] + sum_k A[i, k] * temp[k, j]. "
        "Assume the low rank is fixed to 16 at compile time; do not introduce a runtime rank parameter. "
        "Avoid variable-length shared-memory arrays and avoid shared-memory declarations whose size depends on runtime variables."
    )


def build_lora_generation_user_prompt(
    *,
    iteration: int,
    best_speedup: float,
    feedback: dict[str, Any] | None,
) -> str:
    payload = {
        "task": "generate_lora_candidate",
        "iteration": iteration,
        "operator": "Y = W X + A(B^T X)",
        "constraints": {
            "single_file_output": True,
            "forbid_external_downloads": True,
            "forbid_hidden_final_answer": True,
            "hidden_dim_range": [3584, 4608],
            "low_rank": 16,
            "layout": {
                "W": "[d, d] row-major",
                "X": "[d, n] row-major",
                "A": "[d, 16] row-major",
                "B": "[d, 16] row-major",
                "Y": "[d, n] row-major",
            },
            "required_math": {
                "temp[k, j]": "sum_i B[i, k] * X[i, j]",
                "Y[i, j]": "sum_t W[i, t] * X[t, j] + sum_k A[i, k] * temp[k, j]",
            },
            "required_entrypoint": {
                "name": "launch_optimized_lora",
                "language_linkage": "extern C",
                "parameters": [
                    "const float* W",
                    "const float* X",
                    "const float* A",
                    "const float* B",
                    "float* Y",
                    "int d",
                    "int n",
                    "cudaStream_t stream",
                ],
            },
            "forbid_main_function": True,
            "forbid_host_side_test_harness": True,
            "forbid_runtime_rank_parameter": True,
            "forbid_variable_length_shared_arrays": True,
        },
        "expected_json": {
            "candidate_id": "short_identifier",
            "rationale": "brief explanation of the design choice",
            "source_code": "complete optimized_lora.cu contents",
        },
        "best_speedup_so_far": best_speedup,
        "previous_feedback": feedback or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
