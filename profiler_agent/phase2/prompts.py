from __future__ import annotations

import json
from typing import Any


def _extract_previous_rel_l2_err(feedback: dict[str, Any] | None) -> float | None:
    if not isinstance(feedback, dict):
        return None
    correctness = feedback.get("correctness")
    if not isinstance(correctness, dict):
        return None
    value = correctness.get("rel_l2_err")
    if isinstance(value, (int, float)):
        return float(value)
    return None


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
        "Avoid variable-length shared-memory arrays and avoid shared-memory declarations whose size depends on runtime variables. "
        "If previous correctness is close but still failing, prioritize correctness over speed: prefer simpler kernels, deterministic writes, full initialization before any += accumulation, and double-precision accumulators for long reductions."
    )


def build_lora_generation_user_prompt(
    *,
    iteration: int,
    best_speedup: float,
    feedback: dict[str, Any] | None,
) -> str:
    previous_rel_l2_err = _extract_previous_rel_l2_err(feedback)
    optimization_priority = "balanced"
    candidate_strategy = "normal_iteration"
    if previous_rel_l2_err is not None and previous_rel_l2_err <= 1e-3:
        optimization_priority = "correctness_first"
        candidate_strategy = "simplify_and_stabilize_numeric_accuracy"
    payload = {
        "task": "generate_lora_candidate",
        "iteration": iteration,
        "operator": "Y = W X + A(B^T X)",
        "optimization_priority": optimization_priority,
        "candidate_strategy": candidate_strategy,
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
            "when_correctness_is_close_but_failing": [
                "prefer correctness over speed",
                "prefer simpler kernels over aggressive fusion",
                "use double accumulators for long dot products when needed",
                "fully initialize outputs before any += accumulation",
                "avoid relying on shared-memory contents across tiles unless they are explicitly reloaded each iteration",
            ],
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
