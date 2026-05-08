from __future__ import annotations

import json
import re
from typing import Any

from profiler_agent.phase2.models import Phase2OptimizerState


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


def _extract_reference_diagnosis_summary(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feedback, dict):
        return None
    notes = feedback.get("notes")
    if not isinstance(notes, list):
        return None
    closer_counts = {"naive": 0, "reference": 0}
    best_note = None
    best_reference_gap = None
    pattern = re.compile(
        r"reference_diagnosis:hidden_dim=(?P<hidden_dim>\d+):"
        r"student_vs_reference_rel_l2=(?P<student_ref>[0-9.eE+-]+):"
        r"student_vs_naive_rel_l2=(?P<student_naive>[0-9.eE+-]+):"
        r"naive_vs_reference_rel_l2=(?P<naive_ref>[0-9.eE+-]+):"
        r"student_closer_to=(?P<closer_to>[a-zA-Z_]+)"
    )
    parsed_notes: list[dict[str, Any]] = []
    for note in notes:
        if not isinstance(note, str):
            continue
        match = pattern.fullmatch(note.strip())
        if not match:
            continue
        payload = {
            "hidden_dim": int(match.group("hidden_dim")),
            "student_vs_reference_rel_l2": float(match.group("student_ref")),
            "student_vs_naive_rel_l2": float(match.group("student_naive")),
            "naive_vs_reference_rel_l2": float(match.group("naive_ref")),
            "student_closer_to": match.group("closer_to"),
        }
        parsed_notes.append(payload)
        closer_to = payload["student_closer_to"]
        if closer_to in closer_counts:
            closer_counts[closer_to] += 1
        ref_gap = payload["student_vs_reference_rel_l2"]
        if best_reference_gap is None or ref_gap < best_reference_gap:
            best_reference_gap = ref_gap
            best_note = payload
    if not parsed_notes:
        return None
    dominant = "mixed"
    if closer_counts["naive"] and not closer_counts["reference"]:
        dominant = "naive"
    elif closer_counts["reference"] and not closer_counts["naive"]:
        dominant = "reference"
    return {
        "dominant_student_closer_to": dominant,
        "closer_to_counts": closer_counts,
        "best_reference_gap_case": best_note,
    }


def _extract_torch_precision_summary(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feedback, dict):
        return None
    notes = feedback.get("notes")
    if not isinstance(notes, list):
        return None
    pattern = re.compile(
        r"torch_precision_env:"
        r"torch_available=(?P<torch_available>[^:]+):"
        r"cuda_available=(?P<cuda_available>[^:]+):"
        r"matmul_allow_tf32=(?P<matmul_allow_tf32>[^:]+):"
        r"cudnn_allow_tf32=(?P<cudnn_allow_tf32>[^:]+):"
        r"float32_matmul_precision=(?P<float32_matmul_precision>.+)"
    )
    for note in notes:
        if not isinstance(note, str):
            continue
        match = pattern.fullmatch(note.strip())
        if not match:
            continue
        return {
            "torch_available": match.group("torch_available"),
            "cuda_available": match.group("cuda_available"),
            "matmul_allow_tf32": match.group("matmul_allow_tf32"),
            "cudnn_allow_tf32": match.group("cudnn_allow_tf32"),
            "float32_matmul_precision": match.group("float32_matmul_precision"),
        }
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
        "If previous correctness is close but still failing, prioritize correctness over speed and match the reference float32 behavior as closely as possible: prefer simpler kernels, deterministic writes, full initialization before any += accumulation, and avoid unnecessary changes in accumulation precision or reassociation that can drift away from float32 matmul semantics. "
        "When in doubt, prefer a simple two-kernel design: one kernel computes temp = B^T X, and one kernel computes Y = W X + A temp. "
        "Avoid shared-memory tiling, fused kernels, warp-specialized tricks, vectorized reinterpret casts, and other aggressive optimizations until correctness is already passing."
    )


def build_lora_generation_user_prompt(
    *,
    state: Phase2OptimizerState,
    iteration: int,
    best_speedup: float,
    feedback: dict[str, Any] | None,
) -> str:
    previous_rel_l2_err = _extract_previous_rel_l2_err(feedback)
    reference_diagnosis = _extract_reference_diagnosis_summary(feedback)
    torch_precision = _extract_torch_precision_summary(feedback)
    optimization_priority = "balanced"
    candidate_strategy = "normal_iteration"
    if previous_rel_l2_err is not None and previous_rel_l2_err <= 1e-3:
        optimization_priority = "correctness_first"
        candidate_strategy = "match_reference_float32_semantics"
        if isinstance(reference_diagnosis, dict) and reference_diagnosis.get("dominant_student_closer_to") == "naive":
            candidate_strategy = "fit_torch_reference_over_naive"
            if isinstance(torch_precision, dict) and torch_precision.get("matmul_allow_tf32") == "True":
                candidate_strategy = "fit_torch_tf32_reference"
    base_candidate: dict[str, Any] = {}
    if state.current_best_candidate_id and state.current_best_source_code:
        base_candidate = {
            "candidate_id": state.current_best_candidate_id,
            "source": state.current_best_source or "unknown",
            "rationale": state.current_best_rationale,
            "source_code": state.current_best_source_code,
            "selection_reason": (
                "best_correct_candidate"
                if state.current_best_correct_candidate_id == state.current_best_candidate_id
                else "best_incorrect_candidate_closest_to_passing"
            ),
            "best_rel_l2_err": state.best_rel_l2_err,
            "best_max_abs_err": state.best_max_abs_err,
        }
    reference_like_candidate: dict[str, Any] = {}
    if state.current_best_reference_candidate_id and state.current_best_reference_source_code:
        reference_like_candidate = {
            "candidate_id": state.current_best_reference_candidate_id,
            "source": state.current_best_reference_source or "unknown",
            "rationale": state.current_best_reference_rationale,
            "source_code": state.current_best_reference_source_code,
            "selection_reason": "best_candidate_measured_closer_to_torch_reference",
            "best_reference_rel_l2_err": state.best_reference_rel_l2_err,
        }
    revision_source_preference = "base_candidate"
    if candidate_strategy == "fit_torch_tf32_reference" and reference_like_candidate:
        revision_source_preference = "reference_like_candidate"
    strategy_specific_guidance: list[str] = []
    if candidate_strategy == "fit_torch_reference_over_naive":
        strategy_specific_guidance = [
            "Your previous candidates were much closer to the naive mathematical loop order than to the PyTorch reference.",
            "Do not simply regenerate the same naive two-kernel global-memory implementation with renamed variables.",
            "Deliberately try a numerically different but still valid reduction path that can move the result toward the PyTorch reference.",
            "Keep the implementation simple enough to compile and reason about, but change the effective accumulation path rather than repeating the same straightforward loop nest.",
        ]
    elif candidate_strategy == "fit_torch_tf32_reference":
        strategy_specific_guidance = [
            "The runtime environment indicates that PyTorch matmul is using a TF32-friendly path.",
            "Do not keep reproducing the same naive float32 loop order; that has already been shown to match the naive reference instead of the PyTorch reference.",
            "Do not treat half precision as equivalent to TF32.",
            "If you use half intrinsics, you must include <cuda_fp16.h>, but prefer simpler numerically altered float-based reduction structures before attempting half-based approximations.",
            "Prefer changes that alter accumulation grouping or staging in a controlled way, while keeping the ABI and required math unchanged.",
            "Avoid cuBLAS or cuBLASLt because the current build/load path is not set up for those dependencies.",
            "If a reference_like_candidate is provided, prefer revising that candidate over the naive-like base_candidate.",
        ]
    payload = {
        "task": "generate_lora_candidate",
        "iteration": iteration,
        "operator": "Y = W X + A(B^T X)",
        "optimization_priority": optimization_priority,
        "candidate_strategy": candidate_strategy,
        "revision_mode": "modify_best_candidate_instead_of_regenerating_from_scratch" if (base_candidate or reference_like_candidate) else "from_scratch",
        "revision_source_preference": revision_source_preference,
        "reference_diagnosis_summary": reference_diagnosis or {},
        "torch_precision_summary": torch_precision or {},
        "base_candidate": base_candidate,
        "reference_like_candidate": reference_like_candidate,
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
                "treat the preferred revision source candidate as the starting point and revise it instead of discarding it",
                "preserve working ABI, indexing structure, and any already-correct math unless you have a specific reason to change them",
                "make the smallest set of changes that can plausibly improve correctness toward the reference",
                "prefer a simple two-kernel design over fused or tiled kernels",
                "prefer one output element per thread with straightforward row-major indexing",
                "prefer simpler kernels over aggressive fusion",
                "match the reference float32 matmul behavior as closely as possible",
                "if diagnosis shows student_closer_to=naive, explicitly shift toward the torch reference path rather than the naive path",
                "if torch matmul TF32 is enabled in the runtime environment, prefer choices that better match the TF32-backed torch reference rather than strict naive float accumulation",
                "do not assume double accumulation is better if it changes the result away from the float32 reference",
                "fully initialize outputs before any += accumulation",
                "avoid unnecessary reassociation of reductions when trying to pass correctness",
                "avoid shared-memory tiling, vectorized casts, and warp-level tricks until correctness is already passing",
                "avoid relying on shared-memory contents across tiles unless they are explicitly reloaded each iteration",
            ],
            "strategy_specific_guidance": strategy_specific_guidance,
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
