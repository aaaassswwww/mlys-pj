from __future__ import annotations

import json
import re
from typing import Any

from profiler_agent.phase2.models import Phase2OptimizerState

_CANDIDATE_FAMILY_SUFFIX_RE = re.compile(r"(?:[_-]?v\d+|[_-]rev\d+|[_-]fix\d+)$", flags=re.IGNORECASE)


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


def _extract_previous_correctness_passed(feedback: dict[str, Any] | None) -> bool:
    if not isinstance(feedback, dict):
        return False
    correctness = feedback.get("correctness")
    if not isinstance(correctness, dict):
        return False
    return bool(correctness.get("passed"))


def _extract_reference_diagnosis_summary(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feedback, dict):
        return None
    per_spec = feedback.get("per_spec")
    if isinstance(per_spec, list):
        parsed_notes: list[dict[str, Any]] = []
        closer_counts = {"naive": 0, "reference": 0}
        best_note = None
        best_reference_gap = None
        for item in per_spec:
            if not isinstance(item, dict):
                continue
            diagnosis = item.get("reference_diagnosis")
            if not isinstance(diagnosis, dict):
                continue
            hidden_dim = item.get("hidden_dim")
            try:
                payload = {
                    "hidden_dim": int(hidden_dim),
                    "student_vs_reference_rel_l2": float(diagnosis["student_vs_reference_rel_l2_err"]),
                    "student_vs_naive_rel_l2": float(diagnosis["student_vs_naive_rel_l2_err"]),
                    "naive_vs_reference_rel_l2": float(diagnosis["naive_vs_reference_rel_l2_err"]),
                    "student_closer_to": str(diagnosis["student_closer_to"]),
                    "full_rel_l2_err": float(item.get("rel_l2_err", float("inf"))),
                    "full_max_abs_err": float(item.get("max_abs_err", float("inf"))),
                }
            except (KeyError, TypeError, ValueError):
                continue
            parsed_notes.append(payload)
            closer_to = payload["student_closer_to"]
            if closer_to in closer_counts:
                closer_counts[closer_to] += 1
            ref_gap = payload["student_vs_reference_rel_l2"]
            if best_reference_gap is None or ref_gap < best_reference_gap:
                best_reference_gap = ref_gap
                best_note = payload
        if parsed_notes:
            dominant = "mixed"
            if closer_counts["naive"] and not closer_counts["reference"]:
                dominant = "naive"
            elif closer_counts["reference"] and not closer_counts["naive"]:
                dominant = "reference"
            worst_note = max(parsed_notes, key=lambda item: float(item["student_vs_reference_rel_l2"]))
            spread = max(float(item["student_vs_reference_rel_l2"]) for item in parsed_notes) - min(
                float(item["student_vs_reference_rel_l2"]) for item in parsed_notes
            )
            return {
                "dominant_student_closer_to": dominant,
                "closer_to_counts": closer_counts,
                "best_reference_gap_case": best_note,
                "worst_reference_gap_case": worst_note,
                "reference_gap_spread": spread,
                "cases": parsed_notes,
                "source": "structured_per_spec",
            }
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
    worst_note = max(parsed_notes, key=lambda item: float(item["student_vs_reference_rel_l2"]))
    spread = max(float(item["student_vs_reference_rel_l2"]) for item in parsed_notes) - min(
        float(item["student_vs_reference_rel_l2"]) for item in parsed_notes
    )
    return {
        "dominant_student_closer_to": dominant,
        "closer_to_counts": closer_counts,
        "best_reference_gap_case": best_note,
        "worst_reference_gap_case": worst_note,
        "reference_gap_spread": spread,
        "cases": parsed_notes,
        "source": "notes",
    }


def _extract_per_spec_summary(feedback: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback, dict):
        return []
    items = feedback.get("per_spec")
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            summary = {
                "hidden_dim": int(item.get("hidden_dim")),
                "num_tokens": int(item.get("num_tokens", 0)),
                "passed": bool(item.get("passed", False)),
                "rel_l2_err": float(item.get("rel_l2_err", float("inf"))),
                "max_abs_err": float(item.get("max_abs_err", float("inf"))),
            }
        except (TypeError, ValueError):
            continue
        diagnosis = item.get("reference_diagnosis")
        if isinstance(diagnosis, dict):
            try:
                summary["reference_diagnosis"] = {
                    "student_vs_reference_rel_l2_err": float(diagnosis["student_vs_reference_rel_l2_err"]),
                    "student_vs_naive_rel_l2_err": float(diagnosis["student_vs_naive_rel_l2_err"]),
                    "naive_vs_reference_rel_l2_err": float(diagnosis["naive_vs_reference_rel_l2_err"]),
                    "student_closer_to": str(diagnosis["student_closer_to"]),
                }
            except (KeyError, TypeError, ValueError):
                pass
        normalized.append(summary)
    return normalized


def _extract_previous_candidate(feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    candidate = feedback.get("previous_candidate")
    if not isinstance(candidate, dict):
        return {}
    source_code = candidate.get("source_code")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(source_code, str) or not source_code.strip():
        return {}
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return {}
    return {
        "candidate_id": candidate_id,
        "rationale": str(candidate.get("rationale", "")),
        "source": str(candidate.get("source", "")),
        "source_code": source_code,
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


def _candidate_family(candidate_id: str) -> str:
    value = candidate_id.strip()
    if not value:
        return ""
    previous = None
    while previous != value:
        previous = value
        value = _CANDIDATE_FAMILY_SUFFIX_RE.sub("", value).rstrip("_-")
    return value


def _extract_stable_patch_family_summary(state: Phase2OptimizerState) -> dict[str, Any] | None:
    history = [item for item in state.llm_revision_history if isinstance(item, dict) and int(item.get("iteration", 0) or 0) > 0]
    if len(history) < 3:
        return None
    recent = history[-3:]
    candidate_ids: list[str] = []
    revision_preferences: list[str] = []
    for item in recent:
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            return None
        candidate_ids.append(candidate_id)
        generation_context = item.get("generation_context")
        if isinstance(generation_context, dict):
            revision_preferences.append(str(generation_context.get("revision_source_preference", "")))
        else:
            revision_preferences.append("")
    families = [_candidate_family(candidate_id) for candidate_id in candidate_ids]
    if not families or any(not family for family in families):
        return None
    if len(set(families)) != 1:
        return None
    if not all(pref == "previous_candidate_patch_first" for pref in revision_preferences):
        return None
    return {
        "family_name": families[0],
        "recent_candidate_ids": candidate_ids,
        "recent_revision_preferences": revision_preferences,
        "stable_patch_chain": True,
    }


def _build_local_mutation_axes(
    *,
    candidate_strategy: str,
    patch_discipline: str,
    reference_diagnosis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if candidate_strategy not in {"fit_torch_tf32_reference", "fit_torch_reference_over_naive", "match_reference_float32_semantics"}:
        return []
    high_balance_pressure = False
    if isinstance(reference_diagnosis, dict):
        spread = reference_diagnosis.get("reference_gap_spread")
        if isinstance(spread, (int, float)) and float(spread) >= 1e-4:
            high_balance_pressure = True
    axes = [
        {
            "axis": "wx_numeric_path",
            "allowed_values": ["keep_current", "tf32ish", "plain_fp32"],
            "intent": "change only the W*X multiply-accumulate path",
        },
        {
            "axis": "temp_numeric_path",
            "allowed_values": ["keep_current", "tf32ish", "plain_fp32"],
            "intent": "change only the B^T X stage numeric path",
        },
        {
            "axis": "lowrank_numeric_path",
            "allowed_values": ["keep_current", "tf32ish", "plain_fp32"],
            "intent": "change only the A*temp stage numeric path",
        },
        {
            "axis": "temp_store_rounding",
            "allowed_values": ["keep_current", "enabled", "disabled"],
            "intent": "toggle whether temp is explicitly rounded when written or read",
        },
        {
            "axis": "accumulation_operator",
            "allowed_values": ["keep_current", "fmaf", "plain_add_mul"],
            "intent": "toggle only the local accumulation primitive without changing loop structure",
        },
        {
            "axis": "accumulation_grouping",
            "allowed_values": ["keep_current", "single_accumulator", "split_accumulator"],
            "intent": "change only how one reduction is grouped",
        },
    ]
    if patch_discipline == "strict_local_patch":
        for axis in axes:
            axis["max_changes_from_previous_candidate"] = 1
        if high_balance_pressure:
            axes.append(
                {
                    "axis": "stage_specific_balance_tuning",
                    "allowed_values": ["keep_current", "wx_only", "temp_only", "lowrank_only"],
                    "intent": "borrow exactly one stage-local idea to improve cross-dimension balance without altering the whole family",
                    "max_changes_from_previous_candidate": 1,
                }
            )
    return axes


def _build_guided_mutation_plans(
    *,
    patch_discipline: str,
    candidate_strategy: str,
    reference_diagnosis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if patch_discipline != "strict_local_patch":
        return []
    if candidate_strategy not in {"fit_torch_tf32_reference", "fit_torch_reference_over_naive", "match_reference_float32_semantics"}:
        return []
    balance_pressure = "normal"
    if isinstance(reference_diagnosis, dict):
        spread = reference_diagnosis.get("reference_gap_spread")
        if isinstance(spread, (int, float)) and float(spread) >= 1e-4:
            balance_pressure = "high"
    plans = [
        {
            "plan_id": "wx_to_tf32ish",
            "changes": [{"axis": "wx_numeric_path", "target": "tf32ish"}],
            "intent": "only move the W*X stage toward a stronger TF32-like path",
        },
        {
            "plan_id": "wx_to_plain_fp32",
            "changes": [{"axis": "wx_numeric_path", "target": "plain_fp32"}],
            "intent": "only move the W*X stage toward a plainer float32 path",
        },
        {
            "plan_id": "temp_to_tf32ish",
            "changes": [{"axis": "temp_numeric_path", "target": "tf32ish"}],
            "intent": "only move the B^T X stage toward a TF32-like path",
        },
        {
            "plan_id": "temp_to_plain_fp32",
            "changes": [{"axis": "temp_numeric_path", "target": "plain_fp32"}],
            "intent": "only move the B^T X stage toward a plainer float32 path",
        },
        {
            "plan_id": "lowrank_to_tf32ish",
            "changes": [{"axis": "lowrank_numeric_path", "target": "tf32ish"}],
            "intent": "only move the A*temp stage toward a TF32-like path",
        },
        {
            "plan_id": "lowrank_to_plain_fp32",
            "changes": [{"axis": "lowrank_numeric_path", "target": "plain_fp32"}],
            "intent": "only move the A*temp stage toward a plainer float32 path",
        },
        {
            "plan_id": "temp_round_enable",
            "changes": [{"axis": "temp_store_rounding", "target": "enabled"}],
            "intent": "only enable explicit temp rounding at the stage boundary",
        },
        {
            "plan_id": "temp_round_disable",
            "changes": [{"axis": "temp_store_rounding", "target": "disabled"}],
            "intent": "only disable explicit temp rounding at the stage boundary",
        },
        {
            "plan_id": "accum_operator_fmaf",
            "changes": [{"axis": "accumulation_operator", "target": "fmaf"}],
            "intent": "only switch the local accumulation primitive toward fmaf",
        },
        {
            "plan_id": "accum_operator_plain",
            "changes": [{"axis": "accumulation_operator", "target": "plain_add_mul"}],
            "intent": "only switch the local accumulation primitive toward plain multiply-add",
        },
        {
            "plan_id": "split_accum",
            "changes": [{"axis": "accumulation_grouping", "target": "split_accumulator"}],
            "intent": "only split one reduction into a grouped accumulation path",
        },
        {
            "plan_id": "single_accum",
            "changes": [{"axis": "accumulation_grouping", "target": "single_accumulator"}],
            "intent": "only collapse one reduction back to a single accumulator",
        },
    ]
    if balance_pressure == "high":
        plans.extend(
            [
                {
                    "plan_id": "balance_wx_tf32_temp_plain",
                    "changes": [
                        {"axis": "wx_numeric_path", "target": "tf32ish"},
                        {"axis": "temp_numeric_path", "target": "plain_fp32"},
                    ],
                    "intent": "borrow a stronger TF32-like path only for W*X while keeping temp plainer for better cross-dimension balance",
                },
                {
                    "plan_id": "balance_temp_tf32_lowrank_plain",
                    "changes": [
                        {"axis": "temp_numeric_path", "target": "tf32ish"},
                        {"axis": "lowrank_numeric_path", "target": "plain_fp32"},
                    ],
                    "intent": "keep the low-rank correction plainer while shifting only temp generation toward TF32-like behavior",
                },
            ]
        )
    return plans


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
        "If a correctness-safe implementation can keep the large GEMMs on cuBLAS-backed paths and only customize the smaller rank-16 update or staging logic, prefer that over a fully hand-written end-to-end numeric path. "
        "Avoid shared-memory tiling, fused kernels, warp-specialized tricks, vectorized reinterpret casts, and other aggressive optimizations until correctness is already passing."
    )


def build_lora_generation_user_prompt(
    *,
    state: Phase2OptimizerState,
    iteration: int,
    best_speedup: float,
    feedback: dict[str, Any] | None,
) -> str:
    previous_correctness_passed = _extract_previous_correctness_passed(feedback)
    previous_rel_l2_err = _extract_previous_rel_l2_err(feedback)
    previous_candidate = _extract_previous_candidate(feedback)
    per_spec_feedback = _extract_per_spec_summary(feedback)
    reference_diagnosis = _extract_reference_diagnosis_summary(feedback)
    torch_precision = _extract_torch_precision_summary(feedback)
    stable_patch_family = _extract_stable_patch_family_summary(state)
    optimization_priority = "balanced"
    candidate_strategy = "normal_iteration"
    if previous_correctness_passed:
        optimization_priority = "speedup_after_correctness"
        candidate_strategy = "speedup_preserve_correctness"
    elif previous_rel_l2_err is not None and previous_rel_l2_err <= 1e-3:
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
    previous_compile_ok = bool(feedback.get("compile_ok")) if isinstance(feedback, dict) else False
    if previous_candidate and previous_compile_ok and not previous_correctness_passed:
        revision_source_preference = "previous_candidate_patch_first"
    patch_discipline = "normal"
    if isinstance(stable_patch_family, dict):
        patch_discipline = "strict_local_patch"
    local_mutation_axes = _build_local_mutation_axes(
        candidate_strategy=candidate_strategy,
        patch_discipline=patch_discipline,
        reference_diagnosis=reference_diagnosis,
    )
    guided_mutation_plans = _build_guided_mutation_plans(
        patch_discipline=patch_discipline,
        candidate_strategy=candidate_strategy,
        reference_diagnosis=reference_diagnosis,
    )
    mutation_execution_mode = "freeform_revision"
    selected_mutation_plan: dict[str, Any] = {}
    if guided_mutation_plans:
        mutation_execution_mode = "guided_local_enumeration"
        selected_mutation_plan = guided_mutation_plans[(max(0, iteration - 1)) % len(guided_mutation_plans)]
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
            "Prefer correctness-safe families that keep the large GEMMs on cuBLAS-compatible paths and only customize the smaller rank-16 update or staging logic.",
            "Do not keep reproducing the same naive float32 loop order; that has already been shown to match the naive reference instead of the PyTorch reference.",
            "Do not revert to a fully plain-float32 two-kernel baseline that removes all reduced-precision or tf32-like behavior from every stage; that fallback path has already been explored and is not the target direction.",
            "Do not treat half precision as equivalent to TF32.",
            "Do not use unsupported pseudo-TF32 intrinsics such as __float2tf32, __nv_float2tf32, or __nv_tf32_to_float.",
            "If you use half intrinsics, you must include <cuda_fp16.h>, but prefer simpler numerically altered float-based reduction structures before attempting half-based approximations.",
            "Prefer the already-working pattern of half-precision multiplication with float accumulation over inventing unavailable TF32-specific intrinsics.",
            "Keep revisions close to the best reference_like_candidate and make local numeric-path changes instead of rewriting the whole implementation.",
            "Explicitly consider mixed numeric paths across stages: for example, use a TF32-like path for W*X but preserve plain float32 behavior for the low-rank path, or vice versa, if that better matches the reference.",
            "Treat the temp = B^T X stage, the W*X stage, and the A*temp stage as independently tunable numeric paths rather than forcing all stages to use the same approximation.",
            "When base_candidate and reference_like_candidate have different strengths, prefer transplanting one local numeric-path idea from the reference_like_candidate into the base_candidate rather than replacing the entire structure.",
            "Avoid whole-program rewrites that switch the complete candidate from one family to another; preserve the base_candidate skeleton and only patch one stage or one accumulation path at a time.",
            "Prefer changes that alter accumulation grouping or staging in a controlled way, while keeping the ABI and required math unchanged.",
            "cuBLAS-backed GEMM paths are allowed and preferred when they reduce correctness risk for the large matrix products.",
            "If a reference_like_candidate is provided, prefer revising that candidate over the naive-like base_candidate.",
            "Optimize for balanced behavior across all tested hidden dimensions rather than overfitting one dimension to an extremely low diagnostic error.",
            "The immediately previous candidate is your latest concrete patch attempt; unless it catastrophically regressed or failed compilation, prefer editing that candidate in place instead of starting a fresh family.",
        ]
    elif candidate_strategy == "speedup_preserve_correctness":
        strategy_specific_guidance = [
            "The previous candidate already passed correctness, so preserve its numerical behavior and optimize speed cautiously.",
            "Do not make broad numeric-path changes that could lose correctness.",
            "Prefer local performance improvements such as safer launch changes, memory-access cleanup, or low-risk staging improvements.",
            "Keep the current best correct candidate as the primary revision base unless there is a specific measured reason to change strategy.",
        ]
    focus_hidden_dim: int | None = None
    if isinstance(reference_diagnosis, dict):
        worst_case = reference_diagnosis.get("worst_reference_gap_case")
        if isinstance(worst_case, dict):
            hidden_dim = worst_case.get("hidden_dim")
            if isinstance(hidden_dim, int):
                focus_hidden_dim = hidden_dim
    balance_priority = "normal"
    if isinstance(reference_diagnosis, dict):
        spread = reference_diagnosis.get("reference_gap_spread")
        if isinstance(spread, (int, float)) and float(spread) >= 1e-4:
            balance_priority = "high"
    payload = {
        "task": "generate_lora_candidate",
        "iteration": iteration,
        "operator": "Y = W X + A(B^T X)",
        "optimization_priority": optimization_priority,
        "candidate_strategy": candidate_strategy,
        "revision_mode": "modify_best_candidate_instead_of_regenerating_from_scratch" if (base_candidate or reference_like_candidate) else "from_scratch",
        "revision_source_preference": revision_source_preference,
        "patch_discipline": patch_discipline,
        "mutation_execution_mode": mutation_execution_mode,
        "selected_mutation_plan": selected_mutation_plan,
        "focus_hidden_dim": focus_hidden_dim,
        "balance_priority": balance_priority,
        "reference_diagnosis_summary": reference_diagnosis or {},
        "torch_precision_summary": torch_precision or {},
        "stable_patch_family_summary": stable_patch_family or {},
        "local_mutation_axes": local_mutation_axes,
        "guided_mutation_plans": guided_mutation_plans,
        "per_spec_feedback": per_spec_feedback,
        "previous_candidate": previous_candidate,
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
                "treat the structured per_spec feedback as the main source of truth for which hidden dimensions regressed or improved",
                "revise the immediately previous candidate in place when it compiled and produced meaningful metrics; do not reset to a new baseline family unless the previous attempt catastrophically failed",
                "if the recent revision history is already a stable patch chain within one candidate family, preserve that family and apply only a local numeric-path patch rather than renaming or re-architecting the candidate",
                "when patch_discipline is strict_local_patch, do not introduce a new candidate family; keep the same structural skeleton and modify only one narrow behavior such as one stage, one rounding rule, one accumulation grouping, or one temp-handling detail",
                "when local_mutation_axes are provided, choose one or at most two axes and keep all other numeric behaviors unchanged",
                "when mutation_execution_mode is guided_local_enumeration, execute the selected_mutation_plan faithfully and do not improvise unrelated changes",
                "in guided_local_enumeration mode, preserve the previous candidate skeleton and translate only the listed axis changes into code edits",
                "prefer a mutation that can be described as a narrow toggle on the previous candidate rather than a fresh rewrite",
                "treat the preferred revision source candidate as the starting point and revise it instead of discarding it",
                "preserve working ABI, indexing structure, and any already-correct math unless you have a specific reason to change them",
                "make the smallest set of changes that can plausibly improve correctness toward the reference",
                "if one hidden_dim is clearly worse than the others, use it as a diagnostic hint, but do not overfit to it at the expense of the other tested hidden_dims",
                "prefer balanced improvements across 3584, 4096, and 4608 over extreme overfitting to any single hidden_dim",
                "treat regressions on already-strong hidden_dims as important failures even if one hidden_dim improves sharply",
                "avoid resetting all stages back to the already-explored naive float32 baseline when reference-like mixed-stage variants are available",
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
