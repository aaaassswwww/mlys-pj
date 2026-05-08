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
_SHARED_RUNTIME_RANK_RE = re.compile(
    r'(?:__shared__|__attribute__\s*\(\(\s*shared\s*\)\))\s+[^;\n\[]+\[[^\]\n]*\br\b[^\]\n]*\]',
    flags=re.IGNORECASE,
)
_FORBIDDEN_HOST_TEST_RE = re.compile(
    r"\b(?:int\s+main\s*\(|run_lora_forward\s*\(|cudaMemcpy(?:Async)?\s*\([^;]*cudaMemcpyHostToDevice)",
    flags=re.IGNORECASE,
)
_HALF_INTRINSIC_RE = re.compile(r"__(?:float2half(?:_rn)?|half2float)\s*\(")
_LIKELY_TF32_HALF_SIM_RE = re.compile(r"\b(?:half_precision|tf32)[A-Za-z0-9_]*\b", flags=re.IGNORECASE)


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
    if not _ENTRYPOINT_SIGNATURE_RE.search(source_code):
        return False, "missing_or_mismatched_launch_optimized_lora_signature"
    if re.search(r"launch_optimized_lora\s*\([^)]*\bint\s+r\b", source_code):
        return False, "launch_optimized_lora_must_not_accept_runtime_rank_parameter"
    if _SHARED_RUNTIME_RANK_RE.search(source_code):
        return False, "contains_runtime_rank_sized_shared_array"
    if ("cublas" in lowered or "cublaslt" in lowered) and "#include <cublas" in lowered:
        return False, "contains_cublas_dependency_not_supported_by_current_build"
    if _HALF_INTRINSIC_RE.search(source_code) and "#include <cuda_fp16.h>" not in source_code:
        return False, "uses_half_intrinsics_without_cuda_fp16_include"
    if _LIKELY_TF32_HALF_SIM_RE.search(source_code) and _HALF_INTRINSIC_RE.search(source_code):
        return False, "contains_unreliable_half_based_tf32_simulation_attempt"
    return True, ""


def build_bootstrap_lora_source() -> str:
    return (
        "#include <cuda_runtime.h>\n"
        "\n"
        "// Bootstrap candidate for Phase 2. This is a compile-oriented placeholder\n"
        "// that keeps optimized_lora.cu present while the agent iterates toward\n"
        "// a correct and faster implementation.\n"
        "__global__ void optimized_lora_placeholder_kernel(\n"
        "    const float* W,\n"
        "    const float* X,\n"
        "    const float* A,\n"
        "    const float* B,\n"
        "    float* Y,\n"
        "    int d,\n"
        "    int n,\n"
        "    int r) {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    int total = d * n;\n"
        "    if (idx < total) {\n"
        "        Y[idx] = 0.0f;\n"
        "    }\n"
        "}\n"
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
        "    (void)W;\n"
        "    (void)X;\n"
        "    (void)A;\n"
        "    (void)B;\n"
        "    const int r = 16;\n"
        "    int total = d * n;\n"
        "    int threads = 256;\n"
        "    int blocks = (total + threads - 1) / threads;\n"
        "    optimized_lora_placeholder_kernel<<<blocks, threads, 0, stream>>>(W, X, A, B, Y, d, n, r);\n"
        "}\n"
    )


class LoraCandidateGenerator:
    def __init__(self, llm_client: LLMClient | None = None, *, debug_dir: Path | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
        self.debug_dir = debug_dir

    def is_enabled(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_enabled()

    def bootstrap_candidate(self) -> GeneratedCandidate:
        return GeneratedCandidate(
            candidate_id="bootstrap-placeholder",
            source_code=build_bootstrap_lora_source(),
            rationale="compile-oriented placeholder that keeps optimized_lora.cu present before optimization converges",
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

        payload = self.llm_client.complete_json(
            system_prompt=build_lora_generation_system_prompt(),
            user_prompt=build_lora_generation_user_prompt(
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
