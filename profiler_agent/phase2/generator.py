from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient
from profiler_agent.phase2.models import GeneratedCandidate, Phase2OptimizerState
from profiler_agent.phase2.prompts import build_lora_generation_system_prompt, build_lora_generation_user_prompt

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


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
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()

    def is_enabled(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_enabled()

    def bootstrap_candidate(self) -> GeneratedCandidate:
        return GeneratedCandidate(
            candidate_id="bootstrap-placeholder",
            source_code=build_bootstrap_lora_source(),
            rationale="compile-oriented placeholder that keeps optimized_lora.cu present before optimization converges",
            source="bootstrap_template",
        )

    def generate_candidate(
        self,
        *,
        state: Phase2OptimizerState,
        feedback: dict[str, Any] | None,
    ) -> GeneratedCandidate:
        default_id = f"iter-{state.iteration:02d}"
        if not self.is_enabled():
            bootstrap = self.bootstrap_candidate()
            return GeneratedCandidate(
                candidate_id=default_id,
                source_code=bootstrap.source_code,
                rationale="llm_disabled_using_bootstrap_candidate",
                source="bootstrap_template",
            )

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
            return GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale="llm_returned_no_json_fallback_to_bootstrap",
                source="bootstrap_template",
            )

        source_code = payload.get("source_code")
        rationale = payload.get("rationale", "")
        candidate_id = payload.get("candidate_id", default_id)
        if not isinstance(source_code, str) or not source_code.strip():
            fallback = self.bootstrap_candidate()
            return GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale="llm_missing_source_code_fallback_to_bootstrap",
                source="bootstrap_template",
            )

        normalized = _normalize_source_code(source_code)
        ok, error = _validate_source_code(normalized)
        if not ok:
            fallback = self.bootstrap_candidate()
            return GeneratedCandidate(
                candidate_id=default_id,
                source_code=fallback.source_code,
                rationale=f"llm_generated_invalid_source_fallback:{error}",
                source="bootstrap_template",
            )

        return GeneratedCandidate(
            candidate_id=_sanitize_candidate_id(str(candidate_id), default=default_id),
            source_code=normalized,
            rationale=str(rationale) if isinstance(rationale, str) else "",
            source="llm_generated",
        )


def write_candidate_snapshot(root_dir: Path, candidate: GeneratedCandidate, *, filename: str) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / filename
    path.write_text(candidate.source_code, encoding="utf-8")
    return path
