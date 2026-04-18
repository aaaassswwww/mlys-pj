from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from profiler_agent.codegen.prompts import (
    build_probe_generation_system_prompt,
    build_probe_generation_user_prompt,
    build_probe_repair_system_prompt,
)
from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient


_METRIC_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]+")
_STL_HEADERS = (
    "<iostream>",
    "<vector>",
    "<algorithm>",
    "<string>",
    "<map>",
    "<unordered_map>",
    "<set>",
)
_STL_PATTERNS = (
    "std::cout",
    "std::cerr",
    "std::vector",
    "std::string",
    "std::sort",
    "std::max_element",
    "std::min_element",
)


@dataclass(frozen=True)
class ProbeGenerationResult:
    ok: bool
    metric: str
    source_path: Path
    source_type: str
    rationale: str
    error: str = ""


def _sanitize_metric(metric: str) -> str:
    cleaned = _METRIC_SAFE_RE.sub("_", metric.strip())
    return cleaned.strip("_") or "unknown_metric"


def _strip_fenced_code(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text, count=1)
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def _normalize_generated_code(code: str) -> str:
    normalized = _strip_fenced_code(code).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("__clock64()", "clock64()")
    normalized = normalized.replace("__clock64(", "clock64(")
    return normalized.strip() + "\n"


def _ensure_output_protocol(code: str) -> bool:
    lowered = code.lower()
    return (
        "metric=" in lowered
        and "value=" in lowered
        and ("samples=" in lowered or "sample" in lowered)
        and "mode=" in lowered
    )


def _basic_cuda_shape_ok(code: str) -> bool:
    return "__global__" in code and ("int main(" in code or "int main()" in code)


def _validate_generated_code(code: str) -> tuple[bool, str]:
    normalized = _normalize_generated_code(code)
    lower = normalized.lower()
    if "cuda_runtime.h" not in lower:
        return False, "missing_cuda_runtime_include"
    if not _basic_cuda_shape_ok(normalized):
        return False, "missing_kernel_or_main"
    if not _ensure_output_protocol(normalized):
        return False, "missing_structured_output_protocol"
    if "__clock64" in normalized:
        return False, "contains_unsupported___clock64"
    for header in _STL_HEADERS:
        if f"#include {header}" in normalized:
            return False, f"contains_nonminimal_header:{header}"
    for pattern in _STL_PATTERNS:
        if pattern in normalized:
            return False, f"contains_nonminimal_cpp_construct:{pattern}"
    if "printf(" not in normalized:
        return False, "missing_printf_output"
    banned = ("wget ", "curl ", "http://", "https://", "git clone")
    for item in banned:
        if item in lower:
            return False, f"contains_banned_pattern:{item.strip()}"
    return True, ""


class ProbeCodeGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()

    def is_enabled(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_enabled()

    def _write_code(self, *, metric: str, code: str, out_dir: Path) -> Path:
        safe_metric = _sanitize_metric(metric)
        metric_dir = out_dir / safe_metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        source_path = metric_dir / "probe.cu"
        source_path.write_text(code, encoding="utf-8")
        return source_path

    @staticmethod
    def _is_workload_counter_metric(metric: str) -> bool:
        return "__" in (metric or "").strip().lower()

    @staticmethod
    def _counter_probe_template(metric: str) -> str:
        lowered = metric.strip().lower()
        if "dram__bytes_read" in lowered:
            kernel_body = (
                "    unsigned long long idx = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);\n"
                "    unsigned long long stride = (unsigned long long)(blockDim.x * gridDim.x);\n"
                "    float acc = 0.0f;\n"
                "    for (int it = 0; it < iterations; ++it) {\n"
                "        for (unsigned long long i = idx; i < count; i += stride) {\n"
                "            acc += src[i];\n"
                "        }\n"
                "    }\n"
                "    if (idx == 0) { sink[0] = acc; }\n"
            )
            bytes_expr = "((double)count * sizeof(float) * iterations)"
        elif "dram__bytes_write" in lowered:
            kernel_body = (
                "    unsigned long long idx = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);\n"
                "    unsigned long long stride = (unsigned long long)(blockDim.x * gridDim.x);\n"
                "    for (int it = 0; it < iterations; ++it) {\n"
                "        for (unsigned long long i = idx; i < count; i += stride) {\n"
                "            dst[i] = (float)(i + it);\n"
                "        }\n"
                "    }\n"
                "    if (idx == 0) { sink[0] = dst[0]; }\n"
            )
            bytes_expr = "((double)count * sizeof(float) * iterations)"
        elif "sm__throughput" in lowered:
            kernel_body = (
                "    unsigned long long idx = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);\n"
                "    float x = (float)(idx % 97) * 0.01f + 1.0f;\n"
                "    float y = (float)(idx % 31) * 0.02f + 0.5f;\n"
                "    #pragma unroll 1\n"
                "    for (int it = 0; it < iterations * 256; ++it) {\n"
                "        x = x * 1.000013f + y;\n"
                "        y = y * 0.999983f + x;\n"
                "        x = x * 0.500001f + y * 0.499999f;\n"
                "    }\n"
                "    if (idx == 0) { sink[0] = x + y; }\n"
            )
            bytes_expr = "((double)grid.x * block.x * iterations * 256.0)"
        else:
            kernel_body = (
                "    unsigned long long idx = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);\n"
                "    unsigned long long stride = (unsigned long long)(blockDim.x * gridDim.x);\n"
                "    float acc = 0.0f;\n"
                "    for (int it = 0; it < iterations; ++it) {\n"
                "        for (unsigned long long i = idx; i < count; i += stride) {\n"
                "            float v = src[i];\n"
                "            v = v * 1.00001f + (float)(i & 7);\n"
                "            dst[i] = v;\n"
                "            acc += v;\n"
                "        }\n"
                "    }\n"
                "    if (idx == 0) { sink[0] = acc + dst[0]; }\n"
            )
            bytes_expr = "((double)count * sizeof(float) * iterations * 2.0)"

        return (
            "#include <cstdio>\n"
            "#include <cuda_runtime.h>\n\n"
            "__global__ void counter_probe_kernel(const float* src, float* dst, float* sink, unsigned long long count, int iterations) {\n"
            f"{kernel_body}"
            "}\n\n"
            "int main() {\n"
            "    const unsigned long long count = 1u << 22;\n"
            "    const int iterations = 128;\n"
            "    float* src = 0;\n"
            "    float* dst = 0;\n"
            "    float* sink = 0;\n"
            "    if (cudaMalloc((void**)&src, count * sizeof(float)) != cudaSuccess) {\n"
            f"        printf(\"metric={metric} value=0 samples=0 mode=ncu_profiled median=0 best=0 std=0\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    if (cudaMalloc((void**)&dst, count * sizeof(float)) != cudaSuccess) {\n"
            "        cudaFree(src);\n"
            f"        printf(\"metric={metric} value=0 samples=0 mode=ncu_profiled median=0 best=0 std=0\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    if (cudaMalloc((void**)&sink, sizeof(float)) != cudaSuccess) {\n"
            "        cudaFree(src);\n"
            "        cudaFree(dst);\n"
            f"        printf(\"metric={metric} value=0 samples=0 mode=ncu_profiled median=0 best=0 std=0\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    cudaMemset(dst, 0, count * sizeof(float));\n"
            "    cudaMemset(sink, 0, sizeof(float));\n"
            "    dim3 block(256);\n"
            "    dim3 grid(160);\n"
            "    cudaEvent_t start;\n"
            "    cudaEvent_t stop;\n"
            "    cudaEventCreate(&start);\n"
            "    cudaEventCreate(&stop);\n"
            "    cudaEventRecord(start);\n"
            "    counter_probe_kernel<<<grid, block>>>(src, dst, sink, count, iterations);\n"
            "    cudaEventRecord(stop);\n"
            "    cudaEventSynchronize(stop);\n"
            "    float ms = 0.0f;\n"
            "    cudaEventElapsedTime(&ms, start, stop);\n"
            "    if (ms <= 0.0f) { ms = 0.001f; }\n"
            f"    double proxy = {bytes_expr} / ((double)ms * 1.0e-3);\n"
            f"    printf(\"metric={metric} value=%0.6f samples=1 mode=ncu_profiled median=%0.6f best=%0.6f std=0\\n\", proxy, proxy, proxy);\n"
            "    cudaEventDestroy(start);\n"
            "    cudaEventDestroy(stop);\n"
            "    cudaFree(src);\n"
            "    cudaFree(dst);\n"
            "    cudaFree(sink);\n"
            "    return 0;\n"
            "}\n"
        )

    def _fallback_generate(self, *, metric: str, out_dir: Path, llm_error: str) -> ProbeGenerationResult:
        if self._is_workload_counter_metric(metric):
            code = _normalize_generated_code(self._counter_probe_template(metric))
            ok, validation_error = _validate_generated_code(code)
            if not ok:
                return ProbeGenerationResult(
                    ok=False,
                    metric=metric,
                    source_path=out_dir / _sanitize_metric(metric) / "probe.cu",
                    source_type="template_generated",
                    rationale="",
                    error=validation_error,
                )
            source_path = self._write_code(metric=metric, code=code, out_dir=out_dir)
            return ProbeGenerationResult(
                ok=True,
                metric=metric,
                source_path=source_path,
                source_type="template_generated",
                rationale=f"template_generated_after_{llm_error or 'llm_failure'}",
                error="",
            )
        return ProbeGenerationResult(
            ok=False,
            metric=metric,
            source_path=out_dir / _sanitize_metric(metric) / "probe.cu",
            source_type="llm_generated",
            rationale="",
            error=llm_error,
        )

    def _llm_generate(self, *, metric: str, prior_error: str | None = None) -> tuple[Optional[dict], str]:
        if not self.is_enabled():
            return None, "llm_disabled"
        system_prompt = (
            build_probe_repair_system_prompt()
            if prior_error
            else build_probe_generation_system_prompt()
        )
        user_prompt = build_probe_generation_user_prompt(metric=metric, prior_error=prior_error)
        payload = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not isinstance(payload, dict):
            return None, "llm_empty_or_invalid_json"
        return payload, ""

    def generate_probe(
        self,
        *,
        metric: str,
        out_dir: Path,
        prior_error: str | None = None,
    ) -> ProbeGenerationResult:
        payload, llm_error = self._llm_generate(metric=metric, prior_error=prior_error)
        if payload is None:
            return self._fallback_generate(metric=metric, out_dir=out_dir, llm_error=llm_error)

        code = payload.get("code")
        rationale = payload.get("rationale")
        if not isinstance(code, str) or not code.strip():
            return self._fallback_generate(metric=metric, out_dir=out_dir, llm_error="missing_code")
        code = _normalize_generated_code(code)
        ok, validation_error = _validate_generated_code(code)
        if not ok:
            if self._is_workload_counter_metric(metric):
                return self._fallback_generate(metric=metric, out_dir=out_dir, llm_error=validation_error)
            return ProbeGenerationResult(
                ok=False,
                metric=metric,
                source_path=out_dir / _sanitize_metric(metric) / "probe.cu",
                source_type="llm_generated",
                rationale=str(rationale) if isinstance(rationale, str) else "",
                error=validation_error,
            )

        source_path = self._write_code(metric=metric, code=code, out_dir=out_dir)
        return ProbeGenerationResult(
            ok=True,
            metric=metric,
            source_path=source_path,
            source_type="llm_generated",
            rationale=str(rationale) if isinstance(rationale, str) else "",
            error="",
        )

