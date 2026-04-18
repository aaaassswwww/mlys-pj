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
    return "metric=" in lowered and "value=" in lowered and ("samples=" in lowered or "sample" in lowered)


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
            return ProbeGenerationResult(
                ok=False,
                metric=metric,
                source_path=out_dir / _sanitize_metric(metric) / "probe.cu",
                source_type="llm_generated",
                rationale="",
                error=llm_error,
            )

        code = payload.get("code")
        rationale = payload.get("rationale")
        if not isinstance(code, str) or not code.strip():
            return ProbeGenerationResult(
                ok=False,
                metric=metric,
                source_path=out_dir / _sanitize_metric(metric) / "probe.cu",
                source_type="llm_generated",
                rationale="",
                error="missing_code",
            )
        code = _normalize_generated_code(code)
        ok, validation_error = _validate_generated_code(code)
        if not ok:
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

