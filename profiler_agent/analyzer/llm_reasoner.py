from __future__ import annotations

import json
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient


def _validate_bound_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value in {"compute_bound", "memory_bound", "balanced_or_mixed", "unknown"}:
        return value
    return None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _validate_bottlenecks(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        severity = item.get("severity")
        reason = item.get("reason")
        suggestion = item.get("suggestion")
        if not all(isinstance(x, str) and x.strip() for x in [category, severity, reason, suggestion]):
            continue
        out.append(
            {
                "category": category.strip(),
                "severity": severity.strip().upper(),
                "reason": reason.strip(),
                "suggestion": suggestion.strip(),
            }
        )
    return out


def build_llm_analysis(
    *,
    results: dict[str, float],
    evidence: dict[str, Any],
    baseline_analysis: dict[str, Any],
    llm_client: LLMClient | None = None,
) -> dict[str, Any] | None:
    client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
    if client is None or not client.is_enabled():
        return None

    system_prompt = (
        "You are a GPU profiling reasoning engine. "
        "Use only provided data, no outside lookup. "
        "Return JSON with keys: bound_type, confidence, bottlenecks, llm_reasoning_summary. "
        "bound_type must be one of compute_bound,memory_bound,balanced_or_mixed,unknown. "
        "bottlenecks is a list of objects with keys category,severity,reason,suggestion."
    )
    user_prompt = json.dumps(
        {
            "results": results,
            "evidence": evidence,
            "baseline_analysis": baseline_analysis,
        },
        ensure_ascii=True,
    )
    payload = client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
    if not isinstance(payload, dict):
        return None

    bound_type = _validate_bound_type(payload.get("bound_type"))
    confidence = _to_float(payload.get("confidence"))
    bottlenecks = _validate_bottlenecks(payload.get("bottlenecks"))
    summary = payload.get("llm_reasoning_summary")
    if bound_type is None or confidence is None or bottlenecks is None:
        return None
    if not isinstance(summary, str) or not summary.strip():
        summary = "LLM reasoning provided without explicit summary."

    return {
        "bound_type": bound_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "bottlenecks": bottlenecks,
        "llm_reasoning_summary": summary.strip(),
    }

