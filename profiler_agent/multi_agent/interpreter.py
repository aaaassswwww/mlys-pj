from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.multi_agent.models import AgentMessage, MultiAgentState


class InterpreterAgent:
    name = "interpreter_agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _normalize_risk_level(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        if lowered in {"low", "medium", "high"}:
            return lowered
        aliases = {
            "med": "medium",
            "mid": "medium",
            "moderate": "medium",
            "critical": "high",
            "severe": "high",
        }
        return aliases.get(lowered)

    @staticmethod
    def _parse_next_actions(llm_json: dict[str, Any]) -> list[str]:
        raw = llm_json.get("next_actions")
        if raw is None:
            raw = llm_json.get("actions")
        if raw is None:
            raw = llm_json.get("recommendations")
        if isinstance(raw, str):
            raw = [token.strip() for token in raw.replace(";", ",").split(",")]
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def summarize_outputs(self, state: MultiAgentState) -> AgentMessage:
        pipeline = state.outputs.get("pipeline", {})
        analysis_path = pipeline.get("analysis_path")
        summary: dict[str, Any] = {"status": "no_pipeline_output"}
        analysis_obj: dict[str, Any] = {}
        fallback_reason = "llm_not_enabled"
        llm_attempted = False
        if isinstance(analysis_path, str) and Path(analysis_path).exists():
            data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
            analysis_obj = data
            summary = {
                "status": "ok",
                "bound_type": data.get("bound_type", "unknown"),
                "confidence_adjusted": data.get("confidence_adjusted", data.get("confidence", 0.0)),
                "bottleneck_count": len(data.get("bottlenecks", [])),
            }

        if self.llm_client is not None and self.llm_client.is_enabled() and analysis_obj:
            llm_attempted = True
            system_prompt = (
                "You are a concise GPU profiling interpreter. Return JSON with keys "
                "'explanation' (string) and 'risk_level' (low|medium|high)."
            )
            user_prompt = json.dumps(
                {"analysis": analysis_obj, "objective": state.request.objective, "targets": state.request.targets},
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if not isinstance(llm_json, dict):
                fallback_reason = "llm_no_response_or_invalid_json"
            else:
                explanation = llm_json.get("explanation")
                risk_level = self._normalize_risk_level(llm_json.get("risk_level"))
                if isinstance(explanation, str) and explanation.strip():
                    summary["llm_explanation"] = explanation.strip()
                if isinstance(risk_level, str):
                    summary["llm_risk_level"] = risk_level
                if "llm_explanation" in summary or "llm_risk_level" in summary:
                    summary["interpretation_source"] = "llm"
                    fallback_reason = ""
                else:
                    fallback_reason = "llm_missing_explanation_and_risk_level"
        if "interpretation_source" not in summary:
            summary["interpretation_source"] = "rules"
        summary["llm_attempted"] = llm_attempted
        if summary["interpretation_source"] == "rules":
            summary["interpretation_fallback_reason"] = fallback_reason

        state.outputs["interpretation"] = summary
        return AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="interpretation_ready",
            content=summary,
        )

    def propose_next_actions(self, state: MultiAgentState) -> AgentMessage:
        interpretation = state.outputs.get("interpretation", {})
        next_actions: list[str] = []
        source = "rules"
        fallback_reason = "llm_not_enabled"
        llm_attempted = False
        if self.llm_client is not None and self.llm_client.is_enabled():
            llm_attempted = True
            system_prompt = (
                "You propose next profiling actions. Return JSON with key 'next_actions' "
                "as a list of concise action strings (max 4)."
            )
            user_prompt = json.dumps(
                {
                    "interpretation": interpretation,
                    "objective": state.request.objective,
                    "targets": state.request.targets,
                },
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if not isinstance(llm_json, dict):
                fallback_reason = "llm_no_response_or_invalid_json"
            else:
                parsed_actions = self._parse_next_actions(llm_json)
                if parsed_actions:
                    next_actions = parsed_actions[:4]
                    source = "llm"
                    fallback_reason = ""
                else:
                    fallback_reason = "llm_missing_next_actions"

        if not next_actions:
            source = "rules"
            bound_type = interpretation.get("bound_type")
            if bound_type == "memory_bound":
                next_actions.append("collect_ncu_memory_metrics_and_optimize_memory_access")
            elif bound_type == "compute_bound":
                next_actions.append("collect_compute_efficiency_metrics_and_optimize_instruction_mix")
            else:
                next_actions.append("improve_signal_coverage_by_using_realistic_workload_run_command")
            next_actions.append("re-run_pipeline_with_same_targets_for_reproducibility_check")

        state.outputs["next_actions"] = next_actions
        return AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="refinement_ready",
            content={
                "next_actions": next_actions,
                "source": source,
                "llm_attempted": llm_attempted,
                "refinement_fallback_reason": fallback_reason if source == "rules" else "",
            },
        )
