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

    def summarize_outputs(self, state: MultiAgentState) -> AgentMessage:
        pipeline = state.outputs.get("pipeline", {})
        analysis_path = pipeline.get("analysis_path")
        summary: dict[str, Any] = {"status": "no_pipeline_output"}
        analysis_obj: dict[str, Any] = {}
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
            system_prompt = (
                "You are a concise GPU profiling interpreter. Return JSON with keys "
                "'explanation' (string) and 'risk_level' (low|medium|high)."
            )
            user_prompt = json.dumps(
                {"analysis": analysis_obj, "objective": state.request.objective, "targets": state.request.targets},
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if isinstance(llm_json, dict):
                explanation = llm_json.get("explanation")
                risk_level = llm_json.get("risk_level")
                if isinstance(explanation, str) and explanation.strip():
                    summary["llm_explanation"] = explanation.strip()
                if isinstance(risk_level, str) and risk_level in {"low", "medium", "high"}:
                    summary["llm_risk_level"] = risk_level
                if "llm_explanation" in summary or "llm_risk_level" in summary:
                    summary["interpretation_source"] = "llm"
        if "interpretation_source" not in summary:
            summary["interpretation_source"] = "rules"

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
        if self.llm_client is not None and self.llm_client.is_enabled():
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
            llm_actions = llm_json.get("next_actions") if isinstance(llm_json, dict) else None
            if isinstance(llm_actions, list):
                parsed_actions = [str(item).strip() for item in llm_actions if str(item).strip()]
                if parsed_actions:
                    next_actions = parsed_actions[:4]
                    source = "llm"

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
            content={"next_actions": next_actions, "source": source},
        )
