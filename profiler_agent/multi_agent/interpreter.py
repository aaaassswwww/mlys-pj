from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient
from profiler_agent.multi_agent.models import AgentMessage, MultiAgentState
from profiler_agent.target_semantics import classify_target


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

    @staticmethod
    def _parse_next_targets(llm_json: dict[str, Any]) -> list[str]:
        raw = llm_json.get("next_targets")
        if raw is None:
            raw = llm_json.get("focus_targets")
        if raw is None:
            raw = llm_json.get("targets")
        if isinstance(raw, str):
            raw = [token.strip() for token in raw.replace(";", ",").split(",")]
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def _extract_probe_refinement(evidence_obj: dict[str, Any]) -> dict[str, Any]:
        targets = evidence_obj.get("targets", {})
        if not isinstance(targets, dict):
            return {"followups": [], "targets": []}
        followups: list[dict[str, str]] = []
        focus_targets: list[str] = []
        supported_modes = {"synthetic_intrinsic_probe", "synthetic_counter_probe"}
        for target, target_evidence in targets.items():
            if not isinstance(target_evidence, dict):
                continue
            if target_evidence.get("measurement_mode") not in supported_modes:
                continue
            probe_iteration = target_evidence.get("probe_iteration", {})
            if not isinstance(probe_iteration, dict):
                continue
            analysis = probe_iteration.get("analysis", {})
            if not isinstance(analysis, dict):
                analysis = {}
            next_action = str(analysis.get("next_action") or probe_iteration.get("final_decision") or "").strip()
            if not next_action or next_action in {"accept_measurement", "unsupported_metric"}:
                continue
            reason = str(analysis.get("reason") or "").strip()
            followups.append({"target": str(target), "next_action": next_action, "reason": reason})
            focus_targets.append(str(target))
        return {"followups": followups, "targets": focus_targets}

    @staticmethod
    def _rule_next_targets(state: MultiAgentState, interpretation: dict[str, Any]) -> list[str]:
        targets = list(state.request.targets)
        if not targets:
            return []
        probe_refinement = interpretation.get("probe_refinement", {})
        if isinstance(probe_refinement, dict):
            raw_targets = probe_refinement.get("targets", [])
            if isinstance(raw_targets, list):
                focused = [str(item).strip() for item in raw_targets if str(item).strip()]
                if focused:
                    return focused
        placeholder_count = interpretation.get("workload_placeholder_count", 0)
        if isinstance(placeholder_count, int) and placeholder_count > 0:
            focused = [target for target in targets if classify_target(target).workload_dependent]
            return focused or targets

        bound_type = interpretation.get("bound_type")
        if bound_type in {"memory_bound", "compute_bound"}:
            focused = [
                target
                for target in targets
                if classify_target(target).semantic_class == "workload_counter"
            ]
            if focused:
                return focused
        return targets

    def summarize_outputs(self, state: MultiAgentState) -> AgentMessage:
        pipeline = state.outputs.get("pipeline", {})
        analysis_path = pipeline.get("analysis_path")
        summary: dict[str, Any] = {"status": "no_pipeline_output"}
        analysis_obj: dict[str, Any] = {}
        evidence_obj: dict[str, Any] = {}
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
        evidence_path = pipeline.get("evidence_path")
        if isinstance(evidence_path, str) and Path(evidence_path).exists():
            data = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                evidence_obj = data
                probe_refinement = self._extract_probe_refinement(data)
                if probe_refinement["followups"]:
                    summary["probe_refinement"] = probe_refinement
                    summary["synthetic_probe_followup_count"] = len(probe_refinement["followups"])

        if self.llm_client is not None and self.llm_client.is_enabled() and analysis_obj:
            llm_attempted = True
            system_prompt = (
                "You are a concise GPU profiling interpreter. Return JSON with keys "
                "'explanation' (string) and 'risk_level' (low|medium|high)."
            )
            user_prompt = json.dumps(
                {
                    "analysis": analysis_obj,
                    "objective": state.request.objective,
                    "targets": state.request.targets,
                    "previous_errors": state.persistent_state.error_history[-3:],
                },
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
        next_targets: list[str] = []
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
                    "pipeline_targets": state.outputs.get("pipeline", {}).get("targets", []),
                    "previous_next_actions": state.persistent_state.recommended_next_actions[-3:],
                    "previous_next_targets": state.persistent_state.recommended_next_targets[-3:],
                    "previous_errors": state.persistent_state.error_history[-3:],
                },
                ensure_ascii=True,
            )
            llm_json = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            if not isinstance(llm_json, dict):
                fallback_reason = "llm_no_response_or_invalid_json"
            else:
                parsed_actions = self._parse_next_actions(llm_json)
                parsed_targets = self._parse_next_targets(llm_json)
                if parsed_actions:
                    next_actions = parsed_actions[:4]
                    allowed_targets = set(state.request.targets)
                    next_targets = [target for target in parsed_targets if target in allowed_targets]
                    source = "llm"
                    fallback_reason = ""
                else:
                    fallback_reason = "llm_missing_next_actions"

        if not next_actions:
            source = "rules"
            probe_refinement = interpretation.get("probe_refinement", {})
            if isinstance(probe_refinement, dict):
                raw_followups = probe_refinement.get("followups", [])
                if isinstance(raw_followups, list) and raw_followups:
                    for item in raw_followups:
                        if not isinstance(item, dict):
                            continue
                        target = str(item.get("target") or "").strip()
                        action = str(item.get("next_action") or "").strip()
                        if not target or not action:
                            continue
                        next_actions.append(f"probe_{action}:{target}")
                    next_targets = self._rule_next_targets(state=state, interpretation=interpretation)
            bound_type = interpretation.get("bound_type")
            placeholder_count = interpretation.get("workload_placeholder_count", 0)
            if next_actions:
                pass
            elif isinstance(placeholder_count, int) and placeholder_count > 0:
                next_actions.append("Finalize workload-dependent counters as placeholders because no run command was provided")
                next_actions.append("Profile SM efficiency metrics when a workload command becomes available")
            elif bound_type == "memory_bound":
                next_actions.append("collect_ncu_memory_metrics_and_optimize_memory_access")
            elif bound_type == "compute_bound":
                next_actions.append("collect_compute_efficiency_metrics_and_optimize_instruction_mix")
            else:
                next_actions.append("improve_signal_coverage_by_using_realistic_workload_run_command")
            next_actions.append("re-run_pipeline_with_same_targets_for_reproducibility_check")
            next_targets = self._rule_next_targets(state=state, interpretation=interpretation)

        state.outputs["next_actions"] = next_actions
        state.outputs["next_targets"] = next_targets or list(state.request.targets)
        return AgentMessage(
            sender=self.name,
            recipient="coordinator",
            action="refinement_ready",
            content={
                "next_actions": next_actions,
                "next_targets": state.outputs["next_targets"],
                "source": source,
                "llm_attempted": llm_attempted,
                "refinement_fallback_reason": fallback_reason if source == "rules" else "",
            },
        )
