from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from profiler_agent.multi_agent.coordinator import MultiAgentCoordinator
from profiler_agent.multi_agent.interpreter import InterpreterAgent
from profiler_agent.multi_agent.models import MultiAgentRequest, MultiAgentState


class ProbeRefinementMultiAgentTests(unittest.TestCase):
    def test_interpreter_extracts_probe_followup_from_evidence(self) -> None:
        out_dir = Path("tests/.tmp") / f"probe_refine_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            evidence_path = out_dir / "evidence.json"
            analysis_path = out_dir / "analysis.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "dram_latency_cycles": {
                                "measurement_mode": "synthetic_intrinsic_probe",
                                "probe_iteration": {
                                    "final_decision": "add_ncu_profile",
                                    "analysis": {
                                        "next_action": "add_ncu_profile",
                                        "reason": "weak_parse_signal",
                                    },
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            analysis_path.write_text(
                json.dumps({"bound_type": "unknown", "confidence": 0.2, "bottlenecks": []}),
                encoding="utf-8",
            )
            state = MultiAgentState(
                request=MultiAgentRequest(targets=["dram_latency_cycles"], run="", out_dir=out_dir),
                outputs={
                    "pipeline": {
                        "evidence_path": str(evidence_path),
                        "analysis_path": str(analysis_path),
                        "targets": ["dram_latency_cycles"],
                    }
                },
            )
            agent = InterpreterAgent()
            agent.summarize_outputs(state)
            msg = agent.propose_next_actions(state)
            self.assertEqual(state.outputs["next_targets"], ["dram_latency_cycles"])
            self.assertIn("probe_add_ncu_profile:dram_latency_cycles", state.outputs["next_actions"])
            self.assertEqual(msg.content["next_targets"], ["dram_latency_cycles"])
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_coordinator_builds_round_directive_for_probe_followup(self) -> None:
        state = MultiAgentState(
            request=MultiAgentRequest(targets=["dram_latency_cycles", "actual_boost_clock_mhz"], run="", out_dir=Path("tests/.tmp")),
            outputs={
                "next_actions": ["probe_add_ncu_profile:dram_latency_cycles"],
                "next_targets": ["dram_latency_cycles"],
                "tool_calls": {},
            },
        )
        directive = MultiAgentCoordinator._build_round_directive(state)
        self.assertEqual(directive["focus_targets"], ["dram_latency_cycles"])
        self.assertIn("microbench", directive["forced_tools"])
        self.assertIn("ncu", directive["forced_tools"])
        self.assertIn("probe_refinement_requested_microbench_rerun", directive["reasons"])
        self.assertIn("probe_refinement_requested_ncu_profile", directive["reasons"])


if __name__ == "__main__":
    unittest.main()
