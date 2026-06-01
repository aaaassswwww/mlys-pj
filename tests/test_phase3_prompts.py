from __future__ import annotations

import json
import unittest

from profiler_agent.phase3.candidate_store import build_candidate_feedback
from profiler_agent.phase3.prompts import build_phase3_generation_user_prompt
from profiler_agent.phase3.models import Phase3OptimizerState


class Phase3PromptTests(unittest.TestCase):
    def test_prompt_includes_performance_focus_breakdown(self) -> None:
        feedback = build_candidate_feedback(
            correctness={"passed": True, "rel_l2_err": 0.0, "max_abs_err": 0.0, "checked_cases": 3},
            benchmark={
                "prefill_tokens_per_s": 100.0,
                "decode_tokens_per_s": 40.0,
                "mixed_tokens_per_s": 55.0,
                "aggregate_tokens_per_s": 65.0,
            },
            baseline_benchmark={
                "prefill_tokens_per_s": 80.0,
                "decode_tokens_per_s": 80.0,
                "mixed_tokens_per_s": 80.0,
                "aggregate_tokens_per_s": 80.0,
            },
            speedup=0.8125,
            notes=["correctness_passed_all_specs"],
            previous_candidate={
                "candidate_id": "runtime-a-v01",
                "rationale": "baseline",
                "source": "llm_runtime_revision",
                "source_code": "def create_engine(model_config, weight_dir, device='cuda'):\n    pass\n",
            },
        )
        state = Phase3OptimizerState(
            iteration=2,
            current_best_candidate_id="runtime-a-v01",
            current_best_correct_candidate_id="runtime-a-v01",
            current_best_source_code="def create_engine(model_config, weight_dir, device='cuda'):\n    pass\n",
            current_best_rationale="baseline",
            current_best_source="llm_runtime_revision",
            best_speedup=0.8125,
        )

        prompt = build_phase3_generation_user_prompt(
            state=state,
            iteration=2,
            feedback=feedback,
            bootstrap_source_code="def create_engine(model_config, weight_dir, device='cuda'):\n    pass\n",
        )
        payload = json.loads(prompt)

        self.assertEqual(payload["performance_feedback"]["weakest_metric"], "decode_speedup")
        self.assertEqual(payload["performance_context"]["current_iteration_focus"], "decode_speedup")
        self.assertIn("cache-length grouping", payload["performance_context"]["suggested_focus_lenses"])
        self.assertTrue(payload["performance_context"]["single_focus_mode"]["enabled"])
        self.assertEqual(payload["performance_context"]["single_focus_mode"]["primary_target_metric"], "decode_speedup")
        self.assertIn(
            "single-target optimization pass",
            payload["performance_context"]["single_focus_mode"]["instruction"],
        )
        self.assertEqual(
            payload["iteration_policy"]["single_target_runtime_optimization"]["primary_target_metric"],
            "decode_speedup",
        )


if __name__ == "__main__":
    unittest.main()
