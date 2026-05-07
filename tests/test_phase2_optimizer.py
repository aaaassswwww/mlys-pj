from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.phase2.generator import LoraCandidateGenerator
from profiler_agent.phase2.harness import BenchmarkResult, CorrectnessResult
from profiler_agent.phase2.models import CandidateEvaluation, Phase2OptimizerState
from profiler_agent.phase2.optimizer import GeneratedCandidate, run_phase2_optimization


class Phase2OptimizerTests(unittest.TestCase):
    def test_optimizer_promotes_best_correct_candidate(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            generated = [
                GeneratedCandidate(candidate_id="cand-a", source_code="// a", rationale="first"),
                GeneratedCandidate(candidate_id="cand-b", source_code="// b", rationale="second"),
            ]

            def generator(state: Phase2OptimizerState, feedback):
                _ = feedback
                return generated[state.iteration - 1]

            def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
                if candidate.candidate_id == "cand-a":
                    speedup = 1.1
                else:
                    speedup = 1.6
                correctness = CorrectnessResult(passed=True, max_abs_err=0.0, rel_l2_err=0.0, rtol=1e-4, atol=1e-4)
                bench_student = BenchmarkResult(
                    warmup_runs=1,
                    measured_runs=3,
                    median_runtime_ms=10.0 / speedup,
                    min_runtime_ms=9.0 / speedup,
                    max_runtime_ms=11.0 / speedup,
                    all_runtime_ms=[10.0 / speedup] * 3,
                )
                bench_ref = BenchmarkResult(
                    warmup_runs=1,
                    measured_runs=3,
                    median_runtime_ms=10.0,
                    min_runtime_ms=9.0,
                    max_runtime_ms=11.0,
                    all_runtime_ms=[10.0] * 3,
                )
                return CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    correctness=correctness,
                    student_benchmark=bench_student,
                    reference_benchmark=bench_ref,
                    speedup=speedup,
                )

            result = run_phase2_optimization(
                root_dir=root,
                max_iterations=2,
                candidate_generator=generator,
                candidate_evaluator=evaluator,
            )
            self.assertEqual(result.best_candidate_id, "cand-b")
            self.assertTrue((root / "optimized_lora.cu").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_optimizer_records_correctness_failures(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_fail_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)

        def generator(state: Phase2OptimizerState, feedback):
            _ = state, feedback
            return GeneratedCandidate(candidate_id="bad", source_code="// bad")

        def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
            correctness = CorrectnessResult(passed=False, max_abs_err=0.3, rel_l2_err=0.2, rtol=1e-4, atol=1e-4)
            bench = BenchmarkResult(
                warmup_runs=1,
                measured_runs=1,
                median_runtime_ms=1.0,
                min_runtime_ms=1.0,
                max_runtime_ms=1.0,
                all_runtime_ms=[1.0],
            )
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                correctness=correctness,
                student_benchmark=bench,
                reference_benchmark=bench,
                speedup=1.0,
            )

        try:
            result = run_phase2_optimization(
                root_dir=root,
                max_iterations=1,
                candidate_generator=generator,
                candidate_evaluator=evaluator,
                bootstrap_candidate=LoraCandidateGenerator(llm_client=None).bootstrap_candidate(),
            )
            self.assertEqual(len(result.state.correctness_failures), 1)
            self.assertTrue((root / "optimized_lora.cu").exists())
            self.assertTrue((root / ".agent_artifacts" / "phase2_report.json").exists())
            state_json = json.loads((root / ".agent_artifacts" / "phase2_state.json").read_text(encoding="utf-8"))
            report_json = json.loads((root / ".agent_artifacts" / "phase2_report.json").read_text(encoding="utf-8"))
            self.assertEqual(state_json["iteration"], 1)
            self.assertEqual(len(state_json["candidate_history"]), 1)
            self.assertEqual(len(state_json["correctness_failures"]), 1)
            self.assertTrue(state_json["done"])
            self.assertEqual(report_json["correctness_failures_count"], 1)
            self.assertEqual(report_json["candidate_history_count"], 1)
            self.assertEqual(len(report_json["recent_candidates"]), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_optimizer_persists_partial_state_when_evaluator_raises(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_exception_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)

        def generator(state: Phase2OptimizerState, feedback):
            _ = state, feedback
            return GeneratedCandidate(candidate_id="boom", source_code="// boom", rationale="broken")

        def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
            _ = candidate
            raise RuntimeError("simulated_evaluator_failure")

        try:
            with self.assertRaises(RuntimeError):
                run_phase2_optimization(
                    root_dir=root,
                    max_iterations=1,
                    candidate_generator=generator,
                    candidate_evaluator=evaluator,
                    bootstrap_candidate=LoraCandidateGenerator(llm_client=None).bootstrap_candidate(),
                )

            state_path = root / ".agent_artifacts" / "phase2_state.json"
            report_path = root / ".agent_artifacts" / "phase2_report.json"
            self.assertTrue(state_path.exists())
            self.assertTrue(report_path.exists())

            state_json = json.loads(state_path.read_text(encoding="utf-8"))
            report_json = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(state_json["iteration"], 1)
            self.assertFalse(state_json["done"])
            self.assertEqual(report_json["iterations_run"], 1)
            self.assertEqual(report_json["candidate_history_count"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_optimizer_uses_time_budget_when_iteration_limit_is_unbounded(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_budget_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            generated_ids: list[str] = []

            def generator(state: Phase2OptimizerState, feedback):
                _ = feedback
                candidate_id = f"cand-{state.iteration}"
                generated_ids.append(candidate_id)
                return GeneratedCandidate(candidate_id=candidate_id, source_code=f"// {candidate_id}")

            def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
                correctness = CorrectnessResult(passed=False, max_abs_err=1.0, rel_l2_err=1.0, rtol=1e-4, atol=1e-4)
                bench = BenchmarkResult(
                    warmup_runs=0,
                    measured_runs=0,
                    median_runtime_ms=0.0,
                    min_runtime_ms=0.0,
                    max_runtime_ms=0.0,
                    all_runtime_ms=[],
                )
                return CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    correctness=correctness,
                    student_benchmark=bench,
                    reference_benchmark=bench,
                    speedup=0.0,
                )

            budget_states = [
                {"enabled": True, "expired": False, "remaining_seconds": 500.0},
                {"enabled": True, "expired": False, "remaining_seconds": 500.0},
                {"enabled": True, "expired": False, "remaining_seconds": 500.0},
                {"enabled": True, "expired": False, "remaining_seconds": 10.0},
            ]

            def fake_budget_status():
                if budget_states:
                    return budget_states.pop(0)
                return {"enabled": True, "expired": True, "remaining_seconds": 0.0}

            with patch("profiler_agent.phase2.optimizer.get_runtime_budget_status", side_effect=fake_budget_status):
                with patch.dict("os.environ", {"PROFILER_AGENT_PHASE2_STOP_BUFFER_SECONDS": "30"}, clear=False):
                    result = run_phase2_optimization(
                        root_dir=root,
                        max_iterations=None,
                        candidate_generator=generator,
                        candidate_evaluator=evaluator,
                    )

            self.assertEqual(result.iterations_run, 2)
            self.assertEqual(result.state.stop_reason, "remaining_runtime_below_stop_buffer:10.000s<=30.000s")
            self.assertEqual(generated_ids, ["cand-1", "cand-2"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_optimizer_stops_early_on_fatal_cuda_runtime_error_note(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_fatal_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            generated_ids: list[str] = []

            def generator(state: Phase2OptimizerState, feedback):
                _ = feedback
                candidate_id = f"cand-{state.iteration}"
                generated_ids.append(candidate_id)
                return GeneratedCandidate(candidate_id=candidate_id, source_code=f"// {candidate_id}")

            def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
                correctness = CorrectnessResult(passed=False, max_abs_err=float("inf"), rel_l2_err=float("inf"), rtol=1e-4, atol=1e-4)
                bench = BenchmarkResult(
                    warmup_runs=0,
                    measured_runs=0,
                    median_runtime_ms=0.0,
                    min_runtime_ms=0.0,
                    max_runtime_ms=0.0,
                    all_runtime_ms=[],
                )
                return CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    correctness=correctness,
                    student_benchmark=bench,
                    reference_benchmark=bench,
                    speedup=0.0,
                    notes=["fatal_cuda_runtime_error:hidden_dim=3584:RuntimeError"],
                )

            result = run_phase2_optimization(
                root_dir=root,
                max_iterations=5,
                candidate_generator=generator,
                candidate_evaluator=evaluator,
            )

            self.assertEqual(result.iterations_run, 1)
            self.assertEqual(result.state.stop_reason, "fatal_cuda_runtime_error:hidden_dim=3584:RuntimeError")
            self.assertEqual(generated_ids, ["cand-1"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_optimizer_persists_partial_state_when_runtime_subprocess_startup_fails(self) -> None:
        root = Path("tests/.tmp") / f"phase2_opt_subproc_startup_fail_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)

        def generator(state: Phase2OptimizerState, feedback):
            _ = state, feedback
            return GeneratedCandidate(candidate_id="cand-startup-fail", source_code="// broken")

        def evaluator(candidate: GeneratedCandidate) -> CandidateEvaluation:
            _ = candidate
            raise RuntimeError("runtime_subprocess_startup_failed:returncode=1")

        try:
            with self.assertRaises(RuntimeError):
                run_phase2_optimization(
                    root_dir=root,
                    max_iterations=2,
                    candidate_generator=generator,
                    candidate_evaluator=evaluator,
                    bootstrap_candidate=LoraCandidateGenerator(llm_client=None).bootstrap_candidate(),
                )

            report_path = root / ".agent_artifacts" / "phase2_report.json"
            state_path = root / ".agent_artifacts" / "phase2_state.json"
            self.assertTrue(report_path.exists())
            self.assertTrue(state_path.exists())
            state_json = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_json["iteration"], 1)
            self.assertFalse(state_json["done"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
