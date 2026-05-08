from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from profiler_agent.phase2.candidate_store import record_candidate_evaluation, write_best_candidate
from profiler_agent.phase2.harness import (
    PythonListBackend,
    benchmark_callable,
    build_reference_diagnosis,
    check_correctness,
    compute_speedup,
    generate_lora_inputs,
    lora_reference_forward,
)
from profiler_agent.phase2.models import CandidateEvaluation, LoraProblemSpec, Phase2OptimizerState


class Phase2HarnessTests(unittest.TestCase):
    def test_lora_reference_forward_matches_manual_formula(self) -> None:
        backend = PythonListBackend()
        inputs = {
            "W": [[1.0, 2.0], [3.0, 4.0]],
            "X": [[1.0, 0.0], [0.0, 1.0]],
            "A": [[1.0], [2.0]],
            "B": [[5.0], [6.0]],
        }
        out = lora_reference_forward(inputs, backend=backend)
        self.assertEqual(out, [[6.0, 8.0], [13.0, 16.0]])

    def test_check_correctness_reports_error_metrics(self) -> None:
        backend = PythonListBackend()
        ref = [[1.0, 2.0], [3.0, 4.0]]
        student = [[1.0, 2.0], [3.0, 4.1]]
        result = check_correctness(student, ref, backend=backend, atol=1e-5, rtol=1e-5)
        self.assertFalse(result.passed)
        self.assertGreater(result.max_abs_err, 0.0)
        self.assertGreater(result.rel_l2_err, 0.0)

    def test_benchmark_callable_and_speedup(self) -> None:
        times = iter([0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.006, 0.009])

        def timer():
            return next(times)

        bench_fast = benchmark_callable(lambda: None, backend=PythonListBackend(), warmup_runs=1, measured_runs=3, timer_fn=timer)

        times2 = iter([0.0, 0.002, 0.003, 0.006, 0.007, 0.011, 0.012, 0.017])

        def timer2():
            return next(times2)

        bench_slow = benchmark_callable(lambda: None, backend=PythonListBackend(), warmup_runs=1, measured_runs=3, timer_fn=timer2)
        speedup = compute_speedup(bench_slow, bench_fast)
        self.assertGreater(speedup, 1.0)

    def test_generate_lora_inputs_shapes(self) -> None:
        spec = LoraProblemSpec(hidden_dim=8, low_rank=2, output_dim=4, num_tokens=3, device="cpu")
        inputs = generate_lora_inputs(spec, backend=PythonListBackend())
        self.assertEqual(len(inputs["W"]), 4)
        self.assertEqual(len(inputs["W"][0]), 8)
        self.assertEqual(len(inputs["X"]), 8)
        self.assertEqual(len(inputs["X"][0]), 3)

    def test_candidate_store_promotes_best_correct_candidate(self) -> None:
        state = Phase2OptimizerState()
        bench_fast = benchmark_callable(lambda: None, backend=PythonListBackend(), warmup_runs=0, measured_runs=1, timer_fn=iter([0.0, 0.001]).__next__)
        bench_ref = benchmark_callable(lambda: None, backend=PythonListBackend(), warmup_runs=0, measured_runs=1, timer_fn=iter([0.0, 0.002]).__next__)
        evaluation = CandidateEvaluation(
            candidate_id="cand-1",
            correctness=check_correctness([[1.0]], [[1.0]], backend=PythonListBackend()),
            student_benchmark=bench_fast,
            reference_benchmark=bench_ref,
            speedup=compute_speedup(bench_ref, bench_fast),
        )
        promoted = record_candidate_evaluation(
            state,
            candidate_id="cand-1",
            source_code="extern \"C\" __global__ void k(){}",
            evaluation=evaluation,
        )
        self.assertTrue(promoted)
        self.assertEqual(state.current_best_candidate_id, "cand-1")
        self.assertEqual(state.current_best_correct_candidate_id, "cand-1")

    def test_candidate_store_promotes_best_incorrect_candidate_until_correct_one_exists(self) -> None:
        state = Phase2OptimizerState()
        bench = benchmark_callable(lambda: None, backend=PythonListBackend(), warmup_runs=0, measured_runs=1, timer_fn=iter([0.0, 0.001]).__next__)

        first = CandidateEvaluation(
            candidate_id="cand-a",
            correctness=check_correctness([[1.1]], [[1.0]], backend=PythonListBackend(), atol=1e-6, rtol=1e-6),
            student_benchmark=bench,
            reference_benchmark=bench,
            speedup=0.0,
        )
        second = CandidateEvaluation(
            candidate_id="cand-b",
            correctness=check_correctness([[1.01]], [[1.0]], backend=PythonListBackend(), atol=1e-6, rtol=1e-6),
            student_benchmark=bench,
            reference_benchmark=bench,
            speedup=0.0,
        )

        promoted_first = record_candidate_evaluation(
            state,
            candidate_id="cand-a",
            source_code="// cand-a",
            evaluation=first,
        )
        promoted_second = record_candidate_evaluation(
            state,
            candidate_id="cand-b",
            source_code="// cand-b",
            evaluation=second,
        )

        self.assertTrue(promoted_first)
        self.assertTrue(promoted_second)
        self.assertEqual(state.current_best_candidate_id, "cand-b")
        self.assertIsNone(state.current_best_correct_candidate_id)

    def test_write_best_candidate_persists_source_and_state(self) -> None:
        root = Path("tests/.tmp") / f"phase2_store_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            state = Phase2OptimizerState(current_best_candidate_id="cand-1", best_speedup=1.5)
            out = write_best_candidate(root, source_code="// cuda", state=state)
            self.assertTrue(out.exists())
            self.assertTrue((root / ".agent_artifacts" / "phase2_state.json").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_build_reference_diagnosis_reports_relative_gaps(self) -> None:
        inputs = {
            "W": [[1.0, 2.0], [3.0, 4.0]],
            "X": [[1.0, 0.0], [0.0, 1.0]],
            "A": [[1.0], [2.0]],
            "B": [[5.0], [6.0]],
        }
        reference = lora_reference_forward(inputs, backend=PythonListBackend())
        student = [[6.1, 8.0], [13.0, 16.0]]
        diagnosis = build_reference_diagnosis(student, reference, inputs, max_rows=2, max_cols=2)
        self.assertIn("student_vs_reference_rel_l2_err", diagnosis)
        self.assertIn("student_vs_naive_rel_l2_err", diagnosis)
        self.assertIn("naive_vs_reference_rel_l2_err", diagnosis)


if __name__ == "__main__":
    unittest.main()
