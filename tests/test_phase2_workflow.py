from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from profiler_agent.phase2.harness import PythonListBackend, lora_reference_forward
from profiler_agent.phase2.models import CompilationResult, LoadResult
from profiler_agent.phase2.workflow import default_problem_specs, run_default_phase2_workflow


class Phase2WorkflowTests(unittest.TestCase):
    def test_default_problem_specs_cover_expected_hidden_dims(self) -> None:
        specs = default_problem_specs(device="cpu")
        self.assertEqual([spec.hidden_dim for spec in specs], [3584, 4096, 4608])
        self.assertTrue(all(spec.low_rank == 16 for spec in specs))

    def test_run_default_phase2_workflow_chains_generator_evaluator_optimizer(self) -> None:
        root = Path("tests/.tmp") / f"phase2_workflow_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            mock_llm = Mock()
            mock_llm.is_enabled.return_value = False

            def candidate_runner(candidate, paths, load_result, spec, inputs, backend):
                _ = candidate, paths, load_result, spec
                return lora_reference_forward(inputs, backend)

            def fake_compile(source_path, library_path):
                library_path.write_text("fake-binary", encoding="utf-8")
                return CompilationResult(
                    ok=True,
                    command=["nvcc", str(source_path), "-shared", "-o", str(library_path)],
                    returncode=0,
                    stdout_tail="compiled",
                    stderr_tail="",
                    output_path=str(library_path),
                )

            with patch("profiler_agent.phase2.evaluator.compile_candidate_source", side_effect=fake_compile):
                with patch(
                    "profiler_agent.phase2.evaluator.load_compiled_candidate",
                    return_value=LoadResult(ok=True, library_path="fake.dll", symbol_name="launch_optimized_lora"),
                ):
                    result = run_default_phase2_workflow(
                        root_dir=root,
                        max_iterations=2,
                        llm_client=mock_llm,
                        problem_specs=default_problem_specs(hidden_dims=[8, 12], num_tokens=3, device="cpu"),
                        candidate_runner=candidate_runner,
                        backend=PythonListBackend(),
                        warmup_runs=0,
                        measured_runs=2,
                    )

            self.assertTrue((root / "optimized_lora.cu").exists())
            self.assertTrue((root / ".agent_artifacts" / "phase2_state.json").exists())
            self.assertTrue((root / ".agent_artifacts" / "phase2_report.json").exists())
            self.assertIsNotNone(result.best_candidate_id)
            self.assertEqual(result.iterations_run, 2)
            self.assertGreaterEqual(len(result.state.candidate_history), 2)
            state_json = json.loads((root / ".agent_artifacts" / "phase2_state.json").read_text(encoding="utf-8"))
            report_json = json.loads((root / ".agent_artifacts" / "phase2_report.json").read_text(encoding="utf-8"))
            self.assertEqual(state_json["iteration"], 2)
            self.assertTrue(state_json["done"])
            self.assertGreaterEqual(report_json["candidate_history_count"], 2)
            self.assertIn("recent_candidates", report_json)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
