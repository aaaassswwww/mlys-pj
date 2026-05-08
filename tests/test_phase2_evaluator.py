from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from profiler_agent.phase2.evaluator import (
    build_candidate_artifact_paths,
    build_compile_checked_candidate_evaluator,
    build_ctypes_candidate_runner,
    build_harness_runtime_evaluator,
    build_nvcc_shared_library_command,
    build_subprocess_runtime_evaluator,
    invoke_launch_optimized_lora,
    load_compiled_candidate,
)
from profiler_agent.phase2.harness import PythonListBackend, empty_benchmark_result, lora_reference_forward
from profiler_agent.phase2.models import CandidateEvaluation, CorrectnessResult, GeneratedCandidate, LoadResult, LoraProblemSpec


class Phase2EvaluatorTests(unittest.TestCase):
    def test_invoke_launch_optimized_lora_passes_tensor_pointers(self) -> None:
        class FakeTensor:
            def __init__(self, ptr: int) -> None:
                self._ptr = ptr

            def data_ptr(self) -> int:
                return self._ptr

        class FakeSymbol:
            def __call__(self, *args):
                self.called_args = args

        symbol = FakeSymbol()
        inputs = {
            "W": FakeTensor(101),
            "X": FakeTensor(202),
            "A": FakeTensor(303),
            "B": FakeTensor(404),
        }
        output = FakeTensor(505)
        spec = LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")
        invoke_launch_optimized_lora(symbol, inputs=inputs, output=output, spec=spec, stream_ptr=77)
        self.assertEqual(symbol.called_args[0].value, 101)
        self.assertEqual(symbol.called_args[4].value, 505)
        self.assertEqual(symbol.called_args[5].value, 8)
        self.assertEqual(symbol.called_args[6].value, 3)
        self.assertEqual(symbol.called_args[7].value, 77)

    def test_build_nvcc_shared_library_command_uses_shared_output(self) -> None:
        source = Path("candidate") / "optimized_lora.cu"
        library = Path("candidate") / "optimized_lora.dll"
        with patch("profiler_agent.phase2.evaluator.resolve_command_path", return_value="C:/CUDA/bin/nvcc.exe"):
            with patch("profiler_agent.phase2.evaluator.os.name", "nt"):
                command = build_nvcc_shared_library_command(source, library)
        self.assertEqual(command[0], "C:/CUDA/bin/nvcc.exe")
        self.assertIn("-shared", command)
        self.assertEqual(command[-1], str(library))
        self.assertIn("/wd4819", command)

    def test_build_nvcc_shared_library_command_adds_fpic_on_posix(self) -> None:
        source = Path("candidate") / "optimized_lora.cu"
        library = Path("candidate") / "optimized_lora.so"
        with patch("profiler_agent.phase2.evaluator.resolve_command_path", return_value="/usr/local/cuda/bin/nvcc"):
            with patch("profiler_agent.phase2.evaluator.os.name", "posix"):
                command = build_nvcc_shared_library_command(source, library)
        self.assertEqual(command[0], "/usr/local/cuda/bin/nvcc")
        self.assertIn("-shared", command)
        self.assertIn("-fPIC", command)

    def test_compile_checked_evaluator_returns_compile_failure_when_nvcc_missing(self) -> None:
        root = Path("tests/.tmp") / f"phase2_eval_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            evaluator = build_compile_checked_candidate_evaluator(root_dir=root)
            candidate = GeneratedCandidate(candidate_id="cand-a", source_code='__global__ void k() {}\n')
            with patch("profiler_agent.phase2.evaluator.resolve_command_path", return_value=None):
                evaluation = evaluator(candidate)
            self.assertFalse(evaluation.correctness.passed)
            self.assertEqual(evaluation.notes, ["compile_failed"])
            self.assertIsNotNone(evaluation.compilation)
            self.assertEqual(evaluation.compilation.returncode, 127)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_load_compiled_candidate_reports_missing_symbol(self) -> None:
        root = Path("tests/.tmp") / f"phase2_load_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            library_path = root / "optimized_lora.dll"
            library_path.write_text("placeholder", encoding="utf-8")
            fake_handle = object()
            with patch("profiler_agent.phase2.evaluator.ctypes.CDLL", return_value=fake_handle):
                result = load_compiled_candidate(library_path)
            self.assertFalse(result.ok)
            self.assertIn("missing_required_symbol", result.error)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_compile_checked_evaluator_passes_to_runtime_evaluator(self) -> None:
        root = Path("tests/.tmp") / f"phase2_eval_runtime_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            runtime = Mock()
            runtime.return_value = CandidateEvaluation(
                candidate_id="cand-b",
                correctness=CorrectnessResult(passed=True, max_abs_err=0.0, rel_l2_err=0.0, rtol=1e-4, atol=1e-4),
                student_benchmark=empty_benchmark_result(),
                reference_benchmark=empty_benchmark_result(),
                speedup=1.0,
                notes=["runtime_connected"],
            )
            evaluator = build_compile_checked_candidate_evaluator(root_dir=root, runtime_evaluator=runtime)
            candidate = GeneratedCandidate(candidate_id="cand-b", source_code='extern "C" __global__ void k() {}\n')

            completed = subprocess.CompletedProcess(args=["nvcc"], returncode=0, stdout="", stderr="")
            fake_handle = type("FakeHandle", (), {"launch_optimized_lora": object()})()
            with patch("profiler_agent.phase2.evaluator.resolve_command_path", return_value="nvcc"):
                with patch("profiler_agent.phase2.evaluator.subprocess.run", return_value=completed):
                    with patch("profiler_agent.phase2.evaluator.load_compiled_candidate", return_value=LoadResult(ok=True, library_path="x", symbol_name="launch_optimized_lora")):
                        evaluation = evaluator(candidate)

            self.assertTrue(runtime.called)
            self.assertEqual(evaluation.notes, ["runtime_connected"])
            self.assertIsNotNone(evaluation.compilation)
            self.assertTrue(evaluation.compilation.ok)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_harness_runtime_evaluator_passes_correct_candidate_across_specs(self) -> None:
        specs = [
            LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu"),
            LoraProblemSpec(hidden_dim=10, output_dim=5, num_tokens=2, device="cpu"),
        ]

        def candidate_runner(candidate, paths, load_result, spec, inputs, backend):
            _ = candidate, paths, load_result, spec
            return lora_reference_forward(inputs, backend)

        runtime = build_harness_runtime_evaluator(
            problem_specs=specs,
            candidate_runner=candidate_runner,
            backend=PythonListBackend(),
            warmup_runs=0,
            measured_runs=2,
        )
        evaluation = runtime(
            GeneratedCandidate(candidate_id="cand-ok", source_code="// x"),
            build_candidate_artifact_paths(Path("tests/.tmp"), "cand-ok"),
            LoadResult(ok=True, library_path="x", symbol_name="launch_optimized_lora"),
        )
        self.assertTrue(evaluation.correctness.passed)
        self.assertGreaterEqual(evaluation.speedup, 0.0)
        self.assertIn("correctness_passed_all_specs", evaluation.notes)

    def test_harness_runtime_evaluator_marks_incorrect_candidate(self) -> None:
        specs = [LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")]

        def candidate_runner(candidate, paths, load_result, spec, inputs, backend):
            _ = candidate, paths, load_result, spec, backend
            ref = lora_reference_forward(inputs, PythonListBackend())
            broken = [row[:] for row in ref]
            broken[0][0] += 1.0
            return broken

        runtime = build_harness_runtime_evaluator(
            problem_specs=specs,
            candidate_runner=candidate_runner,
            backend=PythonListBackend(),
            warmup_runs=0,
            measured_runs=2,
        )
        evaluation = runtime(
            GeneratedCandidate(candidate_id="cand-bad", source_code="// x"),
            build_candidate_artifact_paths(Path("tests/.tmp"), "cand-bad"),
            LoadResult(ok=True, library_path="x", symbol_name="launch_optimized_lora"),
        )
        self.assertFalse(evaluation.correctness.passed)
        self.assertEqual(evaluation.speedup, 0.0)
        self.assertIn("correctness_not_yet_passing", evaluation.notes)
        self.assertTrue(any(note.startswith("reference_diagnosis:hidden_dim=8:") for note in evaluation.notes))

    def test_harness_runtime_evaluator_reports_candidate_runner_error(self) -> None:
        specs = [LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")]

        def candidate_runner(candidate, paths, load_result, spec, inputs, backend):
            _ = candidate, paths, load_result, spec, inputs, backend
            raise RuntimeError("boom")

        runtime = build_harness_runtime_evaluator(
            problem_specs=specs,
            candidate_runner=candidate_runner,
            backend=PythonListBackend(),
            warmup_runs=0,
            measured_runs=2,
        )
        evaluation = runtime(
            GeneratedCandidate(candidate_id="cand-err", source_code="// x"),
            build_candidate_artifact_paths(Path("tests/.tmp"), "cand-err"),
            LoadResult(ok=True, library_path="x", symbol_name="launch_optimized_lora"),
        )
        self.assertFalse(evaluation.correctness.passed)
        self.assertIn("candidate_runner_error:hidden_dim=8:RuntimeError", evaluation.notes)

    def test_harness_runtime_evaluator_converts_fatal_cuda_error_to_failure_result(self) -> None:
        specs = [LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")]

        def candidate_runner(candidate, paths, load_result, spec, inputs, backend):
            _ = candidate, paths, load_result, spec, inputs, backend
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        runtime = build_harness_runtime_evaluator(
            problem_specs=specs,
            candidate_runner=candidate_runner,
            backend=PythonListBackend(),
            warmup_runs=0,
            measured_runs=2,
        )
        evaluation = runtime(
            GeneratedCandidate(candidate_id="cand-fatal", source_code="// x"),
            build_candidate_artifact_paths(Path("tests/.tmp"), "cand-fatal"),
            LoadResult(ok=True, library_path="x", symbol_name="launch_optimized_lora"),
        )
        self.assertFalse(evaluation.correctness.passed)
        self.assertIn("fatal_cuda_runtime_error:hidden_dim=8:RuntimeError", evaluation.notes)

    def test_build_ctypes_candidate_runner_invokes_exported_symbol(self) -> None:
        class FakeTensor:
            def __init__(self, ptr: int) -> None:
                self._ptr = ptr

            def data_ptr(self) -> int:
                return self._ptr

        class FakeBackend:
            def __init__(self) -> None:
                self.synchronized = False

            def synchronize(self) -> None:
                self.synchronized = True

        class FakeSymbol:
            def __call__(self, *args):
                self.called_args = args

        fake_symbol = FakeSymbol()
        fake_handle = type("FakeHandle", (), {"launch_optimized_lora": fake_symbol})()
        runner = build_ctypes_candidate_runner(
            output_allocator=lambda spec, inputs: FakeTensor(999),
            stream_resolver=lambda inputs: 1234,
        )
        inputs = {
            "W": FakeTensor(11),
            "X": FakeTensor(22),
            "A": FakeTensor(33),
            "B": FakeTensor(44),
        }
        backend = FakeBackend()
        with patch("profiler_agent.phase2.evaluator.ctypes.CDLL", return_value=fake_handle):
            output = runner(
                GeneratedCandidate(candidate_id="cand-ctypes", source_code="// x"),
                build_candidate_artifact_paths(Path("tests/.tmp"), "cand-ctypes"),
                LoadResult(ok=True, library_path="fake.dll", symbol_name="launch_optimized_lora"),
                LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu"),
                inputs,
                backend,
            )
        self.assertEqual(output.data_ptr(), 999)
        self.assertTrue(backend.synchronized)
        self.assertEqual(fake_symbol.called_args[0].value, 11)
        self.assertEqual(fake_symbol.called_args[7].value, 1234)

    def test_subprocess_runtime_evaluator_reads_child_response(self) -> None:
        root = Path("tests/.tmp") / f"phase2_subproc_eval_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            paths = build_candidate_artifact_paths(root, "cand-subproc")
            paths.source_path.write_text("// x\n", encoding="utf-8")
            paths.library_path.write_text("fake-lib", encoding="utf-8")
            runtime = build_subprocess_runtime_evaluator(
                root_dir=root,
                problem_specs=[LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")],
                warmup_runs=0,
                measured_runs=2,
            )
            seen_command: list[str] = []

            def fake_run(command, capture_output, text, check, timeout, cwd, env):
                _ = capture_output, text, check, timeout, cwd, env
                seen_command[:] = command
                response_path = paths.source_path.parent / "runtime_eval_response.json"
                response_path.write_text(
                    '{"candidate_id":"cand-subproc","correctness":{"passed":true,"max_abs_err":0.0,"rel_l2_err":0.0,"rtol":0.0001,"atol":0.0001},"student_benchmark":{"warmup_runs":0,"measured_runs":2,"median_runtime_ms":1.0,"min_runtime_ms":1.0,"max_runtime_ms":1.0,"all_runtime_ms":[1.0,1.0]},"reference_benchmark":{"warmup_runs":0,"measured_runs":2,"median_runtime_ms":2.0,"min_runtime_ms":2.0,"max_runtime_ms":2.0,"all_runtime_ms":[2.0,2.0]},"speedup":2.0,"notes":["child_ok"]}',
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")

            with patch("profiler_agent.phase2.evaluator.subprocess.run", side_effect=fake_run):
                evaluation = runtime(
                    GeneratedCandidate(candidate_id="cand-subproc", source_code="// x"),
                    paths,
                    LoadResult(ok=True, library_path=str(paths.library_path), symbol_name="launch_optimized_lora"),
                )
            self.assertTrue(evaluation.correctness.passed)
            self.assertEqual(evaluation.speedup, 2.0)
            self.assertEqual(evaluation.notes, ["child_ok"])
            self.assertEqual(seen_command[1], "-c")
            self.assertIn("runtime_eval_worker", seen_command[2])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_subprocess_runtime_evaluator_returns_failure_on_child_crash(self) -> None:
        root = Path("tests/.tmp") / f"phase2_subproc_fail_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            paths = build_candidate_artifact_paths(root, "cand-subproc-fail")
            paths.source_path.write_text("// x\n", encoding="utf-8")
            paths.library_path.write_text("fake-lib", encoding="utf-8")
            runtime = build_subprocess_runtime_evaluator(
                root_dir=root,
                problem_specs=[LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")],
                warmup_runs=0,
                measured_runs=2,
            )
            completed = subprocess.CompletedProcess(args=["python"], returncode=7, stdout="", stderr="child boom")
            with patch("profiler_agent.phase2.evaluator.subprocess.run", return_value=completed):
                evaluation = runtime(
                    GeneratedCandidate(candidate_id="cand-subproc-fail", source_code="// x"),
                    paths,
                    LoadResult(ok=True, library_path=str(paths.library_path), symbol_name="launch_optimized_lora"),
                )
            self.assertFalse(evaluation.correctness.passed)
            self.assertIn("runtime_subprocess_failed:returncode=7", evaluation.notes)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_subprocess_runtime_evaluator_raises_on_child_startup_failure(self) -> None:
        root = Path("tests/.tmp") / f"phase2_subproc_startup_fail_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            paths = build_candidate_artifact_paths(root, "cand-subproc-startup-fail")
            paths.source_path.write_text("// x\n", encoding="utf-8")
            paths.library_path.write_text("fake-lib", encoding="utf-8")
            runtime = build_subprocess_runtime_evaluator(
                root_dir=root,
                problem_specs=[LoraProblemSpec(hidden_dim=8, output_dim=4, num_tokens=3, device="cpu")],
                warmup_runs=0,
                measured_runs=2,
            )
            completed = subprocess.CompletedProcess(
                args=["python"],
                returncode=1,
                stdout="",
                stderr="/usr/bin/python3: can't open file '/workspace/profiler_agent/phase2/runtime_eval_worker.py': [Errno 2] No such file or directory\n",
            )
            with patch("profiler_agent.phase2.evaluator.subprocess.run", return_value=completed):
                with self.assertRaises(RuntimeError) as ctx:
                    runtime(
                        GeneratedCandidate(candidate_id="cand-subproc-startup-fail", source_code="// x"),
                        paths,
                        LoadResult(ok=True, library_path=str(paths.library_path), symbol_name="launch_optimized_lora"),
                    )
            self.assertIn("runtime_subprocess_startup_failed", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
