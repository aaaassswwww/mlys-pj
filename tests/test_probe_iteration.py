from __future__ import annotations

import unittest
from unittest.mock import patch

from profiler_agent.probe_iteration import run_probe_iteration
from profiler_agent.tool_adapters.microbench_adapter import ProbeResult


def _probe_result(
    *,
    value: float | None,
    source: str,
    compile_returncode: int = 0,
    run_returncode: int = 0,
    generation_attempts: int = 1,
    sample_count: int | None = None,
    std_value: float | None = None,
) -> ProbeResult:
    return ProbeResult(
        value=value,
        source=source,
        compile_returncode=compile_returncode,
        run_returncode=run_returncode,
        compile_stdout_tail="",
        compile_stderr_tail="compile_error" if compile_returncode else "",
        run_stdout_tail="",
        run_stderr_tail="run_error" if run_returncode else "",
        compile_command=["nvcc", "probe.cu"],
        run_command=["probe.exe"],
        parsed_from="structured_metric_value" if value is not None else "none",
        metric_name="dram_latency_cycles",
        sample_count=sample_count,
        best_value=value,
        median_value=value,
        std_value=std_value,
        run_values=[value] * sample_count if value is not None and sample_count else None,
        source_path="outputs/generated_probes/dram_latency_cycles/probe.cu",
        generation_source="llm_generated",
        generation_attempts=generation_attempts,
        generation_error="compile_error" if compile_returncode else "",
        generation_trace=[],
        profile_source="skipped_not_requested",
        profile_returncode=0,
        profile_parse_mode="none",
        profile_command=[],
        profile_stdout_tail="",
        profile_stderr_tail="",
    )


class ProbeIterationTests(unittest.TestCase):
    @patch("profiler_agent.probe_iteration.measure_metric_with_evidence")
    def test_probe_iteration_retries_after_compile_failure(self, mock_measure: unittest.mock.Mock) -> None:
        mock_measure.side_effect = [
            _probe_result(value=None, source="compile_failed", compile_returncode=1),
            _probe_result(value=420.0, source="microbench_probe", sample_count=5, std_value=3.0),
        ]

        result = run_probe_iteration(target="dram_latency_cycles", max_probe_iterations=2)
        self.assertEqual(result.value, 420.0)
        self.assertEqual(result.state.iteration, 2)
        self.assertTrue(result.state.done)
        self.assertEqual(result.evidence["probe_iteration"]["final_decision"], "accept_measurement")

    @patch("profiler_agent.probe_iteration.measure_metric_with_evidence")
    def test_probe_iteration_records_change_probe_shape_when_signal_missing(
        self,
        mock_measure: unittest.mock.Mock,
    ) -> None:
        mock_measure.return_value = _probe_result(value=None, source="llm_generation_failed", compile_returncode=0)

        result = run_probe_iteration(target="dram_latency_cycles", max_probe_iterations=1)
        self.assertIsNone(result.value)
        self.assertEqual(result.evidence["probe_iteration"]["final_decision"], "change_probe_shape")
        self.assertEqual(result.evidence["measurement_mode"], "synthetic_intrinsic_probe")

    @patch("profiler_agent.probe_iteration.measure_metric_with_evidence")
    def test_probe_iteration_can_request_ncu_profile_on_weak_parse_signal(
        self,
        mock_measure: unittest.mock.Mock,
    ) -> None:
        weak = _probe_result(value=420.0, source="microbench_probe", sample_count=5, std_value=2.0)
        weak = ProbeResult(**{**weak.__dict__, "parsed_from": "stdout_last_numeric"})
        strong = ProbeResult(
            **{
                **weak.__dict__,
                "parsed_from": "structured_metric_value",
                "profile_source": "ncu_csv",
                "profile_returncode": 0,
                "profile_parse_mode": "csv_metric_row",
            }
        )
        mock_measure.side_effect = [weak, strong]

        result = run_probe_iteration(target="dram_latency_cycles", max_probe_iterations=2)
        self.assertEqual(result.state.iteration, 2)
        self.assertEqual(result.state.analysis_history[0]["next_action"], "add_ncu_profile")
        self.assertEqual(result.state.profile_history[1]["source"], "ncu_csv")


if __name__ == "__main__":
    unittest.main()
