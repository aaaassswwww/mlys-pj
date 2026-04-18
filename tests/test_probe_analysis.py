from __future__ import annotations

import unittest

from profiler_agent.probe_analysis import analyze_probe_round
from profiler_agent.tool_adapters.microbench_adapter import ProbeResult


def _probe(
    *,
    value: float | None,
    source: str = "microbench_probe",
    parsed_from: str = "structured_metric_value",
    sample_count: int | None = 5,
    std_value: float | None = 2.0,
    profile_source: str | None = "skipped_not_requested",
) -> ProbeResult:
    return ProbeResult(
        value=value,
        source=source,
        compile_returncode=0,
        run_returncode=0,
        compile_stdout_tail="",
        compile_stderr_tail="",
        run_stdout_tail="",
        run_stderr_tail="",
        compile_command=["nvcc", "probe.cu"],
        run_command=["probe.exe"],
        parsed_from=parsed_from,
        metric_name="dram_latency_cycles",
        sample_count=sample_count,
        best_value=value,
        median_value=value,
        std_value=std_value,
        run_values=[value] * sample_count if value is not None and sample_count else [],
        source_path="probe.cu",
        generation_source="llm_generated",
        generation_attempts=1,
        generation_error="",
        generation_trace=[],
        profile_source=profile_source,
        profile_returncode=0,
        profile_parse_mode="none",
        profile_command=[],
        profile_stdout_tail="",
        profile_stderr_tail="",
    )


class ProbeAnalysisTests(unittest.TestCase):
    def test_accepts_stable_measurement(self) -> None:
        decision = analyze_probe_round(
            target="dram_latency_cycles",
            result=_probe(value=420.0),
            iteration=1,
            history=[],
        )
        self.assertTrue(decision.done)
        self.assertEqual(decision.next_action, "accept_measurement")

    def test_requests_ncu_profile_for_weak_parse_signal(self) -> None:
        decision = analyze_probe_round(
            target="dram_latency_cycles",
            result=_probe(value=420.0, parsed_from="stdout_last_numeric"),
            iteration=1,
            history=[],
        )
        self.assertFalse(decision.done)
        self.assertEqual(decision.next_action, "add_ncu_profile")
        self.assertTrue(decision.needs_ncu_profile)

    def test_requests_probe_shape_change_for_unstable_measurement(self) -> None:
        decision = analyze_probe_round(
            target="dram_latency_cycles",
            result=_probe(value=420.0, std_value=150.0),
            iteration=1,
            history=[],
        )
        self.assertFalse(decision.done)
        self.assertEqual(decision.next_action, "change_probe_shape")


if __name__ == "__main__":
    unittest.main()
