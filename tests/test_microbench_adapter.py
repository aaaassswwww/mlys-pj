from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from profiler_agent.tool_adapters.microbench_adapter import (
    compile_probe_source,
    measure_metric_with_evidence,
    profile_probe_with_ncu,
    run_probe_binary,
)


class MicrobenchAdapterTests(unittest.TestCase):
    def test_unsupported_metric(self) -> None:
        result = measure_metric_with_evidence("unknown_metric", "dummy")
        self.assertIsNone(result.value)
        self.assertEqual(result.source, "unsupported_metric")

    @patch("profiler_agent.tool_adapters.microbench_adapter.query_metric_with_evidence")
    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._select_probe_source")
    def test_workload_counter_prefers_ncu_profiled_probe_value(
        self,
        mock_select_probe_source: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_run_probe: unittest.mock.Mock,
        mock_query: unittest.mock.Mock,
    ) -> None:
        from profiler_agent.tool_adapters.ncu_adapter import NcuQueryResult

        tmp_dir = Path("tests/.tmp/microbench_counter")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        source_path = tmp_dir / "probe.cu"
        source_path.write_text("// probe", encoding="utf-8")
        try:
            mock_select_probe_source.return_value = (
                source_path,
                "template_generated",
                1,
                "llm_empty_or_invalid_json",
                [],
            )
            mock_compile_probe.return_value = (0, "", "", ["nvcc", str(source_path), "-o", "probe.exe"])
            mock_run_probe.return_value = (
                0,
                "metric=dram__bytes_read.sum.per_second value=1 samples=1 mode=ncu_profiled median=1 best=1 std=0\n",
                "",
                ["probe.exe"],
            )
            mock_query.return_value = NcuQueryResult(
                value=1234.5,
                source="ncu_csv",
                returncode=0,
                parse_mode="csv_metric_row",
                command=["ncu", "--metrics", "dram__bytes_read.sum.per_second", "probe.exe"],
                stdout_tail="csv",
                stderr_tail="",
            )

            result = measure_metric_with_evidence("dram__bytes_read.sum.per_second", "")
            self.assertEqual(result.source, "ncu_profiled_probe")
            self.assertEqual(result.value, 1234.5)
            self.assertEqual(result.profile_source, "ncu_csv")
            self.assertEqual(result.generation_source, "template_generated")
            self.assertEqual(result.parsed_from, "ncu_profiled_probe[counter_metric]")
        finally:
            if source_path.exists():
                source_path.unlink()

    @patch("profiler_agent.tool_adapters.microbench_adapter._probe_source_path")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    def test_successful_probe_parse(
        self,
        mock_run_probe: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_source_path: unittest.mock.Mock,
    ) -> None:
        mock_source_path.return_value.exists.return_value = True
        mock_compile_probe.return_value = (0, "", "")
        mock_run_probe.return_value = (
            0,
            "metric=dram_latency_cycles value=412 samples=5 median=415 best=412 std=2.5\n",
            "",
        )

        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.source, "microbench_probe")
        self.assertEqual(result.value, 412.0)
        self.assertTrue(result.parsed_from.startswith("multi_run_median["))
        self.assertEqual(result.sample_count, 5)
        self.assertEqual(result.best_value, 412.0)
        self.assertEqual(result.median_value, 412.0)
        self.assertEqual(result.std_value, 0.0)
        self.assertEqual(result.run_values, [412.0] * 5)

    @patch.dict(os.environ, {"PROFILER_AGENT_PROBE_REPEAT": "3"}, clear=False)
    @patch("profiler_agent.tool_adapters.microbench_adapter._probe_source_path")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    def test_probe_repeat_env_applies(
        self,
        mock_run_probe: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_source_path: unittest.mock.Mock,
    ) -> None:
        mock_source_path.return_value.exists.return_value = True
        mock_compile_probe.return_value = (0, "", "")
        mock_run_probe.return_value = (
            0,
            "metric=dram_latency_cycles value=500\n",
            "",
        )

        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.sample_count, 3)
        self.assertEqual(mock_run_probe.call_count, 3)

    @patch.dict(
        os.environ,
        {
            "PROFILER_AGENT_PROBE_SOURCE_MODE": "llm_generated",
            "PROFILER_AGENT_DISABLE_STATIC_FALLBACK": "1",
        },
        clear=False,
    )
    @patch("profiler_agent.tool_adapters.microbench_adapter.ProbeCodeGenerator.is_enabled")
    def test_disable_static_fallback_returns_llm_generation_failed(
        self,
        mock_is_enabled: unittest.mock.Mock,
    ) -> None:
        mock_is_enabled.return_value = False
        result = measure_metric_with_evidence("dram_latency_cycles", "dummy")
        self.assertEqual(result.source, "llm_generation_failed")
        self.assertEqual(result.generation_source, "llm_generated_only")
        self.assertIsInstance(result.generation_trace, list)
        assert result.generation_trace is not None
        self.assertTrue(any(item.get("error") == "static_fallback_disabled" for item in result.generation_trace))

    @patch("profiler_agent.tool_adapters.microbench_adapter._generated_probe_binary_path")
    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    @patch("profiler_agent.tool_adapters.microbench_adapter._select_probe_source")
    def test_compile_failure_surfaces_diagnostic_for_generated_probe(
        self,
        mock_select_probe_source: unittest.mock.Mock,
        mock_compile_probe: unittest.mock.Mock,
        mock_generated_binary_path: unittest.mock.Mock,
    ) -> None:
        mock_select_probe_source.return_value = (
            Path(os.path.abspath("outputs/generated_probes/actual_boost_clock_mhz/probe.cu")),
            "llm_generated",
            1,
            "",
            [],
        )
        mock_generated_binary_path.return_value = Path(os.path.abspath("outputs/generated_probes/actual_boost_clock_mhz/probe.exe"))
        mock_compile_probe.return_value = (1, "", 'error: identifier "__clock64" is undefined')

        result = measure_metric_with_evidence("actual_boost_clock_mhz", "dummy")
        self.assertEqual(result.source, "compile_failed")
        self.assertIn("__clock64", result.compile_stderr_tail)
        self.assertEqual(result.compile_returncode, 1)
        self.assertIsInstance(result.compile_command, list)

    @patch("profiler_agent.tool_adapters.microbench_adapter._compile_probe")
    def test_compile_probe_source_returns_typed_stage_result(self, mock_compile_probe: unittest.mock.Mock) -> None:
        mock_compile_probe.return_value = (0, "compiled", "", ["nvcc", "probe.cu", "-o", "probe.exe"])
        stage = compile_probe_source(Path("probe.cu"), Path("probe.exe"))
        self.assertTrue(stage.ok)
        self.assertEqual(stage.returncode, 0)
        self.assertEqual(stage.command[0], "nvcc")

    @patch("profiler_agent.tool_adapters.microbench_adapter._run_probe")
    def test_run_probe_binary_returns_typed_stage_result(self, mock_run_probe: unittest.mock.Mock) -> None:
        mock_run_probe.return_value = (0, "ok", "", ["probe.exe"])
        stage = run_probe_binary(Path("probe.exe"))
        self.assertTrue(stage.ok)
        self.assertEqual(stage.stdout_tail, "ok")

    @patch("profiler_agent.tool_adapters.microbench_adapter.query_metric_with_evidence")
    def test_profile_probe_with_ncu_returns_typed_stage_result(self, mock_query: unittest.mock.Mock) -> None:
        from profiler_agent.tool_adapters.ncu_adapter import NcuQueryResult

        mock_query.return_value = NcuQueryResult(
            value=12.0,
            source="ncu_csv",
            returncode=0,
            parse_mode="csv_metric_row",
            command=["ncu", "--metrics", "dram_latency_cycles", "probe.exe"],
            stdout_tail="csv",
            stderr_tail="",
        )
        stage = profile_probe_with_ncu("dram_latency_cycles", Path("probe.exe"))
        self.assertTrue(stage.ok)
        self.assertEqual(stage.value, 12.0)
        self.assertEqual(stage.source, "ncu_csv")


if __name__ == "__main__":
    unittest.main()
