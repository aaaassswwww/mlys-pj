from __future__ import annotations

import unittest
from unittest.mock import patch

from profiler_agent.tool_adapters.nvml_adapter import query_named_device_attribute


class _FakeCudart:
    def __init__(self, get_device_rc: int = 0, attr_rc: int = 0, raw_value: int = 0) -> None:
        self._get_device_rc = get_device_rc
        self._attr_rc = attr_rc
        self._raw_value = raw_value

    def cudaGetDevice(self, ptr) -> int:
        ptr._obj.value = 0
        return self._get_device_rc

    def cudaGetDeviceCount(self, ptr) -> int:
        ptr._obj.value = 1
        return 0

    def cudaDeviceGetAttribute(self, ptr, attribute: int, device: int) -> int:
        _ = attribute, device
        ptr._obj.value = self._raw_value
        return self._attr_rc


class DeviceAttributeAdapterTests(unittest.TestCase):
    @patch("profiler_agent.tool_adapters.nvml_adapter._load_cudart")
    def test_query_prefers_cuda_runtime_attribute(self, mock_load_cudart: unittest.mock.Mock) -> None:
        mock_load_cudart.return_value = (_FakeCudart(raw_value=2100000), "cudart64_130.dll", None)

        result = query_named_device_attribute("device__attribute_max_gpu_frequency_khz")
        self.assertEqual(result["source"], "cuda_runtime_attribute")
        self.assertEqual(result["value"], 2100000.0)
        self.assertEqual(result["backend_chain"], ["cuda_runtime_attribute"])

    @patch("profiler_agent.tool_adapters.nvml_adapter._query_gpu_field_once")
    @patch("profiler_agent.tool_adapters.nvml_adapter._load_cudart")
    def test_query_falls_back_to_nvidia_smi(
        self,
        mock_load_cudart: unittest.mock.Mock,
        mock_query_gpu_field: unittest.mock.Mock,
    ) -> None:
        mock_load_cudart.return_value = (None, None, "cudart_not_found")
        mock_query_gpu_field.return_value = (
            2100.0,
            {
                "field": "clocks.max.sm",
                "command": ["nvidia-smi"],
                "source": "nvidia_smi_query",
                "returncode": 0,
                "stdout_tail": "2100",
                "stderr_tail": "",
            },
        )

        result = query_named_device_attribute("device__attribute_max_gpu_frequency_khz")
        self.assertEqual(result["source"], "nvidia_smi_query")
        self.assertEqual(result["value"], 2100000.0)
        self.assertEqual(result["backend_chain"], ["cuda_runtime_unavailable", "nvidia_smi_query"])

    @patch("profiler_agent.tool_adapters.nvml_adapter._query_gpu_field_once")
    @patch("profiler_agent.tool_adapters.nvml_adapter._load_cudart")
    def test_query_preserves_both_backend_failures(
        self,
        mock_load_cudart: unittest.mock.Mock,
        mock_query_gpu_field: unittest.mock.Mock,
    ) -> None:
        mock_load_cudart.return_value = (None, None, "cudart_not_found")
        mock_query_gpu_field.return_value = (
            None,
            {
                "field": "clocks.max.sm",
                "command": ["nvidia-smi"],
                "source": "nvidia_smi_query",
                "returncode": 127,
                "stdout_tail": "",
                "stderr_tail": "nvidia_smi_missing",
            },
        )

        result = query_named_device_attribute("device__attribute_max_gpu_frequency_khz")
        self.assertEqual(result["source"], "cuda_runtime_unavailable")
        self.assertIsNone(result["value"])
        self.assertIn("cudart_not_found", result["stderr_tail"])
        self.assertIn("nvidia_smi_missing", result["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
