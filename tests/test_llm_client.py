from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from profiler_agent.multi_agent.llm_client import OpenAICompatibleConfig, OpenAICompatibleLLMClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False


class LLMClientTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "API_KEY": "k1",
            "BASE_URL": "https://example.com/v1",
            "BASE_MODEL": "gpt-5.4",
        },
        clear=True,
    )
    def test_from_env_prefers_submission_variable_names(self) -> None:
        client = OpenAICompatibleLLMClient.from_env()
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.config.api_key, "k1")
        self.assertEqual(client.config.base_url, "https://example.com/v1")
        self.assertEqual(client.config.model, "gpt-5.4")

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "k2",
            "OPENAI_BASE_URL": "https://fallback.example/v1",
            "OPENAI_MODEL": "gpt-4o-mini",
        },
        clear=True,
    )
    def test_from_env_falls_back_to_openai_variable_names(self) -> None:
        client = OpenAICompatibleLLMClient.from_env()
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.config.api_key, "k2")
        self.assertEqual(client.config.base_url, "https://fallback.example/v1")
        self.assertEqual(client.config.model, "gpt-4o-mini")

    @patch("profiler_agent.multi_agent.llm_client.urllib.request.urlopen")
    def test_complete_json_parses_chat_completion(self, mock_urlopen: unittest.mock.Mock) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "{\"intent\":\"gpu_profiling\"}",
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleLLMClient(
            OpenAICompatibleConfig(api_key="k", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
        )
        out = client.complete_json(system_prompt="s", user_prompt="u")
        self.assertEqual(out, {"intent": "gpu_profiling"})

    @patch("profiler_agent.multi_agent.llm_client.urllib.request.urlopen")
    def test_complete_json_returns_none_for_invalid_response(self, mock_urlopen: unittest.mock.Mock) -> None:
        mock_urlopen.return_value = _FakeResponse({"choices": []})
        client = OpenAICompatibleLLMClient(
            OpenAICompatibleConfig(api_key="k", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
        )
        out = client.complete_json(system_prompt="s", user_prompt="u")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
