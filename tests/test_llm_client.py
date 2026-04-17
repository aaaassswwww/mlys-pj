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
    @patch.dict(os.environ, {"API_KEY": "k-test", "OPENAI_BASE_URL": "https://api.openai.com/v1"}, clear=False)
    def test_from_env_uses_api_key_env_var(self) -> None:
        client = OpenAICompatibleLLMClient.from_env()
        self.assertIsNotNone(client)
        assert client is not None
        self.assertEqual(client.config.api_key, "k-test")

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
