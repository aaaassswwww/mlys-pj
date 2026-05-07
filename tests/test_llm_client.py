from __future__ import annotations

import json
import os
import io
import shutil
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from profiler_agent.multi_agent.llm_client import OpenAICompatibleConfig, OpenAICompatibleLLMClient


class _DummyResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _ = (exc_type, exc_val, exc_tb)
        return None


class LlmClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenAICompatibleLLMClient(
            OpenAICompatibleConfig(
                api_key="test_key",
                base_url="https://example.invalid/v1",
                model="gpt-test",
                timeout_s=3,
            )
        )
        self.tmp_dir = Path("tests/.tmp") / f"llm_client_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run_with_payload(self, payload: dict) -> dict | None:
        raw = json.dumps(payload, ensure_ascii=False)
        with patch("urllib.request.urlopen", return_value=_DummyResponse(raw)):
            return self.client.complete_json(system_prompt="s", user_prompt="u")

    def test_complete_json_accepts_plain_json_text(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"selected_tools":["executor","ncu"]}',
                    }
                }
            ]
        }
        parsed = self._run_with_payload(payload)
        self.assertEqual(parsed, {"selected_tools": ["executor", "ncu"]})

    def test_complete_json_accepts_markdown_fenced_json(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"next_actions\": [\"a\", \"b\"]}\n```",
                    }
                }
            ]
        }
        parsed = self._run_with_payload(payload)
        self.assertEqual(parsed, {"next_actions": ["a", "b"]})

    def test_complete_json_accepts_json_embedded_in_text(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Here is the result.\n"
                            "{\"intent\":\"gpu_profiling\"}\n"
                            "Use it safely."
                        ),
                    }
                }
            ]
        }
        parsed = self._run_with_payload(payload)
        self.assertEqual(parsed, {"intent": "gpu_profiling"})

    def test_complete_json_accepts_output_array_shape(self) -> None:
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "{\"risk_level\":\"high\"}"},
                    ]
                }
            ]
        }
        parsed = self._run_with_payload(payload)
        self.assertEqual(parsed, {"risk_level": "high"})

    def test_complete_json_writes_debug_record_when_enabled(self) -> None:
        debug_path = self.tmp_dir / "llm_debug.jsonl"
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"intent\":\"gpu_profiling\"}\n```",
                    }
                }
            ]
        }
        raw = json.dumps(payload, ensure_ascii=False)
        with patch.dict(os.environ, {"PROFILER_AGENT_LLM_DEBUG_PATH": str(debug_path)}):
            with patch("urllib.request.urlopen", return_value=_DummyResponse(raw)):
                parsed = self.client.complete_json(system_prompt="sys", user_prompt="usr")
        self.assertEqual(parsed, {"intent": "gpu_profiling"})
        self.assertTrue(debug_path.exists())
        lines = debug_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        row = json.loads(lines[-1])
        self.assertEqual(row.get("phase"), "json_parsed")
        self.assertIn("raw_response", row)

    def test_complete_json_writes_disabled_record(self) -> None:
        debug_path = self.tmp_dir / "llm_debug_disabled.jsonl"
        disabled_client = OpenAICompatibleLLMClient(
            OpenAICompatibleConfig(
                api_key="",
                base_url="https://example.invalid/v1",
                model="gpt-test",
                timeout_s=3,
            )
        )
        with patch.dict(os.environ, {"PROFILER_AGENT_LLM_DEBUG_PATH": str(debug_path)}):
            parsed = disabled_client.complete_json(system_prompt="sys", user_prompt="usr")
        self.assertIsNone(parsed)
        self.assertTrue(debug_path.exists())
        row = json.loads(debug_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(row.get("phase"), "llm_disabled")

    def test_complete_json_writes_http_error_details(self) -> None:
        debug_path = self.tmp_dir / "llm_debug_http_error.jsonl"
        err = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"invalid_api_key"}'),
        )
        with patch.dict(os.environ, {"PROFILER_AGENT_LLM_DEBUG_PATH": str(debug_path)}):
            with patch("urllib.request.urlopen", side_effect=err):
                parsed = self.client.complete_json(system_prompt="sys", user_prompt="usr")
        self.assertIsNone(parsed)
        row = json.loads(debug_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(row.get("phase"), "http_error")
        self.assertEqual(row.get("error_type"), "HTTPError")
        self.assertEqual(row.get("http_status"), 401)
        self.assertEqual(row.get("error_category"), "auth_error")
        body_excerpt = row.get("error_body_excerpt", "")
        if body_excerpt:
            self.assertIn("invalid_api_key", body_excerpt)

    def test_complete_json_retries_on_timeout_then_succeeds(self) -> None:
        debug_path = self.tmp_dir / "llm_debug_retry.jsonl"
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"intent":"gpu_profiling"}',
                    }
                }
            ]
        }
        raw = json.dumps(payload, ensure_ascii=False)
        attempts = [
            TimeoutError("timed out once"),
            _DummyResponse(raw),
        ]
        with patch.dict(os.environ, {"PROFILER_AGENT_LLM_DEBUG_PATH": str(debug_path)}):
            with patch("urllib.request.urlopen", side_effect=attempts):
                with patch("time.sleep", return_value=None):
                    parsed = self.client.complete_json(system_prompt="sys", user_prompt="usr")
        self.assertEqual(parsed, {"intent": "gpu_profiling"})
        lines = debug_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        self.assertEqual(first.get("phase"), "http_error")
        self.assertEqual(first.get("retryable"), True)
        self.assertEqual(first.get("will_retry"), True)
        self.assertEqual(first.get("error_category"), "network_timeout")
        self.assertEqual(last.get("phase"), "json_parsed")


if __name__ == "__main__":
    unittest.main()
