from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    def is_enabled(self) -> bool: ...

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 30


class OpenAICompatibleLLMClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMClient | None":
        api_key = os.environ.get("API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = (
            os.environ.get("BASE_URL", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or "https://api.openai.com/v1"
        ).rstrip("/")
        model = (
            os.environ.get("BASE_MODEL", "").strip()
            or os.environ.get("OPENAI_MODEL", "").strip()
            or "gpt-5.4"
        )
        timeout = int(os.environ.get("OPENAI_TIMEOUT_S", "30"))
        return cls(OpenAICompatibleConfig(api_key=api_key, base_url=base_url, model=model, timeout_s=timeout))

    def is_enabled(self) -> bool:
        return bool(self.config.api_key.strip())

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message", {})
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        return ""

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.is_enabled():
            return None

        body = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            return None

        try:
            payload = json.loads(raw)
            text = self._extract_text(payload)
            if not text:
                return None
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return None
