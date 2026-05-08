from __future__ import annotations

import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    max_retries: int = 2
    retry_base_s: float = 1.0


class OpenAICompatibleLLMClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    @classmethod
    def _build_from_values(
        cls,
        *,
        api_key: str,
        base_url: str,
        model: str,
    ) -> "OpenAICompatibleLLMClient | None":
        if not api_key:
            return None
        timeout = int(os.environ.get("OPENAI_TIMEOUT_S", "30"))
        max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        retry_base_s = float(os.environ.get("OPENAI_RETRY_BASE_S", "1.0"))
        return cls(
            OpenAICompatibleConfig(
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                model=model,
                timeout_s=timeout,
                max_retries=max(0, max_retries),
                retry_base_s=max(0.0, retry_base_s),
            )
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMClient | None":
        api_key = os.environ.get("API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = (
            os.environ.get("BASE_URL", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or "https://api.openai.com/v1"
        )
        model = (
            os.environ.get("BASE_MODEL", "").strip()
            or os.environ.get("OPENAI_MODEL", "").strip()
            or "gpt-5.4"
        )
        return cls._build_from_values(api_key=api_key, base_url=base_url, model=model)

    @classmethod
    def from_secret_file(
        cls,
        secret_file: str | os.PathLike[str],
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> "OpenAICompatibleLLMClient | None":
        raw = Path(secret_file).read_text(encoding="utf-8").strip()
        if not raw:
            return None
        file_api_key = raw
        file_base_url = ""
        file_model = ""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            file_api_key = str(parsed.get("api_key", "")).strip()
            file_base_url = str(parsed.get("base_url", "")).strip()
            file_model = str(parsed.get("model", "")).strip()
        resolved_base_url = (base_url or "").strip() or file_base_url or "https://api.openai.com/v1"
        resolved_model = (model or "").strip() or file_model or "gpt-5.4"
        return cls._build_from_values(
            api_key=file_api_key.strip(),
            base_url=resolved_base_url,
            model=resolved_model,
        )

    def is_enabled(self) -> bool:
        return bool(self.config.api_key.strip())

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            block = match.group(1).strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts: list[str] = []
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            txt = item.get("text")
                            if isinstance(txt, str):
                                parts.append(txt)
                        if parts:
                            return "\n".join(parts)

        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text

        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                    continue
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        txt = part.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
            if parts:
                return "\n".join(parts)
        return ""

    def _write_debug_record(self, record: dict[str, Any]) -> None:
        debug_path = os.environ.get("PROFILER_AGENT_LLM_DEBUG_PATH", "").strip()
        if not debug_path:
            return
        try:
            path = Path(debug_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        except OSError:
            # Debug recording must never break production flow.
            return

    @staticmethod
    def _truncate(value: str, *, max_len: int = 4000) -> str:
        if len(value) <= max_len:
            return value
        return value[:max_len] + "...<truncated>"

    def _annotate_http_error(self, debug_record: dict[str, Any], exc: Exception) -> None:
        debug_record["error_type"] = type(exc).__name__
        debug_record["error_message"] = str(exc)

        if isinstance(exc, urllib.error.HTTPError):
            debug_record["http_status"] = exc.code
            debug_record["error_reason"] = str(exc.reason)
            if exc.code in {401, 403}:
                debug_record["error_category"] = "auth_error"
            elif exc.code == 429:
                debug_record["error_category"] = "rate_limited"
            elif 500 <= exc.code <= 599:
                debug_record["error_category"] = "upstream_server_error"
            else:
                debug_record["error_category"] = "http_status_error"
            try:
                raw = exc.read()
                if isinstance(raw, bytes):
                    debug_record["error_body_excerpt"] = self._truncate(raw.decode("utf-8", errors="replace"))
            except Exception:
                pass
            return

        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            debug_record["error_reason"] = str(reason)
            if isinstance(reason, TimeoutError):
                debug_record["error_category"] = "network_timeout"
            elif isinstance(reason, socket.gaierror):
                debug_record["error_category"] = "dns_error"
            elif isinstance(reason, ConnectionRefusedError):
                debug_record["error_category"] = "connection_refused"
            else:
                debug_record["error_category"] = "url_error"
            return

        if isinstance(exc, TimeoutError):
            debug_record["error_category"] = "network_timeout"
            return

        if isinstance(exc, http.client.IncompleteRead):
            debug_record["error_category"] = "incomplete_http_read"
            partial = getattr(exc, "partial", b"")
            if isinstance(partial, bytes) and partial:
                debug_record["partial_response_excerpt"] = self._truncate(partial.decode("utf-8", errors="replace"))
            return

        if isinstance(exc, ValueError):
            debug_record["error_category"] = "request_value_error"
            return

        debug_record["error_category"] = "unknown_error"

    @staticmethod
    def _is_retryable_error(exc: Exception, category: str | None) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(exc, http.client.IncompleteRead):
            return True
        if isinstance(exc, urllib.error.URLError):
            return True
        if isinstance(exc, urllib.error.HTTPError):
            if category == "auth_error":
                return False
            if exc.code in {408, 425, 429}:
                return True
            return 500 <= exc.code <= 599
        return False

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.is_enabled():
            self._write_debug_record(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "model": self.config.model,
                    "phase": "llm_disabled",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
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
        base_debug_record: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "url": req.full_url,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_attempts": self.config.max_retries + 1,
        }
        raw = ""
        debug_record: dict[str, Any] = {}
        max_attempts = self.config.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            debug_record = dict(base_debug_record)
            debug_record["attempt"] = attempt
            debug_record["phase"] = "request_built"
            start_ts = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                    debug_record["http_status"] = getattr(resp, "status", None)
                    debug_record["raw_response"] = raw
                    debug_record["elapsed_ms"] = int((time.perf_counter() - start_ts) * 1000)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, http.client.IncompleteRead) as exc:
                debug_record["phase"] = "http_error"
                debug_record["elapsed_ms"] = int((time.perf_counter() - start_ts) * 1000)
                self._annotate_http_error(debug_record, exc)
                category = debug_record.get("error_category")
                retryable = self._is_retryable_error(exc, category if isinstance(category, str) else None)
                will_retry = retryable and attempt < max_attempts
                debug_record["retryable"] = retryable
                debug_record["will_retry"] = will_retry
                if will_retry:
                    sleep_s = self.config.retry_base_s * (2 ** (attempt - 1))
                    debug_record["retry_delay_s"] = round(sleep_s, 3)
                    self._write_debug_record(debug_record)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue
                self._write_debug_record(debug_record)
                return None

        try:
            payload = json.loads(raw)
            debug_record["phase"] = "payload_loaded"
            text = self._extract_text(payload)
            debug_record["extracted_text"] = text
            if not text:
                debug_record["phase"] = "empty_text"
                self._write_debug_record(debug_record)
                return None
            parsed = self._extract_json_from_text(text)
            if parsed is not None:
                debug_record["phase"] = "json_parsed"
                debug_record["parsed_json"] = parsed
                self._write_debug_record(debug_record)
                return parsed
            debug_record["phase"] = "json_not_found_in_text"
            self._write_debug_record(debug_record)
        except (json.JSONDecodeError, TypeError, ValueError):
            debug_record["phase"] = "payload_json_decode_error"
            self._write_debug_record(debug_record)
            return None
        return None
