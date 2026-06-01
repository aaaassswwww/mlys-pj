from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from profiler_agent.multi_agent.llm_client import LLMClient, OpenAICompatibleLLMClient
from profiler_agent.phase3.models import GeneratedRuntimeCandidate, Phase3OptimizerState
from profiler_agent.phase3.prompts import (
    build_phase3_generation_system_prompt,
    build_phase3_generation_user_prompt,
)


def _engine_variant_search_space() -> list[dict[str, Any]]:
    return [
        {"name": "grouped_compile", "group_prefill": True, "group_decode": True, "enable_compile": True},
        {"name": "grouped_eager", "group_prefill": True, "group_decode": True, "enable_compile": False},
        {"name": "decode_group_only", "group_prefill": False, "group_decode": True, "enable_compile": False},
    ]


def _focused_engine_variant_search_space(state: Phase3OptimizerState) -> list[dict[str, Any]]:
    best_id = (state.current_best_correct_candidate_id or state.current_best_candidate_id or "").lower()
    if "grouped_eager" in best_id:
        return [
            {"name": "grouped_eager", "group_prefill": True, "group_decode": True, "enable_compile": False},
            {"name": "grouped_compile", "group_prefill": True, "group_decode": True, "enable_compile": True},
        ]
    if "grouped_compile" in best_id:
        return [
            {"name": "grouped_compile", "group_prefill": True, "group_decode": True, "enable_compile": True},
            {"name": "grouped_eager", "group_prefill": True, "group_decode": True, "enable_compile": False},
        ]
    return _engine_variant_search_space()


def build_runtime_engine_source(
    *,
    group_prefill: bool,
    group_decode: bool,
    enable_compile: bool,
) -> str:
    prefill_block = (
        "        logits_by_request: dict[int, torch.Tensor] = {}\n"
        "        if self.group_prefill_by_length:\n"
        "            grouped_requests = group_request_ids_by_sequence_length(request_ids, sequence_lengths)\n"
        "            for _, grouped_request_ids in grouped_requests.items():\n"
        "                batch_input = torch.stack([sequences_by_request[request_id] for request_id in grouped_request_ids], dim=0)\n"
        "                if len(grouped_request_ids) == 1:\n"
        "                    logits, kv_cache = self.model.logits_and_cache_for_prefill(batch_input)\n"
        "                    request_id = grouped_request_ids[0]\n"
        "                    self.requests.upsert_prompt(request_id, sequences_by_request[request_id])\n"
        "                    self.requests.update_kv_cache(request_id, kv_cache)\n"
        "                    logits_by_request[request_id] = logits.squeeze(0)\n"
        "                    continue\n"
        "                logits, batch_cache = self.model.logits_and_cache_for_prefill_batch(batch_input)\n"
        "                per_request_caches = split_request_cache(batch_cache)\n"
        "                for request_id, row_logits, request_cache in zip(grouped_request_ids, logits, per_request_caches):\n"
        "                    self.requests.upsert_prompt(request_id, sequences_by_request[request_id])\n"
        "                    self.requests.update_kv_cache(request_id, request_cache)\n"
        "                    logits_by_request[request_id] = row_logits\n"
        "        else:\n"
        "            for request_id in request_ids:\n"
        "                sequence = sequences_by_request[request_id].view(1, -1)\n"
        "                logits, kv_cache = self.model.logits_and_cache_for_prefill(sequence)\n"
        "                self.requests.upsert_prompt(request_id, sequences_by_request[request_id])\n"
        "                self.requests.update_kv_cache(request_id, kv_cache)\n"
        "                logits_by_request[request_id] = logits.squeeze(0)\n"
    )
    decode_block = (
        "        grouped_state_pairs = group_pairs_by_sequence_length(\n"
        "            cached_request_ids,\n"
        "            cached_indices,\n"
        "            cached_sequence_lengths,\n"
        "        )\n"
        "        if self.group_decode_by_cache_length:\n"
        "            for cache_len, request_index_pairs in grouped_state_pairs.items():\n"
        "                request_group = [request_id for request_id, _ in request_index_pairs]\n"
        "                group_indices = [token_index for _, token_index in request_index_pairs]\n"
        "                items = [state_by_request_id[request_id] for request_id in request_group]\n"
        "                batch_tokens = token_ids[group_indices].unsqueeze(1)\n"
        "                position_ids = torch.full((len(request_group), 1), fill_value=cache_len, device=self.device, dtype=torch.long)\n"
        "                batch_cache = stack_request_caches([state.kv_cache for state in items])\n"
        "                logits, next_cache = self.model.logits_and_cache_for_decode_batch(\n"
        "                    batch_tokens,\n"
        "                    kv_cache=batch_cache,\n"
        "                    position_ids=position_ids,\n"
        "                )\n"
        "                per_request_caches = split_request_cache(next_cache)\n"
        "                for state, row_logits, request_cache in zip(items, logits, per_request_caches):\n"
        "                    self.requests.update_kv_cache(state.request_id, request_cache)\n"
        "                    logits_by_request[state.request_id] = row_logits\n"
        "        else:\n"
        "            for request_id, token_index in zip(cached_request_ids, cached_indices):\n"
        "                state = state_by_request_id[request_id]\n"
        "                step_token = token_ids[token_index].view(1, 1)\n"
        "                position_ids = torch.full((1, 1), fill_value=state.seq_len - 1, device=self.device, dtype=torch.long)\n"
        "                logits, next_cache = self.model.logits_and_cache_for_decode_batch(\n"
        "                    step_token,\n"
        "                    kv_cache=state.kv_cache,\n"
        "                    position_ids=position_ids,\n"
        "                )\n"
        "                self.requests.update_kv_cache(state.request_id, next_cache)\n"
        "                logits_by_request[state.request_id] = logits.squeeze(0)\n"
    )
    enable_compile_literal = "True" if enable_compile else "False"
    group_prefill_literal = "True" if group_prefill else "False"
    group_decode_literal = "True" if group_decode else "False"
    return f'''"""Generated Phase 3 runtime engine."""\n
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

try:
    from runtime.cache import split_request_cache, stack_request_caches
    from runtime.loader import build_config, load_model
    from runtime.request_state import RequestStateTable
    from runtime.scheduler import group_pairs_by_sequence_length, group_request_ids_by_sequence_length
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from runtime.cache import split_request_cache, stack_request_caches
    from runtime.loader import build_config, load_model
    from runtime.request_state import RequestStateTable
    from runtime.scheduler import group_pairs_by_sequence_length, group_request_ids_by_sequence_length


def create_engine(model_config: dict, weight_dir: str, device: str = "cuda") -> "Engine":
    return Engine(model_config=model_config, weight_dir=weight_dir, device=device)


class Engine:
    def __init__(self, model_config: Dict, weight_dir: str, device: str = "cuda") -> None:
        self.model_config = model_config
        self.weight_dir = weight_dir
        self.device = _resolve_device(device)
        self.runtime_config = build_config(model_config)
        self.model = load_model(model_config, weight_dir, device=self.device)
        self.group_prefill_by_length = {group_prefill_literal}
        self.group_decode_by_cache_length = {group_decode_literal}
        self.enable_compile_warmup = {enable_compile_literal}
        self.requests = RequestStateTable(device=torch.device(self.device))
        self._maybe_prepare_compiled_paths()

    @torch.inference_mode()
    def prefill(self, request_ids: Iterable[int], input_ids: List[object]):
        request_ids = [int(request_id) for request_id in request_ids]
        if len(request_ids) != len(input_ids):
            raise ValueError("request_ids and input_ids must have the same length")
        sequences_by_request: dict[int, torch.Tensor] = {{}}
        sequence_lengths: list[int] = []
        for request_id, tokens in zip(request_ids, input_ids):
            sequence = _normalize_sequence(tokens, device=self.device)
            if sequence.numel() == 0:
                raise ValueError("prefill sequences must be non-empty")
            sequences_by_request[request_id] = sequence
            sequence_lengths.append(sequence.numel())
{prefill_block}
        return torch.stack([logits_by_request[request_id] for request_id in request_ids], dim=0)

    @torch.inference_mode()
    def decode(self, request_ids: Iterable[int], token_ids: object):
        request_ids = [int(request_id) for request_id in request_ids]
        token_ids = _normalize_decode_tokens(token_ids, expected=len(request_ids), device=self.device)
        token_values = _tensor_to_int_list(token_ids)
        state_by_request_id = {{}}
        cached_request_ids: list[int] = []
        cached_sequence_lengths: list[int] = []
        cached_indices: list[int] = []
        fallback_request_ids: list[int] = []
        for index, request_id in enumerate(request_ids):
            token_value = token_values[index]
            state = self.requests.append_token(request_id, token_value)
            state_by_request_id[request_id] = state
            if state.kv_cache is None:
                fallback_request_ids.append(request_id)
            else:
                cached_request_ids.append(request_id)
                cached_sequence_lengths.append(state.seq_len - 1)
                cached_indices.append(index)
        logits_by_request: dict[int, torch.Tensor] = {{}}
        for request_id in fallback_request_ids:
            state = state_by_request_id[request_id]
            if state.tokens is None:
                raise RuntimeError("Missing full token history for non-cached request")
            logits, kv_cache = self.model.logits_and_cache_for_prefill(state.tokens.view(1, -1))
            self.requests.update_kv_cache(state.request_id, kv_cache)
            logits_by_request[state.request_id] = logits.squeeze(0)
{decode_block}
        return torch.stack([logits_by_request[request_id] for request_id in request_ids], dim=0)

    @torch.inference_mode()
    def remove(self, request_ids: Iterable[int]):
        self.requests.remove([int(request_id) for request_id in request_ids])

    def _maybe_prepare_compiled_paths(self) -> None:
        if not self.enable_compile_warmup:
            return
        if os.environ.get("MLSYS_DISABLE_COMPILE", "").lower() in {{"1", "true", "yes", "on"}}:
            return
        if not str(self.device).startswith("cuda"):
            return
        if not self.model.try_enable_compile():
            return
        try:
            self._warmup_compiled_paths()
        except Exception:
            self.model.disable_compile()

    @torch.inference_mode()
    def _warmup_compiled_paths(self) -> None:
        prompt_len = min(8, self.runtime_config.max_position_embeddings)
        prompt = torch.zeros((1, prompt_len), device=self.device, dtype=torch.long)
        _, kv_cache = self.model.logits_and_cache_for_prefill(prompt)
        decode_tokens = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        position_ids = torch.full((1, 1), prompt_len, device=self.device, dtype=torch.long)
        self.model.logits_and_cache_for_decode_batch(
            decode_tokens,
            kv_cache=kv_cache,
            position_ids=position_ids,
        )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _normalize_sequence(tokens: object, device: str) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
    return tokens.to(device=device, dtype=torch.long).view(-1)


def _normalize_decode_tokens(token_ids: object, expected: int, device: str) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_ids = token_ids.to(device=device, dtype=torch.long).view(-1)
    if token_ids.numel() != expected:
        raise ValueError(f"Expected {{expected}} decode tokens, got {{token_ids.numel()}}")
    return token_ids


def _tensor_to_int_list(token_ids: torch.Tensor) -> list[int]:
    if token_ids.device.type == "cpu":
        return token_ids.tolist()
    return token_ids.detach().to(device="cpu").tolist()
'''


def build_baseline_engine_source() -> str:
    return '''"""Naive baseline engine used for local Phase 3 speedup comparisons."""\n
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

try:
    from runtime.loader import build_config, load_model
    from runtime.request_state import RequestStateTable
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from runtime.loader import build_config, load_model
    from runtime.request_state import RequestStateTable


def create_engine(model_config: dict, weight_dir: str, device: str = "cuda") -> "Engine":
    return Engine(model_config=model_config, weight_dir=weight_dir, device=device)


class Engine:
    def __init__(self, model_config: Dict, weight_dir: str, device: str = "cuda") -> None:
        self.model_config = model_config
        self.weight_dir = weight_dir
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "cuda" and not torch.cuda.is_available() else device)
        self.runtime_config = build_config(model_config)
        self.model = load_model(model_config, weight_dir, device=self.device)
        self.requests = RequestStateTable(device=torch.device(self.device))

    @torch.inference_mode()
    def prefill(self, request_ids: Iterable[int], input_ids: List[object]):
        request_ids = [int(request_id) for request_id in request_ids]
        logits = []
        for request_id, tokens in zip(request_ids, input_ids):
            sequence = _normalize_sequence(tokens, device=self.device)
            self.requests.upsert_prompt(request_id, sequence)
            logits.append(self.model.logits_for_last_token(sequence.view(1, -1)).squeeze(0))
        return torch.stack(logits, dim=0)

    @torch.inference_mode()
    def decode(self, request_ids: Iterable[int], token_ids: object):
        request_ids = [int(request_id) for request_id in request_ids]
        token_ids = _normalize_decode_tokens(token_ids, expected=len(request_ids), device=self.device)
        logits = []
        for request_id, token_id in zip(request_ids, token_ids.tolist()):
            state = self.requests.append_token(request_id, int(token_id))
            if state.tokens is None:
                raise RuntimeError("naive baseline requires full token history")
            logits.append(self.model.logits_for_last_token(state.tokens.view(1, -1)).squeeze(0))
        return torch.stack(logits, dim=0)

    @torch.inference_mode()
    def remove(self, request_ids: Iterable[int]):
        self.requests.remove([int(request_id) for request_id in request_ids])


def _normalize_sequence(tokens: object, device: str) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
    return tokens.to(device=device, dtype=torch.long).view(-1)


def _normalize_decode_tokens(token_ids: object, expected: int, device: str) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_ids = token_ids.to(device=device, dtype=torch.long).view(-1)
    if token_ids.numel() != expected:
        raise ValueError(f"Expected {expected} decode tokens, got {token_ids.numel()}")
    return token_ids
'''


_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _normalize_candidate_stem(candidate_id: str, fallback: str) -> str:
    raw = (candidate_id or "").strip()
    if not raw:
        return fallback
    raw = _IDENTIFIER_RE.sub("-", raw).strip("-_")
    return raw or fallback


def _looks_like_runtime_source(source_code: str) -> bool:
    required_markers = [
        "def create_engine(",
        "def prefill(",
        "def decode(",
        "def remove(",
    ]
    return all(marker in source_code for marker in required_markers)


class Phase3CandidateGenerator:
    def __init__(self, llm_client: LLMClient | None = None, *, debug_dir: Path | None = None) -> None:
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleLLMClient.from_env()
        self.debug_dir = debug_dir

    def bootstrap_candidate(self) -> GeneratedRuntimeCandidate:
        return GeneratedRuntimeCandidate(
            candidate_id="grouped_compile-v00",
            source_code=build_runtime_engine_source(group_prefill=True, group_decode=True, enable_compile=True),
            rationale="phase3 bootstrap runtime with KV cache, grouped prefill/decode, and optional compile warmup",
            source="bootstrap_template",
        )

    def _write_generation_debug(
        self,
        *,
        iteration: int,
        candidate: GeneratedRuntimeCandidate,
        reason: str,
        system_prompt: str = "",
        user_prompt: str = "",
        llm_payload: dict[str, Any] | None = None,
    ) -> None:
        if self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "iteration": iteration,
            "candidate_id": candidate.candidate_id,
            "candidate_source": candidate.source,
            "reason": reason,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_payload": llm_payload or {},
        }
        (self.debug_dir / f"phase3_codegen_iter_{iteration:02d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    def _generate_deterministic_candidate(self, *, state: Phase3OptimizerState) -> GeneratedRuntimeCandidate:
        configs = _focused_engine_variant_search_space(state) if state.current_best_correct_candidate_id else _engine_variant_search_space()
        tried = {
            str(item.get("candidate_id")).rsplit("-v", 1)[0]
            for item in state.llm_revision_history
            if isinstance(item, dict) and item.get("candidate_id")
        }
        config = next((item for item in configs if item["name"] not in tried), None)
        if config is None:
            history_len = len([item for item in state.llm_revision_history if isinstance(item, dict)])
            config = configs[history_len % len(configs)]
        return GeneratedRuntimeCandidate(
            candidate_id=f"{config['name']}-v{state.iteration:02d}",
            source_code=build_runtime_engine_source(
                group_prefill=bool(config["group_prefill"]),
                group_decode=bool(config["group_decode"]),
                enable_compile=bool(config["enable_compile"]),
            ),
            rationale=(
                "phase3 fallback runtime candidate varying grouped prefill/decode and compile warmup "
                "around the cache-backed decoder runtime"
            ),
            source="deterministic_runtime_fallback",
        )

    def _generate_llm_candidate(
        self,
        *,
        state: Phase3OptimizerState,
        feedback: dict[str, Any] | None,
    ) -> tuple[GeneratedRuntimeCandidate | None, str, str, dict[str, Any] | None]:
        if self.llm_client is None or not self.llm_client.is_enabled():
            return None, "", "", None
        bootstrap_source = self.bootstrap_candidate().source_code
        system_prompt = build_phase3_generation_system_prompt()
        user_prompt = build_phase3_generation_user_prompt(
            state=state,
            iteration=state.iteration,
            feedback=feedback,
            bootstrap_source_code=bootstrap_source,
        )
        payload = self.llm_client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not isinstance(payload, dict):
            return None, system_prompt, user_prompt, payload
        source_code = payload.get("source_code")
        if not isinstance(source_code, str) or not source_code.strip():
            return None, system_prompt, user_prompt, payload
        if not _looks_like_runtime_source(source_code):
            return None, system_prompt, user_prompt, payload
        stem = _normalize_candidate_stem(str(payload.get("candidate_id", "")), "llm-runtime")
        rationale = str(payload.get("rationale", "")).strip()
        candidate = GeneratedRuntimeCandidate(
            candidate_id=f"{stem}-v{state.iteration:02d}",
            source_code=source_code,
            rationale=rationale or "phase3 llm runtime revision candidate",
            source="llm_runtime_revision",
        )
        return candidate, system_prompt, user_prompt, payload

    def generate_candidate(self, *, state: Phase3OptimizerState, feedback: dict[str, Any] | None) -> GeneratedRuntimeCandidate:
        llm_candidate, system_prompt, user_prompt, payload = self._generate_llm_candidate(state=state, feedback=feedback)
        if llm_candidate is not None:
            self._write_generation_debug(
                iteration=state.iteration,
                candidate=llm_candidate,
                reason="llm_runtime_revision",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                llm_payload=payload,
            )
            return llm_candidate

        candidate = self._generate_deterministic_candidate(state=state)
        self._write_generation_debug(
            iteration=state.iteration,
            candidate=candidate,
            reason="deterministic_runtime_fallback",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            llm_payload=payload,
        )
        return candidate
