"""Request lifecycle and request-to-cache mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from .cache import RequestKVCache


@dataclass
class RequestState:
    request_id: int
    tokens: torch.Tensor | None
    seq_len: int
    kv_cache: RequestKVCache | None = None
    cache_slot: int | None = None
    active: bool = True


class RequestStateTable:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._states: Dict[int, RequestState] = {}
        self._free_cache_slots: List[int] = []
        self._next_cache_slot = 0

    def upsert_prompt(self, request_id: int, tokens: torch.Tensor) -> RequestState:
        existing = self._states.get(int(request_id))
        if existing is not None:
            self._release_cache_slot(existing.cache_slot)
        state = RequestState(
            request_id=request_id,
            tokens=tokens.to(self.device, dtype=torch.long).clone(),
            seq_len=int(tokens.numel()),
            cache_slot=self._allocate_cache_slot(),
            active=True,
        )
        self._states[request_id] = state
        return state

    def append_token(self, request_id: int, token_id: int) -> RequestState:
        state = self.require(request_id)
        state.seq_len += 1
        # We only keep the full token history until the first cache-backed
        # prefill finishes. After that, decode advances by seq_len alone.
        if state.kv_cache is None and state.tokens is not None:
            next_token = torch.tensor([token_id], device=self.device, dtype=torch.long)
            state.tokens = torch.cat([state.tokens, next_token], dim=0)
        return state

    def update_kv_cache(self, request_id: int, kv_cache: RequestKVCache) -> RequestState:
        state = self.require(request_id)
        state.kv_cache = kv_cache
        # Once cache length matches the request length, the full prompt tokens
        # are no longer needed for incremental decode and can be dropped.
        if state.tokens is not None and state.tokens.numel() == state.seq_len:
            state.tokens = None
        return state

    def remove(self, request_ids: Iterable[int]) -> None:
        for request_id in request_ids:
            state = self._states.pop(int(request_id), None)
            if state is not None:
                state.active = False
                self._release_cache_slot(state.cache_slot)

    def require(self, request_id: int) -> RequestState:
        request_id = int(request_id)
        if request_id not in self._states:
            raise KeyError(f"Unknown request_id: {request_id}")
        return self._states[request_id]

    def active_request_ids(self) -> list[int]:
        return list(self._states.keys())

    def active_states(self) -> list[RequestState]:
        return list(self._states.values())

    def _allocate_cache_slot(self) -> int:
        if self._free_cache_slots:
            return self._free_cache_slots.pop()
        slot = self._next_cache_slot
        self._next_cache_slot += 1
        return slot

    def _release_cache_slot(self, cache_slot: int | None) -> None:
        if cache_slot is None:
            return
        self._free_cache_slots.append(cache_slot)
