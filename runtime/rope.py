"""RoPE utilities."""

from __future__ import annotations

import torch


class RotaryEmbeddingCache:
    def __init__(self, head_dim: int, theta: float = 10000.0, max_position_embeddings: int = 4096) -> None:
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        self.head_dim = head_dim
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings
        self._inv_freq = None
        self._cos_cached = None
        self._sin_cached = None
        self._cache_device = None
        self._cos_cached_by_dtype: dict[torch.dtype, torch.Tensor] = {}
        self._sin_cached_by_dtype: dict[torch.dtype, torch.Tensor] = {}

    def get_cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        self._ensure_cache(device=device, required_positions=None)

        max_position = int(position_ids.max().item())
        if max_position >= self.max_position_embeddings:
            raise ValueError(
                f"Position id {max_position} exceeds max_position_embeddings={self.max_position_embeddings}"
            )

        cos_cached = self._cos_cached_by_dtype.get(dtype)
        sin_cached = self._sin_cached_by_dtype.get(dtype)
        if cos_cached is None or sin_cached is None:
            assert self._cos_cached is not None and self._sin_cached is not None
            cos_cached = self._cos_cached.to(dtype)
            sin_cached = self._sin_cached.to(dtype)
            self._cos_cached_by_dtype[dtype] = cos_cached
            self._sin_cached_by_dtype[dtype] = sin_cached

        flat_positions = position_ids.reshape(-1)
        cos = cos_cached.index_select(0, flat_positions).view(*position_ids.shape, -1)
        sin = sin_cached.index_select(0, flat_positions).view(*position_ids.shape, -1)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def _ensure_cache(self, device: torch.device, required_positions: int | None) -> None:
        if (
            self._cos_cached is not None
            and self._sin_cached is not None
            and self._cache_device == device
        ):
            return

        cache_len = self.max_position_embeddings if required_positions is None else max(required_positions, self.max_position_embeddings)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(cache_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._inv_freq = inv_freq
        self._cos_cached = freqs.cos()
        self._sin_cached = freqs.sin()
        self._cache_device = device
        self._cos_cached_by_dtype.clear()
        self._sin_cached_by_dtype.clear()


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    rope_cache: RotaryEmbeddingCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope_cache.get_cos_sin(position_ids, query_states.dtype)
    return rotate_interleaved(query_states, cos, sin), rotate_interleaved(key_states, cos, sin)


def rotate_interleaved(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    rotated = torch.empty_like(hidden_states)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated
