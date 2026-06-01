"""Layer building blocks such as attention, RMSNorm, and MLP."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import LayerKVCache
from .rope import RotaryEmbeddingCache, apply_rotary_pos_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope_cache = RotaryEmbeddingCache(
            head_dim=self.head_dim,
            theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor | None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            position_ids=position_ids,
            rope_cache=self.rope_cache,
        )

        if past_key_value is not None:
            # Decode appends one new token to the cached prefix for each layer.
            key_states = torch.cat([past_key_value.key, key_states], dim=-2)
            value_states = torch.cat([past_key_value.value, value_states], dim=-2)

        present_key_value = None
        if use_cache:
            present_key_value = LayerKVCache(key=key_states, value=value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = self._attention(query_states, key_states, value_states, causal_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output), present_key_value

    def _attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            if causal_mask is not None:
                attn_mask = causal_mask.view(1, 1, query_states.size(-2), key_states.size(-2))
            else:
                attn_mask = None
            # Prefer SDPA on modern PyTorch/CUDA builds; fall back to the
            # explicit matmul path if the backend or shape combination refuses it.
            return F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
        except Exception:
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask.view(1, 1, query_states.size(-2), key_states.size(-2))
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            return torch.matmul(attn_weights, value_states)


class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor | None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            causal_mask=causal_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, present_key_value


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, num_repeats, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * num_repeats, seq_len, head_dim)
