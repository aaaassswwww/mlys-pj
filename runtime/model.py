"""Model execution path for the decoder-only runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .cache import LayerKVCache, RequestKVCache
from .layers import DecoderLayer, RMSNorm


@dataclass(frozen=True)
class LlamaLikeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaLikeConfig":
        def pick(*names: str, default: Optional[Any] = None) -> Any:
            for name in names:
                if name in data:
                    return data[name]
            if default is not None:
                return default
            raise KeyError(f"Missing required config field. Tried: {names}")

        return cls(
            vocab_size=int(pick("vocab_size")),
            hidden_size=int(pick("hidden_size", "dim")),
            intermediate_size=int(pick("intermediate_size", "ffn_hidden_size")),
            num_hidden_layers=int(pick("num_hidden_layers", "n_layers")),
            num_attention_heads=int(pick("num_attention_heads", "n_heads")),
            num_key_value_heads=int(pick("num_key_value_heads", "n_kv_heads", default=pick("num_attention_heads", "n_heads"))),
            rms_norm_eps=float(pick("rms_norm_eps", "norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=10000.0)),
            max_position_embeddings=int(pick("max_position_embeddings", default=4096)),
        )

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


class LlamaLikeModel(nn.Module):
    def __init__(self, config: LlamaLikeConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: RequestKVCache | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, RequestKVCache | None]:
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            # Prefill uses a fresh contiguous range. Decode passes explicit
            # positions so the new token lands after the cached prefix.
            start = past_key_values.seq_len if past_key_values is not None else 0
            position_ids = torch.arange(start, start + input_ids.size(1), device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.size(0), -1)

        causal_mask = None
        if input_ids.size(1) > 1 and past_key_values is None:
            # Only prompt prefill needs a causal matrix. Single-token decode
            # relies on the cache and does not need an explicit square mask.
            causal_mask = torch.triu(
                torch.full((input_ids.size(1), input_ids.size(1)), float("-inf"), device=input_ids.device),
                diagonal=1,
            )

        next_caches: list[LayerKVCache] = []
        for layer_index, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values.layers[layer_index]
            hidden_states, present_key_value = layer(
                hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present_key_value is not None:
                next_caches.append(present_key_value)
        hidden_states = self.norm(hidden_states)
        return hidden_states, RequestKVCache(next_caches) if use_cache else None


class LlamaLikeForCausalLM(nn.Module):
    def __init__(self, config: LlamaLikeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaLikeModel(config)
        object.__setattr__(self, "_eager_model", self.model)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._compile_enabled = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.model(input_ids)
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def logits_for_last_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return logits[:, -1, :]

    @torch.inference_mode()
    def logits_and_cache_for_prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, RequestKVCache]:
        hidden_states, kv_cache = self.model(input_ids, use_cache=True)
        logits = self.lm_head(hidden_states)[:, -1, :]
        assert kv_cache is not None
        return logits, kv_cache

    @torch.inference_mode()
    def logits_and_cache_for_prefill_batch(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, RequestKVCache]:
        hidden_states, kv_cache = self.model(input_ids, use_cache=True)
        logits = self.lm_head(hidden_states)[:, -1, :]
        assert kv_cache is not None
        return logits, kv_cache

    @torch.inference_mode()
    def logits_and_cache_for_decode_step(
        self,
        token_ids: torch.Tensor,
        kv_cache: RequestKVCache,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, RequestKVCache]:
        hidden_states, next_cache = self.model(
            token_ids,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = self.lm_head(hidden_states)[:, -1, :]
        assert next_cache is not None
        return logits, next_cache

    @torch.inference_mode()
    def logits_and_cache_for_decode_batch(
        self,
        token_ids: torch.Tensor,
        kv_cache: RequestKVCache,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, RequestKVCache]:
        hidden_states, next_cache = self.model(
            token_ids,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = self.lm_head(hidden_states)[:, -1, :]
        assert next_cache is not None
        return logits, next_cache

    def try_enable_compile(self) -> bool:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return False
        if self._compile_enabled:
            return True
        try:
            dynamo = getattr(torch, "_dynamo", None)
            if dynamo is not None and hasattr(dynamo, "config"):
                dynamo.config.suppress_errors = True
            # The eager model is retained so compile can be treated as an
            # optional acceleration path instead of a hard dependency.
            self.model = compile_fn(self._eager_model, mode="default", fullgraph=False, dynamic=True)
            self._compile_enabled = True
            return True
        except Exception:
            self.disable_compile()
            return False

    def disable_compile(self) -> None:
        self.model = self._eager_model
        self._compile_enabled = False
