"""KV cache abstractions for incremental decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.key.size(-2))


@dataclass
class RequestKVCache:
    layers: list[LayerKVCache]

    @property
    def seq_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].seq_len

    def num_layers(self) -> int:
        return len(self.layers)


def stack_request_caches(caches: Iterable[RequestKVCache]) -> RequestKVCache:
    caches = list(caches)
    if not caches:
        raise ValueError("Cannot stack an empty cache list")
    seq_lens = {cache.seq_len for cache in caches}
    if len(seq_lens) != 1:
        raise ValueError("All caches must have the same sequence length to stack")

    num_layers = caches[0].num_layers()
    stacked_layers: list[LayerKVCache] = []
    for layer_index in range(num_layers):
        # Each per-request cache stores batch dimension = 1. Concatenating on
        # dim 0 gives us a temporary decode batch without changing layout.
        keys = [cache.layers[layer_index].key for cache in caches]
        values = [cache.layers[layer_index].value for cache in caches]
        stacked_layers.append(
            LayerKVCache(
                key=torch.cat(keys, dim=0),
                value=torch.cat(values, dim=0),
            )
        )
    return RequestKVCache(stacked_layers)


def split_request_cache(cache: RequestKVCache) -> list[RequestKVCache]:
    if not cache.layers:
        return []
    batch_size = cache.layers[0].key.size(0)
    per_request_layers: list[list[LayerKVCache]] = [[] for _ in range(batch_size)]
    for layer in cache.layers:
        # The inverse of stack_request_caches(): restore batch=1 caches so the
        # request table can own them independently after this decode step.
        key_chunks = layer.key.split(1, dim=0)
        value_chunks = layer.value.split(1, dim=0)
        for request_layers, key_chunk, value_chunk in zip(per_request_layers, key_chunks, value_chunks):
            request_layers.append(
                LayerKVCache(
                    key=key_chunk,
                    value=value_chunk,
                )
            )
    return [RequestKVCache(layers) for layers in per_request_layers]
