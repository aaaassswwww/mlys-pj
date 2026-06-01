"""Helpers for creating reproducible local runtime benchmark fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from runtime.model import LlamaLikeConfig, LlamaLikeForCausalLM


def create_toy_runtime_artifacts(
    fixture_name: str,
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 3,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    seed: int = 17,
) -> tuple[dict, str]:
    config = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }

    model = LlamaLikeForCausalLM(LlamaLikeConfig.from_dict(config))
    torch.manual_seed(seed)
    for parameter in model.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.02)

    temp_root = Path(__file__).resolve().parents[1] / ".tmp_tools"
    temp_root.mkdir(exist_ok=True)
    weight_dir = temp_root / fixture_name
    shutil.rmtree(weight_dir, ignore_errors=True)
    weight_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), weight_dir / "model.pt")
    return config, str(weight_dir)
