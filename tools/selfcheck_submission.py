"""Lightweight submission self-check for the generated root engine entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime_fixture import create_toy_runtime_artifacts


def main() -> None:
    engine_module = load_engine_module(ROOT / "engine.py")
    if not hasattr(engine_module, "create_engine"):
        raise RuntimeError("engine.py does not define create_engine")

    config, weight_dir = create_toy_runtime_artifacts("selfcheck_submission", seed=23)
    engine = engine_module.create_engine(config, weight_dir, device="cpu")

    logits = engine.prefill([1, 2], [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([4, 5, 6], dtype=torch.long),
    ])
    assert tuple(logits.shape) == (2, config["vocab_size"])

    logits = engine.decode([1, 2], torch.tensor([7, 8], dtype=torch.long))
    assert tuple(logits.shape) == (2, config["vocab_size"])

    engine.remove([1])
    logits = engine.decode([2], torch.tensor([9], dtype=torch.long))
    assert tuple(logits.shape) == (1, config["vocab_size"])

    print("[selfcheck] import=ok")
    print("[selfcheck] prefill_decode_remove=ok")


def load_engine_module(engine_path: Path):
    spec = importlib.util.spec_from_file_location("submission_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load engine module from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
