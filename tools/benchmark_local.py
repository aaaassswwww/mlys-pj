"""Local benchmark entrypoints for prefill, decode, and mixed traces."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import create_engine
from tools.runtime_fixture import create_toy_runtime_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local runtime benchmarks.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    config, weight_dir = create_toy_runtime_artifacts("benchmark_local")
    engine = create_engine(config, weight_dir, device=args.device)
    vocab_size = config["vocab_size"]

    prompts = [
        torch.randint(0, vocab_size, (args.prompt_len,), dtype=torch.long)
        for _ in range(args.batch_size)
    ]
    request_ids = list(range(args.batch_size))

    for _ in range(args.warmup):
        _run_prefill_case(engine, request_ids, prompts)
        _run_decode_case(engine, request_ids, vocab_size, args.decode_steps)

    prefill_results = []
    decode_results = []
    mixed_results = []
    for iteration in range(args.repeat):
        engine = create_engine(config, weight_dir, device=args.device)
        prefill_results.append(_run_prefill_case(engine, request_ids, prompts))

        engine = create_engine(config, weight_dir, device=args.device)
        engine.prefill(request_ids, prompts)
        decode_results.append(_run_decode_case(engine, request_ids, vocab_size, args.decode_steps))

        engine = create_engine(config, weight_dir, device=args.device)
        mixed_results.append(_run_mixed_case(engine, args.batch_size, args.prompt_len, args.decode_steps, vocab_size))

    print(_format_result("prefill", prefill_results))
    print(_format_result("decode", decode_results))
    print(_format_result("mixed", mixed_results))


def _run_prefill_case(engine, request_ids, prompts):
    start = time.perf_counter()
    engine.prefill(request_ids, prompts)
    elapsed = time.perf_counter() - start
    tokens = sum(int(prompt.numel()) for prompt in prompts)
    return tokens / max(elapsed, 1e-9)


def _run_decode_case(engine, request_ids, vocab_size: int, decode_steps: int):
    start = time.perf_counter()
    for _ in range(decode_steps):
        token_ids = torch.randint(0, vocab_size, (len(request_ids),), dtype=torch.long)
        engine.decode(request_ids, token_ids)
    elapsed = time.perf_counter() - start
    tokens = len(request_ids) * decode_steps
    return tokens / max(elapsed, 1e-9)


def _run_mixed_case(engine, batch_size: int, prompt_len: int, decode_steps: int, vocab_size: int):
    active_request_ids: list[int] = []
    next_request_id = 0
    total_tokens = 0
    start = time.perf_counter()

    for _ in range(batch_size // 2):
        request_id = next_request_id
        next_request_id += 1
        prompt = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long)
        engine.prefill([request_id], [prompt])
        active_request_ids.append(request_id)
        total_tokens += int(prompt.numel())

    for step in range(decode_steps):
        if active_request_ids:
            token_ids = torch.randint(0, vocab_size, (len(active_request_ids),), dtype=torch.long)
            engine.decode(active_request_ids, token_ids)
            total_tokens += len(active_request_ids)

        if step % 3 == 0 and len(active_request_ids) < batch_size:
            request_id = next_request_id
            next_request_id += 1
            prompt = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long)
            engine.prefill([request_id], [prompt])
            active_request_ids.append(request_id)
            total_tokens += int(prompt.numel())

        if step % 4 == 0 and len(active_request_ids) > 1:
            remove_id = active_request_ids.pop(0)
            engine.remove([remove_id])

    elapsed = time.perf_counter() - start
    return total_tokens / max(elapsed, 1e-9)


def _format_result(label: str, results: list[float]) -> str:
    avg = sum(results) / len(results)
    minimum = min(results)
    maximum = max(results)
    return f"{label}: avg_tokens_per_s={avg:.2f} min={minimum:.2f} max={maximum:.2f}"


if __name__ == "__main__":
    main()
