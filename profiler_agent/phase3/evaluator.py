from __future__ import annotations

import importlib.util
import math
import tempfile
import time
from pathlib import Path

import torch

from profiler_agent.phase3.generator import build_baseline_engine_source
from profiler_agent.phase3.models import (
    GeneratedRuntimeCandidate,
    RuntimeBenchmarkResult,
    RuntimeCorrectnessResult,
    RuntimeEvaluation,
)
from tools.runtime_fixture import create_toy_runtime_artifacts
from runtime.loader import load_model


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _compare_logits(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).float()
    max_abs_err = float(diff.abs().max().item())
    rel_l2_err = float((diff.norm() / (rhs.float().norm() + 1e-12)).item())
    return max_abs_err, rel_l2_err


def _reference_last_logits(model, sequences: list[torch.Tensor]) -> torch.Tensor:
    rows = []
    for sequence in sequences:
        rows.append(model.logits_for_last_token(sequence.view(1, -1)).squeeze(0))
    return torch.stack(rows, dim=0)


def _run_correctness_cases(engine_module, *, fixture_name: str) -> RuntimeCorrectnessResult:
    config, weight_dir = create_toy_runtime_artifacts(fixture_name, seed=29)
    device = _resolve_device()
    engine = engine_module.create_engine(config, weight_dir, device=device)
    reference = load_model(config, weight_dir, device=device)

    prompts = [
        torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device),
        torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device),
    ]
    request_ids = [11, 22]
    logits = engine.prefill(request_ids, prompts)
    reference_logits = _reference_last_logits(reference, prompts)
    max_abs_err, rel_l2_err = _compare_logits(logits, reference_logits)
    passed = torch.allclose(logits, reference_logits, rtol=1e-4, atol=1e-4)

    next_tokens = torch.tensor([9, 10], dtype=torch.long, device=device)
    decode_logits = engine.decode(request_ids, next_tokens)
    decode_sequences = [torch.cat([prompt, next_tokens[i].view(1)], dim=0) for i, prompt in enumerate(prompts)]
    reference_decode_logits = _reference_last_logits(reference, decode_sequences)
    decode_max_abs_err, decode_rel_l2_err = _compare_logits(decode_logits, reference_decode_logits)
    passed = passed and torch.allclose(decode_logits, reference_decode_logits, rtol=1e-4, atol=1e-4)

    engine.remove([11])
    single_logits = engine.decode([22], torch.tensor([11], dtype=torch.long, device=device))
    single_sequence = [torch.cat([decode_sequences[1], torch.tensor([11], dtype=torch.long, device=device)], dim=0)]
    reference_single_logits = _reference_last_logits(reference, single_sequence)
    single_max_abs_err, single_rel_l2_err = _compare_logits(single_logits, reference_single_logits)
    passed = passed and torch.allclose(single_logits, reference_single_logits, rtol=1e-4, atol=1e-4)

    return RuntimeCorrectnessResult(
        passed=bool(passed),
        max_abs_err=max(max_abs_err, decode_max_abs_err, single_max_abs_err),
        rel_l2_err=max(rel_l2_err, decode_rel_l2_err, single_rel_l2_err),
        checked_cases=3,
    )


def _run_prefill_case(engine, request_ids, prompts) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    engine.prefill(request_ids, prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = sum(int(prompt.numel()) for prompt in prompts)
    return tokens / max(elapsed, 1e-9)


def _run_decode_case(engine, request_ids, vocab_size: int, decode_steps: int, device: str) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(decode_steps):
        token_ids = torch.randint(0, vocab_size, (len(request_ids),), dtype=torch.long, device=device)
        engine.decode(request_ids, token_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = len(request_ids) * decode_steps
    return tokens / max(elapsed, 1e-9)


def _run_mixed_case(engine, batch_size: int, prompt_len: int, decode_steps: int, vocab_size: int, device: str) -> float:
    active_request_ids: list[int] = []
    next_request_id = 0
    total_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(1, batch_size // 2)):
        request_id = next_request_id
        next_request_id += 1
        prompt = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        engine.prefill([request_id], [prompt])
        active_request_ids.append(request_id)
        total_tokens += int(prompt.numel())
    for step in range(decode_steps):
        if active_request_ids:
            token_ids = torch.randint(0, vocab_size, (len(active_request_ids),), dtype=torch.long, device=device)
            engine.decode(active_request_ids, token_ids)
            total_tokens += len(active_request_ids)
        if step % 3 == 0 and len(active_request_ids) < batch_size:
            request_id = next_request_id
            next_request_id += 1
            prompt = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
            engine.prefill([request_id], [prompt])
            active_request_ids.append(request_id)
            total_tokens += int(prompt.numel())
        if step % 4 == 0 and len(active_request_ids) > 1:
            remove_id = active_request_ids.pop(0)
            engine.remove([remove_id])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return total_tokens / max(elapsed, 1e-9)


def _benchmark_engine(engine_module, *, fixture_name: str) -> RuntimeBenchmarkResult:
    config, weight_dir = create_toy_runtime_artifacts(fixture_name, seed=31)
    device = _resolve_device()
    engine = engine_module.create_engine(config, weight_dir, device=device)
    vocab_size = config["vocab_size"]
    batch_size = 4
    prompt_len = 16
    decode_steps = 8
    prompts = [torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device) for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    prefill = _run_prefill_case(engine, request_ids, prompts)
    engine = engine_module.create_engine(config, weight_dir, device=device)
    engine.prefill(request_ids, prompts)
    decode = _run_decode_case(engine, request_ids, vocab_size, decode_steps, device)
    engine = engine_module.create_engine(config, weight_dir, device=device)
    mixed = _run_mixed_case(engine, batch_size, prompt_len, decode_steps, vocab_size, device)
    aggregate = (prefill + decode + mixed) / 3.0
    return RuntimeBenchmarkResult(
        prefill_tokens_per_s=prefill,
        decode_tokens_per_s=decode,
        mixed_tokens_per_s=mixed,
        aggregate_tokens_per_s=aggregate,
    )


def build_phase3_candidate_evaluator(*, root_dir: Path):
    def evaluate(candidate: GeneratedRuntimeCandidate) -> RuntimeEvaluation:
        candidate_dir = root_dir / ".agent_artifacts" / "phase3_candidates" / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        source_path = candidate_dir / "engine.py"
        source_path.write_text(candidate.source_code, encoding="utf-8")
        notes: list[str] = []
        try:
            compile(candidate.source_code, str(source_path), "exec")
        except SyntaxError as exc:
            correctness = RuntimeCorrectnessResult(False, math.inf, math.inf, 0)
            empty = RuntimeBenchmarkResult(0.0, 0.0, 0.0, 0.0)
            notes.append(f"compile_failed:{type(exc).__name__}:{exc}")
            return RuntimeEvaluation(candidate.candidate_id, correctness, empty, empty, 0.0, notes)

        try:
            module = _load_module_from_path(f"phase3_candidate_{candidate.candidate_id.replace('-', '_')}", source_path)
        except Exception as exc:
            correctness = RuntimeCorrectnessResult(False, math.inf, math.inf, 0)
            empty = RuntimeBenchmarkResult(0.0, 0.0, 0.0, 0.0)
            notes.append(f"import_failed:{type(exc).__name__}:{exc}")
            return RuntimeEvaluation(candidate.candidate_id, correctness, empty, empty, 0.0, notes)

        try:
            correctness = _run_correctness_cases(module, fixture_name=f"phase3_correctness_{candidate.candidate_id}")
        except Exception as exc:
            correctness = RuntimeCorrectnessResult(False, math.inf, math.inf, 0)
            empty = RuntimeBenchmarkResult(0.0, 0.0, 0.0, 0.0)
            notes.append(f"correctness_runtime_error:{type(exc).__name__}:{exc}")
            return RuntimeEvaluation(candidate.candidate_id, correctness, empty, empty, 0.0, notes)

        baseline_path = candidate_dir / "baseline_engine.py"
        baseline_path.write_text(build_baseline_engine_source(), encoding="utf-8")
        baseline_module = _load_module_from_path(f"phase3_baseline_{candidate.candidate_id.replace('-', '_')}", baseline_path)
        baseline_benchmark = _benchmark_engine(baseline_module, fixture_name=f"phase3_baseline_{candidate.candidate_id}")

        if not correctness.passed:
            empty = RuntimeBenchmarkResult(0.0, 0.0, 0.0, 0.0)
            notes.append("correctness_failed")
            return RuntimeEvaluation(candidate.candidate_id, correctness, empty, baseline_benchmark, 0.0, notes)

        try:
            benchmark = _benchmark_engine(module, fixture_name=f"phase3_benchmark_{candidate.candidate_id}")
        except Exception as exc:
            empty = RuntimeBenchmarkResult(0.0, 0.0, 0.0, 0.0)
            notes.append(f"benchmark_failed:{type(exc).__name__}:{exc}")
            return RuntimeEvaluation(candidate.candidate_id, correctness, empty, baseline_benchmark, 0.0, notes)

        speedup = 0.0
        if baseline_benchmark.aggregate_tokens_per_s > 0:
            speedup = benchmark.aggregate_tokens_per_s / baseline_benchmark.aggregate_tokens_per_s
        notes.append("correctness_passed_all_specs")
        return RuntimeEvaluation(candidate.candidate_id, correctness, benchmark, baseline_benchmark, speedup, notes)

    return evaluate
