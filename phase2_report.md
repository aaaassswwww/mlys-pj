# Phase 2 Report

## Overview

Phase 2 is responsible for taking the current LoRA kernel candidate, evaluating it for correctness and speed, and iteratively improving it within a runtime budget.

The work in this phase evolved substantially during debugging. The biggest change was a shift away from a permissive local CUDA evaluation path toward a workflow that more closely matches the real submission contract.

At a high level, Phase 2 now does the following:

1. Generates or mutates candidate implementations.
2. Evaluates candidates under a PyTorch-extension-style loading path.
3. Tracks the best overall candidate and the best correctness-passing candidate.
4. Switches from correctness-first search to speedup-focused search once correctness is achieved.
5. Persists state and report artifacts so the run can be inspected after completion or interruption.

## Main Problems We Encountered

### 1. Local evaluation did not match the real submission contract

Earlier Phase 2 evaluation compiled CUDA code with a command like:

```bash
nvcc optimized_lora.cu -O3 -std=c++14 -Xcompiler -fPIC -lcublas -shared -o optimized_lora.so
```

and then loaded the library with `ctypes`.

This was too permissive. It allowed candidates that directly used raw cuBLAS symbols such as:

- `cublasGemmEx`
- `cublasSetMathMode`
- `cublasSaxpy`

Those candidates could pass local evaluation but fail under the real evaluator, which compiles through `torch.utils.cpp_extension.load(...)` and does not automatically link raw cuBLAS symbols the same way.

This mismatch explained hidden-eval failures such as:

- `undefined symbol: cublasSetMathMode`
- `undefined symbol: cublasGemmEx`

### 2. Correctness-safe cuBLAS families were locally good but submission-unsafe

We explored several correctness-safe families built around explicit cuBLAS calls. These were effective at passing local correctness checks, but they were ultimately not reliable for final submission because their linkage assumptions did not hold in the PyTorch extension toolchain.

### 3. ATen/PyTorch extension candidates initially failed for toolchain reasons

After shifting to a middle route based on ATen operations such as `torch::matmul` and `torch::addmm`, we hit several issues:

- old evaluator still tried to precompile them with raw `nvcc`
- missing PyTorch header include paths in that path
- wrapper duplication causing multiple `PyInit_*` definitions
- inconsistent source detection between the main process and the runtime worker
- mixed C++ standard settings in the extension build path

These problems were fixed incrementally.

## Architectural Direction We Settled On

The current preferred Phase 2 strategy is the ATen-backed middle route:

1. Avoid direct raw cuBLAS API calls in final candidates.
2. Generate single-file PyTorch extension CUDA sources.
3. Export:
   - `torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)`
   - `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`
4. Evaluate candidates using `torch.utils.cpp_extension.load(...)`.

This is much closer to the actual submission/evaluation contract.

## Important Code Changes

### Candidate generation

File:

- `profiler_agent/phase2/generator.py`

Key changes:

1. Added direct-extension ATen candidates.
2. Bootstrap candidate now uses the ATen/PyTorch extension path.
3. Source validation now accepts two contracts:
   - legacy `launch_optimized_lora(...)`
   - direct PyTorch extension `forward(...) + PYBIND11_MODULE(...)`
4. Deterministic middle-route candidates were added, including variants over:
   - `bt_view`
   - `bt_contiguous`
   - `inplace`
   - `out`
   - `functional`

### Runtime evaluation

File:

- `profiler_agent/phase2/evaluator.py`

Key changes:

1. ATen/PyTorch-extension candidates now skip the old raw `nvcc` precompile path.
2. Runtime evaluation uses `torch.utils.cpp_extension.load(...)`.
3. Direct-extension candidates are loaded as a single source file.
4. Legacy `launch_optimized_lora(...)` candidates still use an automatically generated wrapper when needed.
5. Candidate runtime error notes now include the actual exception message instead of only the exception type.

### Worker consistency fix

Files:

- `profiler_agent/phase2/evaluator.py`
- `profiler_agent/phase2/runtime_eval_worker.py`

Problem fixed:

The subprocess worker rebuilt candidates with `source_code=""`, which caused the evaluator to misclassify direct-extension candidates as legacy wrapped candidates. That produced duplicate `PyInit_*` definitions at link time.

Fix:

When `candidate.source_code` is empty, the evaluator now reads `optimized_lora.cu` from disk and detects whether the candidate is already a direct PyTorch extension source.

### State/report bookkeeping

Files:

- `profiler_agent/phase2/models.py`
- `profiler_agent/phase2/optimizer.py`
- `profiler_agent/phase2/candidate_store.py`

Key change:

Added:

- `last_completed_iteration`

This distinguishes:

- the iteration that has started (`iterations_run`)
- the last iteration that actually completed generation, evaluation, and recording (`last_completed_iteration`)

This was necessary because some runs were terminated externally and produced mid-run snapshots with:

- `iterations_run > last_completed_iteration`
- empty `stop_reason`

### Correctness-first to speedup-after-correctness transition

Files:

- `profiler_agent/phase2/optimizer.py`
- `profiler_agent/phase2/prompts.py`

Key changes:

1. Once a correctness-passing candidate exists, speedup iterations prefer:
   - `current_best_correct_candidate`
2. Fatal runtime errors no longer immediately terminate the whole search if a known-correct candidate exists.
3. The search now behaves more like:
   - find a correct anchor
   - optimize cautiously around that anchor

## Current ATen Middle-Route Search Behavior

The best-performing and correctness-passing family in recent runs is a `bt_view` family, particularly:

- `aten_inplace_addmm_bt_view-*`
- `aten_out_addmm_bt_view-*`

Observed behavior:

1. `bt_contiguous` variants frequently fail correctness.
2. `functional` variants tend to be slower.
3. `bt_view` variants are more stable.
4. `inplace bt_view` has often been the best speedup candidate.

Because of this, we narrowed the deterministic speedup search.

### Search narrowing that is now in place

1. Once `bt_view` wins as the best correct family, deterministic speedup search no longer bounces back to `bt_contiguous`.
2. The speedup search now strongly favors `inplace bt_view`.
3. Low-value branches such as repeated `functional` exploration are heavily reduced.

## Recent Template-Level Optimizations

The ATen template was simplified to reduce unnecessary overhead:

1. Removed unconditional:
   - `W.contiguous()`
   - `X.contiguous()`
   - `A.contiguous()`
   - `B.contiguous()`
2. Removed repeated per-call checks that were less valuable on the hot path:
   - `is_cuda()`
   - `scalar_type()`
3. Kept essential shape validation.
4. Preserved the option to use `B.transpose(0, 1).contiguous()` only in the explicit `bt_contiguous` variant.
5. Removed extra copy-heavy patterns from earlier templates.

Current template style:

- `inplace`: compute `out = torch::matmul(W, X)`, then `out.addmm_(A, temp, 1.0, 1.0)`
- `out`: compute `wx`, then `at::addmm_out(...)`
- `functional`: directly return `torch::addmm(...)`

## What the Reports Mean Now

Phase 2 reports now include enough information to interpret incomplete runs more safely.

Important fields:

- `iterations_run`: the iteration number the optimizer has started
- `last_completed_iteration`: the most recent iteration that finished fully
- `current_best_correct_candidate_id`: the current correctness-passing anchor
- `best_speedup`: best speedup measured by the local harness
- `stop_reason`: reason for a normal stop, when available

Interpretation examples:

1. If:
   - `iterations_run = 8`
   - `last_completed_iteration = 7`

   then iteration 8 started but did not finish recording.

2. If:
   - `stop_reason = ""`

   and `iterations_run > last_completed_iteration`, the file is probably a mid-run snapshot or the process was externally interrupted.

## Why Recent Runs Complete Fewer Iterations

Earlier versions sometimes completed 30+ iterations because many candidates failed early:

- compile failure
- load failure
- immediate runtime failure

Those failures made each iteration cheap.

Now that evaluation is closer to the real contract, each completed iteration is more expensive because it often includes:

1. real `torch.utils.cpp_extension.load(...)` compilation
2. real correctness evaluation over multiple shapes
3. benchmark runs for correctness-passing candidates

This is why fewer iterations can still represent better, more realistic progress.

## Current Strengths

1. Local evaluation is much closer to the actual submission contract.
2. The system can now find correctness-passing ATen-based candidates.
3. The search is more disciplined after correctness is achieved.
4. Reports are significantly easier to interpret.
5. The best current family is simpler, safer, and more submission-aligned than the earlier raw-cuBLAS families.

## Remaining Limitations

1. Local `best_speedup` in the Phase 2 report does not necessarily numerically match external evaluation speedup.
2. Outer process timeout can still interrupt runs and leave mid-run snapshots.
3. The search still spends time on some lower-value variants, though much less than before.
4. Plateau detection is not yet fully automated.

## Recommended Next Steps

If we continue improving Phase 2, the most promising next steps are:

1. Add plateau early stopping once the best-correct ATen family has clearly stabilized.
2. Further align local speed benchmarking with the final evaluator’s observed timing behavior.
3. Keep search centered on:
   - `aten_inplace_addmm_bt_view-*`
4. Avoid re-expanding the space toward:
   - raw cuBLAS direct-call families
   - `bt_contiguous` branches
   - broad freeform LLM exploration

## Current Overall Assessment

Phase 2 is in a much better state than its original form.

The most important improvements were not cosmetic. They were contract-level fixes:

1. align candidate format with PyTorch extension loading
2. align runtime evaluation with the real build/load path
3. move away from submission-unsafe raw cuBLAS candidates
4. narrow the speedup search around the best correctness-passing ATen family

At this point, Phase 2 is no longer primarily blocked by correctness architecture. It is now mainly in the regime of targeted performance refinement around a known-good ATen-backed implementation family.
