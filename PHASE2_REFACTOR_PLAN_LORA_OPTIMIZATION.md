# Phase 2 Refactor Plan - LoRA Optimization Agent

## Goal

Refactor the current Phase 1 hardware intrinsic profiling agent into a Phase 2 LoRA CUDA optimization agent for the operator:

`Y = W X + A(B^T X)`

The Phase 2 system must:

- run from `bash run.sh`
- iteratively generate, compile, verify, benchmark, profile, and improve CUDA code
- always keep a latest valid `./optimized_lora.cu` in the submission root
- optimize across multiple hidden sizes `d in [3584, 4608]`
- satisfy correctness before performance matters

This is a workflow shift, not a small feature addition.

## Official Requirement Summary

## 1. Required Deliverable

The evaluator will:

- run `bash run.sh` at repository root
- read the final `./optimized_lora.cu`
- benchmark that file using the official harness

Therefore `optimized_lora.cu` is the true evaluated artifact.

## 2. Correctness Requirement

Correctness is a hard gate.

The student implementation must match a PyTorch reference using:

```python
torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
```

Additional recorded metrics:

- `max_abs_err`
- `rel_l2_err`

If correctness fails, performance score is effectively lost.

## 3. Performance Requirement

Among correct submissions, score is:

- `70%` speedup
- `30%` agent implementation / engineering methodology

Speedup is based on median latency:

- warmup first
- CUDA events
- repeated runs
- median runtime vs standard PyTorch implementation

## 4. Prohibited Patterns

The following are disallowed:

- submitting only a static final kernel
- hardcoding a hidden final `optimized_lora.cu` template and dumping it at runtime
- depending on extra source files for the final measured implementation
- breaking the `bash run.sh` + `./optimized_lora.cu` evaluator contract

## Current Repository Status

## 1. What Can Be Reused

The current codebase already provides useful infrastructure:

- LLM client with OpenAI-compatible API support via `API_KEY`
- multi-agent coordinator, planner, executor, interpreter
- compile/run/profile tool adapters
- iterative probe generation and repair concepts
- evidence/analysis tracing
- detector and reliability logic
- runtime budget / execution shell structure in `run.sh`

This means we do not need to build the agent shell from scratch.

## 2. What Is Still Phase-1-Centric

The current mainline is still designed around:

- `target_spec.json`
- `results.json`
- `evidence.json`
- `analysis.json`
- hardware metric measurement

That is the wrong primary abstraction for Phase 2, where the main object is:

- candidate CUDA implementation quality
- correctness
- speedup
- current best `optimized_lora.cu`

## 3. Existing Phase-2-Oriented Prework

The repository already contains hints of upcoming adaptation:

- `profiler_agent/target_strategies/device_attributes.py`
- `profiler_agent/runtime_tools.py`
- `profiler_agent/probe_iteration.py`
- `tests/test_metric_specs_phase2.py`
- `tests/test_executor_phase3.py`

These are useful supporting pieces, but they do not yet form a full Phase 2 LoRA optimization loop.

## Gap Analysis

## 1. Missing Final Artifact Pipeline

Current project does not revolve around maintaining:

- root-level `optimized_lora.cu`

Needed behavior:

- every successful improvement round updates the current best `optimized_lora.cu`
- timeout or interruption still leaves the latest valid version in place

## 2. Missing LoRA-Specific Correctness Harness

Current analyzer and probe logic do not provide a dedicated Phase 2 correctness harness for:

- building `W, X, A, B`
- evaluating `Y_ref`
- checking `torch.allclose`
- recording `max_abs_err`, `rel_l2_err`

This is mandatory.

## 3. Missing LoRA-Specific Benchmark Harness

Current project lacks a dedicated workload benchmarker that:

- warms up
- runs repeated CUDA event timing
- computes median latency
- compares PyTorch baseline vs generated CUDA candidate
- computes speedup

This is mandatory.

## 4. Wrong LLM Codegen Target

Current LLM codegen is focused on intrinsic probes / microbenchmarks.

Phase 2 needs LLM to generate:

- final candidate `optimized_lora.cu`
- single-file implementation only
- operator-specific kernel + binding code
- code revisions based on correctness/benchmark/profile feedback

## 5. Missing Candidate Management

Current project lacks a first-class candidate evolution loop for:

- candidate id / version
- compile status
- correctness result
- benchmark result
- profile summary
- best-so-far decision

Phase 2 requires genuine iterative optimization, not just one-shot generation.

## 6. `run.sh` Contract Mismatch

Current `run.sh` still packages Phase 1 style outputs into `/workspace/output.json`.

For Phase 2, `run.sh` must instead prioritize:

- running the optimization loop
- maintaining root `optimized_lora.cu`
- optionally still writing traces/artifacts for debugging

The Phase 1 JSON artifacts may still be useful internally, but they are no longer the primary product.

## Proposed Refactor Direction

## 1. Add a Dedicated Phase 2 Workflow

Add a new module family, for example:

- `profiler_agent/phase2/optimizer.py`
- `profiler_agent/phase2/harness.py`
- `profiler_agent/phase2/candidate_store.py`
- `profiler_agent/phase2/prompts.py`
- `profiler_agent/phase2/reporting.py`

This workflow should be separate from the Phase 1 metric pipeline.

## 2. Preserve Existing Infrastructure as Tooling

Reuse current code as supporting layers:

- `multi_agent/*` for orchestration
- `multi_agent/llm_client.py` for LLM calls
- `runtime_tools.py` for command probing
- `tool_adapters/*` for compile/run/profile helpers
- detector/fusion ideas where useful

But do not force the Phase 2 logic into the current target-metric pipeline.

## 3. Make `optimized_lora.cu` a First-Class State Object

The optimizer loop should always manage:

- current candidate source
- current best valid source
- root-level `optimized_lora.cu`

Recommended rule:

- only promote a candidate to `optimized_lora.cu` after it passes correctness
- if several pass correctness, keep the fastest

## 4. Introduce a LoRA Optimization State

Suggested persisted artifact:

- `.agent_artifacts/phase2_state.json`

Suggested fields:

- `iteration`
- `candidate_history`
- `current_best_candidate_id`
- `best_correctness`
- `best_runtime_ms`
- `best_speedup`
- `compile_errors`
- `correctness_failures`
- `profile_summaries`
- `llm_revision_history`
- `done`

## 5. Separate Core Stages Explicitly

Each iteration should have distinct stages:

1. generate candidate CUDA code
2. compile candidate
3. load/execute candidate
4. run correctness check against PyTorch
5. benchmark with warmup + median latency
6. optionally profile with `ncu`
7. ask LLM to revise based on structured feedback
8. decide whether to keep candidate as current best

This separation makes debugging and judging easier.

## Proposed Module Map

## 1. New Files

- `profiler_agent/phase2/optimizer.py`
  - top-level iterative optimization loop
- `profiler_agent/phase2/harness.py`
  - correctness + benchmark harness
- `profiler_agent/phase2/candidate_store.py`
  - best candidate tracking and promotion
- `profiler_agent/phase2/prompts.py`
  - LLM prompts for operator optimization
- `profiler_agent/phase2/models.py`
  - typed state/result objects
- `profiler_agent/phase2/io.py`
  - write `optimized_lora.cu`, state snapshots, summaries

## 2. Existing Files to Reuse / Adapt

- `profiler_agent/multi_agent/llm_client.py`
- `profiler_agent/runtime_tools.py`
- `profiler_agent/multi_agent/executor.py`
- `run.sh`

## 3. Existing Files to Demote to Phase-1-Specific

These should no longer be the mainline for Phase 2:

- `profiler_agent/orchestrator/pipeline.py`
- `profiler_agent/target_strategies/*`
- `profiler_agent/analyzer/*`
- `results.json / evidence.json / analysis.json` as primary outputs

They can remain in repo for Phase 1 continuity, but Phase 2 should not be architecturally blocked by them.

## Implementation Plan

## Phase A - Harness First

Goal:

- create a local correctness + benchmark harness before changing orchestration

Tasks:

- implement PyTorch reference for LoRA operator
- implement compile/load path for candidate CUDA
- implement correctness checker
- implement latency benchmarker
- define structured result object

Acceptance:

- given a candidate `optimized_lora.cu`, local harness can report:
  - compile ok/fail
  - correctness pass/fail
  - `max_abs_err`
  - `rel_l2_err`
  - student median latency
  - PyTorch median latency
  - speedup

## Phase B - Candidate Loop

Goal:

- create genuine iterative candidate generation and comparison

Tasks:

- add candidate history state
- generate first candidate via LLM
- feed back compiler/runtime/correctness/profile errors
- maintain best valid candidate

Acceptance:

- optimizer can run multiple iterations and keep best passing candidate

## Phase C - `run.sh` Contract Shift

Goal:

- align evaluation entrypoint with Phase 2 requirements

Tasks:

- switch `run.sh` primary behavior to Phase 2 optimizer
- ensure `optimized_lora.cu` exists in repo root after every successful round
- optionally keep artifact logging under `.agent_artifacts`

Acceptance:

- evaluator can run `bash run.sh` and then consume `./optimized_lora.cu`

## Phase D - Profiling-Guided Revision

Goal:

- use `ncu` and benchmark feedback as real optimization signal

Tasks:

- collect selected `ncu` metrics for candidate implementations
- summarize bottlenecks for LLM revision prompts
- compare candidate profile signatures

Acceptance:

- later candidates are revised from measured evidence, not only compiler fixes

## Phase E - Submission Hardening

Goal:

- make the system evaluator-ready

Tasks:

- ensure single-file final implementation
- verify no hidden hardcoded final kernel
- verify key comes only from `API_KEY`
- add pre-submit checks for `optimized_lora.cu` presence and contract

Acceptance:

- repository is Phase 2 submission-ready

## Recommended Immediate Next Steps

1. Build the Phase 2 harness first.
2. Create a root `optimized_lora.cu` management flow.
3. Add candidate history and best-candidate promotion.
4. Repoint `run.sh` to the new Phase 2 loop only after harness stability.

## Success Criteria

The Phase 2 refactor is successful when:

- `bash run.sh` drives a LoRA optimization loop
- `optimized_lora.cu` is always present and self-contained
- correctness is locally verifiable
- speedup is locally measurable
- the agent iteratively improves candidates rather than dumping a hidden final answer
- current Phase 1 infrastructure is reused as tooling, not forced as the core workflow

