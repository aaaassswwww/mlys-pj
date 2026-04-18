# GPU Profiling Agent Refactor Plan

## Goal

Refactor the current target-driven GPU profiling agent toward a stronger iterative agent workflow inspired by the example `agent.py + run.py`, while preserving the current evaluator-facing contract:

- input: `target_spec.json`
- output: `results.json`, `evidence.json`, `analysis.json`
- `run.sh` still produces a single `/workspace/output.json`

This plan is designed for continuation in a new session. It emphasizes backward-compatible, staged refactoring rather than a disruptive rewrite.

## What The Example Workflow Does Better

The example workflow has a tighter control loop:

1. Load persistent state
2. Generate or update benchmark CUDA code
3. Compile and run a benchmark
4. Profile the benchmark with `ncu`
5. Ask LLM to analyze the result
6. Let LLM recommend the next focused metric set and next benchmark revision
7. Retry generation with explicit error context when code/profiling fails

The example runner also separates concerns clearly:

- compile benchmark
- run benchmark
- profile benchmark with explicit metric list
- return raw profiling output for analysis

## Current Project Strengths

The current project is stronger than the example in several ways:

- richer evaluator-facing output structure
- multi-source evidence (`ncu`, microbench, `nvml`)
- detector layer and confidence penalties
- LLM-aware probe generation and analysis
- explicit multi-agent trace and plan artifacts
- stable single/multi mode entrypoints

## Current Project Gaps Compared With The Example

### 1. Multi-agent loop is not a true iterative loop

Current flow is mostly:

- route
- plan
- run tools
- run pipeline
- interpret

The `iterative_refinement` step currently produces suggestions, not a real next execution round.

### 2. No durable agent state

The example keeps a persistent state file with:

- iteration number
- metrics history
- analysis history
- error history
- recommended metrics history
- current bottleneck
- done flag

Current project writes outputs and traces, but does not maintain a single durable control-state object that can drive the next iteration.

### 3. No benchmark/workload evolution loop

The example evolves benchmark code versions over time.

Current project only generates per-target probe code. It does not evolve workload-level measurement programs or benchmark variants based on prior profiling outcomes.

### 4. Target semantic routing is too weak

Current unknown targets mostly fall into `GenericMetricStrategy`.

That is insufficient for several metric families:

- device attributes
- Nsight Compute counters
- runtime throughput counters
- probe-friendly intrinsic measurements

This is the largest current architecture gap exposed by the server test.

### 5. Error context is recorded but not used as a primary planning signal

We do record:

- probe compile errors
- generation errors
- LLM HTTP/debug records
- tool failures in evidence

But planner and executor do not yet make strong routing decisions from that error history.

### 6. Tool execution is not isolated enough

The example runner isolates:

- compile
- binary run
- profiling

Current project does this partially, but Linux vs Windows behavior and tool invocation diagnostics are still too fragile, especially around dynamic probe compilation.

### 7. `run` became optional, but metric semantics are not adjusted enough

Supporting no-`run` specs was correct for schema flexibility, but some metric families should not be treated as equally measurable without an actual workload command.

Examples:

- `dram__bytes_read.sum.per_second`
- `dram__bytes_write.sum.per_second`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`

These should be explicitly marked workload-dependent.

## What The Server Test Revealed

The uploaded server output exposed four concrete weaknesses:

1. Empty `run` means no workload-backed counters were actually observable.
2. `ncu` failures were not diagnostic enough.
3. Device attribute targets were misrouted into generic/probe logic instead of dedicated attribute queries.
4. Linux probe compilation had command-line/build invocation issues that were hard to diagnose from current evidence.

## Refactor Direction

The right direction is not to copy the example literally.

Instead:

- keep the current evaluator contract
- keep the current evidence-rich outputs
- add a durable iterative agent state
- strengthen target-type routing
- isolate compile/run/profile operations like the example runner
- turn refinement from a suggestion step into an actual next-iteration execution path

## Proposed Target Architecture

### Layer 1: Control State

Add a persistent state object for multi-round execution.

Suggested file:

- `profiler_agent/agent_state.py`

Suggested persisted fields:

- `iteration`
- `request_targets`
- `request_run`
- `target_categories`
- `selected_tools_history`
- `metrics_history`
- `analysis_history`
- `error_history`
- `recommended_next_targets`
- `recommended_next_metrics`
- `probe_history`
- `current_bottleneck`
- `done`

### Layer 2: Explicit Target-Type Classification

Introduce target semantic classes before strategy selection.

Suggested classes:

- `device_attribute`
- `ncu_counter`
- `runtime_throughput_counter`
- `intrinsic_microbench`
- `telemetry_metric`
- `unknown`

Suggested new module:

- `profiler_agent/target_semantics.py`

This should be used before strategy lookup.

### Layer 3: Tool Execution Split

Split current tool execution into example-style explicit steps:

- compile benchmark/probe
- run binary or workload
- profile with `ncu`
- collect telemetry (`nvml`)
- evaluate outputs

Suggested new modules:

- `profiler_agent/tool_adapters/probe_compiler.py`
- `profiler_agent/tool_adapters/workload_runner.py`
- `profiler_agent/tool_adapters/ncu_profiler.py`

These can wrap current adapters without deleting them immediately.

### Layer 4: Iterative Planning

Planner should produce not only tools, but next-iteration actions.

Examples:

- retry same target with different tool path
- switch from generic metric to device-attribute path
- request focused `ncu` metric subset
- regenerate probe with explicit compile error context
- stop early if metric family is unsupported in no-`run` mode

### Layer 5: Dedicated Strategies For Missing Metric Families

Do not keep routing these through `GenericMetricStrategy`:

- `device__attribute_max_gpu_frequency_khz`
- `device__attribute_max_mem_frequency_khz`
- `device__attribute_fb_bus_width`
- `launch__sm_count`

These should get dedicated strategies.

Suggested implementation path:

- device attributes via `cudaGetDeviceProperties` / `cudaDeviceGetAttribute`
- `launch__sm_count` via device property / hardware query path

For workload-dependent `ncu` metrics, do not rely on generic probe fallback as the primary path.

## Incremental Refactor Plan

### Phase 0: Stabilization And Diagnostic Hardening

Goal:

- make failures explainable before changing behavior

Tasks:

- preserve full tool invocation diagnostics for Linux probe builds
- preserve full `ncu` stderr/stdout tails on failure
- record target semantic category in evidence
- record whether a target is workload-dependent

Files likely touched:

- `profiler_agent/tool_adapters/microbench_adapter.py`
- `profiler_agent/tool_adapters/ncu_adapter.py`
- `profiler_agent/target_strategies/generic.py`
- `profiler_agent/orchestrator/pipeline.py`

Acceptance:

- Linux compile failure shows exact command and exact compiler failure
- `ncu_failed` entries preserve actionable stderr
- evidence shows why a target was routed a certain way

### Phase 1: Add Target Semantic Routing

Goal:

- stop misrouting device attributes and workload counters into the same generic path

Tasks:

- add semantic classifier
- attach semantic class to each target
- gate fallback behavior by semantic class

Rules:

- `device_attribute`: dedicated query strategy, no probe-first fallback
- `ncu_counter` or `runtime_throughput_counter`: require `run` or mark low-confidence/no-signal
- `intrinsic_microbench`: allow LLM probe path

Files likely touched:

- `profiler_agent/target_semantics.py` (new)
- `profiler_agent/target_strategies/registry.py`
- `profiler_agent/target_strategies/generic.py`
- `profiler_agent/orchestrator/task_planner.py`

Acceptance:

- server-test metric set no longer routes all targets through `generic_metric`

### Phase 2: Add Dedicated Attribute Strategies

Goal:

- support hardware property targets without LLM probe generation

Initial targets:

- `device__attribute_max_gpu_frequency_khz`
- `device__attribute_max_mem_frequency_khz`
- `device__attribute_fb_bus_width`
- `launch__sm_count`

Files likely touched:

- `profiler_agent/target_strategies/device_attributes.py` (new)
- `profiler_agent/target_strategies/registry.py`
- possibly new CUDA/NVML helper adapter

Acceptance:

- these targets can produce values without probe generation
- evidence clearly indicates attribute-query source

### Phase 3: Refactor Executor Toward Example-Style Runner Separation

Goal:

- isolate compile/run/profile behaviors so they can be retried and reasoned about independently

Tasks:

- separate compile, run, profile into distinct executor sub-operations
- add typed results for each stage
- stop hiding build errors behind generic `compile_failed`

Files likely touched:

- `profiler_agent/multi_agent/executor.py`
- `profiler_agent/tool_adapters/binary_runner.py`
- `profiler_agent/tool_adapters/ncu_adapter.py`
- new `probe_compiler.py` or equivalent

Acceptance:

- one failing stage does not erase diagnostic context from others

### Phase 4: Introduce Persistent Iterative Agent State

Goal:

- make multi-agent execution truly multi-round

Tasks:

- define state schema
- persist state into artifact directory
- use previous errors and analyses as planning inputs

Suggested artifact:

- `.agent_artifacts/agent_state.json`

Files likely touched:

- `profiler_agent/multi_agent/coordinator.py`
- `profiler_agent/multi_agent/planner.py`
- `profiler_agent/multi_agent/interpreter.py`
- new `profiler_agent/agent_state.py`

Acceptance:

- a second iteration can consume outputs from iteration 1

### Phase 5: Convert Refinement Into Real Re-execution

Goal:

- move from “advice only” to real iterative profiling

Tasks:

- allow planner/interpreter to propose next-round targets/tools
- coordinator can launch iteration 2 when conditions justify it
- add max iteration guard

Suggested initial policy:

- only iterate when:
  - at least one tool path failed in a recoverable way
  - LLM provided a tighter metric/tool recommendation
  - probe generation can plausibly be repaired

Acceptance:

- multi-agent mode can complete more than one measurement round

### Phase 6: Probe/Benchmark Evolution Policy

Goal:

- selectively adopt the example’s evolving benchmark idea without breaking current project scope

Tasks:

- distinguish:
  - intrinsic per-target probe generation
  - optional workload benchmark generation
- only introduce benchmark evolution where truly helpful

This phase is optional and should come after target routing and iterative control are stable.

## Recommended First Concrete Deliverables

If starting fresh in a new session, do these first:

1. implement target semantic classifier
2. add dedicated device-attribute strategy
3. improve Linux compile and `ncu` diagnostics
4. persist minimal `agent_state.json`

These four changes provide the highest leverage with the least architecture churn.

## Suggested File Ownership For Refactor

### Core control flow

- `profiler_agent/main.py`
- `profiler_agent/multi_agent/coordinator.py`
- `profiler_agent/multi_agent/planner.py`
- `profiler_agent/multi_agent/executor.py`
- `profiler_agent/multi_agent/interpreter.py`

### Measurement routing

- `profiler_agent/target_strategies/registry.py`
- `profiler_agent/target_strategies/generic.py`
- `profiler_agent/target_strategies/*`
- `profiler_agent/target_semantics.py` (new)

### Tool isolation

- `profiler_agent/tool_adapters/binary_runner.py`
- `profiler_agent/tool_adapters/ncu_adapter.py`
- `profiler_agent/tool_adapters/microbench_adapter.py`
- new compile/profile helpers as needed

### State and artifacts

- new `profiler_agent/agent_state.py`
- artifact JSON schema in `.agent_artifacts`

## Risks To Avoid During Refactor

1. Do not break the evaluator-facing output contract.
2. Do not hard-wire benchmark-generation behavior into all targets.
3. Do not keep treating device attributes as generic probe metrics.
4. Do not hide Linux tool invocation failures behind generic fallback zeros.
5. Do not make no-`run` mode look equally valid for workload-dependent counters.

## Definition Of Success

The refactor is successful when:

- target families are routed by semantics rather than by fallback accident
- device attribute targets work without LLM probe generation
- Linux compile and `ncu` failures are fully diagnosable
- multi-agent mode can reuse prior-round state
- refinement can trigger at least one real follow-up measurement round
- evaluator-facing files remain stable

## Session Handoff Note

When continuing in a new session, start from:

1. this plan
2. `PROJECT_HANDOFF.md`
3. latest server-produced output artifact showing Linux failures
4. current implementations of:
   - `profiler_agent/multi_agent/executor.py`
   - `profiler_agent/orchestrator/pipeline.py`
   - `profiler_agent/target_strategies/registry.py`
   - `profiler_agent/target_strategies/generic.py`
   - `profiler_agent/tool_adapters/microbench_adapter.py`
   - `profiler_agent/tool_adapters/ncu_adapter.py`

Suggested next-session prompt seed:

> Continue the refactor from `REFACTOR_PLAN_EXAMPLE_ALIGNMENT.md`. Start with Phase 0 and Phase 1 only: improve Linux tool diagnostics, add target semantic classification, and stop routing device attribute metrics through generic probe generation. Preserve the current evaluator output contract.
