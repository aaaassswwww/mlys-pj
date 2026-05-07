# GPU Profiling Agent

Extensible agent framework originally built for Phase 1 hardware intrinsic profiling, now being prepared for Phase 2 LoRA CUDA optimization.

## Contract

Current stable contract in this repository is still the Phase 1 evaluator contract:

- Input: `target_spec.json`
  - Required fields:
    - `targets`: list of metric names to identify
    - `run`: executable command path/string provided by evaluator
- Output:
  - `results.json`: numeric values only, keyed by target
  - `evidence.json`: optional debugging/provenance data
  - `analysis.json`: bottleneck analysis (`bound_type`, confidence, bottlenecks)

Phase 2 is a different evaluator contract centered on:
- `bash run.sh`
- final root-level `optimized_lora.cu`

See [PHASE2_REFACTOR_PLAN_LORA_OPTIMIZATION.md](</workspace/PHASE2_REFACTOR_PLAN_LORA_OPTIMIZATION.md>) for the migration plan.

## Quick Start

```powershell
python -m profiler_agent.main --spec inputs/target_spec.json --out outputs
```

Multi-agent mode (keeps the same pipeline outputs and adds coordinator artifacts):

```powershell
python -m profiler_agent.main --mode multi --objective "analyze gpu bottlenecks" --spec inputs/target_spec.json --out outputs
```

Optional LLM integration for multi-agent decision-making (OpenAI-compatible Chat Completions API):

```powershell
$env:API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-5.4"
# optional:
# $env:OPENAI_BASE_URL="https://api.openai.com/v1"
# $env:OPENAI_TIMEOUT_S="30"
```

When `API_KEY` is set, LLM is used for:
- Router/Planner/Interpreter decisions
- LLM-driven microbenchmark code generation (`PROFILER_AGENT_PROBE_SOURCE_MODE=llm_generated`)
- LLM-led analyzer reasoning (with rule-based guardrails and fallback)
Any API error automatically falls back to rule-based logic.

Strict codegen mode (disable static probe fallback):
```powershell
$env:PROFILER_AGENT_PROBE_SOURCE_MODE="llm_generated"
$env:PROFILER_AGENT_DISABLE_STATIC_FALLBACK="1"
```
In strict mode, probe generation failures are surfaced as `llm_generation_failed`.

## Current MVP Status

Phase 1 status:

- Dynamic target planning from `target_spec.json` (`targets` is not hardcoded).
- Executes the provided `run` command once and records run evidence.
- Strategy registry supports target-specific implementations plus generic fallback.
- `actual_boost_clock_mhz` can use median SM clock sampling via `nvidia-smi` when available.
- `ncu` adapter is wired for real invocation with metric-aware CSV parsing and structured evidence.
- Micro-benchmark probes are added for:
  - `dram_latency_cycles` (`probes/dram_latency_cycles/probe.cu`)
  - `max_shmem_per_block_kb` (`probes/max_shmem_per_block/probe.cu`)
  These compile with `nvcc` at runtime and are used as first-choice source in target-specific strategies.
- Fusion is robust-median based with outlier filtering (MAD threshold) and confidence reporting.
- Analyzer layer is integrated in pipeline:
  - Classifies `compute_bound` / `memory_bound` / `balanced_or_mixed` / `unknown`
  - Emits bottleneck reasons and actionable suggestions in `analysis.json`
- Metric normalization is enabled before writing `results.json`:
  - per-target unit/range specs
  - clamp + round behavior with quality issues logged into `evidence.json.result_quality`
- Probe output protocol supports structured lines:
  - `metric=<name> value=<num> samples=<n> median=<num> best=<num> std=<num>`
  - backward-compatible legacy `<metric>=<value>` parsing
- Multi-agent framework scaffold is available under `profiler_agent/multi_agent/`:
  - `RouterAgent`: routes user objective to intent
  - `PlannerAgent`: builds staged workflow and tool selection
  - `ExecutorAgent`: runs tool execution stage (`ncu`, `microbench`, `nvml`, `nsys`, `torch_profiler`) and pipeline stage
  - `InterpreterAgent`: summarizes outputs and proposes next iteration actions
  - `MultiAgentCoordinator`: orchestrates the end-to-end multi-agent loop

Phase 2 status:

- Some supporting infrastructure already exists for iterative agent workflows:
  - runtime tool probing
  - compile/run/profile separation
  - probe iteration and repair patterns
  - LLM code generation and revision loops
- Dedicated Phase 2 foundation modules now exist under `profiler_agent/phase2/`:
  - `harness.py`: LoRA reference math, correctness checks, benchmark helpers
  - `generator.py`: LLM-first candidate generation for single-file `optimized_lora.cu`
  - `optimizer.py`: iteration loop and best-candidate promotion
  - `candidate_store.py`: root `optimized_lora.cu`, state, and report persistence
- Phase 2 bootstrap behavior is now implemented:
  - even before full CUDA evaluator integration, the optimizer can keep a root `optimized_lora.cu`
  - `.agent_artifacts/phase2_state.json` and `.agent_artifacts/phase2_report.json` are written alongside it
- The repository is not yet fully switched to the Phase 2 LoRA optimization contract.
- The next architecture step is to add a dedicated LoRA optimization workflow rather than extending the Phase 1 metric pipeline further.

## Run Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Golden Regression Self-Check

Use these golden tests before submission to ensure the output contract remains stable
for `results.json`, `evidence.json`, and `analysis.json` under representative scenarios.

```powershell
python -m unittest tests.test_golden_fixtures -v
```

What this validates:
- `success` case: expected resolved values and analysis behavior when signals are available.
- `degraded` case: expected fallback, detector findings, and confidence penalties when tools fail.
- evidence projection stability: selected source and fusion metadata for each target.

## Pre-Submit Compatibility Check

Run static checks for evaluator contract:

```powershell
python scripts/pre_submit_check.py --workspace .
```

Optional (container-like environment only): execute `run.sh` and validate single `output.*` artifact.

```powershell
python scripts/pre_submit_check.py --workspace /workspace --run-container-check
```

## Extending

1. Add a strategy under `profiler_agent/target_strategies/`.
2. Register it in `profiler_agent/target_strategies/registry.py`.
3. Add/upgrade adapters under `profiler_agent/tool_adapters/`.

## Phase 2

The recommended next workstream is documented in:

- [PHASE2_REFACTOR_PLAN_LORA_OPTIMIZATION.md](</workspace/PHASE2_REFACTOR_PLAN_LORA_OPTIMIZATION.md>)
- [PROJECT_HANDOFF.md](</workspace/PROJECT_HANDOFF.md>)
