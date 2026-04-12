# GPU Profiling Agent (MVP Skeleton)

Minimal, extensible scaffold for a target-driven GPU profiling agent.

## Contract

- Input: `target_spec.json`
  - Required fields:
    - `targets`: list of metric names to identify
    - `run`: executable command path/string provided by evaluator
- Output:
  - `results.json`: numeric values only, keyed by target
  - `evidence.json`: optional debugging/provenance data
  - `analysis.json`: bottleneck analysis (`bound_type`, confidence, bottlenecks)

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
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
# optional:
# $env:OPENAI_BASE_URL="https://api.openai.com/v1"
# $env:OPENAI_TIMEOUT_S="30"
```

When `OPENAI_API_KEY` is set, `Router/Planner/Interpreter` will use LLM outputs first
and fall back to rule-based logic on any API error or invalid response.

## Current MVP Status

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

## Extending

1. Add a strategy under `profiler_agent/target_strategies/`.
2. Register it in `profiler_agent/target_strategies/registry.py`.
3. Add/upgrade adapters under `profiler_agent/tool_adapters/`.
