# GPU Profiling Agent - Project Handoff Context (Updated 2026-04-13)

## 1) Purpose
Build a target-driven GPU hardware intrinsic profiling agent for MLSYS project phase 1.7.
System goals:
- Accept dynamic evaluator targets.
- Output numeric results in required format.
- Provide structured evidence for LLM-as-a-Judge.
- Stay robust under tool/environment variation.

## 2) Requirement Mapping (Section 1.7)

### 2.1 1.7.1 Hardware Intrinsic Profiling Objectives
Current registered target scope:
- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `shared_peak_bandwidth_gbps`
- `global_peak_bandwidth_gbps`
- `l2_cache_capacity_kb`
- `actual_boost_clock_mhz`
- `shmem_bank_conflict_penalty_cycles`
- `max_shmem_per_block_kb` (project practical intrinsic signal)

### 2.2 1.7.2 Submission Workflow
Evaluator input:
- `target_spec.json` with `targets` + `run`

Agent outputs:
- `results.json` (required numeric values)
- `evidence.json` (method + tool traces)
- `analysis.json` (reasoning output)

### 2.3 1.7.3 Anti-Hacking
Implemented direction:
- Multi-strategy measurement (ncu/microbench/nvml)
- Detector layer for suspicious runtime/tool states
- Confidence penalties when reliability risk is detected

### 2.4 1.7.4 LLM-as-a-Judge
Implemented direction:
- Rich evidence artifacts (tool traces, fusion metadata, detector findings)
- LLM-based analysis path (with rule guardrails/fallback)

## 3) Latest Policy Compliance (Important)

Recent clarified policy requires:
1. No external downloaded benchmarks; agent must autonomously generate microbench code.
2. OpenAI-compatible API format for LLM calls; final eval uses GPT-5.4 API.
3. No key in source/config; key must be read from env var `API_KEY`.

Current compliance status:
- `API_KEY` env var is now used in LLM client.
- LLM-generated probe path is integrated.
- Strict mode can disable static probe fallback completely.

## 4) I/O Contract

### 4.1 Input
`target_spec.json`
```json
{
  "targets": ["..."],
  "run": "path/or/command"
}
```

### 4.2 Outputs
Single mode:
- `outputs/results.json`
- `outputs/evidence.json`
- `outputs/analysis.json`

Multi-agent mode additionally writes:
- `outputs/multi_agent_plan.json`
- `outputs/multi_agent_trace.json`

## 5) Architecture Snapshot

### 5.1 Mainline Pipeline
`main(single) -> load spec -> run target binary -> strategy measure -> normalize -> detectors -> analysis -> write outputs`

Core files:
- `profiler_agent/main.py`
- `profiler_agent/orchestrator/pipeline.py`
- `profiler_agent/target_strategies/*`
- `profiler_agent/tool_adapters/*`
- `profiler_agent/detectors/service.py`
- `profiler_agent/analyzer/*`

### 5.2 Multi-Agent Framework
`main(multi) -> coordinator -> router -> planner -> executor(tool + pipeline) -> interpreter`

Core files:
- `profiler_agent/multi_agent/coordinator.py`
- `profiler_agent/multi_agent/router.py`
- `profiler_agent/multi_agent/planner.py`
- `profiler_agent/multi_agent/executor.py`
- `profiler_agent/multi_agent/interpreter.py`
- `profiler_agent/multi_agent/llm_client.py`

## 6) LLM-Driven Refactor Status

### 6.1 LLM Probe Codegen (Integrated)
New module:
- `profiler_agent/codegen/prompts.py`
- `profiler_agent/codegen/generator.py`

Behavior:
- Default probe source mode is `llm_generated`.
- Generator requests CUDA C++ source from LLM.
- Generated source is validated (self-contained, no external-download patterns, required output protocol).
- Compile/runtime error supports iterative repair requests.

### 6.2 Probe Source Modes
Env vars:
- `PROFILER_AGENT_PROBE_SOURCE_MODE=llm_generated|static_fallback` (default `llm_generated`)
- `PROFILER_AGENT_DISABLE_STATIC_FALLBACK=1` enables strict mode (no static fallback)

Strict mode behavior:
- If LLM generation fails and fallback disabled, measurement returns `llm_generation_failed`.

### 6.3 Probe Evidence Trace
`ProbeResult` now carries:
- `generation_source`
- `generation_attempts`
- `generation_error`
- `generation_trace` (per-attempt records)
- `source_path`

These fields are propagated into `evidence.json` tool/probe sections.

## 7) Analyzer Refactor Status

### 7.1 LLM-Led Analysis Path
New module:
- `profiler_agent/analyzer/llm_reasoner.py`

`analyzer/service.py` now:
- Builds baseline rule analysis.
- Attempts LLM analysis override.
- Applies guardrails (`rule_guardrail_flags`) for suspicious LLM outputs.
- Falls back to rule analysis when LLM unavailable/invalid.

Output includes:
- `analysis_source: llm|rules`
- optional `llm_reasoning_summary`
- detector-adjusted confidence fields.

## 8) Reliability / Fusion / Detectors

### 8.1 Fusion
Current fusion supports reliability-weighted robust median style combination with outlier handling.

### 8.2 Detectors
Implemented detectors include:
- `source_divergence`
- `tool_path_blocking`
- `clock_lock_or_static_state`
- `resource_mask_suspected`

Detector outputs in evidence:
- `finding_count`
- `findings[]`
- `total_confidence_penalty`

## 9) Metric Normalization / Quality

Before writing `results.json`, normalization applies target specs:
- unit/range clamp
- rounding/integer-like normalization

Evidence includes:
- `result_quality.units`
- `result_quality.issue_count`
- `result_quality.issues`

## 10) Current Runtime Reality

On environments without GPU tools (`nvcc`, `ncu`) or without `API_KEY`:
- Pipeline still runs end-to-end.
- Measurement likely degrades to fallback/default values.
- Evidence records exact failure paths and detector penalties.

## 11) Testing Status (Latest)

- Current suite: **35/35 passing**
- Coverage includes:
  - schema/registry
  - pipeline smoke
  - ncu/microbench adapters
  - probe repeat logic
  - strict no-static-fallback behavior
  - detector logic
  - analyzer logic + LLM analysis path
  - golden fixtures
  - multi-agent framework and LLM usage path
  - main single/multi mode

## 12) Remaining Gaps / Risks

1. Probe numeric precision calibration still needed for final scoring quality (especially bandwidth + L2 capacity).
2. Real workload/evaluator-like end-to-end validation remains limited.
3. `nsys`/`torch_profiler` are integrated as tool calls but not deeply fused into per-target scoring logic.
4. Submission packaging (method report + evidence examples + error analysis) still pending.

## 13) Recommended Next Priorities

1. Run evaluator-like workloads and capture reproducibility/variance.
2. Calibrate target-specific probe parameters and confidence mapping.
3. Expand score fusion with workload-level signals (`nsys`, `torch_profiler`) where meaningful.
4. Finalize submission report and demo script.

## 14) Quick Commands

Single mode:
```powershell
python -m profiler_agent.main --mode single --spec inputs/target_spec.json --out outputs
```

Multi-agent mode:
```powershell
python -m profiler_agent.main --mode multi --objective "analyze gpu bottlenecks" --spec inputs/target_spec.json --out outputs
```

Tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Strict codegen mode (no static fallback):
```powershell
$env:API_KEY="<your_key>"
$env:PROFILER_AGENT_PROBE_SOURCE_MODE="llm_generated"
$env:PROFILER_AGENT_DISABLE_STATIC_FALLBACK="1"
python -m profiler_agent.main --mode single --spec inputs/target_spec.json --out outputs
```

## 15) Machine-Readable Snapshot
```yaml
project: gpu-prof-agent
phase: "1.7 hardware intrinsic profiling"
updated_at: "2026-04-13"
io_contract:
  input: target_spec.json
  required_input_fields: [targets, run]
  outputs: [results.json, evidence.json, analysis.json]
  multi_agent_outputs: [multi_agent_plan.json, multi_agent_trace.json]
status:
  single_pipeline: implemented
  multi_agent_framework: implemented
  llm_probe_codegen: implemented
  llm_analysis_path: implemented_with_guardrails
  static_probe_fallback: configurable
  strict_codegen_mode: implemented
  detector_layer: implemented
  result_normalization: implemented
  tests: "35/35 passing"
llm_config:
  api_key_env: API_KEY
  base_url_env: OPENAI_BASE_URL
  model_env: OPENAI_MODEL
  timeout_env: OPENAI_TIMEOUT_S
targets_registered:
  - l1_latency_cycles
  - l2_latency_cycles
  - dram_latency_cycles
  - shared_peak_bandwidth_gbps
  - global_peak_bandwidth_gbps
  - l2_cache_capacity_kb
  - actual_boost_clock_mhz
  - max_shmem_per_block_kb
  - shmem_bank_conflict_penalty_cycles
open_risks:
  - "probe precision calibration still needed"
  - "real workload validation breadth limited"
  - "nsys/torch_profiler not deeply fused into scoring"
next_priority:
  - "evaluator-like e2e validation"
  - "probe parameter calibration"
  - "final submission report packaging"
```
