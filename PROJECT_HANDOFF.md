# GPU Profiling Agent - Project Handoff Context (Updated 2026-04-12)

## 1) Purpose
Build a target-driven GPU hardware intrinsic profiling agent for MLSYS project phase 1.7.
The system must:
- Accept dynamic evaluator targets.
- Output numeric results.
- Provide structured evidence for LLM-based judging.
- Remain robust under environment/tool variations.

## 2) Official Requirement Summary (Section 1.7)

### 2.1 1.7.1 Hardware Intrinsic Profiling Objectives
Targets include:
- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `shared_peak_bandwidth_gbps`
- `global_peak_bandwidth_gbps`
- `l2_cache_capacity_kb`
- `actual_boost_clock_mhz`
- `shmem_bank_conflict_penalty_cycles`
- `max_shmem_per_block_kb` (project-added practical intrinsic signal)

### 2.2 1.7.2 Submission & Evaluation Workflow
- Evaluator provides `target_spec.json`.
- Agent runs profiling and writes:
  - `results.json` (required numeric output)
  - `evidence.json`
  - `analysis.json`
- Server compares results against ground truth.

### 2.3 1.7.3 Anti-Hacking / Environment Variations
Must handle unstable environments:
- Frequency lock or static clock behavior
- Resource masking
- Tool/API interception or missing tools
- Non-standard runtime behavior

### 2.4 1.7.4 LLM-Based Evaluation
Scoring includes both:
- Numeric closeness
- Evidence quality / explainability

## 3) Current I/O Contract

### 3.1 Input
`target_spec.json`
```json
{
  "targets": ["..."],
  "run": "path/or/command to executable"
}
```

### 3.2 Outputs
- `outputs/results.json`
- `outputs/evidence.json`
- `outputs/analysis.json`

If run in multi-agent mode, additional artifacts are generated:
- `outputs/multi_agent_plan.json`
- `outputs/multi_agent_trace.json`

## 4) Architecture (Current)

### 4.1 Single-Pipeline Path (Stable Mainline)
`main(single) -> load spec -> run target binary -> per-target strategy -> normalize -> detectors -> analysis -> write outputs`

Core files:
- `profiler_agent/main.py`
- `profiler_agent/orchestrator/pipeline.py`
- `profiler_agent/target_strategies/*`
- `profiler_agent/tool_adapters/*`
- `profiler_agent/detectors/service.py`
- `profiler_agent/analyzer/*`
- `profiler_agent/schema/*`

### 4.2 Multi-Agent Framework (Integrated, Backward-Compatible)
`main(multi) -> coordinator -> router -> planner -> executor(tool_execution + pipeline_execution) -> interpreter`

Core files:
- `profiler_agent/multi_agent/coordinator.py`
- `profiler_agent/multi_agent/router.py`
- `profiler_agent/multi_agent/planner.py`
- `profiler_agent/multi_agent/executor.py`
- `profiler_agent/multi_agent/interpreter.py`
- `profiler_agent/multi_agent/llm_client.py`

## 5) Implemented Target Strategies
- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `shared_peak_bandwidth_gbps`
- `global_peak_bandwidth_gbps`
- `l2_cache_capacity_kb`
- `actual_boost_clock_mhz`
- `max_shmem_per_block_kb`
- `shmem_bank_conflict_penalty_cycles`

Unknown targets use generic fallback strategy.

## 6) Probe System Status

### 6.1 Protocol
Structured probe output:
`metric=<name> value=<num> samples=<n> median=<num> best=<num> std=<num>`
Legacy `<metric>=<value>` still supported.

### 6.2 Statistical Rigor (Implemented)
Microbench now supports:
- Multi-run sampling (default repeat = 5)
- Aggregation by median
- `sample_count`, `best_value`, `std_value`, `run_values`
- Env override: `PROFILER_AGENT_PROBE_REPEAT` (1-20)

## 7) Fusion, Reliability, and Evidence

### 7.1 Fusion
Current fusion method:
- `robust_weighted_median`
- Source-level reliability weighting (`ncu`, `microbench`, `nvml`)
- Outlier handling + confidence output

### 7.2 Evidence
`evidence.json` includes:
- `run` execution tails / return code
- Per-target `selected_source`, `candidates`, `candidate_reliability`
- `fusion` metadata (`method`, `confidence`, retained/dropped/source_reliability)
- Tool-level execution details
- `result_quality` summary
- `detectors` output

## 8) Detector Layer (Implemented)
Explicit anti-hacking / reliability detectors:
- `source_divergence`
- `tool_path_blocking`
- `clock_lock_or_static_state`
- `resource_mask_suspected`

Detector outputs:
- `evidence.detectors.findings`
- `total_confidence_penalty`
Penalty is applied in analyzer to produce:
- `confidence_penalty`
- `confidence_adjusted`
- `detector_summary`

## 9) Analyzer Layer
Bound classification:
- `compute_bound`
- `memory_bound`
- `balanced_or_mixed`
- `unknown`

Analysis output includes:
- confidence (+ adjusted confidence after detector penalties)
- observed metrics and missing signals
- bottleneck list and suggestions

## 10) Multi-Agent + LLM Integration Status

### 10.1 Executor Tool-Calling (Implemented)
Executor supports tool stage before pipeline stage:
- `executor` (run target command)
- `ncu`
- `microbench`
- `nvml`
- `nsys` (availability/version probe)
- `torch_profiler` (torch availability/version probe)

Results stored in:
- `state.outputs.tool_calls`

### 10.2 LLM Interface (Implemented with Fallback)
LLM client:
- OpenAI-compatible Chat Completions API
- Env-configured

Used in:
- Router (intent decision)
- Planner (tool selection)
- Interpreter (summary + next actions)

Fallback behavior:
- If key missing / API failure / invalid response, rules-based logic is used automatically.

Env vars:
- `OPENAI_API_KEY` (required to enable)
- `OPENAI_BASE_URL` (optional, default `https://api.openai.com/v1`)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_TIMEOUT_S` (optional, default `30`)

## 11) Testing Status
- Current test suite: **30/30 passing**
- Includes:
  - schema validation
  - registry mapping
  - pipeline smoke
  - ncu / microbench adapters
  - fusion reliability behavior
  - analyzer behavior
  - detector logic
  - no-GPU-tools fallback
  - golden fixtures (`success` / `degraded`)
  - multi-agent framework
  - main single/multi mode integration
  - LLM client parsing
  - multi-agent LLM usage path

## 12) Remaining Gaps / Risks
1. Probe precision calibration still needed for final scoring quality (especially bandwidth/L2 capacity).
2. `nsys` and `torch_profiler` are integrated at executor-call level but not yet deeply fused into target scoring logic.
3. Need broader real-workload validation (current ad-hoc runs include synthetic command/probe executable).
4. Need final report packaging (methodology, evidence examples, error analysis) for submission readiness.

## 13) Recommended Next Priorities
1. Build calibration runner + benchmark set for per-target parameter tuning.
2. Integrate `nsys`/`torch_profiler` signals into analysis/fusion where meaningful.
3. Add real evaluator-like end-to-end scenarios and capture final evidence examples.
4. Prepare submission report and demo script.

## 14) Quick Commands

Single pipeline mode:
```powershell
python -m profiler_agent.main --mode single --spec inputs/target_spec.json --out outputs
```

Multi-agent mode:
```powershell
python -m profiler_agent.main --mode multi --objective "analyze gpu bottlenecks" --spec inputs/target_spec.json --out outputs
```

Run full tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Golden regression only:
```powershell
python -m unittest tests.test_golden_fixtures -v
```

## 15) Machine-Readable Snapshot
```yaml
project: gpu-prof-agent
phase: "1.7 hardware intrinsic profiling"
updated_at: "2026-04-12"
io_contract:
  input: target_spec.json
  required_input_fields: [targets, run]
  outputs:
    - results.json
    - evidence.json
    - analysis.json
  multi_agent_outputs:
    - multi_agent_plan.json
    - multi_agent_trace.json
status:
  single_pipeline: implemented
  dynamic_target_dispatch: implemented
  probe_framework: implemented
  probe_multi_run_stats: implemented
  detector_module: implemented
  reliability_weighted_fusion: implemented
  analyzer_confidence_adjustment: implemented
  multi_agent_framework: implemented
  llm_client_integration: implemented_with_fallback
  executor_tool_calling: implemented
  tests: "30/30 passing"
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
  - "probe precision calibration needed for final scoring"
  - "nsys/torch_profiler not yet deeply fused into scoring logic"
  - "real-workload validation breadth still limited"
next_priority:
  - "build calibration runner and benchmark set"
  - "deepen nsys/torch_profiler integration into analysis/fusion"
  - "prepare final report and evaluator-like end-to-end runs"
```

