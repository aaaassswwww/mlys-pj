# GPU Profiling Agent - Project Handoff Context

## 1) Purpose
Build a target-driven GPU hardware profiling agent for MLSYS course project phase 1.7, with dynamic targets from evaluator input, measurable numeric outputs, and evidence for LLM-based judging.

## 2) Official Requirement Summary (Section 1.7)

### 2.1 1.7.1 Hardware Intrinsic Profiling Objectives
Agent should probe hardware-intrinsic metrics (not static lookup):
- Memory latency hierarchy: `l1_latency_cycles`, `l2_latency_cycles`, `dram_latency_cycles`
- Effective peak bandwidth: shared/global memory bandwidth
- L2 cache capacity via latency-vs-size cliff
- Actual boost clock under load
- Shared memory bank conflict penalty

### 2.2 1.7.2 Submission & Evaluation Workflow
- Student submits agent system code.
- Evaluator provides `target_spec.json`.
- Agent outputs numeric `results.json`.
- Output compared to server-side ground truth benchmarks.

### 2.3 1.7.3 Anti-Hacking & Environment Variations
Environment may be altered to break static/spec lookup:
- Non-standard frequency lock
- SM/resource masking
- API virtualization/interception
Recommended: multi-strategy fusion (microbench + binary execution + ncu cross-check).

### 2.4 1.7.4 LLM-Based Evaluation
Judge uses:
- Student output + reasoning/logs
- Ground truth
- Experimental evidence
Therefore explainability/evidence quality is part of score, not just numeric closeness.

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
- `outputs/results.json` (required; numeric target values only)
- `outputs/evidence.json` (structured evidence/log)
- `outputs/analysis.json` (bound/bottleneck analysis)

## 4) Current Architecture (Implemented)

### 4.1 Pipeline
`main -> load spec -> run executable -> per-target strategy -> normalize results -> analysis -> write outputs`

### 4.2 Core Modules
- `profiler_agent/main.py`
- `profiler_agent/orchestrator/pipeline.py`
- `profiler_agent/target_strategies/*`
- `profiler_agent/tool_adapters/{microbench,ncu,nvml,binary_runner}.py`
- `profiler_agent/analyzer/*`
- `profiler_agent/schema/*`

### 4.3 Strategy System
- Dynamic strategy lookup via registry by target name.
- Default fallback strategy exists for unknown targets.
- Probe-first strategy base implemented for hardware-intrinsic targets.

## 5) Implemented Targets in Registry
- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `shared_peak_bandwidth_gbps`
- `global_peak_bandwidth_gbps`
- `l2_cache_capacity_kb`
- `actual_boost_clock_mhz`
- `max_shmem_per_block_kb`
- `shmem_bank_conflict_penalty_cycles`

## 6) Probe System Status

### 6.1 Output Protocol (standardized)
Probes now emit structured line format:
`metric=<name> value=<num> samples=<n> median=<num> best=<num> std=<num>`
Backward compatibility for legacy `<metric>=<value>` still supported.

### 6.2 Probe Files Present
- `probes/l1_latency_cycles/probe.cu`
- `probes/l2_latency_cycles/probe.cu`
- `probes/dram_latency_cycles/probe.cu`
- `probes/shared_peak_bandwidth_gbps/probe.cu`
- `probes/global_peak_bandwidth_gbps/probe.cu`
- `probes/l2_cache_capacity_kb/probe.cu`
- `probes/max_shmem_per_block/probe.cu`
- `probes/shmem_bank_conflict_penalty_cycles/probe.cu`

### 6.3 Accuracy Maturity
- Logic and plumbing are complete.
- Some probes are still heuristic/first-pass and need further calibration for high-precision evaluation.

## 7) Result Normalization & Quality

### 7.1 Metric Specs
`profiler_agent/schema/metric_specs.py` defines per-target:
- unit
- min/max clamp range
- rounding behavior
- integer-like behavior (if needed)

### 7.2 Normalization
Pipeline applies normalization before writing `results.json`.
Quality report is appended to `evidence.json` as:
- `result_quality.units`
- `result_quality.issue_count`
- `result_quality.issues`

## 8) Analyzer Layer (Implemented)
- Bound classification: `compute_bound`, `memory_bound`, `balanced_or_mixed`, `unknown`
- Outputs confidence, observed metrics, missing signals, bottlenecks/suggestions
- File: `outputs/analysis.json`

## 9) Fusion & Evidence
- Multi-source candidate fusion exists (`robust_median` + outlier handling).
- Evidence includes tool-level status and parse modes.
- `ncu` adapter supports metric-aware CSV parsing with fallback modes.

## 10) Testing Status
- Test suite currently passes: 17/17.
- Includes:
  - schema validation
  - registry mapping
  - pipeline smoke
  - ncu adapter parsing
  - microbench parser (structured protocol)
  - fusion logic
  - analyzer behavior
  - no-GPU-tools fallback logic via mocks

## 11) Known Gaps / Risks
- Real runtime precision depends on availability of `nvcc`/`ncu`.
- In missing-tool environments, logic falls back safely but may produce low-confidence/default-like outputs.
- Some probes (especially capacity cliff and bandwidth) need tighter experimental rigor for final scoring quality.
- Anti-hacking detectors are not yet fully explicit as a dedicated module (partially covered by multi-strategy evidence).

## 12) Recommended Next Priorities
1. Improve probe statistical rigor (multiple runs, median/std for all).
2. Add explicit anti-hacking detectors (freq-lock/resource-mask/API inconsistency).
3. Strengthen confidence scoring with detector outputs.
4. Add golden-case fixtures for `results/evidence/analysis` schema regression.

## 13) Quick Commands
Run pipeline:
```powershell
python -m profiler_agent.main --spec inputs/target_spec.json --out outputs
```

Run tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## 14) Machine-Readable Snapshot
```yaml
project: gpu-prof-agent
phase: "1.7 hardware intrinsic profiling"
io_contract:
  input: target_spec.json
  required_input_fields: [targets, run]
  outputs: [results.json, evidence.json, analysis.json]
status:
  pipeline: implemented
  dynamic_target_dispatch: implemented
  probe_framework: implemented
  analyzer: implemented
  normalization: implemented
  tests: "17/17 passing"
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
  - "probe precision still needs calibration for final scoring"
  - "tool unavailability (ncu/nvcc) lowers practical accuracy"
next_priority:
  - "probe statistical rigor"
  - "explicit anti-hacking detector module"
```

