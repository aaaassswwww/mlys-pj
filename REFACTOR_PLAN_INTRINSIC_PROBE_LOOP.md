# Intrinsic Probe Loop Refactor Plan

## Goal

Promote intrinsic probe measurement to a first-class path alongside device-attribute queries and workload-backed profiling, while preserving:

- `results.json`
- `evidence.json`
- `analysis.json`
- `run.sh -> /workspace/output.json`

## Current Status

Implemented in this phase:

- `target_semantics.py` now distinguishes:
  - `device_attribute`
  - `intrinsic_probe`
  - `workload_counter`
  - `unknown`
- semantic evidence now carries:
  - `semantic_class`
  - `semantic_subclass`
  - `measurement_mode_candidate`
- intrinsic measurements now route through `probe_iteration.py`
- evidence now distinguishes:
  - `device_attribute_query`
  - `synthetic_intrinsic_probe`
  - `workload_profile`
  - `placeholder_no_run`

## Phase Progress

### Phase A

Done:

- explicit probe-first intrinsic semantics
- workload counters no longer share the intrinsic semantic class

### Phase B

Done for minimum viable loop:

- added `ProbeIterationState`
- added `ProbeIterationResult`
- compile/run failure feedback now drives next-round probe regeneration via `prior_error`

Remaining:

- optional `ncu` profiling of generated probe binaries
- richer probe-stage analysis and next-action selection

### Next Recommended Steps

1. Expand probe prompts for measurement-specific experiment design.
2. Split probe compile/run/profile helpers into explicit typed adapters.
3. Add probe-result analysis for stability/range checking.
4. Feed multi-agent refinement directives into probe-iteration next actions.
