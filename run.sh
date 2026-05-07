#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/workspace"
SPEC_PATH="/target/target_spec.json"
ARTIFACT_DIR="${WORKSPACE_DIR}/.agent_artifacts"
FINAL_OUTPUT="${WORKSPACE_DIR}/output.json"
FINAL_KERNEL="${WORKSPACE_DIR}/optimized_lora.cu"
export PROFILER_AGENT_ENABLE_TIME_BUDGET="${PROFILER_AGENT_ENABLE_TIME_BUDGET:-1}"
export PROFILER_AGENT_MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS:-1740}"
export PROFILER_AGENT_SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS:-1755}"
export PROFILER_AGENT_PHASE2_ITERATIONS="${PROFILER_AGENT_PHASE2_ITERATIONS:-2}"
TIME_BUDGET_ENABLED="${PROFILER_AGENT_ENABLE_TIME_BUDGET}"
MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS}"
SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS}"
PHASE2_ITERATIONS="${PROFILER_AGENT_PHASE2_ITERATIONS}"

cd "${WORKSPACE_DIR}"

# Keep exactly one /workspace/output.* artifact for evaluator pickup.
find "${WORKSPACE_DIR}" -maxdepth 1 -type f -name "output.*" -delete || true
rm -f "${FINAL_KERNEL}"

echo "[run.sh] Using target spec: ${SPEC_PATH}"
if [[ -f "${SPEC_PATH}" ]]; then
  echo "[run.sh] ---- begin target_spec.json ----"
  cat "${SPEC_PATH}"
  echo
  echo "[run.sh] ---- end target_spec.json ----"
else
  echo "[run.sh] target spec not found: ${SPEC_PATH}" >&2
fi

rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}"

# Phase 2 entrypoint: optimize and keep root-level optimized_lora.cu updated.
RUN_RC=0
if [[ "${TIME_BUDGET_ENABLED}" == "1" || "${TIME_BUDGET_ENABLED}" == "true" || "${TIME_BUDGET_ENABLED}" == "yes" || "${TIME_BUDGET_ENABLED}" == "on" ]]; then
  echo "[run.sh] Time budget enabled: ${MAX_RUNTIME_SECONDS}s"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SHELL_TIMEOUT_SECONDS}s" python3 -m profiler_agent.main \
      --mode phase2 \
      --out "${WORKSPACE_DIR}" \
      --phase2-iterations "${PHASE2_ITERATIONS}" || RUN_RC=$?
  else
    python3 -m profiler_agent.main \
      --mode phase2 \
      --out "${WORKSPACE_DIR}" \
      --phase2-iterations "${PHASE2_ITERATIONS}" || RUN_RC=$?
  fi
else
  python3 -m profiler_agent.main \
    --mode phase2 \
    --out "${WORKSPACE_DIR}" \
    --phase2-iterations "${PHASE2_ITERATIONS}" || RUN_RC=$?
fi

# Package compatibility report artifact while keeping optimized_lora.cu as primary output.
export RUN_RC
python3 - <<'PY'
import json
import os
from pathlib import Path

workspace = Path("/workspace")
artifacts = workspace / ".agent_artifacts"
spec_path = Path("/target/target_spec.json")
kernel_path = workspace / "optimized_lora.cu"
phase2_state_path = artifacts / "phase2_state.json"
phase2_report_path = artifacts / "phase2_report.json"

raw_spec = {"targets": []}
if spec_path.exists():
    try:
        raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        raw_spec = {"targets": []}

def load_or_default(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

phase2_state = load_or_default(phase2_state_path, {})
phase2_report = load_or_default(phase2_report_path, {})

run_rc = int(os.environ.get("RUN_RC", "0") or "0")
timed_out = run_rc == 124

payload = {
    "mode": "phase2",
    "phase2": {
        "optimized_lora_exists": kernel_path.exists(),
        "optimized_lora_path": str(kernel_path),
        "phase2_state": phase2_state,
        "phase2_report": phase2_report,
        "timed_out": timed_out,
    },
    "target_spec_echo": raw_spec,
    "run_rc": run_rc,
}

(workspace / "output.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

if [[ "${RUN_RC}" -ne 0 && "${RUN_RC}" -ne 124 ]]; then
  exit "${RUN_RC}"
fi
