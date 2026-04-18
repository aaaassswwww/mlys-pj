#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/workspace"
SPEC_PATH="/target/target_spec.json"
ARTIFACT_DIR="${WORKSPACE_DIR}/.agent_artifacts"
FINAL_OUTPUT="${WORKSPACE_DIR}/output.json"
export PROFILER_AGENT_ENABLE_TIME_BUDGET="${PROFILER_AGENT_ENABLE_TIME_BUDGET:-1}"
export PROFILER_AGENT_MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS:-1740}"
export PROFILER_AGENT_SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS:-1755}"
TIME_BUDGET_ENABLED="${PROFILER_AGENT_ENABLE_TIME_BUDGET}"
MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS}"
SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS}"

cd "${WORKSPACE_DIR}"

# Keep exactly one /workspace/output.* artifact for evaluator pickup.
find "${WORKSPACE_DIR}" -maxdepth 1 -type f -name "output.*" -delete || true

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

# Multi-agent entrypoint (internally reuses stable pipeline execution).
RUN_RC=0
if [[ "${TIME_BUDGET_ENABLED}" == "1" || "${TIME_BUDGET_ENABLED}" == "true" || "${TIME_BUDGET_ENABLED}" == "yes" || "${TIME_BUDGET_ENABLED}" == "on" ]]; then
  echo "[run.sh] Time budget enabled: ${MAX_RUNTIME_SECONDS}s"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SHELL_TIMEOUT_SECONDS}s" python3 -m profiler_agent.main \
      --mode multi \
      --spec "${SPEC_PATH}" \
      --out "${ARTIFACT_DIR}" || RUN_RC=$?
  else
    python3 -m profiler_agent.main \
      --mode multi \
      --spec "${SPEC_PATH}" \
      --out "${ARTIFACT_DIR}" || RUN_RC=$?
  fi
else
  python3 -m profiler_agent.main \
    --mode multi \
    --spec "${SPEC_PATH}" \
    --out "${ARTIFACT_DIR}" || RUN_RC=$?
fi

# Package required single-file report artifact.
export RUN_RC
python3 - <<'PY'
import json
import os
from pathlib import Path

workspace = Path("/workspace")
artifacts = workspace / ".agent_artifacts"
spec_path = Path("/target/target_spec.json")

raw_spec = {"targets": []}
if spec_path.exists():
    try:
        raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        raw_spec = {"targets": []}

requested_targets = raw_spec.get("targets", [])
if not isinstance(requested_targets, list):
    requested_targets = []

def load_or_default(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

results_default = {str(target): 0.0 for target in requested_targets if str(target).strip()}
evidence_default = {
    "run": {"command": "", "returncode": 0, "stdout_tail": "", "stderr_tail": ""},
    "targets": {},
}
analysis_default = {
    "bound_type": "unknown",
    "confidence": 0.0,
    "bottlenecks": [],
}

results = load_or_default(artifacts / "results.json", results_default)
evidence = load_or_default(artifacts / "evidence.json", evidence_default)
analysis = load_or_default(artifacts / "analysis.json", analysis_default)

run_rc = int(os.environ.get("RUN_RC", "0") or "0")
time_budget = evidence.get("time_budget")
timed_out = isinstance(time_budget, dict) and bool(time_budget.get("timed_out"))
if run_rc == 124 and timed_out:
    timeout_note = "outer_shell_timeout_triggered_partial_artifacts_were_packaged"
    evidence.setdefault("time_budget", {})
    evidence["time_budget"]["shell_timeout_triggered"] = True
    evidence["time_budget"]["reason"] = timeout_note
    notes = analysis.get("analysis_notes")
    if not isinstance(notes, list):
        notes = []
    notes.append(timeout_note)
    analysis["analysis_notes"] = notes

payload = {
    "results": results,
    "evidence": evidence,
    "analysis": analysis,
}

plan = artifacts / "multi_agent_plan.json"
trace = artifacts / "multi_agent_trace.json"
agent_state = artifacts / "agent_state.json"
if plan.exists():
    payload["multi_agent_plan"] = json.loads(plan.read_text(encoding="utf-8"))
if trace.exists():
    payload["multi_agent_trace"] = json.loads(trace.read_text(encoding="utf-8"))
if agent_state.exists():
    payload["agent_state"] = json.loads(agent_state.read_text(encoding="utf-8"))

(workspace / "output.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

if [[ "${RUN_RC}" -ne 0 && "${RUN_RC}" -ne 124 ]]; then
  exit "${RUN_RC}"
fi
