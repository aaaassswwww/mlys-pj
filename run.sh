#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/workspace"
SPEC_PATH="/target/target_spec.json"
ARTIFACT_DIR="${WORKSPACE_DIR}/.agent_artifacts"
FINAL_OUTPUT="${WORKSPACE_DIR}/output.json"

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
python3 -m profiler_agent.main \
  --mode multi \
  --spec "${SPEC_PATH}" \
  --out "${ARTIFACT_DIR}"

# Package required single-file report artifact.
python3 - <<'PY'
import json
from pathlib import Path

workspace = Path("/workspace")
artifacts = workspace / ".agent_artifacts"

payload = {
    "results": json.loads((artifacts / "results.json").read_text(encoding="utf-8")),
    "evidence": json.loads((artifacts / "evidence.json").read_text(encoding="utf-8")),
    "analysis": json.loads((artifacts / "analysis.json").read_text(encoding="utf-8")),
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
