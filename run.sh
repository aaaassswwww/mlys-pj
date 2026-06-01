#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/.agent_artifacts"
FINAL_OUTPUT="${ROOT_DIR}/output3.json"
FINAL_ENGINE="${ROOT_DIR}/engine.py"
RESULT_LOG="${ROOT_DIR}/results.log"
BENCHMARK_CAPTURE="${ROOT_DIR}/benchmark_results.json"

RUN_SH_LLM_API_KEY=""
RUN_SH_LLM_BASE_URL=""
RUN_SH_LLM_MODEL="gpt-5.4"

LLM_SECRET_FILE=""
LLM_BASE_URL_ARG=""
LLM_MODEL_ARG=""
PURGE_LLM_SECRET_FILE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-secret-file)
      LLM_SECRET_FILE="${2:-}"
      shift 2
      ;;
    --llm-base-url)
      LLM_BASE_URL_ARG="${2:-}"
      shift 2
      ;;
    --llm-model)
      LLM_MODEL_ARG="${2:-}"
      shift 2
      ;;
    --purge-llm-secret-file)
      PURGE_LLM_SECRET_FILE="1"
      shift
      ;;
    *)
      echo "[run.sh] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cleanup_secret_file() {
  if [[ "${PURGE_LLM_SECRET_FILE}" == "1" && -n "${LLM_SECRET_FILE}" && -f "${LLM_SECRET_FILE}" ]]; then
    if command -v shred >/dev/null 2>&1; then
      shred -u "${LLM_SECRET_FILE}" || rm -f "${LLM_SECRET_FILE}" || true
    else
      rm -f "${LLM_SECRET_FILE}" || true
    fi
  fi
}

if [[ -z "${LLM_SECRET_FILE}" && -n "${RUN_SH_LLM_API_KEY}" ]]; then
  SECRET_DIR="/dev/shm"
  if [[ ! -d "${SECRET_DIR}" || ! -w "${SECRET_DIR}" ]]; then
    SECRET_DIR="${TMPDIR:-/tmp}"
  fi
  umask 077
  LLM_SECRET_FILE="$(mktemp "${SECRET_DIR%/}/profiler_agent_llm_secret.XXXXXX")"
  printf '%s\n' "${RUN_SH_LLM_API_KEY}" > "${LLM_SECRET_FILE}"
  PURGE_LLM_SECRET_FILE="1"
  if [[ -z "${LLM_BASE_URL_ARG}" && -n "${RUN_SH_LLM_BASE_URL}" ]]; then
    LLM_BASE_URL_ARG="${RUN_SH_LLM_BASE_URL}"
  fi
  if [[ -z "${LLM_MODEL_ARG}" && -n "${RUN_SH_LLM_MODEL}" ]]; then
    LLM_MODEL_ARG="${RUN_SH_LLM_MODEL}"
  fi
fi

trap cleanup_secret_file EXIT

LLM_ARGS=()
if [[ -n "${LLM_SECRET_FILE}" ]]; then
  LLM_ARGS+=(--llm-secret-file "${LLM_SECRET_FILE}")
fi
if [[ -n "${LLM_BASE_URL_ARG}" ]]; then
  LLM_ARGS+=(--llm-base-url "${LLM_BASE_URL_ARG}")
fi
if [[ -n "${LLM_MODEL_ARG}" ]]; then
  LLM_ARGS+=(--llm-model "${LLM_MODEL_ARG}")
fi

export PROFILER_AGENT_ENABLE_TIME_BUDGET="${PROFILER_AGENT_ENABLE_TIME_BUDGET:-1}"
export PROFILER_AGENT_MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS:-1740}"
export PROFILER_AGENT_SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS:-1755}"
export PROFILER_AGENT_PHASE3_ITERATIONS="${PROFILER_AGENT_PHASE3_ITERATIONS:-0}"
export PROFILER_AGENT_PHASE3_STOP_BUFFER_SECONDS="${PROFILER_AGENT_PHASE3_STOP_BUFFER_SECONDS:-135}"
export OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-120}"

TIME_BUDGET_ENABLED="${PROFILER_AGENT_ENABLE_TIME_BUDGET}"
MAX_RUNTIME_SECONDS="${PROFILER_AGENT_MAX_RUNTIME_SECONDS}"
SHELL_TIMEOUT_SECONDS="${PROFILER_AGENT_SHELL_TIMEOUT_SECONDS}"
PHASE3_ITERATIONS="${PROFILER_AGENT_PHASE3_ITERATIONS}"

mkdir -p "${ARTIFACT_DIR}"
: > "${RESULT_LOG}"
rm -f "${FINAL_OUTPUT}" "${BENCHMARK_CAPTURE}"

exec > >(tee -a "${RESULT_LOG}") 2>&1

echo "[run.sh] start"
echo "[run.sh] root=${ROOT_DIR}"
echo "[run.sh] phase3_iterations=${PHASE3_ITERATIONS}"

cd "${ROOT_DIR}"

if [[ -n "${LLM_SECRET_FILE}" && -f "${LLM_SECRET_FILE}" ]]; then
  echo "[run.sh] llm_secret_file_configured=yes"
else
  echo "[run.sh] llm_secret_file_configured=no"
fi

python3 - <<'PY'
import json
from pathlib import Path

payload = {}
try:
    import torch
    payload = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    payload = {"probe_error": f"{type(exc).__name__}: {exc}"}

print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
Path(".agent_artifacts/runtime_env.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

RUN_RC=0
if [[ "${TIME_BUDGET_ENABLED}" == "1" || "${TIME_BUDGET_ENABLED}" == "true" || "${TIME_BUDGET_ENABLED}" == "yes" || "${TIME_BUDGET_ENABLED}" == "on" ]]; then
  echo "[run.sh] Time budget enabled: ${MAX_RUNTIME_SECONDS}s"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SHELL_TIMEOUT_SECONDS}s" python3 -m profiler_agent.main \
      --mode phase3 \
      --out "${ROOT_DIR}" \
      --phase3-iterations "${PHASE3_ITERATIONS}" \
      "${LLM_ARGS[@]}" || RUN_RC=$?
  else
    python3 -m profiler_agent.main \
      --mode phase3 \
      --out "${ROOT_DIR}" \
      --phase3-iterations "${PHASE3_ITERATIONS}" \
      "${LLM_ARGS[@]}" || RUN_RC=$?
  fi
else
  python3 -m profiler_agent.main \
    --mode phase3 \
    --out "${ROOT_DIR}" \
    --phase3-iterations "${PHASE3_ITERATIONS}" \
    "${LLM_ARGS[@]}" || RUN_RC=$?
fi

if [[ -f "${FINAL_ENGINE}" ]]; then
  echo "[run.sh] running selfcheck"
  python3 tools/selfcheck_submission.py
  echo "[run.sh] selfcheck=passed"

  if [[ "${RUN_RC}" -eq 0 ]]; then
    echo "[run.sh] running local benchmark capture"
    python3 tools/benchmark_local.py --device auto --batch-size 4 --prompt-len 16 --decode-steps 8 --warmup 0 --repeat 1 | tee "${BENCHMARK_CAPTURE}" || true
  fi
fi

export RUN_RC
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(".").resolve()
artifacts = root / ".agent_artifacts"
engine_path = root / "engine.py"
report_path = artifacts / "phase3_report.json"
state_path = artifacts / "phase3_state.json"
benchmark_path = root / "benchmark_results.json"

def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

payload = {
    "mode": "phase3",
    "engine_exists": engine_path.exists(),
    "engine_path": str(engine_path),
    "phase3_state": load_json(state_path),
    "phase3_report": load_json(report_path),
    "benchmark_capture": load_text(benchmark_path),
    "run_rc": int(os.environ.get("RUN_RC", "0") or "0"),
    "timed_out": int(os.environ.get("RUN_RC", "0") or "0") == 124,
}

(root / "output3.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

echo "[run.sh] output_file=${FINAL_OUTPUT}"
echo "[run.sh] end"

if [[ "${RUN_RC}" -ne 0 && "${RUN_RC}" -ne 124 ]]; then
  exit "${RUN_RC}"
fi
