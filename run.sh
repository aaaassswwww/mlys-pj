#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/workspace"
SPEC_PATH="/target/target_spec.json"
ARTIFACT_DIR="${WORKSPACE_DIR}/.agent_artifacts"
FINAL_OUTPUT="${WORKSPACE_DIR}/output.json"
FINAL_KERNEL="${WORKSPACE_DIR}/optimized_lora.cu"

# Optional inline LLM config for local/server runs.
# Set RUN_SH_LLM_API_KEY to a real value when you want run.sh itself to drive Phase 2.
# After the run, clear this line back to "" if you do not want the key to remain in the file.
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
export PROFILER_AGENT_PHASE2_ITERATIONS="${PROFILER_AGENT_PHASE2_ITERATIONS:-15}"
export PROFILER_AGENT_PHASE2_SPEEDUP_ITERATIONS="${PROFILER_AGENT_PHASE2_SPEEDUP_ITERATIONS:-30}"
export PROFILER_AGENT_PHASE2_STOP_BUFFER_SECONDS="${PROFILER_AGENT_PHASE2_STOP_BUFFER_SECONDS:-135}"
export OPENAI_TIMEOUT_S="${OPENAI_TIMEOUT_S:-120}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-2}"
export OPENAI_RETRY_BASE_S="${OPENAI_RETRY_BASE_S:-2}"
export PROFILER_AGENT_LLM_PRECHECK="${PROFILER_AGENT_LLM_PRECHECK:-1}"
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

echo "[run.sh] ---- begin torch/cuda precision probe ----"
python3 - <<'PY'
import json
from pathlib import Path

payload = {}
try:
    import torch  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "matmul_allow_tf32": (
            getattr(torch.backends.cuda.matmul, "allow_tf32", None)
            if cuda_available and hasattr(torch.backends, "cuda")
            else None
        ),
        "cudnn_allow_tf32": (
            getattr(torch.backends.cudnn, "allow_tf32", None)
            if cuda_available and hasattr(torch.backends, "cudnn")
            else None
        ),
        "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: "unsupported")(),
    }
except Exception as exc:
    payload = {
        "probe_error": f"{type(exc).__name__}: {exc}",
    }

print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
Path("/workspace/.agent_artifacts/tf32_env.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
echo "[run.sh] ---- end torch/cuda precision probe ----"

if [[ "${PROFILER_AGENT_LLM_PRECHECK}" == "1" || "${PROFILER_AGENT_LLM_PRECHECK}" == "true" || "${PROFILER_AGENT_LLM_PRECHECK}" == "yes" || "${PROFILER_AGENT_LLM_PRECHECK}" == "on" ]]; then
  if [[ -n "${LLM_SECRET_FILE}" && -f "${LLM_SECRET_FILE}" ]]; then
    echo "[run.sh] ---- begin llm precheck ----"
    export PROFILER_AGENT_LLM_DEBUG_PATH="/workspace/.agent_artifacts/llm_precheck_debug.jsonl"
    rm -f "${PROFILER_AGENT_LLM_DEBUG_PATH}"
    python3 - "${LLM_SECRET_FILE}" "${LLM_BASE_URL_ARG}" "${LLM_MODEL_ARG}" <<'PY'
import json
import os
import sys
from pathlib import Path

from profiler_agent.multi_agent.llm_client import OpenAICompatibleLLMClient

secret_file = Path(sys.argv[1])
base_url = sys.argv[2] or None
model = sys.argv[3] or None
payload = {
    "secret_file_exists": secret_file.exists(),
    "base_url_override": base_url,
    "model_override": model,
}
client = OpenAICompatibleLLMClient.from_secret_file(secret_file, base_url=base_url, model=model)
if client is None:
    payload["ok"] = False
    payload["error"] = "unable_to_build_llm_client_from_secret_file"
else:
    payload["resolved_base_url"] = client.config.base_url
    payload["resolved_model"] = client.config.model
    try:
        result = client.complete_json(
            system_prompt="Return JSON only.",
            user_prompt='{"task":"precheck","response_schema":{"ok":"bool","message":"short_string"},"instruction":"Return a tiny JSON object confirming the API call succeeded."}',
        )
        payload["ok"] = isinstance(result, dict)
        payload["result"] = result
        if result is None:
            payload["error"] = "llm_complete_json_returned_none"
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"

probe_path = Path("/workspace/.agent_artifacts/llm_precheck.json")
probe_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
debug_path = Path(os.environ.get("PROFILER_AGENT_LLM_DEBUG_PATH", "")).expanduser()
if debug_path and debug_path.exists():
    try:
        last_line = debug_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        debug_payload = json.loads(last_line)
        summary = {
            "phase": debug_payload.get("phase"),
            "http_status": debug_payload.get("http_status"),
            "error_category": debug_payload.get("error_category"),
            "error_type": debug_payload.get("error_type"),
            "error_message": debug_payload.get("error_message"),
            "url": debug_payload.get("url"),
            "retryable": debug_payload.get("retryable"),
            "will_retry": debug_payload.get("will_retry"),
        }
        print(json.dumps({"llm_precheck_debug": summary}, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"llm_precheck_debug_error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, sort_keys=True))
PY
    echo "[run.sh] ---- end llm precheck ----"
  else
    echo "[run.sh] llm precheck skipped: no secret file configured"
  fi
fi

# Phase 2 entrypoint: optimize and keep root-level optimized_lora.cu updated.
RUN_RC=0
if [[ "${TIME_BUDGET_ENABLED}" == "1" || "${TIME_BUDGET_ENABLED}" == "true" || "${TIME_BUDGET_ENABLED}" == "yes" || "${TIME_BUDGET_ENABLED}" == "on" ]]; then
  echo "[run.sh] Time budget enabled: ${MAX_RUNTIME_SECONDS}s"
  echo "[run.sh] LLM timeout: ${OPENAI_TIMEOUT_S}s, retries: ${OPENAI_MAX_RETRIES}, retry base: ${OPENAI_RETRY_BASE_S}s"
  echo "[run.sh] Phase2 iterations: ${PHASE2_ITERATIONS} (0 means time-budget driven), speedup-stage floor: ${PROFILER_AGENT_PHASE2_SPEEDUP_ITERATIONS}, stop buffer: ${PROFILER_AGENT_PHASE2_STOP_BUFFER_SECONDS}s"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SHELL_TIMEOUT_SECONDS}s" python3 -m profiler_agent.main \
      --mode phase2 \
      --out "${WORKSPACE_DIR}" \
      --phase2-iterations "${PHASE2_ITERATIONS}" \
      "${LLM_ARGS[@]}" || RUN_RC=$?
  else
    python3 -m profiler_agent.main \
      --mode phase2 \
      --out "${WORKSPACE_DIR}" \
      --phase2-iterations "${PHASE2_ITERATIONS}" \
      "${LLM_ARGS[@]}" || RUN_RC=$?
  fi
else
  python3 -m profiler_agent.main \
    --mode phase2 \
    --out "${WORKSPACE_DIR}" \
    --phase2-iterations "${PHASE2_ITERATIONS}" \
    "${LLM_ARGS[@]}" || RUN_RC=$?
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
