#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="${RLINF_REPO_PATH:-}"
if [[ -z "${REPO_PATH}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../RLinf/rlinf" ]]; then
    REPO_PATH="$( cd "${SCRIPT_DIR}/../RLinf" && pwd )"
  elif [[ -d "${SCRIPT_DIR}/RLinf/rlinf" ]]; then
    REPO_PATH="$( cd "${SCRIPT_DIR}/RLinf" && pwd )"
  else
    echo "Could not locate RLinf repo. Set RLINF_REPO_PATH to the RLinf repository root." >&2
    exit 1
  fi
fi

MODEL_PATH="${MODEL_PATH:-${REPO_PATH}/weight/RLinf-Pi05-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
CONFIG_NAME="libero_10_ppo_openpi_pi05"
NUM_EPISODES=1
SAVE_FRACTION=1.0
SEED=""
SHUFFLE="false"
VLM_CHECK_INTERVAL=40
VLM_API_URL="${VLM_API_URL:-http://127.0.0.1:8972/v1/chat/completions}"
VLM_API_KEY="${VLM_API_KEY:-}"
VLM_X_AUTH_TOKEN="${VLM_X_AUTH_TOKEN:-}"
VLM_MODEL="${VLM_MODEL:-Qwen3.5-27B}"
VLM_PROMPT='You are judging whether a robot manipulation task is already complete from a single camera image. Reply with strict JSON only: {"terminate": true/false, "reason": "short reason"}. Set terminate=true only when the task goal is clearly finished in the image.'
VLM_TIMEOUT=30

# Task selection: use exactly one of the following modes.
LIST_TASKS="false"
TASK_ID=0
TASK_NAME=""

export RLINF_REPO_PATH="${REPO_PATH}"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to your local checkpoint directory before running." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

ARGS=(
  --config-name "${CONFIG_NAME}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-episodes "${NUM_EPISODES}"
  --save-fraction "${SAVE_FRACTION}"
)

if [[ "${LIST_TASKS}" == "true" ]]; then
  ARGS+=(--list-tasks)
elif [[ -n "${TASK_NAME}" ]]; then
  ARGS+=(--task-name "${TASK_NAME}")
else
  ARGS+=(--task-id "${TASK_ID}")
fi

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

if [[ "${SHUFFLE}" == "true" ]]; then
  ARGS+=(--shuffle)
fi

if [[ "${VLM_CHECK_INTERVAL}" != "0" ]]; then
  ARGS+=(--vlm-check-interval "${VLM_CHECK_INTERVAL}")
fi

if [[ -n "${VLM_API_URL}" ]]; then
  ARGS+=(--vlm-api-url "${VLM_API_URL}")
fi

if [[ -n "${VLM_API_KEY}" ]]; then
  ARGS+=(--vlm-api-key "${VLM_API_KEY}")
fi

if [[ -n "${VLM_X_AUTH_TOKEN}" ]]; then
  ARGS+=(--vlm-x-auth-token "${VLM_X_AUTH_TOKEN}")
fi

if [[ -n "${VLM_MODEL}" ]]; then
  ARGS+=(--vlm-model "${VLM_MODEL}")
fi

if [[ -n "${VLM_PROMPT}" ]]; then
  ARGS+=(--vlm-prompt "${VLM_PROMPT}")
fi

if [[ -n "${VLM_TIMEOUT}" ]]; then
  ARGS+=(--vlm-timeout "${VLM_TIMEOUT}")
fi

python "${SCRIPT_DIR}/simple_eval_libero10_pi05.py" "${ARGS[@]}"
