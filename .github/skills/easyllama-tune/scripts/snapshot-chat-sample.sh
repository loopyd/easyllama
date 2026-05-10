#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  echo "Usage: $0 <output.json> [prompt] [--prompt-file path] [--mode NAME] [--model-id ID]" >&2
  exit 1
}

OUTPUT_PATH="${1:-}"
shift || true

PROMPT_TEXT="Explain why lowering KV cache precision can change long-context behavior in three concise bullet points."
PROMPT_FILE=""

[[ -n "${OUTPUT_PATH}" ]] || usage

if (( $# > 0 )) && [[ "$1" != --* ]]; then
  PROMPT_TEXT="$1"
  shift
fi

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      [[ -n "${MODE}" ]] || usage
      shift 2
      ;;
    --model-id)
      MODEL_ID="${2:-}"
      [[ -n "${MODEL_ID}" ]] || usage
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="${2:-}"
      [[ -n "${PROMPT_FILE}" ]] || usage
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -n "${PROMPT_FILE}" ]]; then
  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "prompt file not found: ${PROMPT_FILE}" >&2
    exit 1
  fi
  PROMPT_TEXT="$(cat -- "${PROMPT_FILE}")"
fi

resolve_mode_config_paths

BASE_URL="${BASE_URL:-http://localhost:8080/v1/chat/completions}"
TARGET_MODEL="${TARGET_MODEL:-$(resolve_chat_model_id "$(config_for_reads)")}"
SEED_VALUE="${SEED_VALUE:-123}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE_VALUE="${TEMPERATURE_VALUE:-0}"
TOP_P_VALUE="${TOP_P_VALUE:-1}"
AUTH_FILE="${AUTH_FILE:-${LLAMACPP_AUTH_FILE:-${REPO_ROOT}/auth.json}}"
API_KEY_VALUE="${API_KEY:-${LLAMACPP_API_KEY:-}}"

mkdir -p "$(dirname -- "${OUTPUT_PATH}")"

payload_file="$(mktemp)"
trap 'rm -f "${payload_file}"' EXIT

if [[ -z "${API_KEY_VALUE}" && -f "${AUTH_FILE}" ]]; then
  API_KEY_VALUE="$(AUTH_FILE="${AUTH_FILE}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["AUTH_FILE"])
try:
    data = json.loads(path.read_text())
except Exception:
    data = {}

print(data.get("api_key", ""))
PY
)"
fi

AUTH_ARGS=()
if [[ -n "${API_KEY_VALUE}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY_VALUE}")
fi

PROMPT_TEXT="${PROMPT_TEXT}" \
TARGET_MODEL="${TARGET_MODEL}" \
SEED_VALUE="${SEED_VALUE}" \
MAX_TOKENS="${MAX_TOKENS}" \
TEMPERATURE_VALUE="${TEMPERATURE_VALUE}" \
TOP_P_VALUE="${TOP_P_VALUE}" \
"${PYTHON_BIN}" - <<'PY' >"${payload_file}"
import json
import os

payload = {
    "model": os.environ["TARGET_MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT_TEXT"]}],
    "temperature": float(os.environ["TEMPERATURE_VALUE"]),
    "top_p": float(os.environ["TOP_P_VALUE"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "seed": int(os.environ["SEED_VALUE"]),
}

print(json.dumps(payload))
PY

echo "+ curl -fsS ${BASE_URL} -H 'Content-Type: application/json' -d @${payload_file}"
curl -fsS "${BASE_URL}" \
  -H 'Content-Type: application/json' \
  "${AUTH_ARGS[@]}" \
  -d @"${payload_file}" | tee "${OUTPUT_PATH}" >/dev/null

echo "+ ${PYTHON_BIN} extract-preview ${OUTPUT_PATH}"
OUTPUT_PATH="${OUTPUT_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["OUTPUT_PATH"]).read_text())
choices = payload.get("choices") or []
first = (choices[0] if choices else {}) or {}
message = first.get("message") or {}
content = message.get("content")

preview = ""
if isinstance(content, list):
  preview = "\n".join(
    str(item.get("text", ""))
    for item in content
    if isinstance(item, dict) and item.get("type") == "text"
  ).strip()
elif isinstance(content, str):
  preview = content.strip()

if not preview:
  preview = str(message.get("reasoning_content") or "").strip()

if not preview:
  preview = str(first.get("text") or "").strip()

print(preview)
PY