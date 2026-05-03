#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <mode> [--messages] [--no-messages]" >&2
  exit 1
}

require_tool() {
  local tool_name="$1"
  command -v "${tool_name}" >/dev/null 2>&1 || {
    echo "missing required tool: ${tool_name}" >&2
    exit 1
  }
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_jq() {
  local description="$1"
  local program="$2"
  local json_input="$3"

  if ! jq -e "${program}" >/dev/null <<<"${json_input}"; then
    echo "failed check: ${description}" >&2
    jq . <<<"${json_input}" >&2 || printf '%s\n' "${json_input}" >&2
    exit 1
  fi
}

request_json() {
  local method="$1"
  local path="$2"
  local body="${3:-}"

  if [[ "${method}" == "GET" ]]; then
    curl -fsS "${AUTH_ARGS[@]}" "${BASE_URL}${path}"
    return
  fi

  curl -fsS "${AUTH_ARGS[@]}" \
    -H 'Content-Type: application/json' \
    -d "${body}" \
    "${BASE_URL}${path}"
}

check_model_present() {
  local models_json="$1"
  local model_id="$2"

  if ! jq -e --arg model_id "${model_id}" '.data | any(.id == $model_id)' >/dev/null <<<"${models_json}"; then
    fail "model id not advertised: ${model_id}"
  fi
}

MODE="${1:-}"
[[ -n "${MODE}" ]] || usage
shift

EXPECT_MESSAGES="0"
if [[ "${MODE}" == "lucebox" ]]; then
  EXPECT_MESSAGES="1"
fi

while (( $# > 0 )); do
  case "$1" in
    --messages)
      EXPECT_MESSAGES="1"
      ;;
    --no-messages)
      EXPECT_MESSAGES="0"
      ;;
    *)
      usage
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

require_tool curl
require_tool jq

cd "${REPO_ROOT}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
AUTH_FILE="${AUTH_FILE:-${REPO_ROOT}/auth.json}"
CHAT_MODEL="${CHAT_MODEL:-qwen3-chat}"
GEN_MODEL="${GEN_MODEL:-qmd-generate}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-embeddings}"
QMD_EMBED_MODEL="${QMD_EMBED_MODEL:-qmd-embed}"
RERANK_MODEL="${RERANK_MODEL:-qmd-rerank}"
EXPECTED_EMBED_DIM="${EXPECTED_EMBED_DIM:-}"

API_KEY_VALUE="${API_KEY:-}"
if [[ -z "${API_KEY_VALUE}" && -f "${AUTH_FILE}" ]]; then
  API_KEY_VALUE="$(jq -r '.api_key // empty' "${AUTH_FILE}")"
fi

AUTH_ARGS=()
if [[ -n "${API_KEY_VALUE}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY_VALUE}")
fi

echo "+ wait for ${BASE_URL}/health"
curl -fsS --retry 30 --retry-all-errors --retry-delay 1 "${BASE_URL}/health" >/dev/null

echo "+ GET /health"
health_json="$(request_json GET /health)"
if ! jq -e 'type == "object"' >/dev/null 2>/dev/null <<<"${health_json}"; then
  health_text="$(tr -d '\r\n' <<<"${health_json}")"
  [[ "${health_text}" == "OK" ]] || fail "unexpected health response: ${health_json}"
fi

echo "+ GET /v1/models"
models_json="$(request_json GET /v1/models)"
assert_jq "models response has data" '.data | type == "array" and length > 0' "${models_json}"
check_model_present "${models_json}" "${CHAT_MODEL}"
check_model_present "${models_json}" "${GEN_MODEL}"
check_model_present "${models_json}" "${EMBED_MODEL}"
check_model_present "${models_json}" "${QMD_EMBED_MODEL}"
check_model_present "${models_json}" "${RERANK_MODEL}"

echo "+ POST /v1/chat/completions"
chat_json="$(request_json POST /v1/chat/completions '{"model":"'"${CHAT_MODEL}"'","messages":[{"role":"user","content":"Reply with exactly ok."}],"max_tokens":16,"stream":false}')"
assert_jq "chat completion has assistant content" '.choices[0].message.content | type == "string" and length > 0' "${chat_json}"

if [[ "${EXPECT_MESSAGES}" == "1" ]]; then
  echo "+ POST /v1/messages"
  messages_json="$(request_json POST /v1/messages '{"model":"'"${CHAT_MODEL}"'","messages":[{"role":"user","content":"Reply with exactly ok."}],"max_tokens":16,"stream":false}')"
  assert_jq "messages response has content" '.content | type == "array" and length > 0' "${messages_json}"
fi

echo "+ POST /v1/completions"
completions_json="$(request_json POST /v1/completions '{"model":"'"${GEN_MODEL}"'","prompt":"Reply with exactly ok.","max_tokens":16}')"
assert_jq "completions response has text" '.choices[0].text | type == "string" and length > 0' "${completions_json}"

echo "+ POST /v1/responses"
responses_json="$(request_json POST /v1/responses '{"model":"'"${GEN_MODEL}"'","input":"Reply with exactly ok.","max_output_tokens":16}')"
assert_jq "responses output contains text" '[.output[]? | .content[]? | .text? // empty] | join("") | length > 0' "${responses_json}"

echo "+ POST /v1/embeddings (${EMBED_MODEL})"
embed_json="$(request_json POST /v1/embeddings '{"model":"'"${EMBED_MODEL}"'","input":"local llama embeddings smoke test"}')"
assert_jq "embedding response has vectors" '.data[0].embedding | type == "array" and length > 0' "${embed_json}"
embed_dim="$(jq -r '.data[0].embedding | length' <<<"${embed_json}")"

echo "+ POST /v1/embeddings (${QMD_EMBED_MODEL})"
qmd_embed_json="$(request_json POST /v1/embeddings '{"model":"'"${QMD_EMBED_MODEL}"'","input":"local llama embeddings smoke test"}')"
assert_jq "qmd embedding response has vectors" '.data[0].embedding | type == "array" and length > 0' "${qmd_embed_json}"
qmd_embed_dim="$(jq -r '.data[0].embedding | length' <<<"${qmd_embed_json}")"

if [[ "${embed_dim}" != "${qmd_embed_dim}" ]]; then
  fail "embedding alias dimension mismatch: ${embed_dim} != ${qmd_embed_dim}"
fi

if [[ -n "${EXPECTED_EMBED_DIM}" && "${embed_dim}" != "${EXPECTED_EMBED_DIM}" ]]; then
  fail "unexpected embedding dimension: got ${embed_dim}, expected ${EXPECTED_EMBED_DIM}"
fi

echo "+ POST /v1/rerank"
rerank_json="$(request_json POST /v1/rerank '{"model":"'"${RERANK_MODEL}"'","query":"best local reranker for qmd search","documents":["Qwen3 Reranker 8B is a cross-encoder reranker served through /v1/rerank.","Qwen3 Embeddings 8B creates vectors for retrieval, not pairwise reranking.","QMD Query Expansion rewrites search prompts before retrieval and reranking."]}')"
assert_jq "rerank returns one result per document" '.results | type == "array" and length == 3 and all(.[]; has("index") and has("relevance_score"))' "${rerank_json}"

echo "+ GET /ui/"
curl -fsS "${AUTH_ARGS[@]}" "${BASE_URL}/ui/" >/dev/null

echo "all public endpoint checks passed for mode: ${MODE}"