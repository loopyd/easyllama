#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_HOST="host"
RUNTIME_CONTAINER="container"
LLAMA_SWAP_BIN="/app/bin/llama-swap"
MODELS_DIR_CONTAINER="/root/.cache/huggingface/hub"
CHAT_TEMPLATE_DIR_CONTAINER="/chat_template"
MMPROJ_DIR_CONTAINER="/mmproj"
HF_URL_BASE="https://huggingface.co"
DEFAULT_AUTH_FILE="${SCRIPT_DIR}/auth.json"
AUTH_EXAMPLE_FILE="${SCRIPT_DIR}/auth.json.example"
AUTH_FILE="${LLAMACPP_AUTH_FILE:-${DEFAULT_AUTH_FILE}}"
DEFAULT_LS_CONFIG_FILE="${SCRIPT_DIR}/config.yml"
LS_CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.yml.example"
LEGACY_LS_CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
LEGACY_LS_CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.yaml.example"
IMAGE_NAME="${LLAMACPP_IMAGE_NAME:-llamacpp-local:cuda13}"
CONTAINER_NAME="${LLAMACPP_CONTAINER_NAME:-llamacpp-server-swap}"
MODELS_DIR="${LLAMACPP_MODELS_DIR:-models}"
CHAT_TEMPLATE_DIR="${LLAMACPP_CHAT_TEMPLATE_DIR:-chat_template}"
MMPROJ_DIR="${LLAMACPP_MMPROJ_DIR:-mmproj}"
LS_CONFIG_FILE="${LLAMACPP_LS_CONFIG_FILE:-}"
HOST_PORT="${LLAMACPP_HOST_PORT:-8080}"
CONTAINER_PORT="${LLAMACPP_CONTAINER_PORT:-8080}"
DEFAULT_CUDA_ARCHITECTURES="${LLAMACPP_DEFAULT_CUDA_ARCHITECTURES:-120}"
CMAKE_CUDA_ARCHITECTURES="${LLAMACPP_CMAKE_CUDA_ARCHITECTURES:-auto}"
LLAMA_CPP_REPO="${LLAMACPP_LLAMA_CPP_REPO:-https://github.com/TheTom/llama-cpp-turboquant.git}"
LLAMA_CPP_REF="${LLAMACPP_LLAMA_CPP_REF:-feature/turboquant-kv-cache}"
HOST_LANG="${LLAMACPP_HOST_LANG:-${LANG:-C.UTF-8}}"
HOST_LC_ALL="${LLAMACPP_HOST_LC_ALL:-${LC_ALL:-${HOST_LANG}}}"
LOG_LEVEL="${LLAMACPP_LOG_LEVEL:-info}"
NO_COLOR="${LLAMACPP_NO_COLOR:-}"
HF_TOKEN_RESOLVED=""
API_KEY_RESOLVED=""
runtime_mode(){ local m="${LLAMACPP_RUNTIME_MODE:-}"; case "${m}" in "") [[ -f /.dockerenv ]] && { printf '%s' "${RUNTIME_CONTAINER}"; return; }; printf '%s' "${RUNTIME_HOST}" ;; "${RUNTIME_HOST}"|"${RUNTIME_CONTAINER}") printf '%s' "${m}" ;; *) printf 'unsupported LLAMACPP_RUNTIME_MODE=%s; allowed: %s,%s\n' "${m}" "${RUNTIME_HOST}" "${RUNTIME_CONTAINER}" >&2; exit 1 ;; esac; }
RUNTIME_MODE="$(runtime_mode)"
colors(){ if [[ -t 1 && -z "${NO_COLOR}" ]]; then C_RESET='\033[0m'; C_INFO='\033[1;34m'; C_WARN='\033[1;33m'; C_ERROR='\033[1;31m'; C_OK='\033[1;32m'; else C_RESET=''; C_INFO=''; C_WARN=''; C_ERROR=''; C_OK=''; fi; }
colors
log(){ local l="${1}" c="${C_INFO}"; shift; case "${LOG_LEVEL}" in debug) ;; info) [[ "${l}" == "DEBUG" ]] && return 0 ;; warn) [[ "${l}" != "WARN" && "${l}" != "ERROR" ]] && return 0 ;; error) [[ "${l}" != "ERROR" ]] && return 0 ;; esac; case "${l}" in WARN) c="${C_WARN}" ;; ERROR) c="${C_ERROR}" ;; OK) c="${C_OK}" ;; esac; printf '%b[%s] %s%b\n' "${c}" "${l}" "$*" "${C_RESET}"; }
info(){ log INFO "$*"; }
warn(){ log WARN "$*"; }
ok(){ log OK "$*"; }
err(){ log ERROR "$*"; }
die(){ err "$*"; exit 1; }
need(){ local -a miss=(); local d; for d in "$@"; do command -v "${d}" >/dev/null 2>&1 || miss+=("${d}"); done; if [[ ${#miss[@]} -gt 0 ]]; then die "missing dependency: ${miss[*]}"; fi; }
apath(){ local p="${1}"; [[ "${p}" = /* ]] && { printf '%s' "${p}"; return; }; printf '%s' "${SCRIPT_DIR}/${p}"; }
get_tz(){ local t; [[ -n "${TZ:-}" ]] && { printf '%s' "${TZ}"; return; }; [[ -r /etc/timezone ]] && { tr -d '\n' < /etc/timezone; return; }; command -v timedatectl >/dev/null 2>&1 && { t="$(timedatectl show -p Timezone --value 2>/dev/null || true)"; [[ -n "${t}" ]] && { printf '%s' "${t}"; return; }; }; printf '%s' "UTC"; }
need_docker(){ command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"; docker info >/dev/null 2>&1 || die "docker daemon is not reachable (start docker and retry)"; }
cfg(){ HOST_TZ="${LLAMACPP_HOST_TZ:-$(get_tz)}"; MODELS_DIR="$(apath "${MODELS_DIR}")"; CHAT_TEMPLATE_DIR="$(apath "${CHAT_TEMPLATE_DIR}")"; MMPROJ_DIR="$(apath "${MMPROJ_DIR}")"; if [[ -n "${LS_CONFIG_FILE}" ]]; then LS_CONFIG_FILE="$(apath "${LS_CONFIG_FILE}")"; fi; }
load_auth(){ local src=""; HF_TOKEN_RESOLVED="${HF_TOKEN:-${LLAMACPP_HF_TOKEN:-}}"; API_KEY_RESOLVED="${LLAMACPP_API_KEY:-${API_KEY:-}}"; [[ -n "${HF_TOKEN_RESOLVED}" && -n "${API_KEY_RESOLVED}" ]] && return 0; if [[ -r "${AUTH_FILE}" ]]; then src="${AUTH_FILE}"; elif [[ -r "${AUTH_EXAMPLE_FILE}" ]]; then src="${AUTH_EXAMPLE_FILE}"; info "using ${AUTH_EXAMPLE_FILE}; create ${DEFAULT_AUTH_FILE} for local credentials"; else warn "no auth file found at ${AUTH_FILE}; private Hugging Face downloads may fail"; return 0; fi; need jq; jq empty "${src}" >/dev/null 2>&1 || die "invalid JSON auth file: ${src}"; [[ -z "${HF_TOKEN_RESOLVED}" ]] && HF_TOKEN_RESOLVED="$(jq -r '.hf_token // empty' "${src}")"; [[ -z "${API_KEY_RESOLVED}" ]] && API_KEY_RESOLVED="$(jq -r '.api_key // empty' "${src}")"; }
listen_url(){ if [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]]; then printf '%s' "http://127.0.0.1:${HOST_PORT}"; else printf '%s' "http://127.0.0.1:${CONTAINER_PORT}"; fi; }
resolved_api_key(){ local f="" k="${API_KEY_RESOLVED:-${LLAMACPP_API_KEY:-${API_KEY:-}}}"; if [[ -z "${k}" ]]; then if [[ "${RUNTIME_MODE}" == "${RUNTIME_CONTAINER}" ]]; then f="$(container_cfg_path 2>/dev/null || true)"; else f="$(resolve_ls_config 2>/dev/null || true)"; fi; if [[ -r "${f}" ]]; then k="$(awk 'BEGIN { in_api = 0 } /^apiKeys:[[:space:]]*$/ { in_api = 1; next } in_api && /^[[:space:]]*-[[:space:]]*/ { line = $0; sub(/^[[:space:]]*-[[:space:]]*/, "", line); sub(/[[:space:]]+#.*$/, "", line); print line; exit } in_api && $0 !~ /^[[:space:]]/ { exit }' "${f}")"; k="${k#\"}"; k="${k%\"}"; if [[ "${k}" =~ ^\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}$ ]]; then k="${!BASH_REMATCH[1]:-}"; fi; fi; fi; printf '%s' "${k}"; }
model_ready(){ local base="${1}" model_id="${2}" json; shift 2; json="$(curl -fsS "$@" "${base}/running")" || return 1; printf '%s' "${json}" | jq -e --arg model_id "${model_id}" '.running[]? | select(.model == $model_id and .state == "ready")' >/dev/null; }
hf_mmproj_url(){ local s="${1}" o r e f; [[ -n "${s}" ]] || die "LLAMACPP_HF_MMPROJ cannot be empty"; o="${s%%/*}"; r="${s#*/}"; e="${r%%/*}"; f="${r#*/}"; [[ "${o}" == "${s}" || "${r}" == "${s}" || -z "${o}" || -z "${e}" || -z "${f}" ]] && die "LLAMACPP_HF_MMPROJ must be <owner>/<repo>/<file.gguf>; got: ${s}"; printf '%s' "${HF_URL_BASE}/${o}/${e}/blob/main/${f}"; }
map_mmproj(){ local p="${1}" u n o t rs ps; [[ -n "${p}" ]] || { printf '%s' ""; return; }; if [[ "${p}" =~ ^https?:// ]]; then need curl; u="${p}"; [[ "${u}" =~ ^https?://huggingface\.co/.*/blob/ ]] && u="${u/\/blob\//\/resolve\/}"; n="$(basename "${u%%\?*}")"; [[ -n "${n}" ]] || die "could not infer mmproj filename from URL: ${p}"; o="${MMPROJ_DIR}/${n}"; t="${o}.part"; mkdir -p -- "${MMPROJ_DIR}"; if [[ -n "${HF_TOKEN_RESOLVED}" ]]; then rs="$(curl -fsIL -H "Authorization: Bearer ${HF_TOKEN_RESOLVED}" "${u}" 2>/dev/null | awk 'tolower($1)=="content-length:" {gsub(/\r/, "", $2); print $2}' | tail -n1 || true)"; else rs="$(curl -fsIL "${u}" 2>/dev/null | awk 'tolower($1)=="content-length:" {gsub(/\r/, "", $2); print $2}' | tail -n1 || true)"; fi; if [[ ! -s "${o}" || ( -n "${rs}" && "$(wc -c < "${o}")" != "${rs}" ) ]]; then info "downloading mmproj from ${p}" >&2; [[ -s "${o}" && ! -s "${t}" ]] && mv -f -- "${o}" "${t}"; if [[ -n "${HF_TOKEN_RESOLVED}" ]]; then curl -fL --retry 3 --retry-delay 2 -C - -H "Authorization: Bearer ${HF_TOKEN_RESOLVED}" -o "${t}" "${u}" >/dev/null; else curl -fL --retry 3 --retry-delay 2 -C - -o "${t}" "${u}" >/dev/null; fi; if [[ -n "${rs}" ]]; then ps="$(wc -c < "${t}")"; [[ "${ps}" == "${rs}" ]] || die "mmproj download incomplete for ${p}: got ${ps} bytes, expected ${rs}"; fi; mv -f -- "${t}" "${o}"; ok "downloaded mmproj to ${o}" >&2; fi; printf '%s' "${MMPROJ_DIR_CONTAINER}/${n}"; return; fi; [[ "${p}" == "${MMPROJ_DIR_CONTAINER}"/* ]] && { printf '%s' "${p}"; return; }; [[ "${p}" == "${MMPROJ_DIR}"/* ]] && { printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#"${MMPROJ_DIR}"/}"; return; }; [[ "${p}" == mmproj/* ]] && { printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#mmproj/}"; return; }; [[ "${p}" != */* ]] && { printf '%s' "${MMPROJ_DIR_CONTAINER}/${p}"; return; }; [[ "${p}" == /* ]] && die "LLAMACPP_MMPROJ_FILE must be in ${MMPROJ_DIR}, use mmproj/<file>, or provide a URL"; printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#./}"; }
mmproj_arg(){ local s="${LLAMACPP_MMPROJ_FILE:-}" m; [[ -n "${LLAMACPP_HF_MMPROJ:-}" && -z "${s}" ]] && s="$(hf_mmproj_url "${LLAMACPP_HF_MMPROJ}")"; [[ -z "${s}" ]] && { printf '%s' ""; return; }; m="$(map_mmproj "${s}")"; printf '%s' "--mmproj ${m}"; }
cuda_arch(){ local a; [[ "${CMAKE_CUDA_ARCHITECTURES}" != "auto" ]] && { printf '%s' "${CMAKE_CUDA_ARCHITECTURES}"; return; }; if ! command -v nvidia-smi >/dev/null 2>&1; then warn "nvidia-smi not found; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"; printf '%s' "${DEFAULT_CUDA_ARCHITECTURES}"; return; fi; a="$({ nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true; } | awk -F'.' 'NF >= 1 { gsub(/[^0-9]/, "", $1); gsub(/[^0-9]/, "", $2); if ($1 != "") { d = ($2 == "" ? "0" : substr($2, 1, 1)); print $1 d; } }' | sort -u -n | paste -sd ';' -)"; [[ -n "${a}" ]] && { printf '%s' "${a}"; return; }; warn "failed to detect compute capability; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"; printf '%s' "${DEFAULT_CUDA_ARCHITECTURES}"; }
resolve_ls_config(){ local f; if [[ -n "${LS_CONFIG_FILE}" ]]; then [[ -r "${LS_CONFIG_FILE}" ]] || die "no llama-swap config found at ${LS_CONFIG_FILE}; set LLAMACPP_LS_CONFIG_FILE to a readable file"; printf '%s' "${LS_CONFIG_FILE}"; return; fi; for f in "${DEFAULT_LS_CONFIG_FILE}" "${LS_CONFIG_EXAMPLE_FILE}" "${LEGACY_LS_CONFIG_FILE}" "${LEGACY_LS_CONFIG_EXAMPLE_FILE}"; do [[ -r "${f}" ]] || continue; case "${f}" in "${LS_CONFIG_EXAMPLE_FILE}") info "using ${LS_CONFIG_EXAMPLE_FILE}; create ${DEFAULT_LS_CONFIG_FILE} for local overrides" >&2 ;; "${LEGACY_LS_CONFIG_FILE}"|"${LEGACY_LS_CONFIG_EXAMPLE_FILE}") warn "using legacy config path ${f}; rename to config.yml/config.yml.example to match the current layout" >&2 ;; esac; printf '%s' "${f}"; return; done; die "no llama-swap config found; set LLAMACPP_LS_CONFIG_FILE or create ${DEFAULT_LS_CONFIG_FILE}"; }
cfg_path(){ local f b e k; f="$(resolve_ls_config)"; if [[ -n "${API_KEY_RESOLVED}" ]]; then e="${SCRIPT_DIR}/.runtime/config.effective.yaml"; mkdir -p -- "${SCRIPT_DIR}/.runtime"; k="${API_KEY_RESOLVED//\\/\\\\}"; k="${k//\"/\\\"}"; { printf 'apiKeys:\n  - "%s"\n' "${k}"; cat "${f}"; } > "${e}"; chmod 600 "${e}" 2>/dev/null || true; f="${e}"; fi; b="$(basename "${f}")"; printf '%s|%s' "${f}" "/app/config.d/${b}"; }
exists(){ docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; }
running(){ [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]; }
container_cfg_path(){ local f; local -a matches=(); if [[ -d /app/config.d ]]; then shopt -s nullglob; matches=(/app/config.d/*.yaml /app/config.d/*.yml); shopt -u nullglob; [[ ${#matches[@]} -gt 0 ]] && { printf '%s' "${matches[0]}"; return; }; fi; for f in /app/config.yml /app/config.yml.example /app/config.yaml /app/config.yaml.example; do [[ -r "${f}" ]] && { printf '%s' "${f}"; return; }; done; die "no container config found under /app/config.d or /app/config.yml(.example)"; }
serve(){ local f; f="$(container_cfg_path)"; [[ -x "${LLAMA_SWAP_BIN}" ]] || die "llama-swap binary not found at ${LLAMA_SWAP_BIN}"; info "starting llama-swap (config=${f}, listen=:${CONTAINER_PORT})"; exec "${LLAMA_SWAP_BIN}" -config "${f}" -listen "0.0.0.0:${CONTAINER_PORT}"; }
build(){ local a; [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "build command is host-only"; need_docker; a="$(cuda_arch)"; info "building ${IMAGE_NAME} (repo=${LLAMA_CPP_REPO} ref=${LLAMA_CPP_REF} CMAKE_CUDA_ARCHITECTURES=${a})"; DOCKER_BUILDKIT=1 docker build --pull --build-arg DEBIAN_FRONTEND=noninteractive --build-arg HOST_TZ="${HOST_TZ}" --build-arg HOST_LANG="${HOST_LANG}" --build-arg HOST_LC_ALL="${HOST_LC_ALL}" --build-arg LLAMA_CPP_REPO="${LLAMA_CPP_REPO}" --build-arg LLAMA_CPP_REF="${LLAMA_CPP_REF}" --build-arg CMAKE_CUDA_ARCHITECTURES="${a}" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"; ok "build complete: ${IMAGE_NAME}"; }
start(){ local h c m; local -a a; [[ "${RUNTIME_MODE}" == "${RUNTIME_CONTAINER}" ]] && { serve; return; }; need_docker; docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' || die "nvidia container runtime is not available in docker"; load_auth; mkdir -p -- "${MODELS_DIR}" "${MMPROJ_DIR}"; IFS='|' read -r h c <<< "$(cfg_path)"; m="$(mmproj_arg)"; running && { warn "container ${CONTAINER_NAME} is already running"; return 0; }; exists && docker rm -f "${CONTAINER_NAME}" >/dev/null; a=(--detach --init --name "${CONTAINER_NAME}" --restart unless-stopped --security-opt no-new-privileges --gpus all --runtime nvidia --publish "${HOST_PORT}:${CONTAINER_PORT}" --volume "${MODELS_DIR}:${MODELS_DIR_CONTAINER}" --volume "${MMPROJ_DIR}:${MMPROJ_DIR_CONTAINER}" --volume "${h}:${c}:ro" --env "LLAMACPP_RUNTIME_MODE=${RUNTIME_CONTAINER}" --env "CONTAINER_PORT=${CONTAINER_PORT}" --env "LLAMACPP_MMPROJ_ARG=${m}" --env "TZ=${HOST_TZ}" --env "LANG=${HOST_LANG}" --env "LC_ALL=${HOST_LC_ALL}"); [[ -n "${HF_TOKEN_RESOLVED}" ]] && a+=(--env "HF_TOKEN=${HF_TOKEN_RESOLVED}"); [[ -d "${CHAT_TEMPLATE_DIR}" ]] && a+=(--volume "${CHAT_TEMPLATE_DIR}:${CHAT_TEMPLATE_DIR_CONTAINER}:ro"); [[ -r /etc/localtime ]] && a+=(--volume /etc/localtime:/etc/localtime:ro); [[ -r /etc/timezone ]] && a+=(--volume /etc/timezone:/etc/timezone:ro); docker run "${a[@]}" "${IMAGE_NAME}" serve >/dev/null; ok "started ${CONTAINER_NAME} on http://localhost:${HOST_PORT}"; }
warmup(){ local base key json id status; local -a headers=() models=(); need curl jq; [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] && load_auth; base="$(listen_url)"; if [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]]; then need_docker; running || die "container ${CONTAINER_NAME} is not running; start it first"; fi; key="$(resolved_api_key)"; [[ -n "${key}" ]] && headers=(-H "Authorization: Bearer ${key}"); curl -fsS -o /dev/null "${base}/health" || die "llama-swap is not reachable at ${base}"; if [[ $# -gt 0 ]]; then models=("$@"); else json="$(curl -fsS "${headers[@]}" "${base}/v1/models")" || die "failed to list models from ${base}/v1/models"; mapfile -t models < <(printf '%s' "${json}" | jq -r '.data[].id' | awk 'NF && !seen[$0]++'); fi; [[ ${#models[@]} -gt 0 ]] || { warn "no models selected for warmup"; return 0; }; for id in "${models[@]}"; do info "warming model ${id}"; status="$(curl -sS -L -o /dev/null -w '%{http_code}' "${headers[@]}" "${base}/upstream/${id}/health")" || die "failed to contact warmup path for model ${id}"; if [[ "${status}" =~ ^[23] ]]; then ok "warmed ${id}"; continue; fi; if model_ready "${base}" "${id}" "${headers[@]}"; then warn "warmup health request returned HTTP ${status} for ${id}, but the model is ready; treating it as warmed"; continue; fi; die "failed to warm model ${id} (HTTP ${status})"; done; }
stop(){ [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "stop command is host-only"; need_docker; if exists; then docker rm -f "${CONTAINER_NAME}" >/dev/null; ok "removed container ${CONTAINER_NAME}"; else warn "container ${CONTAINER_NAME} does not exist"; fi; }
logs(){ [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "logs command is host-only"; need_docker; docker logs -f "${CONTAINER_NAME}"; }
status(){ [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "status command is host-only"; need_docker; docker ps -a --filter "name=^/${CONTAINER_NAME}$"; }
clean(){ [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "clean command is host-only"; need_docker; exists && docker rm -f "${CONTAINER_NAME}" >/dev/null; if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then docker image rm -f "${IMAGE_NAME}" >/dev/null; ok "removed image ${IMAGE_NAME}"; else warn "image ${IMAGE_NAME} does not exist"; fi; }
usage(){ cat <<'EOF'
Usage: ./run.sh <command>
Commands: build,start,warmup,stop,restart,logs,status,clean,serve,help
  build    Build local llama.cpp CUDA image from Dockerfile
  start    Start llama-swap container (host) or run server (container)
  warmup   Load one or more configured models via llama-swap's upstream route
  stop     Stop and remove llama-swap container
  restart  Restart container (stop then start)
  logs     Follow container logs
  status   Show container status
  clean    Remove container and local image
  serve    Run llama-swap directly (container entrypoint mode)
  help     Show this help text
Env:
  LLAMACPP_LS_CONFIG_FILE path to llama-swap YAML
  LLAMACPP_AUTH_FILE path to auth JSON with {"hf_token":"..."}
  HF_TOKEN or LLAMACPP_HF_TOKEN overrides auth JSON token
  LLAMACPP_API_KEY or API_KEY overrides auth JSON api_key
  LLAMACPP_MMPROJ_FILE local/path|mmproj/file|URL for --mmproj
  LLAMACPP_HF_MMPROJ shorthand owner/repo/file.gguf for auto-download
Warmup:
  ./run.sh warmup           Warm all model IDs currently listed by /v1/models
  ./run.sh warmup model-a   Warm only the named model IDs
EOF
}
main(){ local c="${1:-help}"; case "${c}" in help|-h|--help) usage; return 0 ;; build|start|warmup|stop|restart|logs|status|clean|serve) cfg ;; *) err "unknown command: ${c}"; usage; return 1 ;; esac; case "${c}" in build) build ;; start) start ;; warmup) shift; warmup "$@" ;; stop) stop ;; restart) stop || true; start ;; logs) logs ;; status) status ;; clean) clean ;; serve) serve ;; esac; }
main "$@"
