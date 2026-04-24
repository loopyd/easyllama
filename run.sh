#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/config.json"
CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.json.example"
CONFIG_FILE="${LLAMACPP_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"

# Pre-config defaults for early logging before JSON config is loaded.
LOG_LEVEL="${LLAMACPP_LOG_LEVEL:-info}"
NO_COLOR="${LLAMACPP_NO_COLOR:-}"

init_colors() {
  if [[ -t 1 && -z "${NO_COLOR}" ]]; then
    C_RESET='\033[0m'
    C_INFO='\033[1;34m'
    C_WARN='\033[1;33m'
    C_ERROR='\033[1;31m'
    C_OK='\033[1;32m'
  else
    C_RESET=''
    C_INFO=''
    C_WARN=''
    C_ERROR=''
    C_OK=''
  fi
}

init_colors

log_enabled() {
  local level="${1}"
  case "${LOG_LEVEL}" in
    debug) return 0 ;;
    info) [[ "${level}" != "DEBUG" ]] ;;
    warn) [[ "${level}" == "WARN" || "${level}" == "ERROR" ]] ;;
    error) [[ "${level}" == "ERROR" ]] ;;
    *) return 0 ;;
  esac
}

log() {
  local level="${1}"
  shift
  log_enabled "${level}" || return 0

  local color="${C_INFO}"
  case "${level}" in
    WARN) color="${C_WARN}" ;;
    ERROR) color="${C_ERROR}" ;;
    OK) color="${C_OK}" ;;
  esac
  printf '%b[%s] %s%b\n' "${color}" "${level}" "$*" "${C_RESET}"
}

info() { log INFO "$*"; }
warn() { log WARN "$*"; }
ok() { log OK "$*"; }
err() { log ERROR "$*"; }
die() { err "$*"; exit 1; }

check_dependencies() {
  local -a missing=()
  local dep
  for dep in "$@"; do
    command -v "${dep}" >/dev/null 2>&1 || missing+=("${dep}")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    die "missing dependency: ${missing[*]}"
  fi
}

detect_host_tz() {
  [[ -n "${TZ:-}" ]] && { echo "${TZ}"; return; }
  [[ -r /etc/timezone ]] && { tr -d '\n' < /etc/timezone; return; }
  if command -v timedatectl >/dev/null 2>&1; then
    local tz
    tz="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
    [[ -n "${tz}" ]] && { echo "${tz}"; return; }
  fi
  echo "UTC"
}

CONFIG_LOADED=0

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

to_on_off() {
  if is_truthy "${1}"; then
    printf '%s' "on"
  else
    printf '%s' "off"
  fi
}

config_has_key() {
  local key="${1}"
  jq -e --arg key "${key}" 'has($key) and .[$key] != null' "${CONFIG_FILE}" >/dev/null 2>&1
}

config_read() {
  local key="${1}"
  jq -r --arg key "${key}" '
    if has($key) and .[$key] != null then
      .[$key] as $v |
      if ($v | type) == "boolean" then
        if $v then "1" else "" end
      elif ($v | type) == "string" then
        $v
      else
        ($v | tostring)
      end
    else
      empty
    end
  ' "${CONFIG_FILE}" 2>/dev/null
}

resolve_setting() {
  local env_name="${1}"
  local config_key="${2}"
  local default_value="${3}"

  if [[ -n "${!env_name+x}" ]]; then
    printf '%s' "${!env_name}"
    return
  fi

  local config_value=""
  if [[ "${CONFIG_LOADED}" == "1" ]] && config_has_key "${config_key}"; then
    config_value="$(config_read "${config_key}")"
    printf '%s' "${config_value}"
    return
  fi

  printf '%s' "${default_value}"
}

parse_extra_server_args() {
  local raw_value="${1}"
  EXTRA_SERVER_ARGS=()

  [[ -z "${raw_value}" ]] && return

  if [[ "${raw_value}" =~ ^[[:space:]]*\[ ]]; then
    check_dependencies jq
    mapfile -t EXTRA_SERVER_ARGS < <(
      printf '%s' "${raw_value}" | jq -r '
        if type == "array" then
          .[] | tostring
        else
          error("extra_server_args must be a JSON array when using JSON syntax")
        end
      ' 2>/dev/null
    ) || die "LLAMACPP_EXTRA_SERVER_ARGS JSON must be a valid array"
    return
  fi

  local -a split_args=()
  read -r -a split_args <<< "${raw_value}"
  EXTRA_SERVER_ARGS=("${split_args[@]}")
}

load_extra_server_args() {
  EXTRA_SERVER_ARGS=()

  if [[ -n "${LLAMACPP_EXTRA_SERVER_ARGS+x}" ]]; then
    parse_extra_server_args "${LLAMACPP_EXTRA_SERVER_ARGS}"
    return
  fi

  if [[ "${CONFIG_LOADED}" != "1" ]] || ! config_has_key extra_server_args; then
    return
  fi

  check_dependencies jq
  local cfg_type
  cfg_type="$(jq -r '.extra_server_args | type' "${CONFIG_FILE}" 2>/dev/null || true)"
  case "${cfg_type}" in
    array)
      mapfile -t EXTRA_SERVER_ARGS < <(jq -r '.extra_server_args[] | tostring' "${CONFIG_FILE}")
      ;;
    string)
      parse_extra_server_args "$(jq -r '.extra_server_args' "${CONFIG_FILE}")"
      ;;
    *)
      die "extra_server_args must be a string or array in ${CONFIG_FILE}"
      ;;
  esac
}

resolve_path_from_script_dir() {
  local path_value="${1}"
  if [[ "${path_value}" = /* ]]; then
    printf '%s' "${path_value}"
  else
    printf '%s' "${SCRIPT_DIR}/${path_value}"
  fi
}

load_config() {
  if [[ ! -r "${CONFIG_FILE}" ]]; then
    if [[ -n "${LLAMACPP_CONFIG_FILE:-}" ]]; then
      die "LLAMACPP_CONFIG_FILE is set but not readable: ${CONFIG_FILE}"
    fi

    if [[ -r "${CONFIG_EXAMPLE_FILE}" ]]; then
      CONFIG_FILE="${CONFIG_EXAMPLE_FILE}"
      info "using ${CONFIG_EXAMPLE_FILE}; create ${DEFAULT_CONFIG_FILE} for local overrides"
    else
      return
    fi
  fi

  check_dependencies jq
  jq empty "${CONFIG_FILE}" >/dev/null 2>&1 || die "invalid JSON config: ${CONFIG_FILE}"
  CONFIG_LOADED=1
  info "loaded config from ${CONFIG_FILE}"
}

load_settings() {
  local host_tz_default
  host_tz_default="$(detect_host_tz)"

  IMAGE_NAME="$(resolve_setting LLAMACPP_IMAGE_NAME image_name 'llamacpp-local:cuda13')"
  CONTAINER_NAME="$(resolve_setting LLAMACPP_CONTAINER_NAME container_name 'llamacpp-server')"
  MODELS_DIR="$(resolve_setting LLAMACPP_MODELS_DIR models_dir 'models')"
  MODELS_DIR="$(resolve_path_from_script_dir "${MODELS_DIR}")"

  HOST_PORT="$(resolve_setting LLAMACPP_HOST_PORT host_port '8080')"
  CONTAINER_PORT="$(resolve_setting LLAMACPP_CONTAINER_PORT container_port '8080')"

  LOG_LEVEL="$(resolve_setting LLAMACPP_LOG_LEVEL log_level 'info')"
  NO_COLOR="$(resolve_setting LLAMACPP_NO_COLOR no_color '')"

  HOST_TZ="$(resolve_setting LLAMACPP_HOST_TZ host_tz "${host_tz_default}")"
  HOST_LANG="$(resolve_setting LLAMACPP_HOST_LANG host_lang "${LANG:-C.UTF-8}")"
  HOST_LC_ALL="$(resolve_setting LLAMACPP_HOST_LC_ALL host_lc_all "${LC_ALL:-${HOST_LANG}}")"

  DEFAULT_CUDA_ARCHITECTURES="$(resolve_setting LLAMACPP_DEFAULT_CUDA_ARCHITECTURES default_cuda_architectures '120')"
  CMAKE_CUDA_ARCHITECTURES="$(resolve_setting LLAMACPP_CMAKE_CUDA_ARCHITECTURES cmake_cuda_architectures 'auto')"

  HF_MODEL="$(resolve_setting LLAMACPP_HF_MODEL hf_model 'mradermacher/Qwen3.6-35B-A3B-abliterated-MAX-i1-GGUF:i1-Q6_K')"
  HF_TOKEN="$(resolve_setting LLAMACPP_HF_TOKEN hf_token '')"
  SERVER_HOST="$(resolve_setting LLAMACPP_SERVER_HOST server_host '0.0.0.0')"

  CTX_SIZE="$(resolve_setting LLAMACPP_CTX_SIZE ctx_size '125000')"
  TEMP="$(resolve_setting LLAMACPP_TEMP temp '0.6')"
  TOP_P="$(resolve_setting LLAMACPP_TOP_P top_p '0.95')"
  TOP_K="$(resolve_setting LLAMACPP_TOP_K top_k '20')"
  MIN_P="$(resolve_setting LLAMACPP_MIN_P min_p '0.01')"
  REPEAT_PENALTY="$(resolve_setting LLAMACPP_REPEAT_PENALTY repeat_penalty '1.05')"
  PRESENCE_PENALTY="$(resolve_setting LLAMACPP_PRESENCE_PENALTY presence_penalty '0.00')"
  BATCH_SIZE="$(resolve_setting LLAMACPP_BATCH_SIZE batch_size '2048')"
  UBATCH_SIZE="$(resolve_setting LLAMACPP_UBATCH_SIZE ubatch_size '512')"
  PARALLEL="$(resolve_setting LLAMACPP_PARALLEL parallel '1')"
  N_CPU_MOE="$(resolve_setting LLAMACPP_N_CPU_MOE n_cpu_moe '2')"
  FIT="$(resolve_setting LLAMACPP_FIT fit 'on')"
  FLASH_ATTN="$(resolve_setting LLAMACPP_FLASH_ATTN flash_attn 'on')"
  CACHE_TYPE_K="$(resolve_setting LLAMACPP_CACHE_TYPE_K cache_type_k 'q8_0')"
  CACHE_TYPE_V="$(resolve_setting LLAMACPP_CACHE_TYPE_V cache_type_v 'q8_0')"
  NO_MMAP="$(resolve_setting LLAMACPP_NO_MMAP no_mmap '1')"
  POLL="$(resolve_setting LLAMACPP_POLL poll '1')"
  JINJA="$(resolve_setting LLAMACPP_JINJA jinja '1')"
  CHAT_TEMPLATE_KWARGS="$(resolve_setting LLAMACPP_CHAT_TEMPLATE_KWARGS chat_template_kwargs '{"preserve_thinking":false}')"

  ENABLE_REASONING="$(resolve_setting LLAMACPP_ENABLE_REASONING enable_reasoning 'off')"
  REASONING_FORMAT="$(resolve_setting LLAMACPP_REASONING_FORMAT reasoning_format "${LLAMACPP_REASONING_PARSER:-auto}")"
  REASONING_BUDGET="$(resolve_setting LLAMACPP_REASONING_BUDGET reasoning_budget '256')"
  REASONING_BUDGET_MESSAGE="$(resolve_setting LLAMACPP_REASONING_BUDGET_MESSAGE reasoning_budget_message 'Answer directly and avoid long hidden reasoning loops.')"

  load_extra_server_args

  case "${CACHE_TYPE_V}" in
    f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1)
      ;;
    *)
      die "unsupported CACHE_TYPE_V=${CACHE_TYPE_V}; allowed: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1"
      ;;
  esac

  case "${ENABLE_REASONING}" in
    on|off|auto)
      ;;
    *)
      die "unsupported ENABLE_REASONING=${ENABLE_REASONING}; allowed: on,off,auto"
      ;;
  esac

  if [[ -n "${CHAT_TEMPLATE_KWARGS}" ]]; then
    check_dependencies jq
    CHAT_TEMPLATE_KWARGS="$(printf '%s' "${CHAT_TEMPLATE_KWARGS}" | jq -c 'if type == "object" then . else error("chat_template_kwargs must be a JSON object") end' 2>/dev/null)" \
      || die "chat_template_kwargs must be a valid JSON object"
  fi

  init_colors
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
  docker info >/dev/null 2>&1 || die "docker daemon is not reachable (start docker and retry)"
}

require_nvidia_runtime() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' || die "nvidia container runtime is not available in docker"
}

detect_cuda_architectures() {
  if [[ "${CMAKE_CUDA_ARCHITECTURES}" != "auto" ]]; then
    echo "${CMAKE_CUDA_ARCHITECTURES}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"
    echo "${DEFAULT_CUDA_ARCHITECTURES}"
    return
  fi

  local arch_list
  arch_list="$({
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true
  } | awk -F'.' '
    NF >= 1 {
      gsub(/[^0-9]/, "", $1)
      gsub(/[^0-9]/, "", $2)
      if ($1 != "") {
        d = ($2 == "" ? "0" : substr($2, 1, 1))
        print $1 d
      }
    }
  ' | sort -u -n | paste -sd ';' -)"

  if [[ -z "${arch_list}" ]]; then
    warn "failed to detect compute capability; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"
    echo "${DEFAULT_CUDA_ARCHITECTURES}"
    return
  fi

  echo "${arch_list}"
}

container_exists() { docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; }

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

build_image() {
  require_docker

  local detected_arch
  detected_arch="$(detect_cuda_architectures)"
  info "building ${IMAGE_NAME} (CMAKE_CUDA_ARCHITECTURES=${detected_arch})"

  DOCKER_BUILDKIT=1 docker build \
    --pull \
    --build-arg DEBIAN_FRONTEND=noninteractive \
    --build-arg HOST_TZ="${HOST_TZ}" \
    --build-arg HOST_LANG="${HOST_LANG}" \
    --build-arg HOST_LC_ALL="${HOST_LC_ALL}" \
    --build-arg CMAKE_CUDA_ARCHITECTURES="${detected_arch}" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

  ok "build complete: ${IMAGE_NAME}"
}

stop_container() {
  require_docker
  if container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    ok "removed container ${CONTAINER_NAME}"
  else
    warn "container ${CONTAINER_NAME} does not exist"
  fi
}

start_container() {
  require_docker
  require_nvidia_runtime
  mkdir -p -- "${MODELS_DIR}"

  if container_running; then
    warn "container ${CONTAINER_NAME} is already running"
    return 0
  fi
  container_exists && docker rm -f "${CONTAINER_NAME}" >/dev/null

  local -a server_args=(
    -hf "${HF_MODEL}"
    --host "${SERVER_HOST}"
    --port "${CONTAINER_PORT}"
    --temp "${TEMP}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --min-p "${MIN_P}"
    --repeat-penalty "${REPEAT_PENALTY}"
    --presence-penalty "${PRESENCE_PENALTY}"
    --ctx-size "${CTX_SIZE}"
    --fit "$(to_on_off "${FIT}")"
    --flash-attn "$(to_on_off "${FLASH_ATTN}")"
    --cache-type-k "${CACHE_TYPE_K}"
    --cache-type-v "${CACHE_TYPE_V}"
    --batch-size "${BATCH_SIZE}"
    --ubatch-size "${UBATCH_SIZE}"
    --parallel "${PARALLEL}"
    --n-cpu-moe "${N_CPU_MOE}"
    --poll "${POLL}"
  )

  is_truthy "${NO_MMAP}" && server_args+=(--no-mmap)
  is_truthy "${JINJA}" && server_args+=(--jinja)
  [[ -n "${CHAT_TEMPLATE_KWARGS}" ]] && server_args+=(--chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")

  [[ -n "${ENABLE_REASONING}" ]] && server_args+=(--reasoning "${ENABLE_REASONING}")
  if [[ "${ENABLE_REASONING}" != "off" ]]; then
    [[ -n "${REASONING_FORMAT}" ]] && server_args+=(--reasoning-format "${REASONING_FORMAT}")
    [[ -n "${REASONING_BUDGET}" ]] && server_args+=(--reasoning-budget "${REASONING_BUDGET}")
    [[ -n "${REASONING_BUDGET_MESSAGE}" ]] && server_args+=(--reasoning-budget-message "${REASONING_BUDGET_MESSAGE}")
  fi

  [[ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]] && server_args+=("${EXTRA_SERVER_ARGS[@]}")

  local -a run_args=(
    --detach
    --init
    --name "${CONTAINER_NAME}"
    --restart unless-stopped
    --security-opt no-new-privileges
    --gpus all
    --runtime nvidia
    --publish "${HOST_PORT}:${CONTAINER_PORT}"
    --volume "${MODELS_DIR}:/root/.cache/huggingface/hub"
    --env "TZ=${HOST_TZ}"
    --env "LANG=${HOST_LANG}"
    --env "LC_ALL=${HOST_LC_ALL}"
  )

  [[ -r /etc/localtime ]] && run_args+=(--volume /etc/localtime:/etc/localtime:ro)
  [[ -r /etc/timezone ]] && run_args+=(--volume /etc/timezone:/etc/timezone:ro)
  [[ -n "${HF_TOKEN}" ]] && run_args+=(--env "HF_TOKEN=${HF_TOKEN}")

  docker run "${run_args[@]}" "${IMAGE_NAME}" "${server_args[@]}" >/dev/null
  ok "started ${CONTAINER_NAME} on http://localhost:${HOST_PORT}"
}

show_logs() { require_docker; docker logs -f "${CONTAINER_NAME}"; }
show_status() { require_docker; docker ps -a --filter "name=^/${CONTAINER_NAME}$"; }

clean_all() {
  require_docker
  container_exists && docker rm -f "${CONTAINER_NAME}" >/dev/null
  if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    docker image rm -f "${IMAGE_NAME}" >/dev/null
    ok "removed image ${IMAGE_NAME}"
  else
    warn "image ${IMAGE_NAME} does not exist"
  fi
}

usage() {
  cat <<'EOF'
Usage: ./run.sh <command>

Commands:
  build      Build local llama.cpp CUDA 13 image from Dockerfile
  start      Start llama-server container
  stop       Stop and remove the running container
  restart    Restart container (stop then start)
  logs       Follow container logs
  status     Show container status
  clean      Remove container and local image
  help       Show this help text

Configuration precedence:
  1) Environment variables (LLAMACPP_* only)
  2) LLAMACPP_CONFIG_FILE path (if set)
  3) config.json (local, gitignored)
  4) config.json.example (tracked template)
  5) Built-in defaults

Important environment variables:
  LLAMACPP_IMAGE_NAME, LLAMACPP_CONTAINER_NAME, LLAMACPP_MODELS_DIR,
  LLAMACPP_HF_MODEL, LLAMACPP_HF_TOKEN, LLAMACPP_HOST_PORT,
  LLAMACPP_HOST_TZ, LLAMACPP_HOST_LANG, LLAMACPP_HOST_LC_ALL,
  LLAMACPP_SERVER_HOST, LLAMACPP_CTX_SIZE, LLAMACPP_FIT,
  LLAMACPP_FLASH_ATTN, LLAMACPP_CACHE_TYPE_K, LLAMACPP_CACHE_TYPE_V,
  LLAMACPP_NO_MMAP, LLAMACPP_POLL, LLAMACPP_JINJA,
  LLAMACPP_CHAT_TEMPLATE_KWARGS, LLAMACPP_EXTRA_SERVER_ARGS,
  LLAMACPP_CMAKE_CUDA_ARCHITECTURES(auto|list),
  LLAMACPP_DEFAULT_CUDA_ARCHITECTURES, LLAMACPP_CONFIG_FILE

LLAMACPP_EXTRA_SERVER_ARGS accepts either:
  - shell words string: --foo bar --baz
  - JSON array string: ["--foo","bar","--baz"]
EOF
}

main() {
  local command="${1:-help}"
  case "${command}" in
    build)
      build_image
      ;;
    start)
      start_container
      ;;
    stop)
      stop_container
      ;;
    restart)
      stop_container || true
      start_container
      ;;
    logs)
      show_logs
      ;;
    status)
      show_status
      ;;
    clean)
      clean_all
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      err "unknown command: ${command}"
      usage
      exit 1
      ;;
  esac
}

load_config
load_settings

main "${@}"