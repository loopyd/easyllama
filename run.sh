#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/config.json"
CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.json.example"
CONFIG_FILE="${LLAMACPP_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
LOG_LEVEL="${LLAMACPP_LOG_LEVEL:-info}"
NO_COLOR="${LLAMACPP_NO_COLOR:-}"

setup_colors() {
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

setup_colors

should_log() {
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
  should_log "${level}" || return 0

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

require_commands() {
  local -a missing=()
  local dep
  for dep in "$@"; do
    command -v "${dep}" >/dev/null 2>&1 || missing+=("${dep}")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    die "missing dependency: ${missing[*]}"
  fi
}

detect_host_timezone() {
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

config_has_path() {
  local path="${1}"
  jq -e --arg path "${path}" '
    getpath($path | split(".")) != null
  ' "${CONFIG_FILE}" >/dev/null 2>&1
}

config_get_path() {
  local path="${1}"
  jq -r --arg path "${path}" '
    getpath($path | split(".")) as $v |
    if $v != null then
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

resolve_config_value() {
  local env_name="${1}"
  local config_path="${2}"
  local default_value="${3}"

  if [[ -n "${!env_name+x}" ]]; then
    printf '%s' "${!env_name}"
    return
  fi

  local config_value=""
  if [[ "${CONFIG_LOADED}" == "1" ]] && config_has_path "${config_path}"; then
    config_value="$(config_get_path "${config_path}")"
    printf '%s' "${config_value}"
    return
  fi

  printf '%s' "${default_value}"
}

parse_extra_server_args_value() {
  local raw_value="${1}"
  EXTRA_SERVER_ARGS=()

  [[ -z "${raw_value}" ]] && return

  if [[ "${raw_value}" =~ ^[[:space:]]*\[ ]]; then
    require_commands jq
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
    parse_extra_server_args_value "${LLAMACPP_EXTRA_SERVER_ARGS}"
    return
  fi

  if [[ "${CONFIG_LOADED}" != "1" ]] || ! config_has_path inference.extra_server_args; then
    return
  fi

  require_commands jq
  local cfg_type
  cfg_type="$(jq -r '.inference.extra_server_args | type' "${CONFIG_FILE}" 2>/dev/null || true)"
  case "${cfg_type}" in
    array)
      mapfile -t EXTRA_SERVER_ARGS < <(jq -r '.inference.extra_server_args[] | tostring' "${CONFIG_FILE}")
      ;;
    string)
      parse_extra_server_args_value "$(jq -r '.inference.extra_server_args' "${CONFIG_FILE}")"
      ;;
    *)
      die "inference.extra_server_args must be a string or array in ${CONFIG_FILE}"
      ;;
  esac
}

resolve_project_path() {
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

  require_commands jq
  jq empty "${CONFIG_FILE}" >/dev/null 2>&1 || die "invalid JSON config: ${CONFIG_FILE}"
  CONFIG_LOADED=1
  info "loaded config from ${CONFIG_FILE}"
}

assign_config_value() {
  local var_name="${1}"
  local env_name="${2}"
  local config_path="${3}"
  local default_value="${4}"
  local resolved
  resolved="$(resolve_config_value "${env_name}" "${config_path}" "${default_value}")"
  printf -v "${var_name}" '%s' "${resolved}"
}

load_settings() {
  local host_tz_default
  local host_lang_default="${LANG:-C.UTF-8}"
  local reasoning_format_default="${LLAMACPP_REASONING_PARSER:-auto}"
  host_tz_default="$(detect_host_timezone)"

  assign_config_value IMAGE_NAME LLAMACPP_IMAGE_NAME container.image_name 'llamacpp-local:cuda13'
  assign_config_value CONTAINER_NAME LLAMACPP_CONTAINER_NAME container.container_name 'llamacpp-server'
  assign_config_value MODELS_DIR LLAMACPP_MODELS_DIR container.models_dir 'models'
  MODELS_DIR="$(resolve_project_path "${MODELS_DIR}")"

  assign_config_value HOST_PORT LLAMACPP_HOST_PORT network.host_port '8080'
  assign_config_value CONTAINER_PORT LLAMACPP_CONTAINER_PORT network.container_port '8080'

  assign_config_value LOG_LEVEL LLAMACPP_LOG_LEVEL logging.log_level 'info'
  assign_config_value NO_COLOR LLAMACPP_NO_COLOR logging.no_color ''

  assign_config_value HOST_TZ LLAMACPP_HOST_TZ locale.host_tz "${host_tz_default}"
  assign_config_value HOST_LANG LLAMACPP_HOST_LANG locale.host_lang "${host_lang_default}"
  assign_config_value HOST_LC_ALL LLAMACPP_HOST_LC_ALL locale.host_lc_all "${LC_ALL:-${HOST_LANG}}"

  assign_config_value DEFAULT_CUDA_ARCHITECTURES LLAMACPP_DEFAULT_CUDA_ARCHITECTURES build.default_cuda_architectures '120'
  assign_config_value CMAKE_CUDA_ARCHITECTURES LLAMACPP_CMAKE_CUDA_ARCHITECTURES build.cmake_cuda_architectures 'auto'
  assign_config_value LLAMA_CPP_REPO LLAMACPP_LLAMA_CPP_REPO build.llama_cpp_repo 'https://github.com/ggml-org/llama.cpp.git'
  assign_config_value LLAMA_CPP_REF LLAMACPP_LLAMA_CPP_REF build.llama_cpp_ref 'master'

  assign_config_value HF_MODEL LLAMACPP_HF_MODEL model.hf_model 'mradermacher/Qwen3.6-35B-A3B-abliterated-MAX-i1-GGUF:i1-Q6_K'
  assign_config_value HF_TOKEN LLAMACPP_HF_TOKEN model.hf_token ''
  assign_config_value SERVER_HOST LLAMACPP_SERVER_HOST network.server_host '0.0.0.0'

  assign_config_value CTX_SIZE LLAMACPP_CTX_SIZE inference.ctx_size '125000'
  assign_config_value TEMP LLAMACPP_TEMP inference.temp '0.6'
  assign_config_value TOP_P LLAMACPP_TOP_P inference.top_p '0.95'
  assign_config_value TOP_K LLAMACPP_TOP_K inference.top_k '20'
  assign_config_value MIN_P LLAMACPP_MIN_P inference.min_p '0.01'
  assign_config_value REPEAT_PENALTY LLAMACPP_REPEAT_PENALTY inference.repeat_penalty '1.05'
  assign_config_value PRESENCE_PENALTY LLAMACPP_PRESENCE_PENALTY inference.presence_penalty '0.00'
  assign_config_value BATCH_SIZE LLAMACPP_BATCH_SIZE inference.batch_size '2048'
  assign_config_value UBATCH_SIZE LLAMACPP_UBATCH_SIZE inference.ubatch_size '512'
  assign_config_value PARALLEL LLAMACPP_PARALLEL inference.parallel '1'
  assign_config_value N_CPU_MOE LLAMACPP_N_CPU_MOE inference.n_cpu_moe '2'
  assign_config_value FIT LLAMACPP_FIT inference.fit 'on'
  assign_config_value FLASH_ATTN LLAMACPP_FLASH_ATTN inference.flash_attn 'on'
  assign_config_value CACHE_TYPE_K LLAMACPP_CACHE_TYPE_K inference.cache_type_k 'q8_0'
  assign_config_value CACHE_TYPE_V LLAMACPP_CACHE_TYPE_V inference.cache_type_v 'q8_0'
  assign_config_value KV_UNIFIED LLAMACPP_KV_UNIFIED inference.kv_unified '0'
  assign_config_value CACHE_IDLE_SLOTS LLAMACPP_CACHE_IDLE_SLOTS inference.cache_idle_slots '0'
  assign_config_value NO_MMAP LLAMACPP_NO_MMAP inference.no_mmap '1'
  assign_config_value POLL LLAMACPP_POLL inference.poll '1'
  assign_config_value JINJA LLAMACPP_JINJA inference.jinja '1'
  assign_config_value CHAT_TEMPLATE_KWARGS LLAMACPP_CHAT_TEMPLATE_KWARGS inference.chat_template_kwargs '{"preserve_thinking":false}'

  assign_config_value ENABLE_REASONING LLAMACPP_ENABLE_REASONING reasoning.enable 'off'
  assign_config_value REASONING_FORMAT LLAMACPP_REASONING_FORMAT reasoning.format "${reasoning_format_default}"
  assign_config_value REASONING_BUDGET LLAMACPP_REASONING_BUDGET reasoning.budget '256'
  assign_config_value REASONING_BUDGET_MESSAGE LLAMACPP_REASONING_BUDGET_MESSAGE reasoning.budget_message 'Answer directly and avoid long hidden reasoning loops.'

  load_extra_server_args

  case "${CACHE_TYPE_V}" in
    f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1|turbo2|turbo3|turbo4)
      ;;
    *)
      die "unsupported CACHE_TYPE_V=${CACHE_TYPE_V}; allowed: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1,turbo2,turbo3,turbo4"
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
    require_commands jq
    CHAT_TEMPLATE_KWARGS="$(printf '%s' "${CHAT_TEMPLATE_KWARGS}" | jq -c 'if type == "object" then . else error("chat_template_kwargs must be a JSON object") end' 2>/dev/null)" \
      || die "chat_template_kwargs must be a valid JSON object"
  fi

  setup_colors
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

get_supported_cache_type_v_list() {
  docker run --rm --gpus all --runtime nvidia "${IMAGE_NAME}" -h 2>/dev/null | awk '
    /--cache-type-v TYPE/ { in_block = 1; next }
    in_block && /allowed values:/ {
      sub(/^.*allowed values:[[:space:]]*/, "")
      values = values $0
      collecting = 1
      next
    }
    in_block && collecting {
      if ($0 ~ /^[[:space:]]+\(default:/) {
        gsub(/[[:space:]]+/, "", values)
        gsub(/,+$/, "", values)
        print values
        exit
      }
      values = values $0
    }
  '
}

validate_cache_type_v_support() {
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    warn "image ${IMAGE_NAME} not found; run ./run.sh build before start"
    return 0
  fi

  local supported_csv
  supported_csv="$(get_supported_cache_type_v_list || true)"
  [[ -z "${supported_csv}" ]] && return 0

  case ",${supported_csv}," in
    *",${CACHE_TYPE_V},"*)
      return 0
      ;;
  esac

  die "CACHE_TYPE_V=${CACHE_TYPE_V} not supported by image ${IMAGE_NAME}. supported: ${supported_csv}. Rebuild with LLAMACPP_LLAMA_CPP_REPO/LLAMACPP_LLAMA_CPP_REF set to a turbo2-capable fork (for example TheTom/llama-cpp-turboquant)."
}

container_exists() { docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; }

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

build_image() {
  require_docker

  local detected_arch
  detected_arch="$(detect_cuda_architectures)"
  info "building ${IMAGE_NAME} (repo=${LLAMA_CPP_REPO} ref=${LLAMA_CPP_REF} CMAKE_CUDA_ARCHITECTURES=${detected_arch})"

  DOCKER_BUILDKIT=1 docker build \
    --pull \
    --build-arg DEBIAN_FRONTEND=noninteractive \
    --build-arg HOST_TZ="${HOST_TZ}" \
    --build-arg HOST_LANG="${HOST_LANG}" \
    --build-arg HOST_LC_ALL="${HOST_LC_ALL}" \
    --build-arg LLAMA_CPP_REPO="${LLAMA_CPP_REPO}" \
    --build-arg LLAMA_CPP_REF="${LLAMA_CPP_REF}" \
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
  validate_cache_type_v_support
  mkdir -p -- "${MODELS_DIR}"

  if container_running; then
    warn "container ${CONTAINER_NAME} is already running"
    return 0
  fi
  if container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi

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
  is_truthy "${KV_UNIFIED}" && server_args+=(--kv-unified)
  is_truthy "${CACHE_IDLE_SLOTS}" && server_args+=(--cache-idle-slots)
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
  if container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi
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
  LLAMACPP_KV_UNIFIED, LLAMACPP_CACHE_IDLE_SLOTS,
  LLAMACPP_NO_MMAP, LLAMACPP_POLL, LLAMACPP_JINJA,
  LLAMACPP_CHAT_TEMPLATE_KWARGS, LLAMACPP_EXTRA_SERVER_ARGS,
  LLAMACPP_LLAMA_CPP_REPO, LLAMACPP_LLAMA_CPP_REF,
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