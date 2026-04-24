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

need_cmds() {
  local -a missing=()
  local dep
  for dep in "$@"; do
    command -v "${dep}" >/dev/null 2>&1 || missing+=("${dep}")
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    die "missing dependency: ${missing[*]}"
  fi
}

detect_tz() {
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

cfg_has() {
  local path="${1}"
  jq -e --arg path "${path}" '
    getpath($path | split(".")) != null
  ' "${CONFIG_FILE}" >/dev/null 2>&1
}

cfg_get() {
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

cfg_set() {
  local var_name="${1}"
  local env_name="${2}"
  local config_path="${3}"
  local default_value="${4-}"
  local value=""

  if [[ -n "${!env_name+x}" ]]; then
    value="${!env_name}"
  elif [[ "${CONFIG_LOADED}" == "1" ]] && cfg_has "${config_path}"; then
    value="$(cfg_get "${config_path}")"
  else
    value="${default_value}"
  fi

  printf -v "${var_name}" '%s' "${value}"
}

parse_extra_server_args_value() {
  local raw_value="${1}"
  EXTRA_SERVER_ARGS=()

  [[ -z "${raw_value}" ]] && return

  if [[ "${raw_value}" =~ ^[[:space:]]*\[ ]]; then
    need_cmds jq
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

load_extra_args() {
  EXTRA_SERVER_ARGS=()

  if [[ -n "${LLAMACPP_EXTRA_SERVER_ARGS+x}" ]]; then
    parse_extra_server_args_value "${LLAMACPP_EXTRA_SERVER_ARGS}"
    return
  fi

  if [[ "${CONFIG_LOADED}" != "1" ]] || ! cfg_has inference.extra_server_args; then
    return
  fi

  need_cmds jq
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

abs_path() {
  local path_value="${1}"
  if [[ "${path_value}" = /* ]]; then
    printf '%s' "${path_value}"
  else
    printf '%s' "${SCRIPT_DIR}/${path_value}"
  fi
}

arg_add() {
  local -n args_ref="${1}"
  local flag="${2}"
  local value="${3-}"
  [[ -n "${value}" ]] || return 0
  args_ref+=("${flag}" "${value}")
}

arg_bool() {
  local -n args_ref="${1}"
  local value="${2-}"
  local on_flag="${3}"
  local off_flag="${4-}"

  [[ -n "${value}" ]] || return 0

  if is_truthy "${value}"; then
    args_ref+=("${on_flag}")
  elif [[ -n "${off_flag}" ]]; then
    args_ref+=("${off_flag}")
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

  need_cmds jq
  jq empty "${CONFIG_FILE}" >/dev/null 2>&1 || die "invalid JSON config: ${CONFIG_FILE}"
  CONFIG_LOADED=1
  info "loaded config from ${CONFIG_FILE}"
}

load_cfg() {
  local host_tz_default
  local host_lang_default="${LANG:-C.UTF-8}"
  local reasoning_format_default="${LLAMACPP_REASONING_PARSER:-auto}"
  host_tz_default="$(detect_tz)"

  cfg_set IMAGE_NAME LLAMACPP_IMAGE_NAME container.image_name 'llamacpp-local:cuda13'
  cfg_set CONTAINER_NAME LLAMACPP_CONTAINER_NAME container.container_name 'llamacpp-server'
  cfg_set MODELS_DIR LLAMACPP_MODELS_DIR container.models_dir 'models'
  MODELS_DIR="$(abs_path "${MODELS_DIR}")"

  cfg_set HOST_PORT LLAMACPP_HOST_PORT network.host_port '8080'
  cfg_set CONTAINER_PORT LLAMACPP_CONTAINER_PORT network.container_port '8080'

  cfg_set LOG_LEVEL LLAMACPP_LOG_LEVEL logging.log_level 'info'
  cfg_set NO_COLOR LLAMACPP_NO_COLOR logging.no_color ''

  cfg_set HOST_TZ LLAMACPP_HOST_TZ locale.host_tz "${host_tz_default}"
  cfg_set HOST_LANG LLAMACPP_HOST_LANG locale.host_lang "${host_lang_default}"
  cfg_set HOST_LC_ALL LLAMACPP_HOST_LC_ALL locale.host_lc_all "${LC_ALL:-${HOST_LANG}}"

  cfg_set DEFAULT_CUDA_ARCHITECTURES LLAMACPP_DEFAULT_CUDA_ARCHITECTURES build.default_cuda_architectures '120'
  cfg_set CMAKE_CUDA_ARCHITECTURES LLAMACPP_CMAKE_CUDA_ARCHITECTURES build.cmake_cuda_architectures 'auto'
  cfg_set LLAMA_CPP_REPO LLAMACPP_LLAMA_CPP_REPO build.llama_cpp_repo 'https://github.com/ggml-org/llama.cpp.git'
  cfg_set LLAMA_CPP_REF LLAMACPP_LLAMA_CPP_REF build.llama_cpp_ref 'master'

  cfg_set HF_MODEL LLAMACPP_HF_MODEL model.hf_model 'mradermacher/Qwen3.6-35B-A3B-abliterated-MAX-i1-GGUF:i1-Q6_K'
  cfg_set HF_TOKEN LLAMACPP_HF_TOKEN model.hf_token ''
  cfg_set SERVER_HOST LLAMACPP_SERVER_HOST network.server_host '0.0.0.0'

  cfg_set CTX_SIZE LLAMACPP_CTX_SIZE inference.ctx_size '0'
  cfg_set N_PREDICT LLAMACPP_N_PREDICT inference.n_predict '-1'
  cfg_set TEMP LLAMACPP_TEMP inference.temp '0.80'
  cfg_set DYNATEMP_RANGE LLAMACPP_DYNATEMP_RANGE inference.dynatemp_range '0.00'
  cfg_set DYNATEMP_EXP LLAMACPP_DYNATEMP_EXP inference.dynatemp_exp '1.00'
  cfg_set TOP_P LLAMACPP_TOP_P inference.top_p '0.95'
  cfg_set TOP_K LLAMACPP_TOP_K inference.top_k '40'
  cfg_set MIN_P LLAMACPP_MIN_P inference.min_p '0.05'
  cfg_set TOP_N_SIGMA LLAMACPP_TOP_N_SIGMA inference.top_n_sigma '-1.00'
  cfg_set XTC_PROBABILITY LLAMACPP_XTC_PROBABILITY inference.xtc_probability '0.00'
  cfg_set XTC_THRESHOLD LLAMACPP_XTC_THRESHOLD inference.xtc_threshold '0.10'
  cfg_set TYPICAL_P LLAMACPP_TYPICAL_P inference.typical_p '1.00'
  cfg_set SAMPLERS LLAMACPP_SAMPLERS inference.samplers ''
  cfg_set SAMPLER_SEQ LLAMACPP_SAMPLER_SEQ inference.sampler_seq 'edskypmxt'
  cfg_set REPEAT_LAST_N LLAMACPP_REPEAT_LAST_N inference.repeat_last_n '64'
  cfg_set REPEAT_PENALTY LLAMACPP_REPEAT_PENALTY inference.repeat_penalty '1.00'
  cfg_set PRESENCE_PENALTY LLAMACPP_PRESENCE_PENALTY inference.presence_penalty '0.00'
  cfg_set FREQUENCY_PENALTY LLAMACPP_FREQUENCY_PENALTY inference.frequency_penalty '0.00'
  cfg_set DRY_MULTIPLIER LLAMACPP_DRY_MULTIPLIER inference.dry_multiplier '0.00'
  cfg_set DRY_BASE LLAMACPP_DRY_BASE inference.dry_base '1.75'
  cfg_set DRY_ALLOWED_LENGTH LLAMACPP_DRY_ALLOWED_LENGTH inference.dry_allowed_length '2'
  cfg_set DRY_PENALTY_LAST_N LLAMACPP_DRY_PENALTY_LAST_N inference.dry_penalty_last_n '-1'
  cfg_set BATCH_SIZE LLAMACPP_BATCH_SIZE inference.batch_size '2048'
  cfg_set UBATCH_SIZE LLAMACPP_UBATCH_SIZE inference.ubatch_size '512'
  cfg_set PARALLEL LLAMACPP_PARALLEL inference.parallel '-1'
  cfg_set N_CPU_MOE LLAMACPP_N_CPU_MOE inference.n_cpu_moe ''
  cfg_set FIT LLAMACPP_FIT inference.fit 'on'
  cfg_set FLASH_ATTN LLAMACPP_FLASH_ATTN inference.flash_attn 'auto'
  cfg_set CACHE_TYPE_K LLAMACPP_CACHE_TYPE_K inference.cache_type_k 'f16'
  cfg_set CACHE_TYPE_V LLAMACPP_CACHE_TYPE_V inference.cache_type_v 'f16'
  cfg_set KV_UNIFIED LLAMACPP_KV_UNIFIED inference.kv_unified '1'
  cfg_set CACHE_IDLE_SLOTS LLAMACPP_CACHE_IDLE_SLOTS inference.cache_idle_slots '1'
  cfg_set BACKEND_SAMPLING LLAMACPP_BACKEND_SAMPLING inference.backend_sampling '0'
  cfg_set WEB_UI LLAMACPP_WEB_UI inference.web_ui '1'
  cfg_set NO_MMAP LLAMACPP_NO_MMAP inference.no_mmap '0'
  cfg_set POLL LLAMACPP_POLL inference.poll '50'
  cfg_set JINJA LLAMACPP_JINJA inference.jinja '1'
  cfg_set CHAT_TEMPLATE_KWARGS LLAMACPP_CHAT_TEMPLATE_KWARGS inference.chat_template_kwargs ''

  cfg_set ENABLE_REASONING LLAMACPP_ENABLE_REASONING reasoning.enable 'auto'
  cfg_set REASONING_FORMAT LLAMACPP_REASONING_FORMAT reasoning.format "${reasoning_format_default}"
  cfg_set REASONING_BUDGET LLAMACPP_REASONING_BUDGET reasoning.budget '-1'
  cfg_set REASONING_BUDGET_MESSAGE LLAMACPP_REASONING_BUDGET_MESSAGE reasoning.budget_message ''

  load_extra_args

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

  case "${FIT,,}" in
    on|off)
      FIT="${FIT,,}"
      ;;
    1|true|yes)
      FIT="on"
      ;;
    0|false|no)
      FIT="off"
      ;;
    *)
      die "unsupported FIT=${FIT}; allowed: on,off,true,false,1,0"
      ;;
  esac

  case "${FLASH_ATTN,,}" in
    on|off|auto)
      FLASH_ATTN="${FLASH_ATTN,,}"
      ;;
    1|true|yes)
      FLASH_ATTN="on"
      ;;
    0|false|no)
      FLASH_ATTN="off"
      ;;
    *)
      die "unsupported FLASH_ATTN=${FLASH_ATTN}; allowed: on,off,auto,true,false,1,0"
      ;;
  esac

  if [[ -n "${SAMPLERS}" && -n "${SAMPLER_SEQ}" ]]; then
    die "set only one of inference.samplers or inference.sampler_seq"
  fi

  if [[ -n "${CHAT_TEMPLATE_KWARGS}" ]]; then
    need_cmds jq
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

  local -a server_args=()
  arg_add server_args -hf "${HF_MODEL}"
  arg_add server_args --host "${SERVER_HOST}"
  arg_add server_args --port "${CONTAINER_PORT}"
  arg_add server_args --ctx-size "${CTX_SIZE}"
  arg_add server_args --n-predict "${N_PREDICT}"
  arg_add server_args --temp "${TEMP}"
  arg_add server_args --dynatemp-range "${DYNATEMP_RANGE}"
  arg_add server_args --dynatemp-exp "${DYNATEMP_EXP}"
  arg_add server_args --top-k "${TOP_K}"
  arg_add server_args --top-p "${TOP_P}"
  arg_add server_args --min-p "${MIN_P}"
  arg_add server_args --top-n-sigma "${TOP_N_SIGMA}"
  arg_add server_args --xtc-probability "${XTC_PROBABILITY}"
  arg_add server_args --xtc-threshold "${XTC_THRESHOLD}"
  arg_add server_args --typical-p "${TYPICAL_P}"
  arg_add server_args --samplers "${SAMPLERS}"
  arg_add server_args --sampler-seq "${SAMPLER_SEQ}"
  arg_add server_args --repeat-last-n "${REPEAT_LAST_N}"
  arg_add server_args --repeat-penalty "${REPEAT_PENALTY}"
  arg_add server_args --presence-penalty "${PRESENCE_PENALTY}"
  arg_add server_args --frequency-penalty "${FREQUENCY_PENALTY}"
  arg_add server_args --dry-multiplier "${DRY_MULTIPLIER}"
  arg_add server_args --dry-base "${DRY_BASE}"
  arg_add server_args --dry-allowed-length "${DRY_ALLOWED_LENGTH}"
  arg_add server_args --dry-penalty-last-n "${DRY_PENALTY_LAST_N}"
  arg_add server_args --batch-size "${BATCH_SIZE}"
  arg_add server_args --ubatch-size "${UBATCH_SIZE}"
  arg_add server_args --parallel "${PARALLEL}"
  arg_add server_args --n-cpu-moe "${N_CPU_MOE}"
  arg_add server_args --fit "${FIT}"
  arg_add server_args --flash-attn "${FLASH_ATTN}"
  arg_add server_args --cache-type-k "${CACHE_TYPE_K}"
  arg_add server_args --cache-type-v "${CACHE_TYPE_V}"
  arg_add server_args --poll "${POLL}"

  arg_bool server_args "${NO_MMAP}" --no-mmap
  arg_bool server_args "${KV_UNIFIED}" --kv-unified --no-kv-unified
  arg_bool server_args "${CACHE_IDLE_SLOTS}" --cache-idle-slots --no-cache-idle-slots
  arg_bool server_args "${BACKEND_SAMPLING}" --backend-sampling
  arg_bool server_args "${WEB_UI}" --webui --no-webui
  arg_bool server_args "${JINJA}" --jinja --no-jinja
  arg_add server_args --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"

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
  LLAMACPP_SERVER_HOST, LLAMACPP_CTX_SIZE, LLAMACPP_N_PREDICT,
  LLAMACPP_TEMP, LLAMACPP_DYNATEMP_RANGE, LLAMACPP_DYNATEMP_EXP,
  LLAMACPP_TOP_K, LLAMACPP_TOP_P, LLAMACPP_MIN_P,
  LLAMACPP_TOP_N_SIGMA, LLAMACPP_TYPICAL_P,
  LLAMACPP_XTC_PROBABILITY, LLAMACPP_XTC_THRESHOLD,
  LLAMACPP_SAMPLERS, LLAMACPP_SAMPLER_SEQ,
  LLAMACPP_REPEAT_LAST_N,
  LLAMACPP_REPEAT_PENALTY, LLAMACPP_PRESENCE_PENALTY,
  LLAMACPP_FREQUENCY_PENALTY, LLAMACPP_DRY_MULTIPLIER,
  LLAMACPP_DRY_BASE, LLAMACPP_DRY_ALLOWED_LENGTH,
  LLAMACPP_DRY_PENALTY_LAST_N, LLAMACPP_FIT,
  LLAMACPP_FLASH_ATTN, LLAMACPP_CACHE_TYPE_K, LLAMACPP_CACHE_TYPE_V,
  LLAMACPP_KV_UNIFIED, LLAMACPP_CACHE_IDLE_SLOTS,
  LLAMACPP_BACKEND_SAMPLING, LLAMACPP_WEB_UI,
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
      require_docker
      docker logs -f "${CONTAINER_NAME}"
      ;;
    status)
      require_docker
      docker ps -a --filter "name=^/${CONTAINER_NAME}$"
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
load_cfg

main "${@}"