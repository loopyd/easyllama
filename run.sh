#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/config.json"
CONFIG_EXAMPLE_FILE="${SCRIPT_DIR}/config.json.example"
CONFIG_FILE="${LLAMACPP_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
CONFIG_CONTAINER_PATH="/app/config.json"
LOG_LEVEL="${LLAMACPP_LOG_LEVEL:-info}"
NO_COLOR="${LLAMACPP_NO_COLOR:-}"
RUNTIME_HOST="host"
RUNTIME_CONTAINER="container"
LLAMA_SERVER_BIN="/app/bin/llama-server"
HF_URL_BASE="https://huggingface.co"
MODELS_DIR_CONTAINER="/root/.cache/huggingface/hub"
CHAT_TEMPLATE_DIR_CONTAINER="/chat_template"
MMPROJ_DIR_CONTAINER="/mmproj"

detect_runtime_mode() {
  case "${LLAMACPP_RUNTIME_MODE:-}" in
    "") ;;
    "${RUNTIME_HOST}"|"${RUNTIME_CONTAINER}") printf '%s' "${LLAMACPP_RUNTIME_MODE}"; return ;;
    *) die "unsupported LLAMACPP_RUNTIME_MODE=${LLAMACPP_RUNTIME_MODE}; allowed: ${RUNTIME_HOST},${RUNTIME_CONTAINER}" ;;
  esac
  [[ -f /.dockerenv ]] && { printf '%s' "${RUNTIME_CONTAINER}"; return; }
  printf '%s' "${RUNTIME_HOST}"
}

RUNTIME_MODE="$(detect_runtime_mode)"

setup_colors() {
  if [[ -t 1 && -z "${NO_COLOR}" ]]; then
    C_RESET='\033[0m'; C_INFO='\033[1;34m'; C_WARN='\033[1;33m'; C_ERROR='\033[1;31m'; C_OK='\033[1;32m'
  else
    C_RESET=''; C_INFO=''; C_WARN=''; C_ERROR=''; C_OK=''
  fi
}

setup_colors

log() {
  local l c
  l="${1}"
  shift
  case "${LOG_LEVEL}" in
    debug) ;;
    info) [[ "${l}" == "DEBUG" ]] && return 0 ;;
    warn) [[ "${l}" != "WARN" && "${l}" != "ERROR" ]] && return 0 ;;
    error) [[ "${l}" != "ERROR" ]] && return 0 ;;
  esac
  c="${C_INFO}"
  case "${l}" in
    WARN) c="${C_WARN}" ;;
    ERROR) c="${C_ERROR}" ;;
    OK) c="${C_OK}" ;;
  esac
  printf '%b[%s] %s%b\n' "${c}" "${l}" "$*" "${C_RESET}"
}

info() { log INFO "$*"; }
warn() { log WARN "$*"; }
ok() { log OK "$*"; }
err() { log ERROR "$*"; }
die() { err "$*"; exit 1; }

need_cmds() {
  local -a m
  local d
  m=()
  for d in "$@"; do
    command -v "${d}" >/dev/null 2>&1 || m+=("${d}")
  done
  if [[ ${#m[@]} -gt 0 ]]; then die "missing dependency: ${m[*]}"; fi
}

detect_tz() {
  local t
  [[ -n "${TZ:-}" ]] && { echo "${TZ}"; return; }
  [[ -r /etc/timezone ]] && { tr -d '\n' < /etc/timezone; return; }
  if command -v timedatectl >/dev/null 2>&1; then
    t="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
    [[ -n "${t}" ]] && { echo "${t}"; return; }
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

normalize_toggle() {
  local n v a o
  n="${1}"
  v="${2}"
  a="${3:-0}"
  case "${v,,}" in
    on|off) o="${v,,}" ;;
    auto)
      [[ "${a}" == "1" ]] || die "unsupported ${n}=${v}; allowed: on,off,true,false,1,0"
      o="auto"
      ;;
    1|true|yes) o="on" ;;
    0|false|no) o="off" ;;
    *)
      [[ "${a}" == "1" ]] && die "unsupported ${n}=${v}; allowed: on,off,auto,true,false,1,0"
      die "unsupported ${n}=${v}; allowed: on,off,true,false,1,0"
      ;;
  esac
  printf '%s' "${o}"
}

cfg_has() {
  local p
  p="${1}"
  jq -e --arg path "${p}" '
    getpath($path | split(".")) != null
  ' "${CONFIG_FILE}" >/dev/null 2>&1
}

cfg_get() {
  local p
  p="${1}"
  jq -r --arg path "${p}" '
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
  local v e p d x
  v="${1}"
  e="${2}"
  p="${3}"
  d="${4-}"
  x=""
  if [[ -n "${!e+x}" ]]; then
    x="${!e}"
  elif [[ "${CONFIG_LOADED}" == "1" ]] && cfg_has "${p}"; then
    x="$(cfg_get "${p}")"
  else
    x="${d}"
  fi
  printf -v "${v}" '%s' "${x}"
}

cfg_bulk() {
  local s v e p d
  for s in "$@"; do
    IFS='|' read -r v e p d <<< "${s}"
    cfg_set "${v}" "${e}" "${p}" "${d}"
  done
}

parse_extra_server_args_value() {
  local r
  r="${1}"
  EXTRA_SERVER_ARGS=()
  [[ -z "${r}" ]] && return
  if [[ "${r}" =~ ^[[:space:]]*\[ ]]; then
    need_cmds jq
    mapfile -t EXTRA_SERVER_ARGS < <(
      printf '%s' "${r}" | jq -r '
        if type == "array" then
          .[] | tostring
        else
          error("extra_server_args must be a JSON array when using JSON syntax")
        end
      ' 2>/dev/null
    ) || die "LLAMACPP_EXTRA_SERVER_ARGS JSON must be a valid array"
    return
  fi
  read -r -a EXTRA_SERVER_ARGS <<< "${r}"
}

load_extra_args() {
  local t
  EXTRA_SERVER_ARGS=()
  if [[ -n "${LLAMACPP_EXTRA_SERVER_ARGS+x}" ]]; then
    parse_extra_server_args_value "${LLAMACPP_EXTRA_SERVER_ARGS}"
    return
  fi
  if [[ "${CONFIG_LOADED}" != "1" ]] || ! cfg_has inference.extra_server_args; then return 0; fi
  need_cmds jq
  t="$(jq -r '.inference.extra_server_args | type' "${CONFIG_FILE}" 2>/dev/null || true)"
  case "${t}" in
    array) mapfile -t EXTRA_SERVER_ARGS < <(jq -r '.inference.extra_server_args[] | tostring' "${CONFIG_FILE}") ;;
    string) parse_extra_server_args_value "$(jq -r '.inference.extra_server_args' "${CONFIG_FILE}")" ;;
    *) die "inference.extra_server_args must be a string or array in ${CONFIG_FILE}" ;;
  esac
}

abs_path() {
  local p
  p="${1}"
  [[ "${p}" = /* ]] && { printf '%s' "${p}"; return; }
  printf '%s' "${SCRIPT_DIR}/${p}"
}

map_chat_template_path() {
  local p
  p="${1}"
  [[ -n "${p}" ]] || { printf '%s' ""; return; }
  [[ "${p}" != */* ]] && { printf '%s' "${CHAT_TEMPLATE_DIR_CONTAINER}/${p}"; return; }
  case "${p}" in
    chat_template/*) printf '%s' "${CHAT_TEMPLATE_DIR_CONTAINER}/${p#chat_template/}"; return ;;
    "${MODELS_DIR_CONTAINER}/chat_template"/*) printf '%s' "${CHAT_TEMPLATE_DIR_CONTAINER}/${p#"${MODELS_DIR_CONTAINER}"/chat_template/}"; return ;;
    "${CHAT_TEMPLATE_DIR_CONTAINER}"/*) printf '%s' "${p}"; return ;;
    /chat_template/*) printf '%s' "${CHAT_TEMPLATE_DIR_CONTAINER}/${p#/chat_template/}"; return ;;
    "${CHAT_TEMPLATE_DIR}"/*) printf '%s' "${CHAT_TEMPLATE_DIR_CONTAINER}/${p#"${CHAT_TEMPLATE_DIR}"/}"; return ;;
  esac
  printf '%s' "${p}"
}

map_mmproj_path() {
  local p u n f t r ps
  p="${1}"
  [[ -n "${p}" ]] || { printf '%s' ""; return; }
  if [[ "${p}" =~ ^https?:// ]]; then
    need_cmds curl
    u="${p}"
    [[ "${u}" =~ ^https?://huggingface\.co/.*/blob/ ]] && u="${u/\/blob\//\/resolve\/}"
    n="$(basename "${u%%\?*}")"
    [[ -n "${n}" ]] || die "could not infer mmproj filename from URL: ${p}"
    f="${MMPROJ_DIR}/${n}"
    t="${f}.part"
    mkdir -p -- "${MMPROJ_DIR}"
    if [[ -n "${HF_TOKEN}" ]]; then
      r="$(curl -fsIL -H "Authorization: Bearer ${HF_TOKEN}" "${u}" 2>/dev/null | awk 'tolower($1)=="content-length:" {gsub(/\r/, "", $2); print $2}' | tail -n1 || true)"
    else
      r="$(curl -fsIL "${u}" 2>/dev/null | awk 'tolower($1)=="content-length:" {gsub(/\r/, "", $2); print $2}' | tail -n1 || true)"
    fi
    if [[ ! -s "${f}" || ( -n "${r}" && "$(wc -c < "${f}")" != "${r}" ) ]]; then
      info "downloading mmproj from ${p}" >&2
      [[ -s "${f}" && ! -s "${t}" ]] && mv -f -- "${f}" "${t}"
      if [[ -n "${HF_TOKEN}" ]]; then
        curl -fL --retry 3 --retry-delay 2 -C - -H "Authorization: Bearer ${HF_TOKEN}" -o "${t}" "${u}" >/dev/null
      else
        curl -fL --retry 3 --retry-delay 2 -C - -o "${t}" "${u}" >/dev/null
      fi
      if [[ -n "${r}" ]]; then
        ps="$(wc -c < "${t}")"
        [[ "${ps}" == "${r}" ]] || die "mmproj download incomplete for ${p}: got ${ps} bytes, expected ${r}"
      fi
      mv -f -- "${t}" "${f}"
      ok "downloaded mmproj to ${f}" >&2
    fi
    printf '%s' "${MMPROJ_DIR_CONTAINER}/${n}"
    return
  fi
  [[ "${p}" != */* ]] && { printf '%s' "${MMPROJ_DIR_CONTAINER}/${p}"; return; }
  case "${p}" in
    mmproj/*) printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#mmproj/}"; return ;;
    "${MMPROJ_DIR_CONTAINER}"/*) printf '%s' "${p}"; return ;;
    "${MMPROJ_DIR}"/*) printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#"${MMPROJ_DIR}"/}"; return ;;
    "${MODELS_DIR_CONTAINER}/mmproj"/*) printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#"${MODELS_DIR_CONTAINER}"/mmproj/}"; return ;;
    "${MODELS_DIR}"/*) printf '%s' "${MODELS_DIR_CONTAINER}/${p#"${MODELS_DIR}"/}"; return ;;
    /*) printf '%s' "${p}"; return ;;
    *) printf '%s' "${MMPROJ_DIR_CONTAINER}/${p#./}"; return ;;
  esac
}

hf_mmproj_url() {
  local s o r e f
  s="${1}"
  [[ -n "${s}" ]] || die "hf_mmproj cannot be empty"
  o="${s%%/*}"
  r="${s#*/}"
  e="${r%%/*}"
  f="${r#*/}"
  [[ "${o}" == "${s}" || "${r}" == "${s}" || -z "${o}" || -z "${e}" || -z "${f}" ]] && die "hf_mmproj must use format <owner>/<repo>/<file.gguf>; got: ${s}"
  printf '%s' "${HF_URL_BASE}/${o}/${e}/blob/main/${f}"
}

arg_add() {
  local t f v qf qv
  t="${1}"
  f="${2}"
  v="${3-}"
  [[ -n "${v}" ]] || return 0
  printf -v qf '%q' "${f}"
  printf -v qv '%q' "${v}"
  eval "${t}+=(${qf} ${qv})"
}

arg_add_pairs() {
  local t="${1}"
  shift
  while [[ "$#" -gt 1 ]]; do
    arg_add "${t}" "${1}" "${2}"
    shift 2
  done
}

arg_bool() {
  local t v on off qf
  t="${1}"
  v="${2-}"
  on="${3}"
  off="${4-}"
  [[ -n "${v}" ]] || return 0
  if is_truthy "${v}"; then
    printf -v qf '%q' "${on}"
    eval "${t}+=(${qf})"
  elif [[ -n "${off}" ]]; then
    printf -v qf '%q' "${off}"
    eval "${t}+=(${qf})"
  fi
}

arg_add_bools() {
  local t="${1}"
  shift
  while [[ "$#" -gt 2 ]]; do
    arg_bool "${t}" "${1}" "${2}" "${3}"
    shift 3
  done
}

load_config() {
  if [[ ! -r "${CONFIG_FILE}" ]]; then
    [[ -n "${LLAMACPP_CONFIG_FILE:-}" ]] && die "LLAMACPP_CONFIG_FILE is set but not readable: ${CONFIG_FILE}"
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
}

load_cfg() {
  local tz lg rf
  tz="$(detect_tz)"
  lg="${LANG:-C.UTF-8}"
  rf="${LLAMACPP_REASONING_PARSER:-auto}"
  cfg_bulk \
    'IMAGE_NAME|LLAMACPP_IMAGE_NAME|container.image_name|llamacpp-local:cuda13' \
    'CONTAINER_NAME|LLAMACPP_CONTAINER_NAME|container.container_name|llamacpp-server' \
    'MODELS_DIR|LLAMACPP_MODELS_DIR|container.models_dir|models' \
    'CHAT_TEMPLATE_DIR|LLAMACPP_CHAT_TEMPLATE_DIR|container.chat_template_dir|chat_template' \
    'MMPROJ_DIR|LLAMACPP_MMPROJ_DIR|container.mmproj_dir|mmproj'
  if [[ "${RUNTIME_MODE}" == "${RUNTIME_CONTAINER}" ]]; then
    MODELS_DIR="${MODELS_DIR_CONTAINER}"; CHAT_TEMPLATE_DIR="${CHAT_TEMPLATE_DIR_CONTAINER}"; MMPROJ_DIR="${MMPROJ_DIR_CONTAINER}"
  else
    MODELS_DIR="$(abs_path "${MODELS_DIR}")"; CHAT_TEMPLATE_DIR="$(abs_path "${CHAT_TEMPLATE_DIR}")"; MMPROJ_DIR="$(abs_path "${MMPROJ_DIR}")"
  fi
  cfg_bulk \
    'HOST_PORT|LLAMACPP_HOST_PORT|network.host_port|8080' \
    'CONTAINER_PORT|LLAMACPP_CONTAINER_PORT|network.container_port|8080' \
    'LOG_LEVEL|LLAMACPP_LOG_LEVEL|logging.log_level|info' \
    'NO_COLOR|LLAMACPP_NO_COLOR|logging.no_color|' \
    "HOST_TZ|LLAMACPP_HOST_TZ|locale.host_tz|${tz}" \
    "HOST_LANG|LLAMACPP_HOST_LANG|locale.host_lang|${lg}" \
    'DEFAULT_CUDA_ARCHITECTURES|LLAMACPP_DEFAULT_CUDA_ARCHITECTURES|build.default_cuda_architectures|120' \
    'CMAKE_CUDA_ARCHITECTURES|LLAMACPP_CMAKE_CUDA_ARCHITECTURES|build.cmake_cuda_architectures|auto' \
    'LLAMA_CPP_REPO|LLAMACPP_LLAMA_CPP_REPO|build.llama_cpp_repo|https://github.com/ggml-org/llama.cpp.git' \
    'LLAMA_CPP_REF|LLAMACPP_LLAMA_CPP_REF|build.llama_cpp_ref|master' \
    'HF_MODEL|LLAMACPP_HF_MODEL|model.hf_model|mradermacher/Qwen3.6-35B-A3B-abliterated-MAX-i1-GGUF:i1-Q6_K' \
    'HF_TOKEN|LLAMACPP_HF_TOKEN|model.hf_token|' \
    'SERVER_HOST|LLAMACPP_SERVER_HOST|network.server_host|0.0.0.0' \
    'CTX_SIZE|LLAMACPP_CTX_SIZE|inference.ctx_size|0' \
    'THREADS|LLAMACPP_THREADS|inference.threads|' \
    'THREADS_BATCH|LLAMACPP_THREADS_BATCH|inference.threads_batch|' \
    'N_PREDICT|LLAMACPP_N_PREDICT|inference.n_predict|-1' \
    'TEMP|LLAMACPP_TEMP|inference.temp|0.80' \
    'DYNATEMP_RANGE|LLAMACPP_DYNATEMP_RANGE|inference.dynatemp_range|0.00' \
    'DYNATEMP_EXP|LLAMACPP_DYNATEMP_EXP|inference.dynatemp_exp|1.00' \
    'TOP_P|LLAMACPP_TOP_P|inference.top_p|0.95' \
    'TOP_K|LLAMACPP_TOP_K|inference.top_k|40' \
    'MIN_P|LLAMACPP_MIN_P|inference.min_p|0.05' \
    'TOP_N_SIGMA|LLAMACPP_TOP_N_SIGMA|inference.top_n_sigma|-1.00' \
    'XTC_PROBABILITY|LLAMACPP_XTC_PROBABILITY|inference.xtc_probability|0.00' \
    'XTC_THRESHOLD|LLAMACPP_XTC_THRESHOLD|inference.xtc_threshold|0.10' \
    'TYPICAL_P|LLAMACPP_TYPICAL_P|inference.typical_p|1.00' \
    'SAMPLERS|LLAMACPP_SAMPLERS|inference.samplers|' \
    'SAMPLER_SEQ|LLAMACPP_SAMPLER_SEQ|inference.sampler_seq|edskypmxt' \
    'REPEAT_LAST_N|LLAMACPP_REPEAT_LAST_N|inference.repeat_last_n|64' \
    'REPEAT_PENALTY|LLAMACPP_REPEAT_PENALTY|inference.repeat_penalty|1.00' \
    'PRESENCE_PENALTY|LLAMACPP_PRESENCE_PENALTY|inference.presence_penalty|0.00' \
    'FREQUENCY_PENALTY|LLAMACPP_FREQUENCY_PENALTY|inference.frequency_penalty|0.00' \
    'DRY_MULTIPLIER|LLAMACPP_DRY_MULTIPLIER|inference.dry_multiplier|0.00' \
    'DRY_BASE|LLAMACPP_DRY_BASE|inference.dry_base|1.75' \
    'DRY_ALLOWED_LENGTH|LLAMACPP_DRY_ALLOWED_LENGTH|inference.dry_allowed_length|2' \
    'DRY_PENALTY_LAST_N|LLAMACPP_DRY_PENALTY_LAST_N|inference.dry_penalty_last_n|-1' \
    'BATCH_SIZE|LLAMACPP_BATCH_SIZE|inference.batch_size|2048' \
    'UBATCH_SIZE|LLAMACPP_UBATCH_SIZE|inference.ubatch_size|512' \
    'PARALLEL|LLAMACPP_PARALLEL|inference.parallel|-1' \
    'N_CPU_MOE|LLAMACPP_N_CPU_MOE|inference.n_cpu_moe|' \
    'FIT|LLAMACPP_FIT|inference.fit|on' \
    'FLASH_ATTN|LLAMACPP_FLASH_ATTN|inference.flash_attn|auto' \
    'CACHE_TYPE_K|LLAMACPP_CACHE_TYPE_K|inference.cache_type_k|f16' \
    'CACHE_TYPE_V|LLAMACPP_CACHE_TYPE_V|inference.cache_type_v|f16' \
    'KV_UNIFIED|LLAMACPP_KV_UNIFIED|inference.kv_unified|1' \
    'CACHE_IDLE_SLOTS|LLAMACPP_CACHE_IDLE_SLOTS|inference.cache_idle_slots|1' \
    'BACKEND_SAMPLING|LLAMACPP_BACKEND_SAMPLING|inference.backend_sampling|0' \
    'WEB_UI|LLAMACPP_WEB_UI|inference.web_ui|1' \
    'NO_WARMUP|LLAMACPP_NO_WARMUP|inference.no_warmup|0' \
    'NO_MMAP|LLAMACPP_NO_MMAP|inference.no_mmap|0' \
    'POLL|LLAMACPP_POLL|inference.poll|50' \
    'JINJA|LLAMACPP_JINJA|inference.jinja|1' \
    'MMPROJ_FILE|LLAMACPP_MMPROJ_FILE|inference.mmproj_file|' \
    'HF_MMPROJ|LLAMACPP_HF_MMPROJ|model.hf_mmproj|' \
    'CHAT_TEMPLATE_FILE|LLAMACPP_CHAT_TEMPLATE_FILE|inference.chat_template_file|' \
    'CHAT_TEMPLATE_KWARGS|LLAMACPP_CHAT_TEMPLATE_KWARGS|inference.chat_template_kwargs|'
  cfg_set HOST_LC_ALL LLAMACPP_HOST_LC_ALL locale.host_lc_all "${LC_ALL:-${HOST_LANG:-${lg}}}"
  if [[ -n "${HF_MMPROJ}" ]] && [[ -z "${LLAMACPP_MMPROJ_FILE+x}" ]]; then
    MMPROJ_FILE="$(hf_mmproj_url "${HF_MMPROJ}")"
  fi
  cfg_bulk \
    'ENABLE_REASONING|LLAMACPP_ENABLE_REASONING|reasoning.enable|auto' \
    "REASONING_FORMAT|LLAMACPP_REASONING_FORMAT|reasoning.format|${rf}" \
    'REASONING_BUDGET|LLAMACPP_REASONING_BUDGET|reasoning.budget|-1' \
    'REASONING_BUDGET_MESSAGE|LLAMACPP_REASONING_BUDGET_MESSAGE|reasoning.budget_message|'
  load_extra_args
  case "${CACHE_TYPE_V}" in
    f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1|turbo2|turbo3|turbo4) ;;
    *) die "unsupported CACHE_TYPE_V=${CACHE_TYPE_V}; allowed: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1,turbo2,turbo3,turbo4" ;;
  esac
  case "${ENABLE_REASONING}" in
    on|off|auto) ;;
    *) die "unsupported ENABLE_REASONING=${ENABLE_REASONING}; allowed: on,off,auto" ;;
  esac
  FIT="$(normalize_toggle FIT "${FIT}" 0)"
  FLASH_ATTN="$(normalize_toggle FLASH_ATTN "${FLASH_ATTN}" 1)"
  if [[ -n "${SAMPLERS}" && -n "${SAMPLER_SEQ}" ]]; then
    die "set only one of inference.samplers or inference.sampler_seq"
  fi
  if [[ -n "${CHAT_TEMPLATE_KWARGS}" ]]; then
    need_cmds jq
    CHAT_TEMPLATE_KWARGS="$(printf '%s' "${CHAT_TEMPLATE_KWARGS}" | jq -c 'if type == "object" then . else error("chat_template_kwargs must be a JSON object") end' 2>/dev/null)" || die "chat_template_kwargs must be a valid JSON object"
  fi
  setup_colors
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
  docker info >/dev/null 2>&1 || die "docker daemon is not reachable (start docker and retry)"
}

detect_cuda_architectures() {
  local a
  [[ "${CMAKE_CUDA_ARCHITECTURES}" != "auto" ]] && { echo "${CMAKE_CUDA_ARCHITECTURES}"; return; }
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"
    echo "${DEFAULT_CUDA_ARCHITECTURES}"
    return
  fi
  a="$({
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
  [[ -n "${a}" ]] && { echo "${a}"; return; }
  warn "failed to detect compute capability; using fallback CUDA arch ${DEFAULT_CUDA_ARCHITECTURES}"
  echo "${DEFAULT_CUDA_ARCHITECTURES}"
}

validate_cache_type_v_support() {
  local s
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    warn "image ${IMAGE_NAME} not found; run ./run.sh build before start"
    return 0
  fi
  s="$(
    docker run --rm --entrypoint "${LLAMA_SERVER_BIN}" --gpus all --runtime nvidia "${IMAGE_NAME}" -h 2>/dev/null | awk '
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
  )" || true
  [[ -z "${s}" ]] && return 0
  case ",${s}," in
    *",${CACHE_TYPE_V},"*) return 0 ;;
  esac
  die "CACHE_TYPE_V=${CACHE_TYPE_V} not supported by image ${IMAGE_NAME}. supported: ${s}. Rebuild with LLAMACPP_LLAMA_CPP_REPO/LLAMACPP_LLAMA_CPP_REF set to a turbo2-capable fork (for example TheTom/llama-cpp-turboquant)."
}

container_exists() { docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; }
container_running() { [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]; }

build_image() {
  local a
  require_docker
  a="$(detect_cuda_architectures)"
  info "building ${IMAGE_NAME} (repo=${LLAMA_CPP_REPO} ref=${LLAMA_CPP_REF} CMAKE_CUDA_ARCHITECTURES=${a})"
  DOCKER_BUILDKIT=1 docker build \
    --pull \
    --build-arg DEBIAN_FRONTEND=noninteractive \
    --build-arg HOST_TZ="${HOST_TZ}" \
    --build-arg HOST_LANG="${HOST_LANG}" \
    --build-arg HOST_LC_ALL="${HOST_LC_ALL}" \
    --build-arg LLAMA_CPP_REPO="${LLAMA_CPP_REPO}" \
    --build-arg LLAMA_CPP_REF="${LLAMA_CPP_REF}" \
    --build-arg CMAKE_CUDA_ARCHITECTURES="${a}" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"
  ok "build complete: ${IMAGE_NAME}"
}

stop_container() {
  require_docker
  if container_exists; then docker rm -f "${CONTAINER_NAME}" >/dev/null; ok "removed container ${CONTAINER_NAME}"; else warn "container ${CONTAINER_NAME} does not exist"; fi
}

start_container() {
  local -a a
  [[ "${RUNTIME_MODE}" == "${RUNTIME_HOST}" ]] || die "start command is host-only; use serve inside container"
  require_docker
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' || die "nvidia container runtime is not available in docker"
  validate_cache_type_v_support
  mkdir -p -- "${MODELS_DIR}" "${MMPROJ_DIR}"
  container_running && { warn "container ${CONTAINER_NAME} is already running"; return 0; }
  container_exists && docker rm -f "${CONTAINER_NAME}" >/dev/null
  a=(
    --detach
    --init
    --name "${CONTAINER_NAME}"
    --restart unless-stopped
    --security-opt no-new-privileges
    --gpus all
    --runtime nvidia
    --publish "${HOST_PORT}:${CONTAINER_PORT}"
    --volume "${MODELS_DIR}:${MODELS_DIR_CONTAINER}"
    --volume "${MMPROJ_DIR}:${MMPROJ_DIR_CONTAINER}"
    --volume "${CONFIG_FILE}:${CONFIG_CONTAINER_PATH}:ro"
    --env "LLAMACPP_CONFIG_FILE=${CONFIG_CONTAINER_PATH}"
    --env "LLAMACPP_RUNTIME_MODE=${RUNTIME_CONTAINER}"
    --env "TZ=${HOST_TZ}"
    --env "LANG=${HOST_LANG}"
    --env "LC_ALL=${HOST_LC_ALL}"
  )
  [[ -d "${CHAT_TEMPLATE_DIR}" ]] && a+=(--volume "${CHAT_TEMPLATE_DIR}:${CHAT_TEMPLATE_DIR_CONTAINER}:ro")
  [[ -r /etc/localtime ]] && a+=(--volume /etc/localtime:/etc/localtime:ro)
  [[ -r /etc/timezone ]] && a+=(--volume /etc/timezone:/etc/timezone:ro)
  [[ -n "${HF_TOKEN}" ]] && a+=(--env "HF_TOKEN=${HF_TOKEN}")
  docker run "${a[@]}" "${IMAGE_NAME}" serve >/dev/null
  ok "started ${CONTAINER_NAME} on http://localhost:${HOST_PORT}"
}

start_server() {
  local c m
  local -a a
  c="$(map_chat_template_path "${CHAT_TEMPLATE_FILE}")"
  m="$(map_mmproj_path "${MMPROJ_FILE}")"
  [[ "${c}" == "${CHAT_TEMPLATE_DIR_CONTAINER}"/* && ! -d "${CHAT_TEMPLATE_DIR}" ]] && die "CHAT_TEMPLATE_FILE points to chat_template/, but ${CHAT_TEMPLATE_DIR} does not exist"
  [[ "${m}" == "${MMPROJ_DIR_CONTAINER}"/* && ! -d "${MMPROJ_DIR}" ]] && die "MMPROJ_FILE points to mmproj/, but ${MMPROJ_DIR} does not exist"
  [[ -x "${LLAMA_SERVER_BIN}" ]] || die "llama-server binary not found at ${LLAMA_SERVER_BIN}"
  a=()
  arg_add_pairs a \
    -hf "${HF_MODEL}" \
    --host "${SERVER_HOST}" \
    --port "${CONTAINER_PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --threads "${THREADS}" \
    --threads-batch "${THREADS_BATCH}" \
    --n-predict "${N_PREDICT}" \
    --temp "${TEMP}" \
    --dynatemp-range "${DYNATEMP_RANGE}" \
    --dynatemp-exp "${DYNATEMP_EXP}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --min-p "${MIN_P}" \
    --top-n-sigma "${TOP_N_SIGMA}" \
    --xtc-probability "${XTC_PROBABILITY}" \
    --xtc-threshold "${XTC_THRESHOLD}" \
    --typical-p "${TYPICAL_P}" \
    --samplers "${SAMPLERS}" \
    --sampler-seq "${SAMPLER_SEQ}" \
    --repeat-last-n "${REPEAT_LAST_N}" \
    --repeat-penalty "${REPEAT_PENALTY}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    --frequency-penalty "${FREQUENCY_PENALTY}" \
    --dry-multiplier "${DRY_MULTIPLIER}" \
    --dry-base "${DRY_BASE}" \
    --dry-allowed-length "${DRY_ALLOWED_LENGTH}" \
    --dry-penalty-last-n "${DRY_PENALTY_LAST_N}" \
    --batch-size "${BATCH_SIZE}" \
    --ubatch-size "${UBATCH_SIZE}" \
    --parallel "${PARALLEL}" \
    --n-cpu-moe "${N_CPU_MOE}" \
    --fit "${FIT}" \
    --flash-attn "${FLASH_ATTN}" \
    --cache-type-k "${CACHE_TYPE_K}" \
    --cache-type-v "${CACHE_TYPE_V}" \
    --poll "${POLL}"
  arg_add_bools a \
    "${NO_MMAP}" --no-mmap "" \
    "${KV_UNIFIED}" --kv-unified --no-kv-unified \
    "${CACHE_IDLE_SLOTS}" --cache-idle-slots --no-cache-idle-slots \
    "${BACKEND_SAMPLING}" --backend-sampling "" \
    "${WEB_UI}" --webui --no-webui \
    "${NO_WARMUP}" --no-warmup "" \
    "${JINJA}" --jinja --no-jinja
  arg_add a --mmproj "${m}"
  arg_add a --chat-template-file "${c}"
  arg_add a --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
  [[ -n "${ENABLE_REASONING}" ]] && a+=(--reasoning "${ENABLE_REASONING}")
  if [[ "${ENABLE_REASONING}" != "off" ]]; then
    [[ -n "${REASONING_FORMAT}" ]] && a+=(--reasoning-format "${REASONING_FORMAT}")
    [[ -n "${REASONING_BUDGET}" ]] && a+=(--reasoning-budget "${REASONING_BUDGET}")
    [[ -n "${REASONING_BUDGET_MESSAGE}" ]] && a+=(--reasoning-budget-message "${REASONING_BUDGET_MESSAGE}")
  fi
  [[ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]] && a+=("${EXTRA_SERVER_ARGS[@]}")
  exec "${LLAMA_SERVER_BIN}" "${a[@]}"
}

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
Commands: build, start, stop, restart, logs, status, clean, serve, help

Command details:
  build    Build local llama.cpp CUDA 13 image from Dockerfile
  start    Start llama-server container
  stop     Stop and remove the running container
  restart  Restart container (stop then start)
  logs     Follow container logs
  status   Show container status
  clean    Remove container and local image
  serve    Run llama-server from config (container entrypoint mode)
  help     Show this help text

Config precedence:
  1) Environment variables (LLAMACPP_* only)
  2) LLAMACPP_CONFIG_FILE path (if set)
  3) config.json (local, gitignored)
  4) config.json.example (tracked template)
  5) Built-in defaults

LLAMACPP_EXTRA_SERVER_ARGS accepts either:
  - shell words string: --foo bar --baz
  - JSON array string: ["--foo","bar","--baz"]
EOF
}

main() {
  local c="${1:-help}"
  case "${c}" in
    help|-h|--help) usage; return 0 ;;
    build|start|stop|restart|logs|status|clean|serve) load_config; load_cfg ;;
    *) err "unknown command: ${c}"; usage; return 1 ;;
  esac
  case "${c}" in
    build) build_image ;;
    start) if [[ "${RUNTIME_MODE}" == "${RUNTIME_CONTAINER}" ]]; then start_server; else start_container; fi ;;
    stop) stop_container ;;
    restart) stop_container || true; start_container ;;
    logs) require_docker; docker logs -f "${CONTAINER_NAME}" ;;
    status) require_docker; docker ps -a --filter "name=^/${CONTAINER_NAME}$" ;;
    clean) clean_all ;;
    serve) start_server ;;
  esac
}

main "${@}"