#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <mode> [model ...]" >&2
  exit 1
}

MODE="${1:-}"
[[ -n "${MODE}" ]] || usage
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

resolve_config_path() {
  if [[ -n "${LLAMACPP_LS_CONFIG_FILE:-}" ]]; then
    printf '%s\n' "${LLAMACPP_LS_CONFIG_FILE}"
    return 0
  fi

  local active_config="config.${MODE}.yml"
  local example_config="${active_config}.example"

  if [[ -f "${active_config}" ]]; then
    printf '%s\n' "${active_config}"
    return 0
  fi

  if [[ -f "${example_config}" ]]; then
    printf '%s\n' "${example_config}"
    return 0
  fi

  printf '%s\n' \
    "no llama-swap config found for ${MODE} mode; set LLAMACPP_LS_CONFIG_FILE or create ${active_config} from ${example_config}" >&2
  return 1
}

cd "${REPO_ROOT}"

CONFIG_PATH="$(resolve_config_path)"

echo "+ ${SCRIPT_DIR}/validate-config-yaml.sh ${CONFIG_PATH}"
"${SCRIPT_DIR}/validate-config-yaml.sh" "${CONFIG_PATH}"

echo "+ ./run.sh --mode ${MODE} build"
./run.sh --mode "${MODE}" build

echo "+ ./run.sh --mode ${MODE} restart"
./run.sh --mode "${MODE}" restart

if (( $# > 0 )); then
  echo "+ ./run.sh --mode ${MODE} warmup $*"
  ./run.sh --mode "${MODE}" warmup "$@"
else
  echo "+ ./run.sh --mode ${MODE} warmup"
  ./run.sh --mode "${MODE}" warmup
fi