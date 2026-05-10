#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  echo "Usage: $0 [--mode NAME]" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      [[ -n "${MODE}" ]] || usage
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

cd "${REPO_ROOT}"
resolve_mode_config_paths

config_path="$(config_for_reads)"
server_bin="$(llama_server_bin_path "${config_path}")"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  stop_running_easyllama_containers
  echo "container ${CONTAINER_NAME} is not running; starting ${MODE} mode first" >&2
  ./run.sh --mode "${MODE}" restart >/dev/null
fi

echo "+ docker exec ${CONTAINER_NAME} ${server_bin} --help | sed -n '/cache-type-k/,+10p'"
docker exec "${CONTAINER_NAME}" "${server_bin}" --help | sed -n '/cache-type-k/,+10p'