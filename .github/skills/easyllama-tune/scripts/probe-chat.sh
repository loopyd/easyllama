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

resolve_mode_config_paths

require_file "${ACTIVE_CONFIG}"
require_file "${VALIDATE_CONFIG_SCRIPT}"

selected_model_id="$(resolve_chat_model_id "${ACTIVE_CONFIG}")"

cd "${REPO_ROOT}"

show_tuning_values "${ACTIVE_CONFIG}"

echo "+ ${VALIDATE_CONFIG_SCRIPT} ${ACTIVE_CONFIG}"
"${VALIDATE_CONFIG_SCRIPT}" "${ACTIVE_CONFIG}"

stop_running_easyllama_containers

echo "+ ./run.sh --mode ${MODE} restart"
./run.sh --mode "${MODE}" restart

echo "+ ./run.sh --mode ${MODE} warmup ${selected_model_id}"
if ./run.sh --mode "${MODE}" warmup "${selected_model_id}"; then
  show_live_server_processes "${ACTIVE_CONFIG}"
else
  status=$?
  echo "+ docker logs --tail 120 ${CONTAINER_NAME} 2>&1" >&2
  logs="$(recent_container_logs 120)"
  printf '%s\n' "${logs}" >&2
  if logs_indicate_fit_boundary "${logs}"; then
    echo "fit-boundary signal detected: upstream 502 or exit status 250 usually means the selected setting ran out of room during startup" >&2
  fi
  exit "${status}"
fi