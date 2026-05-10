#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  echo "Usage: $0 [--mode NAME] --good N --bad N [--cache-type TYPE] [--ctx-size N] [--fit on|off] [--sync-example]" >&2
  exit 1
}

known_good=""
known_bad=""
cache_type=""
ctx_size=""
fit_mode=""
sync_example=0

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      [[ -n "${MODE}" ]] || usage
      shift 2
      ;;
    --good)
      known_good="${2:-}"
      shift 2
      ;;
    --bad)
      known_bad="${2:-}"
      shift 2
      ;;
    --cache-type)
      cache_type="${2:-}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:-}"
      shift 2
      ;;
    --fit)
      fit_mode="${2:-}"
      shift 2
      ;;
    --sync-example)
      sync_example=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

[[ -n "${known_good}" && -n "${known_bad}" ]] || usage
(( known_good < known_bad )) || usage

resolve_mode_config_paths

set_script="${SCRIPT_DIR}/set-chat-tuning.sh"
probe_script="${SCRIPT_DIR}/probe-chat.sh"

set_args=(--mode "${MODE}" --layers "${known_good}")
[[ -n "${cache_type}" ]] && set_args+=(--cache-type "${cache_type}")
[[ -n "${ctx_size}" ]] && set_args+=(--ctx-size "${ctx_size}")
[[ -n "${fit_mode}" ]] && set_args+=(--fit "${fit_mode}")

echo "+ ${set_script} ${set_args[*]}"
"${set_script}" "${set_args[@]}"

echo "+ ${probe_script} --mode ${MODE}"
"${probe_script}" --mode "${MODE}"

current_good="${known_good}"
current_bad="${known_bad}"

while (( current_good + 1 < current_bad )); do
  candidate=$(((current_good + current_bad + 1) / 2))
  candidate_args=(--mode "${MODE}" --layers "${candidate}")
  [[ -n "${cache_type}" ]] && candidate_args+=(--cache-type "${cache_type}")
  [[ -n "${ctx_size}" ]] && candidate_args+=(--ctx-size "${ctx_size}")
  [[ -n "${fit_mode}" ]] && candidate_args+=(--fit "${fit_mode}")

  echo "==> probing gpu-layers=${candidate} with good=${current_good} bad=${current_bad}"
  echo "+ ${set_script} ${candidate_args[*]}"
  "${set_script}" "${candidate_args[@]}"

  if "${probe_script}" --mode "${MODE}"; then
    current_good="${candidate}"
  else
    current_bad="${candidate}"
  fi
done

final_args=(--mode "${MODE}" --layers "${current_good}")
[[ -n "${cache_type}" ]] && final_args+=(--cache-type "${cache_type}")
[[ -n "${ctx_size}" ]] && final_args+=(--ctx-size "${ctx_size}")
[[ -n "${fit_mode}" ]] && final_args+=(--fit "${fit_mode}")
(( sync_example == 1 )) && final_args+=(--sync-example)

echo "+ ${set_script} ${final_args[*]}"
"${set_script}" "${final_args[@]}"

if (( sync_example == 1 )); then
  echo "+ ${VALIDATE_CONFIG_SCRIPT} ${EXAMPLE_CONFIG}"
  "${VALIDATE_CONFIG_SCRIPT}" "${EXAMPLE_CONFIG}"
fi

echo "highest verified working gpu-layers=${current_good}"
echo "next failing bound gpu-layers=${current_bad}"