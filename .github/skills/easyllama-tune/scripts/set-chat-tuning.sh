#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  echo "Usage: $0 [--mode NAME] [--layers N] [--ctx-size N] [--cache-type TYPE] [--fit on|off] [--sync-example] [--show]" >&2
  exit 1
}

layers=""
ctx_size=""
cache_type=""
fit_mode=""
sync_example=0
show_only=0

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      [[ -n "${MODE}" ]] || usage
      shift 2
      ;;
    --layers)
      layers="${2:-}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:-}"
      shift 2
      ;;
    --cache-type)
      cache_type="${2:-}"
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
    --show)
      show_only=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

resolve_mode_config_paths

if [[ -z "${layers}" && -z "${ctx_size}" && -z "${cache_type}" && -z "${fit_mode}" ]]; then
  if (( show_only == 1 )); then
    show_tuning_values "$(config_for_reads)"
    exit 0
  fi

  usage
fi

require_file "${ACTIVE_CONFIG}"
if (( sync_example == 1 )); then
  require_file "${EXAMPLE_CONFIG}"
fi

if [[ -n "${cache_type}" ]]; then
  validate_cache_type "${cache_type}"
fi

apply_updates() {
  local target_file="$1"

  [[ -n "${layers}" ]] && set_macro_value "${target_file}" qwen3_chat_gpu_layers "${layers}"
  [[ -n "${ctx_size}" ]] && set_macro_value "${target_file}" qwen3_chat_ctx_size "${ctx_size}"
  [[ -n "${fit_mode}" ]] && set_macro_value "${target_file}" qwen3_chat_fit "${fit_mode}"
  if [[ -n "${cache_type}" ]]; then
    set_macro_value "${target_file}" cache_type_k "${cache_type}"
    set_macro_value "${target_file}" cache_type_v "${cache_type}"
  fi
}

apply_updates "${ACTIVE_CONFIG}"
if (( sync_example == 1 )); then
  apply_updates "${EXAMPLE_CONFIG}"
fi

show_tuning_values "${ACTIVE_CONFIG}"
if (( sync_example == 1 )); then
  show_tuning_values "${EXAMPLE_CONFIG}"
fi