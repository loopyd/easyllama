#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() { echo "Usage: $0 [--mode NAME]" >&2; exit 1; }
while (( $# > 0 )); do
  case "$1" in
    --mode) MODE="${2:-}"; [[ -n "${MODE}" ]] || usage; shift 2 ;;
    *) usage ;;
  esac
done

config_path="$(config_for_reads)"
if ! grep -q -- '--cache-type-k' "${config_path}"; then
  echo "mode ${MODE} chat backend does not expose llama.cpp KV cache types" >&2
  exit 1
fi

cd "${REPO_ROOT}"
./run.sh --mode "${MODE}" server "${MODE}" --help | sed -n '/cache-type-k/,+10p'
