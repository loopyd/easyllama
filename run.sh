#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EASYLLAMA_ROOT="${EASYLLAMA_ROOT:-${SCRIPT_DIR}}"

if [[ -x "${SCRIPT_DIR}/.venv/bin/easyllama" ]]; then
  exec "${SCRIPT_DIR}/.venv/bin/easyllama" "$@"
fi

if [[ -x "/opt/venv/bin/easyllama" ]]; then
  exec "/opt/venv/bin/easyllama" "$@"
fi

if command -v easyllama >/dev/null 2>&1; then
  exec "$(command -v easyllama)" "$@"
fi

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  printf '%s\n' "python3 is required to run easyllama" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m easyllama "$@"
