#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

cd "${REPO_ROOT}"

echo "+ bash -n run.sh"
bash -n run.sh

echo "+ ${PYTHON_BIN} -m ruff check easyllama"
"${PYTHON_BIN}" -m ruff check easyllama

echo "+ ${PYTHON_BIN} -m compileall easyllama"
"${PYTHON_BIN}" -m compileall easyllama

echo "+ ./run.sh help"
./run.sh help >/dev/null