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

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

resolve_config_path() {
  EASYLLAMA_ROOT="${REPO_ROOT}" MODE_NAME="${MODE}" "${PYTHON_BIN}" - <<'PY'
import os

from easyllama.config import Config

print(Config.load(mode_override=os.environ["MODE_NAME"]).resolve_ls_config())
PY
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