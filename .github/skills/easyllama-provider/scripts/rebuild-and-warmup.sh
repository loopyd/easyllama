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

cd "${REPO_ROOT}"

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