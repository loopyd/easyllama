#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SKILL_ROOT}/../../.." && pwd)"

MODE="${MODE:-llamacpp}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-}"
MODEL_ID="${MODEL_ID:-}"
VALIDATE_CONFIG_SCRIPT="${VALIDATE_CONFIG_SCRIPT:-${REPO_ROOT}/.github/skills/easyllama-provider/scripts/validate-config-yaml.sh}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

resolve_mode_config_paths() {
  local -a paths
  mapfile -t paths < <(EASYLLAMA_ROOT="${REPO_ROOT}" MODE_NAME="${MODE}" "${PYTHON_BIN}" - <<'PY'
import os
from easyllama.config import Config

settings = Config.load(mode_override=os.environ["MODE_NAME"])
active = settings.dirs.root / f"config/config.{settings.runtime.mode}.yml"
print(settings.llama_swap_override or active)
print(active.with_suffix(".yml.example"))
PY
)
  ACTIVE_CONFIG="${ACTIVE_CONFIG:-${paths[0]}}"
  EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-${paths[1]}}"
}

config_for_reads() {
  resolve_mode_config_paths
  [[ -f "${ACTIVE_CONFIG}" ]] && printf '%s\n' "${ACTIVE_CONFIG}" || printf '%s\n' "${EXAMPLE_CONFIG}"
}

require_file() {
  [[ -f "$1" ]] || { echo "file not found: $1" >&2; exit 1; }
}

resolve_chat_model_id() {
  local file="${1:-$(config_for_reads)}"
  [[ -n "${MODEL_ID}" ]] && { printf '%s\n' "${MODEL_ID}"; return; }
  FILE="${file}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import yaml

models = (yaml.safe_load(Path(os.environ["FILE"]).read_text()) or {}).get("models", {})
for alias, model in models.items():
    command = str(model.get("cmd", ""))
    if "--embedding" not in command and "--reranking" not in command:
        print(alias)
        break
else:
    raise SystemExit(f"no chat model alias found in {os.environ['FILE']}")
PY
}

chat_flag() {
  local file="$1" flag="$2"
  FILE="${file}" FLAG="${flag}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import shlex
import yaml

models = (yaml.safe_load(Path(os.environ["FILE"]).read_text()) or {}).get("models", {})
for model in models.values():
    command = str(model.get("cmd", ""))
    if "--embedding" in command or "--reranking" in command:
        continue
    words = shlex.split(command)
    flag = os.environ["FLAG"]
    aliases = ("--gpu-layers", "--n-gpu-layers") if flag == "--gpu-layers" else (flag,)
    found = next((candidate for candidate in aliases if candidate in words), None)
    if found is None:
        raise SystemExit(f"{flag} is not supported by the selected chat command")
    print(words[words.index(found) + 1])
    break
else:
    raise SystemExit("no chat command found")
PY
}

set_chat_flag() {
  local file="$1" flag="$2" value="$3"
  FILE="${file}" FLAG="${flag}" VALUE="${value}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import re

path = Path(os.environ["FILE"])
text = path.read_text()
flag = os.environ["FLAG"]
aliases = ("--gpu-layers", "--n-gpu-layers") if flag == "--gpu-layers" else (flag,)
for candidate in aliases:
    escaped = re.escape(candidate)
    updated, count = re.subn(rf"({escaped}\s+)[^\s'\"]+", rf"\g<1>{os.environ['VALUE']}", text, count=1)
    if count == 1:
        path.write_text(updated)
        break
else:
    raise SystemExit(f"{flag} is not supported by the selected chat command")
PY
}

optional_chat_flag() {
  chat_flag "$1" "$2" 2>/dev/null || printf 'not set\n'
}

show_tuning_values() {
  local file="${1:-$(config_for_reads)}"
  printf 'config=%s\n' "${file}"
  printf '  chat_model_alias=%s\n' "$(resolve_chat_model_id "${file}")"
  printf '  ctx_size=%s\n' "$(chat_flag "${file}" --ctx-size)"
  printf '  gpu_layers=%s\n' "$(chat_flag "${file}" --gpu-layers)"
  printf '  fit=%s\n' "$(optional_chat_flag "${file}" --fit)"
  printf '  cache_type_k=%s\n' "$(chat_flag "${file}" --cache-type-k)"
  printf '  cache_type_v=%s\n' "$(chat_flag "${file}" --cache-type-v)"
}

stop_running_easyllama_containers() {
  echo "+ ./run.sh --mode ${MODE} stop" >&2
  "${REPO_ROOT}/run.sh" --mode "${MODE}" stop >/dev/null
}

recent_container_logs() {
  "${REPO_ROOT}/run.sh" --mode "${MODE}" logs --tail "${1:-120}" 2>&1 || true
}

logs_indicate_fit_boundary() {
  grep -Eq 'HTTP status=502|HTTP 502|exit status 250|exit code: 250|CUDA error: out of memory' <<<"${1:-}"
}

show_live_server_processes() {
  "${REPO_ROOT}/run.sh" --mode "${MODE}" logs --tail 40
}

validate_cache_type() {
  case "$1" in
    f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
    *) echo "unsupported cache type: $1" >&2; exit 1 ;;
  esac
}
