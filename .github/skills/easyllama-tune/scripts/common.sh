#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SKILL_ROOT}/../../.." && pwd)"

MODE="${MODE:-mtp}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-}"
MODEL_ID="${MODEL_ID:-}"
CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-server-swap}"
VALIDATE_CONFIG_SCRIPT="${VALIDATE_CONFIG_SCRIPT:-${REPO_ROOT}/.github/skills/easyllama-provider/scripts/validate-config-yaml.sh}"

default_python() {
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
  else
    command -v python3
  fi
}

PYTHON_BIN="${PYTHON_BIN:-$(default_python)}"

resolve_mode_config_paths() {
  if [[ -n "${ACTIVE_CONFIG}" && -n "${EXAMPLE_CONFIG}" ]]; then
    return 0
  fi

  mapfile -t resolved_paths < <(MODE_NAME="${MODE}" OVERRIDE="${LLAMACPP_LS_CONFIG_FILE:-}" "${PYTHON_BIN}" - <<'PY'
import os

from easyllama.config import load_settings

settings = load_settings(mode_override=os.environ["MODE_NAME"])
config_pair = settings.configs[settings.mode]
override = os.environ.get("OVERRIDE") or ""

print(override or config_pair.active)
print(config_pair.example)
PY
)

if [[ -z "${ACTIVE_CONFIG}" ]]; then
    ACTIVE_CONFIG="${resolved_paths[0]}"
  fi

  if [[ -z "${EXAMPLE_CONFIG}" ]]; then
    EXAMPLE_CONFIG="${resolved_paths[1]}"
  fi
}

config_for_reads() {
  resolve_mode_config_paths
  if [[ -f "${ACTIVE_CONFIG}" ]]; then
    printf '%s\n' "${ACTIVE_CONFIG}"
  else
    printf '%s\n' "${EXAMPLE_CONFIG}"
  fi
}

resolve_chat_model_id() {
  local file="${1:-$(config_for_reads)}"

  if [[ -n "${MODEL_ID}" ]]; then
    printf '%s\n' "${MODEL_ID}"
    return 0
  fi

  FILE="${file}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import re
import sys


def finalize(alias: str | None, cmd_lines: list[str]) -> str | None:
    if alias is None:
        return None
    command = "\n".join(cmd_lines)
    if "--embedding" in command or "--reranking" in command:
        return None
    return alias


lines = Path(os.environ["FILE"]).read_text().splitlines()
in_models = False
current_alias: str | None = None
cmd_lines: list[str] = []
collecting_cmd = False
cmd_indent = 0

for line in lines:
    if not in_models:
        if re.match(r"^models:\s*(?:#.*)?$", line):
            in_models = True
        continue

    if line and not line.startswith((" ", "\t")):
        break

    alias_match = re.match(r'^\s{2}"([^"]+)":\s*(?:#.*)?$', line)
    if alias_match:
        candidate = finalize(current_alias, cmd_lines)
        if candidate is not None:
            print(candidate)
            sys.exit(0)
        current_alias = alias_match.group(1)
        cmd_lines = []
        collecting_cmd = False
        cmd_indent = 0
        continue

    if current_alias is None:
        continue

    if not collecting_cmd:
        cmd_match = re.match(r'^(\s*)cmd:\s*\|\s*$', line)
        if cmd_match:
            collecting_cmd = True
            cmd_indent = len(cmd_match.group(1))
        continue

    if line.strip() and len(line) - len(line.lstrip(" ")) <= cmd_indent:
        collecting_cmd = False
        continue

    cmd_lines.append(line)

candidate = finalize(current_alias, cmd_lines)
if candidate is not None:
    print(candidate)
    sys.exit(0)

raise SystemExit(f"no chat model alias found in {os.environ['FILE']}")
PY
}

require_file() {
  local path="$1"

  if [[ ! -f "${path}" ]]; then
    echo "file not found: ${path}" >&2
    exit 1
  fi
}

container_family_prefix() {
  if [[ "${CONTAINER_NAME}" == *-* ]]; then
    printf '%s\n' "${CONTAINER_NAME%%-*}"
  else
    printf '%s\n' "${CONTAINER_NAME}"
  fi
}

running_easyllama_containers() {
  local prefix

  prefix="$(container_family_prefix)"
  docker ps --format '{{.Names}}' | grep -E "^${prefix}(-|$)" || true
}

stop_running_easyllama_containers() {
  local containers=()

  mapfile -t containers < <(running_easyllama_containers)
  if (( ${#containers[@]} == 0 )); then
    return 0
  fi

  echo "+ docker stop ${containers[*]}" >&2
  docker stop "${containers[@]}" >/dev/null
}

recent_container_logs() {
  local tail_lines="${1:-120}"

  docker logs --tail "${tail_lines}" "${CONTAINER_NAME}" 2>&1 || true
}

logs_indicate_fit_boundary() {
  local log_text="${1:-}"

  grep -Eq 'HTTP status=502|HTTP 502|exit status 250|exit code: 250|CUDA error: out of memory' <<<"${log_text}"
}

read_macro_value() {
  local file="$1"
  local key="$2"

  FILE="${file}" KEY="${key}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import re

text = Path(os.environ["FILE"]).read_text()
pattern = rf'^[ \t]*{re.escape(os.environ["KEY"])}:[ \t]*"([^"]*)"[ \t]*(?:#.*)?$'
match = re.search(pattern, text, re.M)
if not match:
    raise SystemExit(f"macro not found: {os.environ['KEY']} in {os.environ['FILE']}")
print(match.group(1))
PY
}

set_macro_value() {
  local file="$1"
  local key="$2"
  local value="$3"

  FILE="${file}" KEY="${key}" VALUE="${value}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import re

path = Path(os.environ["FILE"])
text = path.read_text()
pattern = rf'^([ \t]*{re.escape(os.environ["KEY"])}:[ \t]*)(?:"[^"]*"|[^#\n]+?)([ \t]*(?:#.*)?$)'
replacement = rf'\1"{os.environ["VALUE"]}"\2'
updated, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
if count != 1:
    raise SystemExit(f"macro not found: {os.environ['KEY']} in {path}")
path.write_text(updated)
PY
}

show_tuning_values() {
  local file="${1:-${ACTIVE_CONFIG}}"

  printf 'config=%s\n' "${file}"
  printf '  chat_model_alias=%s\n' "$(resolve_chat_model_id "${file}")"
  printf '  qwen3_chat_ctx_size=%s\n' "$(read_macro_value "${file}" qwen3_chat_ctx_size)"
  printf '  qwen3_chat_gpu_layers=%s\n' "$(read_macro_value "${file}" qwen3_chat_gpu_layers)"
  printf '  qwen3_chat_fit=%s\n' "$(read_macro_value "${file}" qwen3_chat_fit)"
  printf '  cache_type_k=%s\n' "$(read_macro_value "${file}" cache_type_k)"
  printf '  cache_type_v=%s\n' "$(read_macro_value "${file}" cache_type_v)"
}

llama_server_bin_path() {
  local file="${1:-$(config_for_reads)}"

  read_macro_value "${file}" llama_server_bin
}

show_live_server_processes() {
  local file="${1:-$(config_for_reads)}"
  local server_bin

  server_bin="$(basename -- "$(llama_server_bin_path "${file}")")"
  echo "+ docker exec ${CONTAINER_NAME} sh -lc \"ps -eo pid,args | grep -F -- '${server_bin}' | grep -v grep\""
  docker exec "${CONTAINER_NAME}" sh -lc "ps -eo pid,args | grep -F -- '${server_bin}' | grep -v grep"
}

validate_cache_type() {
  local cache_type="$1"

  case "${cache_type}" in
    f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
    *)
      echo "unsupported cache type: ${cache_type}" >&2
      exit 1
      ;;
  esac
}