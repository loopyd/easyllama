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

echo "+ ${PYTHON_BIN} -m ruff check easyllama tests"
"${PYTHON_BIN}" -m ruff check easyllama tests

echo "+ ${PYTHON_BIN} -m ruff format --check easyllama tests"
"${PYTHON_BIN}" -m ruff format --check easyllama tests

echo "+ ${PYTHON_BIN} -m pytest -q"
if "${PYTHON_BIN}" -c 'import pytest' 2>/dev/null; then
  "${PYTHON_BIN}" -m pytest -q
else
  python3 -m pytest -q
fi

echo "+ ${PYTHON_BIN} -m compileall -q easyllama tests"
"${PYTHON_BIN}" -m compileall -q easyllama tests

echo "+ ./run.sh --help"
./run.sh --help >/dev/null

echo "+ validate config.json.example and documentation structure"
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

from easyllama.config import Config

example = json.loads(Path("config.json.example").read_text())
settings = Config.model_validate(example, context={"root": Path.cwd()})
assert Config.model_validate_json(settings.model_dump_json()) == settings
expected = {
    "dirs", "runtime", "docker", "resources", "modes", "locale", "lmcache",
    "credentials", "warmup", "llama_swap_override",
}
assert set(example) == expected
readme = Path("README.md").read_text()
api = Path("API.md").read_text()
template_readme = Path("chat_template/README.md").read_text()
tune_skill = Path(".github/skills/easyllama-tune/SKILL.md").read_text()
qwen_server = Path("easyllama/servers/qwen.py").read_text()
for key in expected:
    assert key in readme, f"README missing config node: {key}"
for stale in ("`repos`", "`hardware`", "`llama_swap`"):
    assert stale not in readme, f"README contains stale config node: {stale}"
for text, stale in (
    (api, "Qwen3.8-27B-NVFP4-RTX5090"),
    (template_readme, "`qwen` vLLM chat route"),
    (tune_skill, "Qwen chat route uses vLLM"),
    (qwen_server, 'backend="vllm"'),
):
    assert stale not in text, f"stale Qwen backend documentation: {stale}"
PY

echo "+ validate project skill scripts"
for script in .github/skills/*/scripts/*.sh; do
  bash -n "${script}"
done