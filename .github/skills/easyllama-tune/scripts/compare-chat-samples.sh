#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/skills/easyllama-tune/scripts/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
  echo "Usage: $0 <before.json> <after.json> [--show-diff]" >&2
  exit 1
}

BEFORE_PATH="${1:-}"
AFTER_PATH="${2:-}"
shift 2 || true

[[ -n "${BEFORE_PATH}" && -n "${AFTER_PATH}" ]] || usage

show_diff=0
while (( $# > 0 )); do
  case "$1" in
    --show-diff)
      show_diff=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

require_file "${BEFORE_PATH}"
require_file "${AFTER_PATH}"

BEFORE_PATH="${BEFORE_PATH}" AFTER_PATH="${AFTER_PATH}" SHOW_DIFF="${show_diff}" "${PYTHON_BIN}" - <<'PY'
import difflib
import json
import os
from pathlib import Path
from typing import Any


def load_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def extract_content(payload: dict[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices") or []
    if not choices:
        return "", "empty"
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return text, "message.content[]"
    elif isinstance(content, str):
        text = content.strip()
        if text:
            return text, "message.content"

    reasoning_content = str(message.get("reasoning_content") or "").strip()
    if reasoning_content:
        return reasoning_content, "message.reasoning_content"

    text_choice = str(first.get("text") or "").strip()
    if text_choice:
        return text_choice, "choice.text"

    return "", "empty"


def classify(text: str) -> tuple[str, Any | None]:
    stripped = text.strip()
    if not stripped:
        return "empty", None
    if stripped.startswith("```"):
        return "markdown-fence", None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return "json-object", parsed
    if isinstance(parsed, list):
        return "json-array", parsed
    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and all(line.startswith("- ") for line in lines):
        return "bullet-list", None
    if lines and all(line[:2].isdigit() and line[2:4] == ". " for line in lines if len(line) >= 4):
        return "numbered-list", None
    if len(lines) > 1:
        return "multiline-text", None
    return "plain-text", None


def content_stats(text: str) -> dict[str, int]:
    stripped = text.strip()
    return {
        "chars": len(text),
        "words": len(stripped.split()) if stripped else 0,
        "lines": len(text.splitlines()) if text else 0,
    }


def usage_tokens(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


before_path = os.environ["BEFORE_PATH"]
after_path = os.environ["AFTER_PATH"]
show_diff = os.environ["SHOW_DIFF"] == "1"

before = load_payload(before_path)
after = load_payload(after_path)

before_content, before_source = extract_content(before)
after_content, after_source = extract_content(after)

before_format, before_json = classify(before_content)
after_format, after_json = classify(after_content)

before_stats = content_stats(before_content)
after_stats = content_stats(after_content)

ratio = difflib.SequenceMatcher(None, before_content, after_content).ratio()
identical = before_content == after_content

before_finish = ((before.get("choices") or [{}])[0] or {}).get("finish_reason")
after_finish = ((after.get("choices") or [{}])[0] or {}).get("finish_reason")
before_model = before.get("model")
after_model = after.get("model")

before_prompt_tokens, before_completion_tokens = usage_tokens(before)
after_prompt_tokens, after_completion_tokens = usage_tokens(after)

print(f"before={before_path}")
print(f"after={after_path}")
print(f"model.before={before_model}")
print(f"model.after={after_model}")
print(f"finish_reason.before={before_finish}")
print(f"finish_reason.after={after_finish}")
print(f"format.before={before_format}")
print(f"format.after={after_format}")
print(f"content_source.before={before_source}")
print(f"content_source.after={after_source}")
print(f"content_identical={'yes' if identical else 'no'}")
print(f"similarity_ratio={ratio:.4f}")
print(
    "content_stats.before="
    f"chars:{before_stats['chars']} words:{before_stats['words']} lines:{before_stats['lines']}"
)
print(
    "content_stats.after="
    f"chars:{after_stats['chars']} words:{after_stats['words']} lines:{after_stats['lines']}"
)

if before_prompt_tokens is not None or after_prompt_tokens is not None:
    print(f"prompt_tokens.before={before_prompt_tokens}")
    print(f"prompt_tokens.after={after_prompt_tokens}")
if before_completion_tokens is not None or after_completion_tokens is not None:
    print(f"completion_tokens.before={before_completion_tokens}")
    print(f"completion_tokens.after={after_completion_tokens}")

if before_json is not None and after_json is not None:
    before_keys = list(before_json.keys()) if isinstance(before_json, dict) else []
    after_keys = list(after_json.keys()) if isinstance(after_json, dict) else []
    removed = [key for key in before_keys if key not in after_keys]
    added = [key for key in after_keys if key not in before_keys]
    print(f"json_keys.removed={removed}")
    print(f"json_keys.added={added}")
    canonical_before = json.dumps(before_json, sort_keys=True, ensure_ascii=False)
    canonical_after = json.dumps(after_json, sort_keys=True, ensure_ascii=False)
    print(f"json_content_identical={'yes' if canonical_before == canonical_after else 'no'}")

diff_lines = list(
    difflib.unified_diff(
        before_content.splitlines(),
        after_content.splitlines(),
        fromfile="before-content",
        tofile="after-content",
        lineterm="",
        n=2,
    )
)

if diff_lines:
    print("content_diff_preview:")
    preview = diff_lines if show_diff else diff_lines[:20]
    for line in preview:
        print(line)
    if not show_diff and len(diff_lines) > len(preview):
        print(f"... truncated {len(diff_lines) - len(preview)} more diff lines")
else:
    print("content_diff_preview: no textual differences")
PY