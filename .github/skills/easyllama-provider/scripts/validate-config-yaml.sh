#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <config.yml>" >&2
  exit 1
}

CONFIG_PATH="${1:-}"
[[ $# -eq 1 && -n "${CONFIG_PATH}" ]] || usage

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v yq >/dev/null 2>&1; then
  echo "yq is required to validate ${CONFIG_PATH}" >&2
  exit 1
fi

if yq --help 2>&1 | grep -qi 'jq wrapper'; then
  YQ_FLAVOR="jq-wrapper"
else
  YQ_FLAVOR="mikefarah"
fi

yq_parse() {
  case "${YQ_FLAVOR}" in
    jq-wrapper) yq '.' "$1" >/dev/null ;;
    mikefarah) yq eval '.' "$1" >/dev/null ;;
  esac
}

yq_macro_keys() {
  case "${YQ_FLAVOR}" in
    jq-wrapper) yq -r '.macros // {} | keys[]' "$1" ;;
    mikefarah) yq eval -r '.macros // {} | keys | .[]' "$1" ;;
  esac
}

yq_string_values() {
  case "${YQ_FLAVOR}" in
    jq-wrapper) yq -r '.. | strings' "$1" ;;
    mikefarah) yq eval -r '.. | select(tag == "!!str")' "$1" ;;
  esac
}

extract_references() {
  yq_string_values "$1" | awk '
    {
      text = $0
      while (match(text, /\$\{[^}]+\}/)) {
        print substr(text, RSTART + 2, RLENGTH - 3)
        text = substr(text, RSTART + RLENGTH)
      }
    }
  ' | sort -u
}

find_duplicate_macro_keys() {
  awk '
    BEGIN { in_macros = 0 }

    /^macros:[[:space:]]*($|#)/ {
      in_macros = 1
      next
    }

    in_macros && /^[^[:space:]#]/ {
      in_macros = 0
    }

    !in_macros {
      next
    }

    /^[[:space:]]*($|#)/ {
      next
    }

    /^  [A-Za-z0-9_.-]+:[[:space:]]*/ {
      key = $1
      sub(/:$/, "", key)
      count[key]++
    }

    END {
      for (key in count) {
        if (count[key] > 1) {
          print key
        }
      }
    }
  ' "$1" | sort
}

echo "+ yq parse ${CONFIG_PATH}"
yq_parse "${CONFIG_PATH}"

mapfile -t duplicate_macro_keys < <(find_duplicate_macro_keys "${CONFIG_PATH}")
if (( ${#duplicate_macro_keys[@]} > 0 )); then
  echo "duplicate macro keys in ${CONFIG_PATH}:" >&2
  printf '  - %s\n' "${duplicate_macro_keys[@]}" >&2
  exit 1
fi

declare -A macro_keys=()
while IFS= read -r key; do
  [[ -n "${key}" ]] || continue
  macro_keys["${key}"]=1
done < <(yq_macro_keys "${CONFIG_PATH}")

mapfile -t references < <(extract_references "${CONFIG_PATH}")
unresolved=()
for reference in "${references[@]}"; do
  [[ "${reference}" == "PORT" ]] && continue
  [[ "${reference}" == env.* ]] && continue
  [[ -n "${macro_keys[${reference}]:-}" ]] && continue
  unresolved+=("${reference}")
done

if (( ${#unresolved[@]} > 0 )); then
  echo "unresolved macro references in ${CONFIG_PATH}:" >&2
  printf '  - %s\n' "${unresolved[@]}" >&2
  exit 1
fi

echo "validated ${CONFIG_PATH}"