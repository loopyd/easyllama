FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ENV TZ=${HOST_TZ}

RUN --mount=type=cache,id=llamacpp-apt-cache-runtime,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-runtime,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && DEBCONF_NOWARNINGS=yes apt-get install -y --no-install-recommends apt-utils \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    python3 \
    python3-venv \
    ca-certificates \
    libssl3 \
    tzdata \
    locales

RUN set -eux; \
    ln -snf "/usr/share/zoneinfo/${HOST_TZ}" /etc/localtime; \
    echo "${HOST_TZ}" > /etc/timezone; \
    locales_to_generate=''; \
    for locale_name in "${HOST_LANG}" "${HOST_LC_ALL}"; do \
        case "${locale_name}" in ''|C|C.UTF-8|POSIX) continue ;; esac; \
        locale_key="$(printf '%s' "${locale_name}" | tr '[:lower:]' '[:upper:]')"; \
        locale_entry="$(awk -v key="${locale_key}" '{ sub(/^#[[:space:]]*/, ""); if (toupper($1) == key) { print $1 " " $2; exit } }' /etc/locale.gen)"; \
        [ -n "${locale_entry}" ] || { echo "unsupported locale: ${locale_name}" >&2; exit 1; }; \
        sed -i "s/^# *${locale_entry}$/${locale_entry}/" /etc/locale.gen; \
        canonical_locale="${locale_entry% UTF-8}"; \
        case " ${locales_to_generate} " in \
            *" ${canonical_locale} "*) ;; \
            *) locales_to_generate="${locales_to_generate} ${canonical_locale}" ;; \
        esac; \
    done; \
    if [ -n "${locales_to_generate## }" ]; then \
        locale-gen ${locales_to_generate}; \
    fi

ENV LANG=${HOST_LANG}
ENV LC_ALL=${HOST_LC_ALL}

RUN python3 -m venv /opt/venv

ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:${PATH}
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8080/health >/dev/null || exit 1
ENTRYPOINT ["/app/run.sh"]
CMD ["serve"]
