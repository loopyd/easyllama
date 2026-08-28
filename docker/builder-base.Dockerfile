FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder-base

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ARG LLAMA_CPP_REPO=https://github.com/Luce-Org/llama.cpp.git
ARG LLAMA_CPP_REF=luce-dflash
ARG LUCEBOX_HUB_REPO=https://github.com/Luce-Org/lucebox-hub.git
ARG LUCEBOX_HUB_REF=main
# Fallback only; run.sh auto-detects host GPU compute capability and overrides this.
ARG CMAKE_CUDA_ARCHITECTURES=120
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs
ENV CCACHE_DIR=/root/.cache/ccache
ENV CCACHE_COMPRESS=true
ENV CCACHE_MAXSIZE=20G
ENV TZ=${HOST_TZ}

RUN --mount=type=cache,id=llamacpp-apt-cache-builder,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-builder,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && DEBCONF_NOWARNINGS=yes apt-get install -y --no-install-recommends apt-utils \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    ccache \
    ca-certificates \
    libssl-dev \
    tzdata \
    locales

RUN set -eux; \
    ln -snf "/usr/share/zoneinfo/${HOST_TZ}" /etc/localtime; \
    echo "${HOST_TZ}" > /etc/timezone; \
    locales_to_generate=''; \
    for locale_name in "${HOST_LANG}" "${HOST_LC_ALL}"; do \
        case "${locale_name}" in ''|C|C.UTF-8|POSIX) continue ;; esac; \
        grep -Fqx "${locale_name} UTF-8" /etc/locale.gen \
            || printf '%s UTF-8\n' "${locale_name}" >> /etc/locale.gen; \
        case " ${locales_to_generate} " in \
            *" ${locale_name} "*) ;; \
            *) locales_to_generate="${locales_to_generate} ${locale_name}" ;; \
        esac; \
    done; \
    if [ -n "${locales_to_generate## }" ]; then \
        locale-gen ${locales_to_generate}; \
    fi

ENV LANG=${HOST_LANG}
ENV LC_ALL=${HOST_LC_ALL}

WORKDIR /src
RUN ln -sf "${CUDA_STUBS}/libcuda.so" "${CUDA_STUBS}/libcuda.so.1"
