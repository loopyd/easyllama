# syntax=docker/dockerfile:1.7-labs
ARG CUDA_VERSION=13.1.0

# ── Download llama-swap binary (multi-model orchestrator) ───
FROM ubuntu:24.04 AS ls-download
ARG LS_VERSION=v208
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /install
RUN ARCH=$(dpkg --print-architecture) && \
    case "${ARCH}" in \
        amd64) A="amd64" ;; \
        arm64) A="arm64" ;; \
        *) echo "Unsupported arch: ${ARCH}"; exit 1 ;; \
    esac && \
    curl -fSL -o /tmp/ls.tar.gz "https://github.com/mostlygeek/llama-swap/releases/download/${LS_VERSION}/llama-swap_${LS_VERSION#v}_linux_${A}.tar.gz" && \
    tar xzf /tmp/ls.tar.gz -C /install/

# ── Shared builder base ────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder-base

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ARG LLAMA_CPP_REPO=https://github.com/Luce-Org/llama.cpp.git
ARG LLAMA_CPP_REF=luce-dflash
ARG TURBOQUANT_LLAMA_CPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG TURBOQUANT_LLAMA_CPP_REF=feature/turboquant-kv-cache
ARG SPIRITBUUN_LLAMA_CPP_REPO=https://github.com/spiritbuun/buun-llama-cpp.git
ARG SPIRITBUUN_LLAMA_CPP_REF=master
ARG LUCEBOX_HUB_REPO=https://github.com/Luce-Org/lucebox-hub.git
ARG LUCEBOX_HUB_REF=main
# Fallback only; run.sh auto-detects host GPU compute capability and overrides this.
ARG CMAKE_CUDA_ARCHITECTURES=120
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs
ENV CCACHE_DIR=/root/.cache/ccache
ENV TZ=${HOST_TZ}

RUN mkdir -p /usr/share/keyrings \
    && gpg --batch --no-default-keyring --keyring /etc/apt/trusted.gpg \
    --export A4B469963BF863CC > /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg \
    && printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /' \
    > /etc/apt/sources.list.d/cuda.list

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

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ENV TZ=${HOST_TZ}

RUN mkdir -p /usr/share/keyrings \
    && gpg --batch --no-default-keyring --keyring /etc/apt/trusted.gpg \
    --export A4B469963BF863CC > /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg \
    && printf '%s\n' \
    'deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /' \
    > /etc/apt/sources.list.d/cuda.list

RUN --mount=type=cache,id=llamacpp-apt-cache-runtime,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-runtime,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && DEBCONF_NOWARNINGS=yes apt-get install -y --no-install-recommends apt-utils \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    jq \
    python3 \
    python3-pip \
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

WORKDIR /app
COPY requirements.txt pyproject.toml /app/
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /app/requirements.txt
COPY easyllama/ /app/easyllama/
COPY run.sh /app/
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-deps /app \
    && chmod 755 /app/run.sh

ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:${PATH}
COPY --from=ls-download /install/llama-swap /app/bin/llama-swap
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8080/health >/dev/null || exit 1
ENTRYPOINT ["/app/run.sh"]
CMD ["serve"]

FROM builder-base AS basic-builder
ARG BUILD_MODE=basic
ARG LLAMA_CPP_REPO=https://github.com/Luce-Org/llama.cpp.git
ARG LLAMA_CPP_REF=luce-dflash
ARG CMAKE_CUDA_ARCHITECTURES=120
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "basic" ] || [ "${BUILD_MODE}" = "lucebox" ]; then \
        git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /src/llama.cpp \
        && cd /src/llama.cpp \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -j"$(nproc)" --target llama-server \
        && ccache --show-stats; \
    else \
        mkdir -p /src/llama.cpp/build/bin /src/llama.cpp/gguf-py /src/llama.cpp/models/templates \
        && : > /src/llama.cpp/convert_hf_to_gguf.py; \
    fi

FROM runtime-base AS runtime-basic
COPY --from=basic-builder /src/llama.cpp/build/bin/ /opt/llama.cpp-basic/bin/
COPY --from=basic-builder /src/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=basic-builder /src/llama.cpp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=basic-builder /src/llama.cpp/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-basic/bin/llama-server /app/bin/llama-server-basic
ENV LD_LIBRARY_PATH=/opt/llama.cpp-basic/bin:/usr/local/cuda/lib64

FROM builder-base AS turboquant-builder
ARG BUILD_MODE=basic
ARG TURBOQUANT_LLAMA_CPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG TURBOQUANT_LLAMA_CPP_REF=feature/turboquant-kv-cache
ARG CMAKE_CUDA_ARCHITECTURES=120
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "turboquant" ]; then \
        git clone --depth 1 --branch "${TURBOQUANT_LLAMA_CPP_REF}" "${TURBOQUANT_LLAMA_CPP_REPO}" /src/llama.cpp-turboquant \
        && cd /src/llama.cpp-turboquant \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CUDA_F16=ON \
        -DGGML_CUDA_FA_ALL_VARIANTS=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -j"$(nproc)" --target llama-server \
        && ccache --show-stats; \
    else \
        mkdir -p /src/llama.cpp-turboquant/build/bin /src/llama.cpp-turboquant/gguf-py /src/llama.cpp-turboquant/models/templates \
        && : > /src/llama.cpp-turboquant/convert_hf_to_gguf.py; \
    fi

FROM runtime-base AS runtime-turboquant
COPY --from=turboquant-builder /src/llama.cpp-turboquant/build/bin/ /opt/llama.cpp-turboquant/bin/
COPY --from=turboquant-builder /src/llama.cpp-turboquant/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=turboquant-builder /src/llama.cpp-turboquant/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=turboquant-builder /src/llama.cpp-turboquant/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-turboquant/bin/llama-server /app/bin/llama-server-turboquant
ENV LD_LIBRARY_PATH=/opt/llama.cpp-turboquant/bin:/usr/local/cuda/lib64

FROM builder-base AS spiritbuun-builder
ARG BUILD_MODE=basic
ARG SPIRITBUUN_LLAMA_CPP_REPO=https://github.com/spiritbuun/buun-llama-cpp.git
ARG SPIRITBUUN_LLAMA_CPP_REF=master
ARG CMAKE_CUDA_ARCHITECTURES=120
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "spiritbuun" ]; then \
        git clone --depth 1 --branch "${SPIRITBUUN_LLAMA_CPP_REF}" "${SPIRITBUUN_LLAMA_CPP_REPO}" /src/llama.cpp-spiritbuun \
        && cd /src/llama.cpp-spiritbuun \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -j"$(nproc)" --target llama-server \
        && ccache --show-stats; \
    else \
        mkdir -p /src/llama.cpp-spiritbuun/build/bin /src/llama.cpp-spiritbuun/gguf-py /src/llama.cpp-spiritbuun/models/templates \
        && : > /src/llama.cpp-spiritbuun/convert_hf_to_gguf.py; \
    fi

FROM runtime-base AS runtime-spiritbuun
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/build/bin/ /opt/llama.cpp-spiritbuun/bin/
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-spiritbuun/bin/llama-server /app/bin/llama-server-spiritbuun
ENV LD_LIBRARY_PATH=/opt/llama.cpp-spiritbuun/bin:/usr/local/cuda/lib64

FROM builder-base AS mtp-builder
ARG BUILD_MODE=basic
ARG MTP_LLAMA_CPP_REPO=https://github.com/am17an/llama.cpp.git
ARG MTP_LLAMA_CPP_REF=mtp-clean
ARG CMAKE_CUDA_ARCHITECTURES=120
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "mtp" ]; then \
        git clone --depth 1 --branch "${MTP_LLAMA_CPP_REF}" "${MTP_LLAMA_CPP_REPO}" /src/llama.cpp-mtp \
        && cd /src/llama.cpp-mtp \
        && mkdir -p build/bin gguf-py models/templates \
        && : > convert_hf_to_gguf.py; \
        cmake -B build \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CCACHE=ON \
        -DGGML_AVX=ON \
        -DGGML_AVX2=ON \
        -DGGML_F16C=ON \
        -DGGML_AVX512=OFF \
        -DGGML_FMA=ON \
        -DGGML_LTO=ON \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_USE_TURING_OPFMA=ON \
        -DGGML_CUDA_PEER=OFF \
        -DGGML_CUDA_NO_VULKAN_SHM=ON \
        -DGGML_RPC=OFF \
        -DLLAMA_CURL=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAVA_SERVER_ENABLE=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -j"$(nproc)" --target llama-server \
        && ccache --show-stats; \
    else \
        mkdir -p /src/llama.cpp-mtp/build/bin /src/llama.cpp-mtp/gguf-py /src/llama.cpp-mtp/models/templates \
        && : > /src/llama.cpp-mtp/convert_hf_to_gguf.py; \
    fi

FROM runtime-base AS runtime-mtp
COPY --from=mtp-builder /src/llama.cpp-mtp/build/bin/ /opt/llama.cpp-mtp/bin/
COPY --from=mtp-builder /src/llama.cpp-mtp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=mtp-builder /src/llama.cpp-mtp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=mtp-builder /src/llama.cpp-mtp/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-mtp/bin/llama-server /app/bin/llama-server-mtp
ENV LD_LIBRARY_PATH=/opt/llama.cpp-mtp/bin:/usr/local/cuda/lib64

FROM builder-base AS lucebox-builder
ARG BUILD_MODE=basic
ARG LUCEBOX_HUB_REPO=https://github.com/Luce-Org/lucebox-hub.git
ARG LUCEBOX_HUB_REF=main
ARG CMAKE_CUDA_ARCHITECTURES=120
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "lucebox" ]; then \
        git clone --depth 1 --branch "${LUCEBOX_HUB_REF}" "${LUCEBOX_HUB_REPO}" /src/lucebox-hub \
        && cd /src/lucebox-hub \
        && git submodule update --init --recursive --depth 1 \
        && cd /src/lucebox-hub/dflash \
        && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DDFLASH27B_ENABLE_BSA=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        && cmake --build build -j"$(nproc)" --target test_dflash test_flashprefill_kernels pflash_daemon \
        && ccache --show-stats; \
    else \
        mkdir -p /src/lucebox-hub/dflash/build /src/lucebox-hub/dflash/scripts; \
    fi

FROM runtime-base AS runtime-lucebox
COPY --from=basic-builder /src/llama.cpp/build/bin/ /opt/llama.cpp-basic/bin/
COPY --from=basic-builder /src/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=basic-builder /src/llama.cpp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=basic-builder /src/llama.cpp/models/templates/ /opt/llama.cpp/models/templates/
COPY --from=lucebox-builder /src/lucebox-hub/dflash/build/ /opt/lucebox/dflash/build/
COPY --from=lucebox-builder /src/lucebox-hub/dflash/scripts/ /opt/lucebox/dflash/scripts/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-basic/bin/llama-server /app/bin/llama-server-basic
ENV LD_LIBRARY_PATH=/opt/llama.cpp-basic/bin:/opt/lucebox/dflash/build:/opt/lucebox/dflash/build/deps/llama.cpp/ggml/src:/opt/lucebox/dflash/build/deps/llama.cpp/ggml/src/ggml-cuda:/usr/local/cuda/lib64
