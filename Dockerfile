# syntax=docker/dockerfile:1.7-labs
ARG CUDA_VERSION=13.1.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=master
# Fallback only; run.sh auto-detects host GPU compute capability and overrides this.
ARG CMAKE_CUDA_ARCHITECTURES=120
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs
ENV TZ=${HOST_TZ}
ENV LANG=${HOST_LANG}
ENV LC_ALL=${HOST_LC_ALL}
ENV CCACHE_DIR=/root/.cache/ccache

RUN --mount=type=cache,id=llamacpp-apt-cache-builder,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-builder,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    ccache \
    ca-certificates \
    libssl-dev \
    tzdata \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN ln -snf "/usr/share/zoneinfo/${HOST_TZ}" /etc/localtime \
    && echo "${HOST_TZ}" > /etc/timezone \
    && grep -q "^${HOST_LANG} UTF-8" /etc/locale.gen || echo "${HOST_LANG} UTF-8" >> /etc/locale.gen \
    && locale-gen "${HOST_LANG}"

WORKDIR /app
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /app
RUN ln -sf "${CUDA_STUBS}/libcuda.so" "${CUDA_STUBS}/libcuda.so.1"

RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    cmake -B build \
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
    && ccache --show-stats

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HOST_TZ=UTC
ARG HOST_LANG=C.UTF-8
ARG HOST_LC_ALL=C.UTF-8
ENV TZ=${HOST_TZ}
ENV LANG=${HOST_LANG}
ENV LC_ALL=${HOST_LC_ALL}

RUN --mount=type=cache,id=llamacpp-apt-cache-runtime,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-runtime,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    jq \
    ca-certificates \
    libssl3 \
    tzdata \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN ln -snf "/usr/share/zoneinfo/${HOST_TZ}" /etc/localtime \
    && echo "${HOST_TZ}" > /etc/timezone \
    && grep -q "^${HOST_LANG} UTF-8" /etc/locale.gen || echo "${HOST_LANG} UTF-8" >> /etc/locale.gen \
    && locale-gen "${HOST_LANG}"

WORKDIR /app
COPY --from=builder /app/build/bin/ /app/bin/
COPY run.sh /app/run.sh
COPY config.json.example /app/config.json.example
RUN chmod 755 /app/run.sh
ENV LD_LIBRARY_PATH=/app/bin:/usr/local/cuda/lib64

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8080/health >/dev/null || exit 1

ENTRYPOINT ["/app/run.sh"]
CMD ["serve"]
