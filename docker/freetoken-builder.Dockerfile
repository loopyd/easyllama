# ── FreeToken MoE runtime builder ────────────────────────────────────────
# Installs FreeToken from the pinned repository into an isolated venv.
# FreeToken JIT-compiles its CUDA kernels on first use, so the final runtime
# stage merges the CUDA 13 compiler (nvcc) and ships this venv to the
# container. The venv stays out of /opt/venv so the shared easyllama runtime
# layer remains identical across modes.
# The builder also needs the CUDA toolkit headers to compile FreeToken's
# native extensions during the wheel build, so it reuses the same toolkit
# stage the final runtime merges.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS freetoken-cuda-toolkit

FROM runtime-python AS freetoken-builder
COPY --from=freetoken-cuda-toolkit /usr/local/cuda/. /usr/local/cuda/
ARG FREETOKEN_REPO=https://github.com/FlashML-org/FreeToken.git
ARG FREETOKEN_REF=main
# FreeToken's wheel build compiles native extensions through torch's C++
# extension toolchain, so the builder needs a C++ toolchain that matches the
# compiler torch was built with (Ubuntu's g++, not the cross-named driver).
ENV CC=gcc CXX=g++
RUN --mount=type=cache,id=freetoken-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=freetoken-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=freetoken-uv-cache,target=/root/.cache/uv,sharing=locked \
    apt-get update \
    && DEBCONF_NOWARNINGS=yes apt-get install -y --no-install-recommends \
        git build-essential ninja-build python3-dev \
    && python3 -m venv /opt/ft-venv \
    && /opt/ft-venv/bin/pip install --upgrade pip uv \
    && git clone --depth 1 --branch "${FREETOKEN_REF}" "${FREETOKEN_REPO}" /src/FreeToken \
    && /opt/ft-venv/bin/uv pip install --python /opt/ft-venv/bin/python "/src/FreeToken[accel]" \
    && /opt/ft-venv/bin/ft --version
