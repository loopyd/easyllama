FROM builder-base AS vllm-builder
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.29.0
ARG BUILD_JOBS=1
# RTX 5090 / Blackwell only: avoid compiling legacy SM75-SM110 kernels.
ENV VLLM_TARGET_DEVICE=cuda
# vLLM derives each component's family-specific gencode from this list.
# Do not also set CMAKE_CUDA_ARCHITECTURES/CUDAARCHS: that duplicates sm_120
# beside sm_120f for Blackwell-family kernels and nvcc rejects the build.
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV MAX_JOBS=${BUILD_JOBS}
ENV NVCC_THREADS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
RUN --mount=type=cache,id=vllm-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,id=vllm-ccache,target=/root/.cache/ccache,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv ninja-build ccache \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_REPO}" /src/vllm \
    && cd /src/vllm \
    && export CC="gcc" CXX="g++" CUDAHOSTCXX="g++" \
    # Model inspection imports a bundled flash-attention extension. Build FA2 \
    # retargeted by TORCH_CUDA_ARCH_LIST to SM120; skip Hopper-specific FA3. \
    && sed -i 's|        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))|        pass  # Skip Hopper-specific FA3 kernels|' setup.py \
    && python3 -m venv /opt/vllm-build \
    && /opt/vllm-build/bin/pip install --upgrade pip wheel \
    && /opt/vllm-build/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 \
    && /opt/vllm-build/bin/pip install -r requirements/build/cuda.txt \
    && /opt/vllm-build/bin/python setup.py bdist_wheel --dist-dir /dist \
    && ccache --show-stats
