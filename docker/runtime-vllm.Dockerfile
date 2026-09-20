FROM runtime-python-deps AS runtime-vllm-deps
# Torch and the CUDA build headers are independent of our vLLM wheel, so they
# are installed before the wheel is copied in: rebuilding vLLM then keeps this
# multi-GB layer cached. The pip cache mount makes even an invalidated install
# a cache hit instead of a re-download.
RUN --mount=type=cache,id=vllm-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-cache-vllm,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-vllm,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc libc6-dev cuda-nvcc-13-0 libcublas-dev-13-0 libcurand-dev-13-0 python3-dev \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0
COPY --from=vllm-builder /dist/ /tmp/vllm-dist/
RUN --mount=type=cache,id=vllm-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install /tmp/vllm-dist/*.whl lmcache==0.5.5 \
    && rm -rf /tmp/vllm-dist
# flashinfer's JIT build (vLLM's one-time MoE kernel warmup) compiles and
# links against NVRTC, but the CUDA runtime image ships no NVRTC dev files:
# nvrtc.h lives in the pip nvidia/cu13 package, and only versioned
# libnvrtc.so.* files exist in the CUDA lib dir. Expose both on the standard
# CUDA paths so the first-run JIT build (tracked in the host jit_cache
# volume, so it happens exactly once) can compile and link.
RUN set -eux; \
    hdr_dir=$(find /opt/venv/lib -type d -path '*/nvidia/cu13/include' | head -n1); \
    [ -n "$hdr_dir" ]; \
    cp -n "$hdr_dir"/*.h /usr/local/cuda/include/; \
    so=$(ls /usr/local/cuda*/targets/x86_64-linux/lib/libnvrtc.so.* 2>/dev/null | sort -V | tail -n1); \
    [ -n "$so" ]; \
    ln -sf "$(readlink -f "$so")" /usr/local/cuda/lib64/libnvrtc.so
ENV PATH=/opt/venv/bin:/usr/local/bin:/app/bin:${PATH}

FROM runtime-vllm-deps AS runtime-vllm
# The application source is the fastest-moving input, so it is installed last:
# an easyllama/ edit only rebuilds this small layer, never the torch/vLLM one.
WORKDIR /app
COPY pyproject.toml /app/
COPY easyllama/ /app/easyllama/
COPY run.sh /app/
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-deps /app \
    && chmod 755 /app/run.sh
