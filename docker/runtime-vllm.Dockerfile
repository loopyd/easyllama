FROM runtime-python AS runtime-vllm
COPY --from=vllm-builder /dist/ /tmp/vllm-dist/
RUN --mount=type=cache,id=llamacpp-apt-cache-vllm,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-vllm,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc libc6-dev cuda-nvcc-13-0 libcublas-dev-13-0 libcurand-dev-13-0 python3-dev \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 \
    && /opt/venv/bin/pip install /tmp/vllm-dist/*.whl lmcache==0.5.4 \
    && rm -rf /tmp/vllm-dist
ENV PATH=/opt/venv/bin:/usr/local/bin:/app/bin:${PATH}
