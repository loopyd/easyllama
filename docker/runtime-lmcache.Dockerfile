FROM runtime-python AS runtime-lmcache
RUN --mount=type=cache,id=llamacpp-apt-cache-lmcache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=llamacpp-apt-lists-lmcache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc libc6-dev cuda-nvcc-13-0 libcublas-dev-13-0 libcurand-dev-13-0 python3-dev \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 lmcache==0.5.4 openai
ENV PATH=/opt/venv/bin:/usr/local/bin:/app/bin:${PATH}
