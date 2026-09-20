FROM runtime-base AS runtime-python-deps

WORKDIR /app
# Third-party runtime dependencies only. Keeping the application source out of
# this stage means editing easyllama/ never invalidates this layer, and lets
# heavy backends (vLLM) base their build on it without dragging the app into
# their cache key.
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install \
        colorama docker fastapi huggingface_hub jinja2 protobuf pydantic pycurl pyyaml \
        sentencepiece tqdm uvicorn

FROM runtime-python-deps AS runtime-python

WORKDIR /app
COPY pyproject.toml /app/
COPY easyllama/ /app/easyllama/
COPY run.sh /app/
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-deps /app \
    && chmod 755 /app/run.sh
