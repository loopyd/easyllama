FROM runtime-base AS runtime-python

WORKDIR /app
COPY pyproject.toml /app/
COPY easyllama/ /app/easyllama/
COPY run.sh /app/
RUN --mount=type=cache,id=llamacpp-pip-cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-deps /app \
    && /opt/venv/bin/pip install \
        colorama docker fastapi huggingface_hub jinja2 protobuf pydantic pycurl pyyaml \
        sentencepiece tqdm uvicorn \
    && chmod 755 /app/run.sh

