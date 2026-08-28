FROM runtime-python AS runtime-llamacpp-lucebox
COPY --from=llamacpp-builder /src/llama.cpp/build/bin/ /opt/llama.cpp-llamacpp/bin/
COPY --from=llamacpp-builder /src/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=llamacpp-builder /src/llama.cpp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=llamacpp-builder /src/llama.cpp/models/templates/ /opt/llama.cpp/models/templates/
COPY --from=lucebox-builder /src/lucebox-hub/dflash/build/ /opt/lucebox/dflash/build/
COPY --from=lucebox-builder /src/lucebox-hub/dflash/scripts/ /opt/lucebox/dflash/scripts/
RUN /opt/venv/bin/pip install transformers \
    && mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-llamacpp/bin/llama-server /app/bin/llama-server-llamacpp
ENV LD_LIBRARY_PATH=/opt/llama.cpp-llamacpp/bin:/opt/lucebox/dflash/build:/opt/lucebox/dflash/build/deps/llama.cpp/ggml/src:/opt/lucebox/dflash/build/deps/llama.cpp/ggml/src/ggml-cuda:/usr/local/cuda/lib64
