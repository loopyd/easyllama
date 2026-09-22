FROM runtime-python AS runtime-llamacpp-bonsai
COPY --from=bonsai-builder /src/llama.cpp-bonsai/build/bin/ /opt/llama.cpp-bonsai/bin/
COPY --from=bonsai-builder /src/llama.cpp-bonsai/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=bonsai-builder /src/llama.cpp-bonsai/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=bonsai-builder /src/llama.cpp-bonsai/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-bonsai/bin/llama-server /app/bin/llama-server-bonsai
ENV LD_LIBRARY_PATH=/opt/llama.cpp-bonsai/bin:/usr/local/cuda/lib64
