FROM runtime-python AS runtime-llamacpp-qwen
COPY --from=qwen-builder /src/llama.cpp-qwen/build/bin/ /opt/llama.cpp-qwen/bin/
COPY --from=qwen-builder /src/llama.cpp-qwen/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=qwen-builder /src/llama.cpp-qwen/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=qwen-builder /src/llama.cpp-qwen/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-qwen/bin/llama-server /app/bin/llama-server-qwen
ENV LD_LIBRARY_PATH=/opt/llama.cpp-qwen/bin:/usr/local/cuda/lib64
