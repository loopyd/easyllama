FROM runtime-python AS runtime-llamacpp-spiritbuun
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/build/bin/ /opt/llama.cpp-spiritbuun/bin/
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=spiritbuun-builder /src/llama.cpp-spiritbuun/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-spiritbuun/bin/llama-server /app/bin/llama-server-spiritbuun
ENV LD_LIBRARY_PATH=/opt/llama.cpp-spiritbuun/bin:/usr/local/cuda/lib64
