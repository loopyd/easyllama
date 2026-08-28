FROM runtime-python AS runtime-llamacpp
COPY --from=llamacpp-builder /src/llama.cpp/build/bin/ /opt/llama.cpp-llamacpp/bin/
COPY --from=llamacpp-builder /src/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=llamacpp-builder /src/llama.cpp/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=llamacpp-builder /src/llama.cpp/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-llamacpp/bin/llama-server /app/bin/llama-server-llamacpp
ENV LD_LIBRARY_PATH=/opt/llama.cpp-llamacpp/bin:/usr/local/cuda/lib64
