FROM runtime-python AS runtime-llamacpp-turboquant
COPY --from=turboquant-builder /src/llama.cpp-turboquant/build/bin/ /opt/llama.cpp-turboquant/bin/
COPY --from=turboquant-builder /src/llama.cpp-turboquant/convert_hf_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py
COPY --from=turboquant-builder /src/llama.cpp-turboquant/gguf-py/ /opt/llama.cpp/gguf-py/
COPY --from=turboquant-builder /src/llama.cpp-turboquant/models/templates/ /opt/llama.cpp/models/templates/
RUN mkdir -p /app/bin \
    && ln -sf /opt/llama.cpp-turboquant/bin/llama-server /app/bin/llama-server-turboquant
ENV LD_LIBRARY_PATH=/opt/llama.cpp-turboquant/bin:/usr/local/cuda/lib64
