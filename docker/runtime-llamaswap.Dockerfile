FROM runtime-python AS runtime-llamaswap
COPY --from=vllm-wrapper-build /install/llama-swap /app/bin/llama-swap
