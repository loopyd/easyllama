# The freetoken-cuda-toolkit stage (defined with the builder, before this one)
# supplies nvcc and the CUDA 13 libraries: FreeToken's CUDA kernels are
# JIT-compiled on first use and need nvcc on PATH at runtime, which the shared
# runtime-python base (CUDA runtime only) does not ship.
FROM runtime-python AS runtime-freetoken
COPY --from=freetoken-cuda-toolkit /usr/local/cuda/. /usr/local/cuda/
COPY --from=freetoken-builder /opt/ft-venv /opt/ft-venv
ENV PATH=/opt/ft-venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
