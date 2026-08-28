FROM builder-base AS turboquant-builder
ARG BUILD_MODE=llamacpp
ARG LLAMA_CPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG LLAMA_CPP_REF=feature/turboquant-kv-cache
ARG CMAKE_CUDA_ARCHITECTURES=120
ARG BUILD_JOBS=1
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "turboquant" ]; then \
        git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /src/llama.cpp-turboquant \
        && cd /src/llama.cpp-turboquant \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CUDA_F16=ON \
        -DGGML_CUDA_FA_ALL_VARIANTS=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build --config Release -j"${BUILD_JOBS}" --target llama-server \
        && ccache --show-stats; \
    else \
        mkdir -p /src/llama.cpp-turboquant/build/bin /src/llama.cpp-turboquant/gguf-py /src/llama.cpp-turboquant/models/templates \
        && : > /src/llama.cpp-turboquant/convert_hf_to_gguf.py; \
    fi
