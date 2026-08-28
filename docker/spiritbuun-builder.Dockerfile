FROM builder-base AS spiritbuun-builder
ARG BUILD_MODE=llamacpp
ARG LLAMA_CPP_REPO=https://github.com/spiritbuun/buun-llama-cpp.git
ARG LLAMA_CPP_REF=master
ARG CMAKE_CUDA_ARCHITECTURES=120
ARG BUILD_JOBS=1
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "spiritbuun" ]; then \
        git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /src/llama.cpp-spiritbuun \
        && cd /src/llama.cpp-spiritbuun \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
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
        mkdir -p /src/llama.cpp-spiritbuun/build/bin /src/llama.cpp-spiritbuun/gguf-py /src/llama.cpp-spiritbuun/models/templates \
        && : > /src/llama.cpp-spiritbuun/convert_hf_to_gguf.py; \
    fi
