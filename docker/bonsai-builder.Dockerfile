FROM builder-base AS bonsai-builder
ARG BUILD_MODE=llamacpp
ARG LLAMA_CPP_REPO=https://github.com/PrismML-Eng/llama.cpp.git
ARG LLAMA_CPP_REF=88c4bc60b9c9578f134385be9535e853f2db9b9f
ARG CMAKE_CUDA_ARCHITECTURES=120
ARG BUILD_JOBS=1
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "bonsai" ]; then \
        git init /src/llama.cpp-bonsai \
        && cd /src/llama.cpp-bonsai \
        && git remote add origin "${LLAMA_CPP_REPO}" \
        && git fetch --depth 1 origin "${LLAMA_CPP_REF}" \
        && git checkout --detach FETCH_HEAD \
        && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
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
        mkdir -p /src/llama.cpp-bonsai/build/bin /src/llama.cpp-bonsai/gguf-py /src/llama.cpp-bonsai/models/templates \
        && : > /src/llama.cpp-bonsai/convert_hf_to_gguf.py; \
    fi
