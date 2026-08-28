FROM builder-base AS lucebox-builder
ARG BUILD_MODE=llamacpp
ARG LUCEBOX_HUB_REPO=https://github.com/Luce-Org/lucebox-hub.git
ARG LUCEBOX_HUB_REF=main
ARG CMAKE_CUDA_ARCHITECTURES=120
ARG BUILD_JOBS=1
RUN --mount=type=cache,id=llamacpp-ccache,target=/root/.cache/ccache,sharing=locked \
    if [ "${BUILD_MODE}" = "lucebox" ]; then \
        git clone --depth 1 --branch "${LUCEBOX_HUB_REF}" "${LUCEBOX_HUB_REPO}" /src/lucebox-hub \
        && cd /src/lucebox-hub \
        && git submodule update --init --recursive --depth 1 \
        && cd /src/lucebox-hub/dflash \
        && cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DDFLASH27B_ENABLE_BSA=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${CUDA_STUBS}" \
        && cmake --build build -j"${BUILD_JOBS}" --target test_dflash test_flashprefill_kernels pflash_daemon \
        && ccache --show-stats; \
    else \
        mkdir -p /src/lucebox-hub/dflash/build /src/lucebox-hub/dflash/scripts; \
    fi
