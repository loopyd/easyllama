FROM ubuntu:24.04 AS ls-download
ARG LS_VERSION=v251
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /install
RUN ARCH=$(dpkg --print-architecture) && \
    case "${ARCH}" in \
        amd64) A="amd64" ;; \
        arm64) A="arm64" ;; \
        *) echo "Unsupported arch: ${ARCH}"; exit 1 ;; \
    esac && \
    curl -fSL -o /tmp/ls.tar.gz "https://github.com/mostlygeek/llama-swap/releases/download/${LS_VERSION}/llama-swap_${LS_VERSION#v}_linux_${A}.tar.gz" && \
    tar xzf /tmp/ls.tar.gz -C /install/

# vllm-wrapper is an upstream llama-swap command but is not in release tarballs.
