FROM golang:alpine AS vllm-wrapper-build
ARG LS_VERSION=v251
RUN apk add --no-cache git \
    && git clone --depth 1 --branch "${LS_VERSION}" https://github.com/mostlygeek/llama-swap.git /src/llama-swap \
    && cd /src/llama-swap \
    # Health polling expects connection refusals while models load; keep the 502 but suppress Go's noisy dial log.
    && sed -i '/\/\/ httputil.ReverseProxy panics/i\	reverseProxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, _ error) { http.Error(w, "upstream unavailable", http.StatusBadGateway) }\n' internal/process/process_command.go \
    && grep -q 'upstream unavailable' internal/process/process_command.go \
    # Preload only needs readiness; vLLM intentionally returns 404 at GET /.
    && sed -i 's|http.MethodGet, "/", nil|http.MethodGet, "/health", nil|' internal/server/api.go \
    && grep -q 'http.MethodGet, "/health", nil' internal/server/api.go \
    && CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /install/llama-swap .

# ── Shared builder base ────────────────────────────────────
