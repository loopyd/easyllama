# easy llama(cpp)

GPU-focused llama.cpp runner that uses llama-swap as the only runtime entrypoint.

## Runtime Model

- build a CUDA image with your chosen llama.cpp repo/ref
- run a single swap container that serves multiple models from config.yaml
- keep credentials in auth.json (Hugging Face token + optional local API key)

Single-model start/stop/serve paths were removed on purpose.

## Requirements

- Docker with daemon running
- NVIDIA container runtime in Docker
- NVIDIA driver and nvidia-smi on host
- jq (for auth/config parsing)

## Commands

| Command | Description |
| --- | --- |
| ./run.sh build | Build local CUDA image from Dockerfile |
| ./run.sh swap | Start llama-swap container |
| ./run.sh stop-swap | Stop and remove llama-swap container |
| ./run.sh restart-swap | Restart llama-swap container |
| ./run.sh status-swap | Show llama-swap container status |
| ./run.sh logs-swap | Follow llama-swap logs |
| ./run.sh clean | Remove llama-swap container and image |

## Quick Start

1. Build image.

```bash
./run.sh build
```

1. Start swap runtime.

The chat projector is configured directly in config.yaml and mounted from the local mmproj directory.
1. Verify.

```bash
./run.sh status-swap
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/models | jq '.data[].id'
```

## Credentials

Create auth.json from auth.json.example and set your credentials:

```json
{
  "hf_token": "hf_...",
  "api_key": "your-local-endpoint-key"
}
```

Precedence for token resolution:

1. HF_TOKEN
2. LLAMACPP_HF_TOKEN
3. auth.json (or LLAMACPP_AUTH_FILE)
4. auth.json.example

Precedence for local endpoint API key:

1. LLAMACPP_API_KEY
2. API_KEY
3. auth.json api_key (or LLAMACPP_AUTH_FILE)

When api_key is set, run.sh generates an effective llama-swap config with top-level apiKeys enabled, so /v1/* endpoints require Authorization: Bearer <api_key> or x-api-key.

## mmproj Integration

Optional mmproj can be set with:

- LLAMACPP_MMPROJ_FILE for local path, mounted path, or URL
- LLAMACPP_HF_MMPROJ using owner/repo/file.gguf shorthand

run.sh resolves the projector, auto-downloads URLs into mmproj/, and exports LLAMACPP_MMPROJ_ARG into the swap container; config.yaml consumes it through a macro.

## Further Docs

See README-ls.md for API usage, model behavior, and swap-specific troubleshooting.

## License

GPL-3.0-only. See LICENSE.
