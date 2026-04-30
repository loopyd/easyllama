# Multi-Model Serving with llama-swap

This document describes how to run **multiple models** simultaneously using [llama-swap](https://github.com/mostlygeek/llama-swap) as an orchestrator proxy on top of your custom turboquant llama.cpp build.

## Architecture

```
Client requests → Port 8080
                    │
              ┌─────┴──────┐
              │ llama-swap   │  ← Go binary, zero-dependency proxy
              │ (orchestrator)│     Auto-spawns/stops upstream servers
              └─────┬──────┘         per incoming model request
                    │
          ┌─────────┼─────────┐
          ▼                     ▼
   ┌─────────────┐     ┌──────────────┐
   │ llama-server │     │ llama-server  │
   │ (qwen3-chat) │     │ (embeddings)  │
   │ turbo4 KV    │     │ dense pool    │
   │ port dynamic │     │ port dynamic  │
   └─────────────┘     └──────────────┘
```

### What gets served?

| Endpoint | Model ID | Underlying Model | Purpose |
|---|---|---|---|
| `/v1/chat/completions`, `/v1/completions`, etc. | `qwen3-chat` | HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive Q5_K_P | Chat & reasoning |
| `/v1/embeddings` | `qwen3-embeddings` | Qwen/Qwen3-Embedding-8B-GGUF Q5_K_M | Dense embeddings for RAG/search |

Both use your **turboquant fork** (`TheTom/llama-cpp-turboquant`) — the same one you already build from.

## Quick Start

### 1. Build (one-time, or after config changes)

```bash
./run.sh build
```

This builds your CUDA image including both `llama-server` (from turboquant source) and `llama-swap` (pre-built binary).

### 2. Start multi-model mode

```bash
./run.sh swap
```

This starts a Docker container named `llamacpp-server-swap` running llama-swap on port 8080.

### 3. Verify

```bash
# List available models
curl -s http://localhost:8080/v1/models | jq '.data[].id'

# Expected output:
# "qwen3-chat"
# "qwen3-embeddings"

# Health check
curl -s http://localhost:8080/health
```

## API Usage

### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-chat",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embeddings",
    "input": ["Hello world", "Another document"]
  }'
```

### Other Endpoints

All standard OpenAI-compatible endpoints work through llama-swap:

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion (chat model) |
| `POST /v1/completions` | Legacy completions (chat model) |
| `POST /v1/embeddings` | Embedding generation |
| `POST /v1/responses` | Responses API (chat model) |
| `GET /health` | Health check |
| `GET /ui` | Web UI (playground & monitoring) |

## Commands Reference

| Command | Description |
|---|---|
| `./run.sh swap` | Start llama-swap container |
| `./run.sh stop-swap` | Stop the llama-swap container |
| `./run.sh restart-swap` | Restart (stop + start) |
| `./run.sh logs-swap` | Follow live logs |
| `./run.sh status-swap` | Show container status |

These are the primary runtime commands. Legacy single-model start/stop commands were removed.

## Configuration

### config.yaml

The file `config.yaml` defines all models and their inference parameters. It uses YAML macros to share common settings across models. Key sections:

- **Global settings**: health timeouts, log level, TTL policies
- **Macros**: reusable parameter blocks and optional runtime args (for example mmproj)
- **Models**: each entry defines a complete server command with its own port assignment

To customize, edit `config.yaml` directly or set:

```bash
LLAMACPP_LS_CONFIG_FILE=/path/to/custom.yaml ./run.sh swap
```

### HF_TOKEN Handling

Your Hugging Face token is resolved from `auth.json` (`hf_token`) and passed as an environment variable to the container. The embedding model downloads automatically on first request via `-hf Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M`.

If you need a different token for swap mode:

```bash
HF_TOKEN=hf_xxxx ./run.sh swap
```

### Model Swap Behavior

By default, llama-swap runs **one upstream model at a time**. When a request arrives for a model that isn't currently loaded:

1. If no model is running → start the requested one immediately
2. If another model is running → unload it, then load the new one
3. All waiting requests queue until the correct model is ready

For concurrent multi-model serving (both chat AND embeddings simultaneously), add a `matrix:` block to `config.yaml`:

```yaml
matrix:
  vars:
    c: qwen3-chat
    e: qwen3-embeddings
  sets:
    # Allow both to run together if GPU memory permits
    dual: "c & e"
```

> ⚠️ With a ~27B param chat model + 8B embedding model, you'll need significant VRAM (~24GB+) to run both concurrently in Q5 quantization.

## Web UI

llama-swap includes a built-in web interface:

```
http://localhost:8080/ui
```

Features:
- Real-time playground for testing models
- Token metrics & performance stats
- Request/response inspection
- Manual model load/unload controls
- Live log streaming

## Troubleshooting

### "No such file or directory: /app/bin/llama-swap"

Rebuild the image after adding llama-swap support:

```bash
./run.sh clean && ./run.sh build
```

### Embedding model slow to appear on first use

The first `/v1/embeddings` request triggers a download of `Qwen3-Embedding-8B-Q5_K_M.gguf` (~4.6 GB). Subsequent loads are cached under your `models/` directory. Monitor with:

```bash
./run.sh logs-swap | grep -i "download\|loading"
```

### Port conflicts

If port 8080 is already in use:

```bash
# Or use a different port for swap mode
LLAMACPP_HOST_PORT=8090 ./run.sh swap
```

### TurboQuant features not available

Ensure your Dockerfile still references the turboquant fork:

```json
{
  "build": {
    "llama_cpp_repo": "https://github.com/TheTom/llama-cpp-turboquant.git",
    "llama_cpp_ref": "feature/turboquant-kv-cache"
  }
}
```

Then rebuild: `./run.sh build`

## Differences from Single-Model Mode

| Aspect | Legacy single server | Multi (`swap`) |
|---|---|---|
| Container name | Removed | `llamacpp-server-swap` |
| Config source | Removed | `config.yaml` (YAML) + `auth.json` |
| Models served | Removed | Multiple, auto-swapped |
| Inference params | Removed | Defined in config.yaml cmd blocks |
| Chat template | Auto-mounted from host | Referenced as `/chat_template/qwen3.6.jinja` inside container |
| Warmup | Per-model at spawn time | Per-model at first request |
