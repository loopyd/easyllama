# API

easy llama(cpp) exposes one `llama-swap` endpoint:

- Base URL: `http://127.0.0.1:8080`
- Model discovery: `GET /v1/models`
- Health check: `GET /health`

For setup and runtime flow, see [README.md](README.md).

## Contents

- [Authentication](#authentication)
- [Default model IDs](#default-model-ids)
- [`qwen3-chat` default by mode](#qwen3-chat-default-by-mode)
- [Endpoint matrix](#endpoint-matrix)
- [Fast smoke tests](#fast-smoke-tests)
- [Which endpoint to use](#which-endpoint-to-use)

## Authentication

If API key protection is enabled, define auth header once:

```bash
API_KEY="$(jq -r '.api_key // empty' auth.json)"
AUTH=()
if [[ -n "${API_KEY}" ]]; then
  AUTH=(-H "Authorization: Bearer ${API_KEY}")
fi
```

If `api_key` is absent, `AUTH` stays empty and examples still work.

## Default model IDs

These stable IDs are exposed through `/v1/models`.

| Model ID | Purpose | Default source |
| --- | --- | --- |
| `qwen3-chat` | Primary chat and generation model | mode-dependent |
| `qwen3-embeddings` | Dense embeddings | `Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M` |

### `qwen3-chat` default by mode

| Mode | Default |
| --- | --- |
| `basic` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` |
| `turboquant` | `unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` |
| `qwen` | `RadixArk/Qwen3.8-27B-NVFP4` (raw ModelOpt weights served by vLLM without speculative decoding, 128K configured context, and FP8 KV cache) |
| `spiritbuun` | target `unsloth/Qwen3.6-27B-GGUF:Q5_K_M`, draft `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` |
| `lucebox` | target `unsloth/Qwen3.6-27B-GGUF:Q4_K_M`, draft `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` |

## Endpoint matrix

Read this table first if choosing route by task or by mode.

| Endpoint | `basic` | `turboquant` | `qwen` | `spiritbuun` | `lucebox` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `GET /health` | ✅ | ✅ | ✅ | ✅ | ✅ | Plain-text health check |
| `GET /v1/models` | ✅ | ✅ | ✅ | ✅ | ✅ | Lists configured model IDs |
| `POST /v1/chat/completions` | ✅ | ✅ | ✅ | ✅ | ✅ | Main OpenAI-compatible chat route |
| `POST /v1/messages` | ❌ | ❌ | ❌ | ❌ | ✅ | Anthropic-style messages route |
| `POST /v1/completions` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-chat` |
| `POST /v1/responses` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-chat` |
| `POST /v1/embeddings` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-embeddings` |
| `GET /ui/` | ✅ | ✅ | ✅ | ✅ | ✅ | Built-in `llama-swap` UI |

Important:

- `POST /v1/messages` is specific to `lucebox`; Spiritbuun launches its upstream llama-server without a project-owned messages adapter.
- No reranking model is shipped by the default profiles.

## Fast smoke tests

Use these after startup, rebuild, config edits, or backend changes.

### Health

```bash
curl -sS http://127.0.0.1:8080/health
```

### List models

```bash
curl -sS "${AUTH[@]}" http://127.0.0.1:8080/v1/models | jq -r '.data[].id'
```

### Chat completion

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-chat",
    "messages": [
      {"role": "user", "content": "Reply with exactly ok."}
    ],
    "max_tokens": 16,
    "stream": false
  }' \
  http://127.0.0.1:8080/v1/chat/completions | jq
```

### Messages (`lucebox` only)

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-chat",
    "messages": [
      {"role": "user", "content": "Reply with exactly ok."}
    ],
    "max_tokens": 16,
    "stream": false
  }' \
  http://127.0.0.1:8080/v1/messages | jq
```

### Responses API

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-chat",
    "input": "Reply with exactly ok.",
    "max_output_tokens": 16
  }' \
  http://127.0.0.1:8080/v1/responses | jq
```

### Embeddings

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embeddings",
    "input": "local llama embeddings smoke test"
  }' \
  http://127.0.0.1:8080/v1/embeddings | jq
```

## Which endpoint to use

Shortest route map for common client tasks.

| Goal | Endpoint | Model |
| --- | --- | --- |
| Chat | `POST /v1/chat/completions` | `qwen3-chat` |
| Messages-style chat (`lucebox`) | `POST /v1/messages` | `qwen3-chat` |
| Plain completion / rewrite | `POST /v1/completions` or `POST /v1/responses` | `qwen3-chat` |
| Embeddings | `POST /v1/embeddings` | `qwen3-embeddings` |
