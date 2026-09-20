# API

easy llama(cpp) exposes one `llama-swap` endpoint:

- Base URL: `${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}`
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
API_KEY="$(jq -r '.credentials.api_key // empty' config.json 2>/dev/null)"
AUTH=()
if [[ -n "${API_KEY}" ]]; then
  AUTH=(-H "Authorization: Bearer ${API_KEY}")
fi
```

If `credentials.api_key` is absent, `AUTH` stays empty and examples still work. Set `EASYLLAMA_BASE_URL` when `runtime.host` or `runtime.host_port` differs from the defaults.

## Default model IDs

These stable IDs are exposed through `/v1/models`.

| Model ID | Purpose | Default source |
| --- | --- | --- |
| `qwen3-chat` | Primary chat and generation model | mode-dependent |
| `qwen3-embeddings` | Dense embeddings | `qwen`: `Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-f16.gguf`; other modes: `Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M` |
| `glm53-chat` | GLM-5.3 Flash 320B MoE chat via FreeToken | `glm5.3-flash` mode only: `RedHatAI/GLM-5.3-Flash-NVFP4` |

In `qwen` mode, embeddings have 1,024 dimensions and a 32,768-token context per
slot. Upgrading from the 8B model requires re-embedding the corpus and rebuilding
vector indexes; unchanged endpoint/model IDs do not imply compatible vectors.

### `qwen3-chat` default by mode

| Mode | Default |
| --- | --- |
| `llamacpp` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` |
| `turboquant` | `unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` |
| `qwen` | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:RVN-Q4_K_M-multilingual-mtp.gguf` (text-only llama.cpp serving at 262,144 tokens with Q8_0 KV cache, native prompt caching, and the embedded MTP head at draft depth two) |
| `spiritbuun` | target `unsloth/Qwen3.6-27B-GGUF:Q5_K_M`, draft `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` |
| `lucebox` | target `unsloth/Qwen3.6-27B-GGUF:Q4_K_M`, draft `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` |

`glm5.3-flash` does not expose `qwen3-chat`; its chat model is `glm53-chat`.

## Endpoint matrix

Read this table first if choosing route by task or by mode.

| Endpoint | `llamacpp` | `turboquant` | `qwen` | `spiritbuun` | `lucebox` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `GET /health` | ✅ | ✅ | ✅ | ✅ | ✅ | Plain-text health check |
| `GET /v1/models` | ✅ | ✅ | ✅ | ✅ | ✅ | Lists configured model IDs |
| `POST /v1/chat/completions` | ✅ | ✅ | ✅ | ✅ | ✅ | Main OpenAI-compatible chat route |
| `POST /v1/messages` | ❌ | ❌ | ❌ | ❌ | ✅ | Anthropic-style messages route |
| `POST /v1/completions` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-chat` |
| `POST /v1/responses` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-chat` |
| `POST /v1/embeddings` | ✅ | ✅ | ✅ | ✅ | ✅ | Use `qwen3-embeddings` |
| `POST /v1/rerank` | ❌ | ❌ | ✅ | ❌ | ❌ | Qwen mode: use `qwen3-reranker` |

The `glm5.3-flash` column is omitted from the matrix on purpose: it is served by the FreeToken runtime, not llama.cpp, and natively exposes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/responses`, plus `GET /v1/stats`. It has no embeddings or reranking routes.

Important:

- `POST /v1/messages` is specific to `lucebox` in the llama.cpp family; Spiritbuun launches its upstream llama-server without a project-owned messages adapter. The `glm5.3-flash` (FreeToken) mode also serves `POST /v1/messages` natively.
- The `qwen` profile keeps GPU embeddings and the BGE reranker together in a non-swapping search group. Chat and search groups are mutually exclusive, so the auxiliary models unload when a large chat request needs the GPU. Both search models inherit the 30-minute `globalTTL`: per-model `ttl: 0` overrides were removed because they stopped llama-swap from ever swapping the auxiliary models in. The proxy owns their start, health-check and evict lifecycle.
- Qwen thinking is controlled by `/chat_template/qwen3.8.jinja` through `reasoning_effort`: native values are `low`, `medium`, and `xhigh`; clients with additional level names must map them to those values.
- Only Qwen mode ships a reranker. Its `qwen3-reranker` identifier denotes the Qwen-mode endpoint, not a Qwen-family model: the weights are BGE reranker v2 M3 Q8_0. Send `model`, `query`, `documents` and optional `top_n`; results include document indices and relevance scores. Use the authenticated proxy rather than private native/lifecycle ports.

## Fast smoke tests

Use these after startup, rebuild, config edits, or backend changes.

### Health

```bash
curl -sS ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/health
```

### List models

```bash
curl -sS "${AUTH[@]}" ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/v1/models | jq -r '.data[].id'
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
  ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/v1/chat/completions | jq
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
  ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/v1/messages | jq
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
  ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/v1/responses | jq
```

### Embeddings

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embeddings",
    "input": "local llama embeddings smoke test"
  }' \
  ${EASYLLAMA_BASE_URL:-http://127.0.0.1:8080}/v1/embeddings | jq
```

## Which endpoint to use

Shortest route map for common client tasks.

| Goal | Endpoint | Model |
| --- | --- | --- |
| Chat | `POST /v1/chat/completions` | `qwen3-chat` |
| Messages-style chat (`lucebox`) | `POST /v1/messages` | `qwen3-chat` |
| Plain completion / rewrite | `POST /v1/completions` or `POST /v1/responses` | `qwen3-chat` |
| Embeddings | `POST /v1/embeddings` | `qwen3-embeddings` |
