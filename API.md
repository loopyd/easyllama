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
- [Reader path](#reader-path)

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
| `qwen3-chat` | Primary chat model | mode-dependent |
| `qwen3-embeddings` | Dense embeddings | `Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M` |
| `qmd-generate` | Query expansion and text generation | `tobil/qmd-query-expansion-1.7B-gguf:Q8_0` |
| `qmd-embed` | Embedding alias for QMD flows | `Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M` |
| `qmd-rerank` | Cross-encoder reranker | `mradermacher/Qwen3-Reranker-8B-GGUF:Q5_K_M` |

### `qwen3-chat` default by mode

| Mode | Default |
| --- | --- |
| `basic` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` |
| `turboquant` | `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive:Q5_K_P` |
| `spiritbuun` | target `unsloth/Qwen3.6-27B-GGUF:Q5_K_M`, draft `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` |
| `lucebox` | target `unsloth/Qwen3.6-27B-GGUF:Q4_K_M`, draft `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` |

## Endpoint matrix

Read this table first if choosing route by task or by mode.

| Endpoint | `basic` | `turboquant` | `spiritbuun` | `lucebox` | Notes |
| --- | --- | --- | --- | --- | --- |
| `GET /health` | ✅ | ✅ | ✅ | ✅ | Plain-text health check |
| `GET /v1/models` | ✅ | ✅ | ✅ | ✅ | Lists configured model IDs |
| `POST /v1/chat/completions` | ✅ | ✅ | ✅ | ✅ | Main OpenAI-compatible chat route |
| `POST /v1/messages` | ❌ | ❌ | ✅ | ✅ | Anthropic-style messages route; proxied in `spiritbuun` |
| `POST /v1/completions` | ✅ | ✅ | ✅ | ✅ | Good fit for `qmd-generate` |
| `POST /v1/responses` | ✅ | ✅ | ✅ | ✅ | Good fit for `qmd-generate` |
| `POST /v1/embeddings` | ✅ | ✅ | ✅ | ✅ | Use `qwen3-embeddings` or `qmd-embed` |
| `POST /v1/rerank` | ✅ | ✅ | ✅ | ✅ | Use `qmd-rerank` |
| `GET /ui/` | ✅ | ✅ | ✅ | ✅ | Built-in `llama-swap` UI |

Important:

- `qmd-rerank` is rerank-only.
- `qmd-generate` is generation model for `/v1/completions` and `/v1/responses`.
- `POST /v1/messages` exists in `spiritbuun` and `lucebox`.

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

### Messages (`spiritbuun`, `lucebox`)

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
    "model": "qmd-generate",
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

### Rerank

```bash
curl -sS "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qmd-rerank",
    "query": "best local reranker for qmd search",
    "documents": [
      "Qwen3 Reranker 8B is cross-encoder reranker served through /v1/rerank.",
      "Qwen3 Embeddings 8B creates vectors for retrieval, not pairwise reranking.",
      "QMD Query Expansion rewrites search prompts before retrieval and reranking."
    ]
  }' \
  http://127.0.0.1:8080/v1/rerank | jq
```

## Which endpoint to use

Shortest route map for common client tasks.

| Goal | Endpoint | Model |
| --- | --- | --- |
| Chat | `POST /v1/chat/completions` | `qwen3-chat` |
| Messages-style chat (`spiritbuun`, `lucebox`) | `POST /v1/messages` | `qwen3-chat` |
| Plain completion / rewrite | `POST /v1/completions` or `POST /v1/responses` | `qmd-generate` |
| Embeddings | `POST /v1/embeddings` | `qwen3-embeddings` or `qmd-embed` |
| Reranking | `POST /v1/rerank` | `qmd-rerank` |

## Reader path

- Setup and runtime: [README.md](README.md)
- Release history and upgrade notes: [CHANGELOG.md](CHANGELOG.md)

Use examples in this file as smoke tests after config or runtime changes.
